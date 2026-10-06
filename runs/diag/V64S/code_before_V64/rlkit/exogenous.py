"""rlkit.exogenous — 外生表の読み出し（v2 第 5-2 節「先読みは確定値方式を維持する」）。

verify/exogenous/summer.npz（1 分刻み・1 週間 = 10,080 要素。20_Shizuku2-RL-kit/verify から複製した
現行の外生表）を読み、任意の時刻・地平の在室者数と気象値を返す。キーは datetime, zone_labels,
zone_occupants, total_occupants, outdoor_drybulb, outdoor_rh, outdoor_abs_humidity の 7 つで、
日射量・夜間放射量の将来値は含まれない（現行実装と同じ制約。観測側で現在値を使う）。

v2 で加えたもの: M-5 の「2 時間先までの在室見込み」、観測 +27「次の在室開始までの時間」、
+28「直前の在室終了からの時間」。いずれもゾーン在室者数 > 0 を在室とみなし、1 分刻みで判定する。
"""
import datetime as _dt
import os

import numpy as np

from rlkit import constants

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXOGENOUS_PATH = os.path.join(_REPO_ROOT, "verify", "exogenous", "summer.npz")

_CACHE = {}


class ExogenousData:
    """夏季 1 週間ぶんの外生表。"""

    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.datetimes = [_dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in d["datetime"]]
        self.zone_labels = [str(s) for s in d["zone_labels"]]
        if self.zone_labels != constants.ZONE_LABELS:
            raise ValueError(f"外生表のゾーンの並びが constants.ZONE_LABELS と異なる: {self.zone_labels}")
        self.zone_occupants = d["zone_occupants"].astype(np.float64)  # (10080, 18)
        self.total_occupants = d["total_occupants"].astype(np.float64)  # (10080,)
        self.outdoor_drybulb = d["outdoor_drybulb"].astype(np.float64)
        self.outdoor_rh = d["outdoor_rh"].astype(np.float64)
        self.outdoor_abs_humidity = d["outdoor_abs_humidity"].astype(np.float64)
        self.t0 = self.datetimes[0]
        self.n = len(self.datetimes)

        occupied = self.zone_occupants > 0
        idx = np.arange(self.n)[:, None]
        no_next = 2 * self.n
        # 各時刻 i 以降で最初に在室する分の索引（無ければ no_next）
        self._next_occupied = np.minimum.accumulate(np.where(occupied, idx, no_next)[::-1], axis=0)[::-1]
        # 各時刻 i 以前で最後に在室した分の索引（無ければ -1）
        self._prev_occupied = np.maximum.accumulate(np.where(occupied, idx, -1), axis=0)
        # 区間和で「区間内に在室があるか」を判定するための累積（先頭に 0 行を置く）
        self._occupied_cumsum = np.vstack([np.zeros((1, constants.N_ZONES), dtype=np.int64),
                                           np.cumsum(occupied.astype(np.int64), axis=0)])
        self._no_next = no_next

    def _index_for(self, dt):
        """dt の分刻みインデックス（切り捨て、範囲外はクランプ）。現行と同じ。"""
        minutes = int((dt - self.t0).total_seconds() // 60)
        return min(max(minutes, 0), self.n - 1)

    def zone_occupants_at(self, dt):
        return self.zone_occupants[self._index_for(dt)].copy()

    def total_occupants_at(self, dt):
        return float(self.total_occupants[self._index_for(dt)])

    def weather_at(self, dt):
        i = self._index_for(dt)
        return {
            "drybulb_c": float(self.outdoor_drybulb[i]),
            "relative_humidity_pct": float(self.outdoor_rh[i]),
            "absolute_humidity_gkg": float(self.outdoor_abs_humidity[i]),
        }

    def at_horizon(self, dt, hours):
        """dt + hours 時点の在室者数・気象値（現行 exogenous.at_horizon と同じ）。"""
        future_dt = dt + _dt.timedelta(hours=hours)
        out = {"datetime": future_dt,
               "zone_occupants": self.zone_occupants_at(future_dt),
               "total_occupants": self.total_occupants_at(future_dt)}
        out.update(self.weather_at(future_dt))
        return out

    def total_occupant_minutes(self):
        """1 週間の建物全体の在室者延べ数 W [人・分]。"""
        return float(np.sum(self.total_occupants))

    def zone_occupied_within(self, dt, hours):
        """(dt, dt + hours] の分に在室者のいるゾーンなら真（18,）。"""
        i = self._index_for(dt)
        j = min(i + int(round(hours * 60)), self.n - 1)
        if j <= i:
            return np.zeros(constants.N_ZONES, dtype=bool)
        return (self._occupied_cumsum[j + 1] - self._occupied_cumsum[i + 1]) > 0

    def hours_until_next_occupancy(self, dt):
        """次の在室開始までの時間[h]（18,）。在室中は 0、表の終わりまで在室が無ければ inf。"""
        i = self._index_for(dt)
        nxt = self._next_occupied[i]
        return np.where(nxt >= self._no_next, np.inf, (nxt - i) / 60.0)

    def hours_since_last_occupancy_end(self, dt):
        """直前の在室終了からの時間[h]（18,）。在室中は 0、表の先頭から在室が無ければ inf。"""
        i = self._index_for(dt)
        prv = self._prev_occupied[i]
        elapsed = np.where(prv == i, 0.0, (i - (prv + 1)) / 60.0)
        return np.where(prv < 0, np.inf, elapsed)


def load(path=None):
    """夏季の外生表を読み込み、プロセス内にキャッシュする。"""
    path = path or EXOGENOUS_PATH
    if path not in _CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"外生表が見つからない: {path}")
        _CACHE[path] = ExogenousData(path)
    return _CACHE[path]
