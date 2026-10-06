"""rlkit.v8.blocks — 区間と兄弟の軌跡の揺らし方（指示書 第 7 章 L-2〜L-4）。

区間（L-3）: テナント（2）× 曜日（7）× 時間帯（0〜7・7〜10・10〜13・13〜16・16〜19・19〜24 時の 6）の 84 区間。
  区間には、そのテナントのゾーンの次元（割合・リモコン・設定温度・風量）、テナントの次元（換気・バイパス）、
  ゾーンがすべてそのテナントに属する系統の次元（蒸発温度）を含める。
  割り当て: 固定の乱数種で並べ替えた 84 区間の周回。毎更新、次の 12 区間を兄弟 0〜11 に割り当てる（7 更新で一巡）。
揺らし方（L-4）: 兄弟の半分（6 本）は「まばら」、残り（6 本）は「まとまり」。
  まばら: 区間の中の各判断を、確率 q = 0.03 でその次元の（有効な）選択肢から一様に選び直す。
  まとまり: 区間に含まれる行動の種類を 1 つ一様に選び、その種類の選択肢から値を 1 つ一様に選び、区間の中の単位（ゾーン・テナント・系統）
    ごとに確率 p = 0.5 で、その単位の区間中のその種類の判断をすべてその値にする。
  区間の外では、その時点の状態での確率最大の選択（乱数は使わない。MUST-4）。
  乱数は（走行の種、更新番号、組番号）から決まる種で作り、記録する。
どちらの揺らし方の兄弟になるかは、更新の中の位置 j と周回の番号 c = (更新 − 1) // 7 で (j + c) が奇数なら「まとまり」とする
（6 本ずつ。周回ごとに入れ替わり、各区間が両方の揺らし方を経験する。指示書に細目が無いため置いた前提）。
"""
import datetime

import numpy as np

from rlkit import constants, timing
from rlkit.v8 import action as act8

WINDOW_EDGES = [0, 7, 10, 13, 16, 19, 24]
N_WINDOWS = 6
N_DAYS = 7
N_BLOCKS = 2 * N_DAYS * N_WINDOWS
N_SIBLINGS = 12
UPDATES_PER_CYCLE = N_BLOCKS // N_SIBLINGS            # 7
SPARSE_Q = 0.03
COHERENT_P = 0.5
EPISODE_START = datetime.datetime(1999, 7, 21, 0, 0)
ACTION_TYPES = ("duty", "setpoint", "fan", "remote", "vent", "bypass", "evap")
TYPE_GROUP = {"duty": "zone_duty", "setpoint": "zone_setpoint", "fan": "zone_fan", "remote": "zone_remote",
              "vent": "tenant_vent", "bypass": "tenant_bypass", "evap": "system_evap"}


def window_of_hour(hour):
    for w in range(N_WINDOWS):
        if WINDOW_EDGES[w] <= hour < WINDOW_EDGES[w + 1]:
            return w
    raise ValueError(hour)


def step_datetimes(start=EPISODE_START, n=constants.ZONE_STEPS_PER_WEEK):
    """378 ステップの開始時刻（二段の時間刻み。実際の夏季の暦）。"""
    out, dt = [], start
    for _ in range(n):
        out.append(dt)
        dt = dt + datetime.timedelta(seconds=timing.zone_step_seconds(dt))
    return out


def block_id(tenant, day, window):
    return (tenant * N_DAYS + day) * N_WINDOWS + window


def block_parts(b):
    t, rest = divmod(b, N_DAYS * N_WINDOWS)
    d, w = divmod(rest, N_WINDOWS)
    return t, d, w


def block_steps_table(datetimes=None):
    """区間ごとの、含まれるステップ番号の配列（84 本）。各ステップは、2 つのテナントに 1 つずつ（計 2 区間）に属する。"""
    datetimes = datetimes or step_datetimes()
    table = [[] for _ in range(N_BLOCKS)]
    for i, dt in enumerate(datetimes):
        d, w = dt.weekday(), window_of_hour(dt.hour)
        for t in range(2):
            table[block_id(t, d, w)].append(i)
    return [np.asarray(x, dtype=np.int64) for x in table]


def units_of_type(tenant, kind):
    """そのテナントの区間に含まれる、行動の種類 kind の次元の番号（行動ベクトルの中の位置）の一覧。"""
    start = act8.LAYOUT[TYPE_GROUP[kind]][0]
    if kind in ("duty", "setpoint", "fan", "remote"):
        return [start + k for k in range(act8.N_ZONES) if act8.ZONE_TENANT[k] == tenant]
    if kind in ("vent", "bypass"):
        return [start + tenant]
    return [start + s for s in range(act8.N_SYSTEMS) if act8.SYSTEM_TENANT[s] == tenant]


def block_dims(tenant):
    """そのテナントの区間に含まれる全次元（昇順）。"""
    return sorted(d for kind in ACTION_TYPES for d in units_of_type(tenant, kind))


def block_order(seed):
    """84 区間の周回の順序（固定の乱数種で並べ替える）。"""
    return np.random.default_rng([seed, 84]).permutation(N_BLOCKS)


def assignment(update, order):
    """更新 u（1 起点）の、兄弟 0〜11 の区間。次の 12 区間を、周回の順序から取る。"""
    return [int(order[((update - 1) * N_SIBLINGS + j) % N_BLOCKS]) for j in range(N_SIBLINGS)]


def sibling_kind(update, position):
    c = (update - 1) // UPDATES_PER_CYCLE
    return "coherent" if (position + c) % 2 == 1 else "sparse"


def sibling_seed(run_seed, update, slot):
    return [int(run_seed), int(update), int(slot)]


class Perturber:
    """兄弟 1 本の、区間の中の揺らし方。apply() は区間の中のステップで、確率最大の行動を揺らした行動を返し、変えた判断を changes に残す。

    L-4R（V8.2-S）: 空振りの防止。
      まばら: 区間の中の各判断を確率 q で選び直す。選び直された判断が 1 つも無い場合は、区間の中の判断を 1 つ一様に選び（ステップと次元の
        組の乱数の順列の先頭から、そのステップで選択肢が 2 つ以上ある最初のもの）、確率最大以外の有効な選択肢から一様に選び直す。
      まとまり: 単位が 1 つも選ばれなかった場合は、単位を 1 つ一様に選ぶ。
    乱数（選び直しの有無と値、順列）は、区間の始めに決まる種から先に引いておく（走行中のマスクや行動に依存しない）。
    """

    def __init__(self, block, kind, seed, steps_of_block):
        self.block = int(block)
        self.kind = kind
        self.seed = list(seed)
        self.tenant, self.day, self.window = block_parts(self.block)
        self.steps = np.asarray(steps_of_block, dtype=np.int64)
        self._step_pos = {int(s): i for i, s in enumerate(self.steps)}
        self.dims = block_dims(self.tenant)
        self.rng = np.random.default_rng(self.seed)
        self.changes = []                                     # (ステップ, 次元, 元の選択, 新しい選択)
        self.coherent = None
        self.fallback = None                                  # L-4R: 空振りの防止が働いたか、その候補
        self._fallback_done = False
        if kind == "coherent":
            ktype = ACTION_TYPES[int(self.rng.integers(len(ACTION_TYPES)))]
            group = TYPE_GROUP[ktype]
            value = int(self.rng.integers(act8.GROUP_K[group]))
            units = units_of_type(self.tenant, ktype)
            picked = [d for d in units if self.rng.random() < COHERENT_P]
            if not picked:                                    # L-4R: 単位が 1 つも選ばれなかったら、単位を 1 つ一様に選ぶ
                picked = [units[int(self.rng.integers(len(units)))]]
                self.fallback = {"kind": "coherent_unit", "unit": picked[0]}
            self.coherent = {"type": ktype, "value": value, "units": picked, "all_units": units}
        elif kind == "sparse":
            n_steps, n_dims = len(self.steps), len(self.dims)
            self._draws = self.rng.random((n_steps, n_dims))
            self._picks = self.rng.random((n_steps, n_dims))
            self._redraw = self._draws < SPARSE_Q
            if not self._redraw.any():                        # L-4R: 選び直しが 1 つも無い → 順列の先頭の判断を 1 つ選び直す
                order = self.rng.permutation(n_steps * n_dims)
                self._candidates = [(int(self.steps[i // n_dims]), self.dims[i % n_dims], float(self._picks[i // n_dims, i % n_dims]))
                                    for i in order]
                self._cand_ptr = 0
                self.fallback = {"kind": "sparse_one_decision", "first_candidate": list(self._candidates[0][:2])}
            else:
                self._candidates = None
        else:
            raise ValueError(kind)

    def in_block(self, step):
        return int(step) in self._step_pos

    def apply(self, step, action, flat_mask):
        """action (80,) は確率最大の行動、flat_mask (372,) はこのステップのマスク。区間の外・区間の中の変更なしは、そのまま返す。"""
        out = np.array(action, dtype=np.int64, copy=True)
        if not self.in_block(step):
            return out
        if self.kind == "sparse":
            i = self._step_pos[int(step)]
            if self._candidates is None:
                for j, d in enumerate(self.dims):
                    if self._redraw[i, j]:
                        o, n = int(act8.OFFSETS[d]), int(act8.NVEC[d])
                        valid = np.flatnonzero(flat_mask[o:o + n])
                        new = int(valid[min(int(self._picks[i, j] * len(valid)), len(valid) - 1)])
                        if new != out[d]:
                            self.changes.append((int(step), int(d), int(out[d]), new))
                            out[d] = new
            elif not self._fallback_done:
                while self._cand_ptr < len(self._candidates):
                    cs, d, pick = self._candidates[self._cand_ptr]
                    if cs < int(step):                        # 過ぎた候補（無効だった）は飛ばす
                        self._cand_ptr += 1
                        continue
                    if cs > int(step):                        # 順列で先の候補が、まだ来ていない
                        break
                    o, n = int(act8.OFFSETS[d]), int(act8.NVEC[d])
                    others = [int(v) for v in np.flatnonzero(flat_mask[o:o + n]) if int(v) != int(out[d])]
                    self._cand_ptr += 1
                    if others:                                # 確率最大以外の有効な選択肢から一様に
                        new = others[min(int(pick * len(others)), len(others) - 1)]
                        self.changes.append((int(step), int(d), int(out[d]), new))
                        out[d] = new
                        self._fallback_done = True
                        self.fallback["applied"] = [int(step), int(d)]
                        break
        else:
            c = self.coherent
            for d in c["units"]:
                o, n = int(act8.OFFSETS[d]), int(act8.NVEC[d])
                if flat_mask[o + c["value"]] and out[d] != c["value"]:
                    self.changes.append((int(step), int(d), int(out[d]), c["value"]))
                    out[d] = c["value"]
        return out

    def record(self):
        rec = {"block": self.block, "tenant": self.tenant, "day": self.day, "window": self.window, "kind": self.kind,
               "seed": self.seed, "n_steps": int(len(self.steps)), "n_changed": len(self.changes),
               "changes": [list(c) for c in self.changes], "fallback": self.fallback}
        if self.coherent is not None:
            rec["coherent"] = {"type": self.coherent["type"], "value": self.coherent["value"],
                               "units": self.coherent["units"], "n_units_available": len(self.coherent["all_units"])}
        return rec
