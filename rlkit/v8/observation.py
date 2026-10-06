"""rlkit.v8.observation — V8-S の観測（指示書 第 6 章 O-1・O-2）。704 次元。

V7 の 648 次元（全体 90 ＋ ゾーン 31 × 18）に、次を足す。
  O-1 時刻の情報。V7 の観測に既にあるもの: 曜日（全体 0〜6）、時刻の周期（全体 7・8）、平日か（全体 9）、エピソードの進行率（全体 10）、
      次の在室の開始までの時間（ゾーン +27）、直前の無人の連続時間（ゾーン +28）、在室者数の予報（全体・ゾーン）。無かったもの:
      前日の在室時間帯のゾーン平均室温（ゾーンブロック +33。直近の在室時間帯の平均。まだ無いあいだは 0）だけを足す。
  O-2 新しい行動の状態。ゾーンごとのリモコンの許可の状態（ゾーン +31。読み戻し）、直前の刻みの運転の割合（ゾーン +32）、
      テナントごとのバイパスの状態（全体 +2: 90 南・91 北）。
全体 92（V7 の 0〜89 + 90・91）、ゾーン 34（V7 の 0〜30 + 31・32・33）。並びは V7 の項目の位置を保つ（中立な初期化の列の対応）。
設定温度の索引（ゾーン +19）は、V7 と同じ正規化 (設定温度 − 22 ℃) ÷ 6 ℃で表す（V7 の格子の値では同じ値、22.5・23.5 ℃も表せる）。
"""
import numpy as np

from rlkit import constants, observation as obs7
from rlkit.v8 import action as act8

GLOBAL_DIM_V7 = constants.GLOBAL_BLOCK_DIM          # 90
ZONE_DIM_V7 = constants.ZONE_BLOCK_DIM              # 31
N_ZONES = constants.N_ZONES
GLOBAL_DIM = GLOBAL_DIM_V7 + 2                      # 92
ZONE_DIM = ZONE_DIM_V7 + 3                          # 34
TOTAL_DIM = GLOBAL_DIM + ZONE_DIM * N_ZONES         # 704
IDX_BYPASS = GLOBAL_DIM_V7                          # 90（南）・91（北）
ZONE_REMOTE, ZONE_PREV_DUTY, ZONE_YESTERDAY_TEMP = ZONE_DIM_V7, ZONE_DIM_V7 + 1, ZONE_DIM_V7 + 2

OBS_LOW = np.zeros(TOTAL_DIM, dtype=np.float64)
OBS_LOW[:GLOBAL_DIM_V7] = obs7.OBS_LOW[:GLOBAL_DIM_V7]          # −1〜1 の項目（CO2 の変化量・予測）は全体ブロックだけ
OBS_HIGH = np.ones(TOTAL_DIM, dtype=np.float64)


def zone_base(k):
    return GLOBAL_DIM + ZONE_DIM * k


def old_zone_base(k):
    return GLOBAL_DIM_V7 + ZONE_DIM_V7 * k


def column_map_v7_to_v8():
    """V7 の観測の位置 → V8 の観測の位置（長さ 648）。中立な初期化で、入力層の列を並べ替えるときに使う。"""
    m = np.arange(constants.OBSERVATION_DIM, dtype=np.int64)
    m[:GLOBAL_DIM_V7] = np.arange(GLOBAL_DIM_V7)
    for k in range(N_ZONES):
        m[old_zone_base(k):old_zone_base(k) + ZONE_DIM_V7] = zone_base(k) + np.arange(ZONE_DIM_V7)
    return m


def clipped_entries(raw):
    return (raw < OBS_LOW) | (raw > OBS_HIGH)


class ObservationBuilderV8(obs7.ObservationBuilder):
    def __init__(self, exo, zone_step_energy_norm_gj):
        super().__init__(exo, zone_step_energy_norm_gj)
        self._reset_v8()

    def _reset_v8(self):
        self._occ_sum = np.zeros(N_ZONES)
        self._occ_n = 0
        self._was_in_window = False
        self._last_occ_mean = np.zeros(N_ZONES)       # 直近の在室時間帯の平均室温 [℃]。まだ無ければ 0

    def reset(self):
        super().reset()
        self._reset_v8()

    @staticmethod
    def _in_occupied_window(dt):
        return dt.weekday() < 5 and constants.OCCUPIED_WEEKDAY_START_HOUR <= dt.hour < constants.OCCUPIED_WEEKDAY_END_HOUR

    def build(self, dt, step_index, stage, snapshot, decoded, zone_step_energy_gj=None, zone_step_dr=None,
              tenant_thermal=None, setpoint_readback=None, remote_status=None, remote_zone=None):
        """V7 の build で 648 次元を作り、V8 の位置へ移して、新しい項目を足す。decoded は v8.action.decode_action の値。
        Returns: (obs float32 (704,) クリップ後, raw float64 (704,) クリップ前)"""
        _o7, raw7 = super().build(dt, step_index, stage, snapshot, decoded, zone_step_energy_gj, zone_step_dr, tenant_thermal,
                                  setpoint_readback, remote_status=None)
        raw = np.zeros(TOTAL_DIM, dtype=np.float64)
        raw[:GLOBAL_DIM_V7] = raw7[:GLOBAL_DIM_V7]
        for k in range(N_ZONES):
            raw[zone_base(k):zone_base(k) + ZONE_DIM_V7] = raw7[old_zone_base(k):old_zone_base(k) + ZONE_DIM_V7]
        # O-2 全体: テナントごとのバイパスの状態
        for j, t in enumerate(act8.TENANT_ORDER):
            raw[IDX_BYPASS + j] = 1.0 if decoded["bypass"][t] else 0.0
        # O-1: 前日（直近）の在室時間帯のゾーン平均室温
        temps = np.array([snapshot["zones"][l]["drybulb_c"] for l in constants.ZONE_LABELS], dtype=np.float64)
        if self._in_occupied_window(dt):
            self._occ_sum += temps
            self._occ_n += 1
            self._was_in_window = True
        elif self._was_in_window:
            if self._occ_n > 0:
                self._last_occ_mean = self._occ_sum / self._occ_n
            self._occ_sum = np.zeros(N_ZONES)
            self._occ_n = 0
            self._was_in_window = False
        for k, (ou, iu, tenant, zn, label, area) in enumerate(constants.ZONE_DEFS):
            base = zone_base(k)
            z = decoded["zones"][label]
            # 設定温度（+19）: V7 と同じ正規化 (設定温度 − 22) ÷ 6。V7 の格子の値では V7 と同じ値になる
            raw[base + 19] = 0.0 if z["setpoint"] is None else (z["setpoint"] - constants.ZONE_SETPOINT_C[1][0]) / (
                constants.ZONE_SETPOINT_C[1][-1] - constants.ZONE_SETPOINT_C[1][0])
            raw[base + ZONE_REMOTE] = 1.0 if (remote_zone is None or remote_zone[k]) else 0.0
            raw[base + ZONE_PREV_DUTY] = z["duty_index"] / (len(act8.DUTY_FRACTIONS) - 1)
            raw[base + ZONE_YESTERDAY_TEMP] = self._last_occ_mean[k] / constants.OUTDOOR_TEMP_NORM_C
        obs = np.clip(raw, OBS_LOW, OBS_HIGH).astype(np.float32)
        return obs, raw
