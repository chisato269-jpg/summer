"""rlkit.observation — 観測ベクトル 642 次元の構成（v2 第 5-2 節、指示書 V6-S 第 3-4 節）。

全体ブロック 84（索引 0–83）＋ ゾーンブロック 31 × 18（基点 84 + 31k、k は constants.ZONE_DEFS の順）。
V6-S で全体 80・81（O-1: テナント別の換気の水準 ÷ 3）と 82・83（O-3: テナント別のリモコン許可の状態）を加えた。
設定温度の索引 +19 は新しい格子（20〜28 ℃の 9 択）で正規化する（A-2）。
正規化定数は constants の固定値で、全項目を [0, 1] にクリップする。build() はクリップ前の値も返し、
環境が項目ごとのクリップ発生率を数える（v2 第 5-2-2 節）。
v2 が「維持」「移動」とした項目の中身は、現行 observation.py・env._system_status（夏季）と同じ計算である。
"""
import math

import numpy as np

from rlkit import action as action_mod
from rlkit import constants, psychro, timing

GLOBAL_DIM = constants.GLOBAL_BLOCK_DIM
ZONE_DIM = constants.ZONE_BLOCK_DIM
N_ZONES = constants.N_ZONES
TOTAL_DIM = constants.OBSERVATION_DIM

IDX_CO2 = 40
IDX_SYSTEM = 44
IDX_BOUNDARY = 64
IDX_STEP_LENGTH = 65
IDX_TENANT_THERMAL = 66
IDX_VENT_LEVEL = 80      # O-1: 80 南・81 北（テナント旗 +2 と同じ順）
IDX_REMOTE = 82          # O-3: 82 南・83 北


def zone_base(k):
    return GLOBAL_DIM + ZONE_DIM * k


def _norm_index(idx, n_values):
    return 0.0 if n_values <= 1 else idx / (n_values - 1)


def system_status(decoded, snapshot):
    """系統状態 × 4 の元の値（現行 env._system_status の夏季分と同じ）。

    運転モードは系統の先頭ゾーンの運転状態、冷媒温度は蒸発温度を 15〜22 ℃で正規化、
    負荷率は系統内ゾーンの熱負荷の絶対値の和を a_s で割った値、ファン風向は夏季の行動に無いため 0。
    """
    out = {}
    evap_grid = constants.SYSTEM_EVAPORATING_C
    for ou in range(1, 5):
        labels = constants.SYSTEM_ZONES[ou]
        states = [decoded["zones"][l]["state"] for l in labels]
        evap = decoded["systems"][ou]["evap_c"]
        out[ou] = {
            "mode_idx": constants.ZONE_STATE.index(states[0]),
            "n_states": len(constants.ZONE_STATE),
            "refrig_norm": (evap - evap_grid[0]) / (evap_grid[-1] - evap_grid[0]),
            "n_on": sum(1 for s in states if s != "off"),
            "n_total": len(labels),
            "load_kw": sum(abs(snapshot["zones"][l]["heatload_kw"]) for l in labels),
        }
    return out


class _ZoneRunning:
    def __init__(self):
        self.prev_drybulb_c = None
        self.cumulative_energy_gj = 0.0
        self.cumulative_dissatisfied_occupant = 0.0
        self.last_step_energy_gj = 0.0
        self.last_step_dr = 0.0


class ObservationBuilder:
    """エピソード開始時に reset() し、ステップごとに build() を呼ぶ。"""

    def __init__(self, exo, zone_step_energy_norm_gj):
        if zone_step_energy_norm_gj is None or not zone_step_energy_norm_gj > 0:
            raise ValueError("未確定項目 J-5: 観測 +25 の正規化定数（1 ステップのゾーン一次エネルギーの最大値[GJ]）が"
                             "設定されていない。rlkit/diag/j1_j5_summer.py の測定値を config に入れること")
        self.exo = exo
        self.energy_norm_gj = float(zone_step_energy_norm_gj)
        self.T = constants.ZONE_STEPS_PER_WEEK
        self.zones = [_ZoneRunning() for _ in range(N_ZONES)]

    def reset(self):
        self.zones = [_ZoneRunning() for _ in range(N_ZONES)]

    def build(self, dt, step_index, stage, snapshot, decoded, zone_step_energy_gj=None, zone_step_dr=None,
              tenant_thermal=None, setpoint_readback=None, remote_status=None):
        """1 ステップぶんの観測を作る。

        Args:
            dt: このステップ開始時刻（直前のステップを進めた後の日時）
            step_index: 0 起点のステップ番号（エピソード進行率に使う）
            stage: 行動の段階（設定温度の格子）
            snapshot: comm.read_step_snapshot() の戻り値
            decoded: 直前に適用した行動の action.decode_action()
            zone_step_energy_gj: 直前ステップのゾーン一次エネルギー e_k,t−1 (18,)。リセット直後は None
            zone_step_dr: 直前ステップの合成不満足率 DR_k,t−1 (18,)。リセット直後は None
            tenant_thermal: comm.read_tenant_thermal_sensation() の戻り値
            setpoint_readback: comm.read_setpoint_readback() の戻り値 (18,)
            remote_status: テナント別のリモコン許可の状態 {テナント: 0〜1}（1114 の読み戻しのテナント平均）
        Returns:
            (obs: float32 (642,) クリップ後、raw: float64 (642,) クリップ前)
        """
        raw = np.zeros(TOTAL_DIM, dtype=np.float64)

        # ---- 全体ブロック 0–79 ----
        raw[dt.weekday()] = 1.0
        theta = 2.0 * math.pi * (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
        raw[7] = (math.sin(theta) + 1.0) / 2.0
        raw[8] = (math.cos(theta) + 1.0) / 2.0
        raw[9] = 1.0 if dt.weekday() < 5 else 0.0
        raw[10] = step_index / self.T  # 指示書 V3-S 第 2-4 節（旧 step/(T−1) は最終ステップで 1 を超えた）

        b = snapshot["building"]
        raw[11] = b["drybulb_c"] / constants.OUTDOOR_TEMP_NORM_C
        raw[12] = psychro.relative_to_absolute_humidity(b["drybulb_c"], b["relative_humidity_pct"]) / constants.ABS_HUMIDITY_NORM_GKG
        raw[13] = b["relative_humidity_pct"] / constants.RELATIVE_HUMIDITY_NORM_PCT
        raw[14] = b["global_horizontal_radiation"] / constants.GLOBAL_RADIATION_NORM_WM2
        raw[15] = b["nocturnal_radiation"] / constants.NOCTURNAL_RADIATION_NORM_WM2

        idx = 16
        for h in constants.FORECAST_HORIZONS_H:
            fc = self.exo.at_horizon(dt, h)
            raw[idx] = fc["drybulb_c"] / constants.OUTDOOR_TEMP_NORM_C
            raw[idx + 1] = fc["absolute_humidity_gkg"] / constants.ABS_HUMIDITY_NORM_GKG
            raw[idx + 2] = raw[14]  # 日射量の将来値は外生表に無いため現在値（現行と同じ）
            idx += 3
        for h in constants.OCCUPANT_FORECAST_HORIZONS_H:
            raw[idx] = self.exo.at_horizon(dt, h)["total_occupants"] / constants.BUILDING_OCCUPANT_NORM
            idx += 1
        assert idx == IDX_CO2

        # 40–43: 南北テナントの CO2 濃度と次ステップ予測（予測は現在値。現行の旧 47–50 と同じ）
        raw[40] = snapshot["co2_ppm"]["South"] / constants.CO2_NORM_PPM
        raw[41] = snapshot["co2_ppm"]["North"] / constants.CO2_NORM_PPM
        raw[42] = raw[40]
        raw[43] = raw[41]

        # 44–63: 系統状態 × 4（現行の旧 51–70 と同じ）
        idx = IDX_SYSTEM
        for ou, st in system_status(decoded, snapshot).items():
            raw[idx] = _norm_index(st["mode_idx"], st["n_states"])
            raw[idx + 1] = st["refrig_norm"]
            raw[idx + 2] = st["n_on"] / st["n_total"]
            # 負荷率の正規化母数は系統内の室内機容量の合計（指示書 V3-S 第 2-4 節。旧 a_s）
            raw[idx + 3] = st["load_kw"] / constants.SYSTEM_CAPACITY_KW[ou - 1]
            raw[idx + 4] = 0.0  # 夏季はファン風向の行動が無い（現行と同じ）
            idx += 5
        assert idx == IDX_BOUNDARY

        raw[IDX_BOUNDARY] = 1.0 if timing.is_system_boundary(dt) else 0.0
        raw[IDX_STEP_LENGTH] = timing.zone_step_seconds(dt) / constants.STEP_LENGTH_NORM_SEC

        # 66–79: テナント別（南 66–72、北 73–79）の温冷感 7 段階の人数 ÷ テナント在室者数。無人時 0
        n_levels = len(constants.THERMAL_SENSATION_LEVELS)
        if tenant_thermal is not None:
            for ti, tenant in enumerate(constants.TENANTS):
                d = tenant_thermal[tenant]
                if d["n_present"] > 0:
                    raw[IDX_TENANT_THERMAL + ti * n_levels:IDX_TENANT_THERMAL + (ti + 1) * n_levels] = \
                        np.asarray(d["counts"], dtype=np.float64) / d["n_present"]

        # 80–81: O-1 実際に適用した換気の水準（マスク後の行動）÷ 3。82–83: O-3 リモコン許可の状態（許可 1、禁止 0）
        n_vent = len(constants.VENT_LEVELS) - 1
        for ti, tenant in enumerate(action_mod.VENT_TENANT_ORDER):
            raw[IDX_VENT_LEVEL + ti] = decoded["tenants"][tenant] / n_vent
            if remote_status is not None:
                raw[IDX_REMOTE + ti] = float(remote_status[tenant])

        # ---- ゾーンブロック ----
        grid = action_mod.setpoint_grid(stage)
        rank = self._thermal_rank([snapshot["zones"][l]["dissatisfied_thermal"] for l in constants.ZONE_LABELS])
        until_next = self.exo.hours_until_next_occupancy(dt)
        since_end = self.exo.hours_since_last_occupancy_end(dt)
        zone_fc = [self.exo.at_horizon(dt, h)["zone_occupants"] for h in constants.ZONE_OCCUPANT_FORECAST_HORIZONS_H]

        for k, (ou, iu, tenant, zn, label, area) in enumerate(constants.ZONE_DEFS):
            base = zone_base(k)
            z = snapshot["zones"][label]
            zs = self.zones[k]
            act = decoded["zones"][label]

            raw[base + 0] = area / constants.ZONE_AREA_NORM_M2
            raw[base + 1] = constants.INDOOR_UNIT_CAPACITY_KW[label] / constants.ZONE_CAPACITY_NORM_KW
            raw[base + 2] = 0.0 if tenant == "South" else 1.0
            raw[base + 3] = (ou - 1) / 3.0

            raw[base + 4] = z["drybulb_c"] / constants.OUTDOOR_TEMP_NORM_C
            raw[base + 5] = z["relative_humidity_pct"] / constants.RELATIVE_HUMIDITY_NORM_PCT
            raw[base + 6] = psychro.relative_to_absolute_humidity(z["drybulb_c"], z["relative_humidity_pct"]) / \
                constants.ABS_HUMIDITY_NORM_GKG
            raw[base + 7] = (z["thermal_sensation"] - constants.THERMAL_SENSATION_MIN) / \
                (constants.THERMAL_SENSATION_MAX - constants.THERMAL_SENSATION_MIN)
            raw[base + 8] = z["clothing_index"] / constants.CLOTHING_INDEX_NORM_CLO

            raw[base + 9] = z["dissatisfied_thermal"]
            raw[base + 10] = z["dissatisfied_draft"]
            raw[base + 11] = z["dissatisfied_vtmp"]

            raw[base + 12] = z["occupant_number"] / constants.ZONE_OCCUPANT_NORM
            for j in range(len(constants.ZONE_OCCUPANT_FORECAST_HORIZONS_H)):
                raw[base + 13 + j] = zone_fc[j][k] / constants.ZONE_OCCUPANT_NORM

            raw[base + 16 + constants.ZONE_STATE.index(act["state"])] = 1.0
            sp_idx = grid.index(act["setpoint"]) if act["setpoint"] is not None else 0
            raw[base + 19] = _norm_index(sp_idx, len(grid))
            raw[base + 20] = _norm_index(constants.ZONE_FAN_SPEED.index(act["fan_speed"]), len(constants.ZONE_FAN_SPEED))

            if zs.prev_drybulb_c is None:
                raw[base + 21] = 0.5
            else:
                delta = z["drybulb_c"] - zs.prev_drybulb_c
                raw[base + 21] = (delta + constants.TEMP_DELTA_HALF_RANGE_C) / (2.0 * constants.TEMP_DELTA_HALF_RANGE_C)
            zs.prev_drybulb_c = z["drybulb_c"]

            if zone_step_energy_gj is not None:
                zs.last_step_energy_gj = float(zone_step_energy_gj[k])
                zs.cumulative_energy_gj += zs.last_step_energy_gj
            raw[base + 22] = zs.cumulative_energy_gj / constants.ZONE_CUM_ENERGY_NORM_GJ
            zs.cumulative_dissatisfied_occupant += z["occupant_number"] * (
                1 - (1 - z["dissatisfied_thermal"]) * (1 - z["dissatisfied_draft"]) * (1 - z["dissatisfied_vtmp"]))
            raw[base + 23] = zs.cumulative_dissatisfied_occupant / constants.ZONE_CUM_DISSATISFIED_OCCUPANT_NORM
            raw[base + 24] = rank[k]

            if zone_step_dr is not None:
                zs.last_step_dr = float(zone_step_dr[k])
            raw[base + 25] = zs.last_step_energy_gj / self.energy_norm_gj
            raw[base + 26] = zs.last_step_dr
            raw[base + 27] = until_next[k] / constants.OCCUPANCY_GAP_NORM_H
            raw[base + 28] = since_end[k] / constants.OCCUPANCY_GAP_NORM_H

            if setpoint_readback is not None:
                rb = setpoint_readback[k]
                # 執務者が行動格子の外へ動かすため、正規化は 16〜32 ℃ を範囲とする（指示書 V3-S 第 2-4 節）
                raw[base + 29] = (rb - constants.SETPOINT_READBACK_MIN_C) / constants.SETPOINT_READBACK_SPAN_C
                if act["state"] != "off" and act["setpoint"] is not None:
                    raw[base + 30] = 1.0 if abs(rb - act["setpoint"]) > constants.SETPOINT_OVERRIDE_TOL_C else 0.0

        obs = np.clip(raw, 0.0, 1.0).astype(np.float32)
        return obs, raw

    @staticmethod
    def _thermal_rank(values):
        """18 ゾーン中の温冷感不満足率の順位 rank/17（昇順。現行と同じ）。"""
        order = sorted(range(len(values)), key=lambda i: values[i])
        n = len(values)
        rank = [0.0] * n
        for r, i in enumerate(order):
            rank[i] = r / (n - 1) if n > 1 else 0.0
        return rank


def clipped_entries(raw):
    """クリップが起きた項目（[0, 1] の外。inf を含む）。"""
    return (raw < 0.0) | (raw > 1.0)
