"""rlkit.env — 夏季の Gymnasium 環境（v2 第 5-1〜5-4 節、指示書 V6-S 第 3 章）。

1 エピソード = 1 回のエミュレータ起動（1 週間、378 ステップ）。現行 rlkit/env.py の駆動・書き込み・
読み出しの手順を維持し、次を v2 に合わせて変えた。
  報酬: 実測電力による按分と、基準点で正規化した報酬（reward.py）。終端補正・報酬スケール c は無い
  観測: 642 次元（observation.py。V6-S O-1・O-3 を含む）
  行動: 単一段階（V6-S A-7）。M-5'・M-9 は旗で適用し、環境は行動を書き換えない
  書き込み: 差分書き込み（A-5）。K-9 結果 B のため発停も差分書き込みのまま
V6-S での変更:
  A-5 換気: 停止はテナントの全熱交換器の発停を Off、Low・Middle・High は風量 1・2・3 を書いてから On。バイパスは無効のまま
  A-6 リモコン許可: テナントの全室内機に同じ値（許可 1、禁止 0）を書く（V5 までの固定値「許可」と同じ関数・同じキー）
  R-6 換気の電力模型: 水準ごと（constants.HEX_POWER_KW_BY_LEVEL）
  M-5' の先読み 12 時間、M-9 を有効化
  読み戻し: リモコン許可の状態（V6-S の観測 82・83。C-5 の後は記録だけ）と発停の状態（第 3-8 節の記録）
V6.2-S C-5: リモコン許可の次元を廃止し、全室内機に許可を固定で書く（V5 と同じ関数・同じキー）。O-3 を観測から除く。
V6.4-S R-8': エピソードの終端で、エミュレータを閉じてから組の分単位の出力（data/）を読み、ステップ別の建物の値
  n_t = (A_t·O_t − A_{t−1}·O_{t−1})/W とゾーンへの配分 n_k,t を作る（rlkit/r8p.py）。終端の info に
  r8p_zone_n (T,18) と、それで計算し直したゾーン別報酬 r8p_zone_rewards (T,18) を入れ、学習器が収益の計算の前に置き換える。
各ステップで info["zone_rewards"]（18）と info["system_reward"] を返す（v2 第 5-5-1 節 手順 1）。
基準点は set_reference() で与える。未設定のあいだ（R-1 の初回評価）は報酬を 0 とする。
"""
import os
import time

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box

from rlkit import action, comm, constants, driver, eci as eci_mod, exogenous, observation, r8p, timing
from rlkit import reward as reward_mod

_VRC = None
_VSC = None


def _szk():
    global _VRC, _VSC
    if _VRC is None:
        from szklib.VRFSystemCommunicator import VRFSystemCommunicator
        from szklib.VentilationSystemCommunicator import VentilationSystemCommunicator
        _VRC, _VSC = VRFSystemCommunicator, VentilationSystemCommunicator
    return _VRC, _VSC


ZONE_INFO = {label: (ou, iu, tenant, zn, area) for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS}
TENANT_ZONE_INDICES = {t: [k for k, z in enumerate(constants.ZONE_DEFS) if z[2] == t] for t in action.VENT_TENANT_ORDER}
KWH_TO_GJ = constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0


def hex_power_kw(levels):
    """R-6: テナント別の換気の水準 {テナント: 0〜3} → ゾーン別の全熱交換器電力 (18,) [kW]。"""
    return np.array([constants.HEX_POWER_KW_BY_LEVEL[area][int(levels[tenant])]
                     for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS], dtype=np.float64)


class SummerEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, stage=1, include_indoor_unit_power=None, zone_step_energy_norm_gj=None, slot=None,
                 m9_enabled=True):
        """include_indoor_unit_power: 未確定項目 J-1。室内機電力を按分に別建てで加えるか。
        zone_step_energy_norm_gj: 未確定項目 J-5。観測 +25 の正規化定数。
        slot: rlkit.parallel.slots.SlotSpec（None は原本のエミュレータ）。"""
        super().__init__()
        if include_indoor_unit_power is None:
            raise ValueError("未確定項目 J-1: 室内機電力を按分に加えるか（include_indoor_unit_power）が設定されていない。"
                             "rlkit/diag/j1_j5_summer.py の判定結果を config に入れること")
        self.include_indoor_unit_power = bool(include_indoor_unit_power)
        self.slot = slot
        self.exo = exogenous.load()
        self.T = constants.ZONE_STEPS_PER_WEEK
        self.W = self.exo.total_occupant_minutes()
        self.observation_space = Box(low=observation.OBS_LOW.astype(np.float32), high=observation.OBS_HIGH.astype(np.float32),
                                     dtype=np.float32)
        self._ob = observation.ObservationBuilder(self.exo, zone_step_energy_norm_gj)
        self.reference = None
        self._reward_calc = None
        self._driver = None
        self._dw = None
        self.m5_enabled = True                    # M-5'（V6 では解除しない）
        self.m9_enabled = bool(m9_enabled)        # M-9（V6-S で有効）
        self.diag_info = False                    # 真なら info に内訳を足す（採取では重いので既定は偽）
        # K-13 (a)（指示書 V6.1-S 第 4 章、診断専用）: 書き込みの前に設定温度（1106）を読み戻し、冷房を指令している
        # ゾーンで読み戻しが指令値と違うときだけ設定温度を書き直す。学習では使わない
        self.setpoint_rewrite_on_mismatch = False
        self.stage = None
        self.set_stage(stage)
        self._reset_episode_state()

    # ---- 学習側から設定する値 ----

    def set_stage(self, stage):
        action.setpoint_grid(stage)
        if self.stage == 1 and stage == 2 and getattr(self, "_prev_action", None) is not None:
            self._prev_action = action.convert_action_stage1_to_stage2(self._prev_action)
        self.stage = stage
        self.action_space = action.action_space(stage)
        return stage

    def set_reference(self, reference_dict):
        """基準（α・β・E*・DR* を持つ辞書。V6.1-S の R-7 では α = 1、β = ρ、E* = E_ref、DR* = DR_ref）。
        次の reset 以降のステップの報酬に使う。"""
        self.reference = reward_mod.RewardReference.from_dict(reference_dict) if reference_dict else None
        self._reward_calc = reward_mod.RewardCalculator(self.reference, self.T, self.W) if self.reference else None
        return True

    def set_setpoint_rewrite_on_mismatch(self, enabled):
        """K-13 (a) の道具（診断専用）。"""
        self.setpoint_rewrite_on_mismatch = bool(enabled)
        return self.setpoint_rewrite_on_mismatch

    def set_diag_info(self, enabled):
        """info に電力と不満足率の内訳を足すか（診断・試験用）。"""
        self.diag_info = bool(enabled)
        return self.diag_info

    def set_m5_enabled(self, enabled):
        """M-5' の適用を切り替える（V6 の学習では常に真。試験用に残す）。"""
        self.m5_enabled = bool(enabled)
        return self.m5_enabled

    # ---- ライフサイクル ----

    def _reset_episode_state(self):
        self._prev_action = action.initial_action(self.stage)
        self._pending_dt = None
        self._pending_boundary = False
        self._no_cooling = np.zeros(constants.N_ZONES, dtype=bool)
        self._prev_building_energy_mj = None
        self._step_index = 0
        self._sum_delta_E = 0.0
        self._sum_zone_energy = 0.0
        self._sum_header_energy = 0.0
        self._sum_raw_resid = 0.0
        self._reward_sum = 0.0
        self._clip_counts = np.zeros(constants.OBSERVATION_DIM, dtype=np.int64)
        self._n_obs = 0
        self._override_count = 0
        self._override_opportunities = 0
        self._cool_while_unoccupied = 0
        self._unoccupied_zone_steps = 0
        self._last_zone_occupants = np.zeros(constants.N_ZONES)
        self._idle_tenants = np.zeros(len(constants.TENANTS), dtype=bool)
        self._remote_status = None
        # 第 3-8 節の記録
        self._m5_zone_steps = 0        # M-5' が効いたゾーン×ステップ数
        self._m9_tenant_steps = 0      # M-9 が効いたテナント×ステップ数
        self._vent_level_steps = np.zeros((len(action.VENT_TENANT_ORDER), len(constants.VENT_LEVELS)), dtype=np.int64)
        self._vent_level_seconds = np.zeros_like(self._vent_level_steps, dtype=np.float64)
        self._remote_permit_steps = np.zeros(len(action.VENT_TENANT_ORDER), dtype=np.int64)
        self._hex_energy_gj = 0.0      # R-6 の模型による全熱交換器の週間一次エネルギー
        self._onoff_mismatch = 0       # 指令した発停と読み戻し（1102）の不一致の件数（ゾーン×ステップ）
        self._setpoint_rewrites = 0    # K-13 (a) で設定温度を書き直した件数（ゾーン×ステップ）
        # R-8'（V6.4-S）: ステップの開始時刻・読み取り時刻・A_t・e_k,t・終端 1 点の n_k,t
        self._r8p_starts, self._r8p_reads, self._r8p_A = [], [], []
        self._r8p_e, self._r8p_n_end = [], []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._driver is not None:
            self._driver.close()
        self._driver = driver.EpisodeDriver(slot=self.slot)
        self._driver.start()
        self._dw = comm.DiffWriter(write_all=not constants.USE_DIFF_WRITER)
        self._ob.reset()
        self._reset_episode_state()
        obs = driver.run_coro(self._driver, self._async_first_obs())
        info = {"action_mask": self.action_masks(), "sim_datetime": self._pending_dt, "stage": self.stage}
        return obs, info

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    async def _read_remote_status(self, comms):
        """O-3: テナント別のリモコン許可の状態（テナントの全室内機の 1114 の平均。全台同じ値を書くため 0 か 1）。"""
        st = await comm.read_remote_permission_status(comms)
        return {t: float(np.mean([st[k] for k in TENANT_ZONE_INDICES[t]])) for t in action.VENT_TENANT_ORDER}

    async def _async_first_obs(self):
        comms = self._driver.comms
        dt = await self._driver.true_sim_datetime()
        self._pending_dt = dt
        self._pending_boundary = timing.is_system_boundary(dt)
        snap = await comm.read_step_snapshot(comms)
        readback = await comm.read_setpoint_readback(comms)
        tenant_thermal = await comm.read_tenant_thermal_sensation(comms)
        self._remote_status = await self._read_remote_status(comms)
        self._prev_building_energy_mj = snap["building"]["total_energy_consumption_mj"]
        decoded = action.decode_action(self._prev_action, self.stage)
        self._update_masks(dt, snap)
        return self._build_obs(dt, snap, decoded, None, None, tenant_thermal, readback)

    # ---- マスク ----

    def _update_masks(self, dt, snap):
        """マスクの対象を更新する。M-5'（ゾーン単位、12 時間先）と M-9（テナント単位、1 時間先）。"""
        occupants = np.array([snap["zones"][l]["occupant_number"] for l in constants.ZONE_LABELS], dtype=np.float64)
        self._last_zone_occupants = occupants
        upcoming = self.exo.zone_occupied_within(dt, constants.M5_FORECAST_WINDOW_H)
        self._no_cooling = (occupants <= 0) & (~upcoming)
        soon = self.exo.zone_occupied_within(dt, constants.M9_FORECAST_WINDOW_H)
        idle = []
        for tenant in action.VENT_TENANT_ORDER:
            zs = TENANT_ZONE_INDICES[tenant]
            idle.append(bool(occupants[zs].sum() <= 0.0) and not bool(np.any(soon[zs])))
        self._idle_tenants = np.array(idle, dtype=bool)

    def action_masks(self):
        mask = action.compute_mask(self._prev_action, self._pending_boundary, self.stage, self._no_cooling,
                                   m5_enabled=self.m5_enabled, idle_tenants=self._idle_tenants,
                                   m9_enabled=self.m9_enabled)
        return action.mask_to_flat(mask)

    # ---- ステップ ----

    def step(self, act):
        obs, r, terminated, truncated, info = driver.run_coro(self._driver, self._async_step(np.asarray(act, dtype=np.int64)))
        if terminated:
            info.update(self._finish_r8p(info))
        return obs, r, terminated, truncated, info

    def minute_output_dir(self):
        """この環境のエミュレータが分単位の出力を書くディレクトリ（組の複製の data/。原本は Shizuku2/data）。"""
        return self.slot.data_dir if self.slot is not None else os.path.join(driver._SHIZUKU2_DIR, "data")

    def _finish_r8p(self, info):
        """R-8'（V6.4-S 第 2 章）: エミュレータを閉じて分単位の出力を確定させてから n_k,t を作る。

        同じ組の次のエピソードのエミュレータは reset() で起動するため、読み終えるまで出力は上書きされない（UT-34）。
        """
        t0 = time.time()
        self.close()
        minute = r8p.load_minute_data(self.minute_output_dir())
        A = np.asarray(self._r8p_A, dtype=np.float64)
        n_kt, n_t, empty = r8p.zone_n(constants.R8P_BRANCH, constants.R8P_DELTA_MIN, A, self._r8p_reads,
                                      self._r8p_starts, minute, self.exo)
        e = np.asarray(self._r8p_e, dtype=np.float64)
        zone_rewards = self._reward_calc.zone_rewards(e, n_kt) if self._reward_calc is not None else np.zeros_like(n_kt)
        DR = float(info["DR_rep"])
        total = float(n_kt.sum())
        total_end = float(np.sum(self._r8p_n_end))
        # 読んだ分単位の出力がこのエピソードのものかの確認（UT-34）: R8-B の週の和と成績の DR の相対差（V6.3-S で 2.5e-7）
        n_B = r8p.building_n_B(minute, self._r8p_starts, self._r8p_reads[-1], self.exo)
        return {
            "r8p_branch": constants.R8P_BRANCH, "r8p_delta_min": constants.R8P_DELTA_MIN,
            "r8p_zone_n": n_kt, "r8p_building_n": n_t, "r8p_zone_rewards": zone_rewards,
            "r8p_sum_n": total, "r8p_sum_rel": abs(total - DR) / DR if DR > 0 else 0.0,
            "r8p_min_n": float(n_t.min()), "r8p_zone_sum_max_abs": float(np.abs(n_kt.sum(axis=1) - n_t).max()),
            "r8p_empty_steps": int(empty.sum()),
            "r8p_empty_max_abs_n": float(np.abs(n_t[empty]).max()) if empty.any() else 0.0,
            "r8p_minute_rows": len(minute["dt"]),
            "r8p_minute_B_week_rel": abs(float(n_B.sum()) - DR) / DR if DR > 0 else 0.0,
            "r8p_seconds": time.time() - t0,
            # MUST-7: 週の恒等式の残差を不満足の側でも記録する（R-8' の値と、参考として終端 1 点の値）
            "raw_resid_dissatisfaction_rel": (total - DR) / DR if DR > 0 else 0.0,
            "raw_resid_dissatisfaction_endpoint_rel": (total_end - DR) / DR if DR > 0 else 0.0,
        }

    def _build_obs(self, dt, snap, decoded, zone_energy, zone_dr, tenant_thermal, readback):
        obs, raw = self._ob.build(dt, self._step_index, self.stage, snap, decoded, zone_energy, zone_dr,
                                  tenant_thermal, readback, remote_status=self._remote_status)
        self._clip_counts += observation.clipped_entries(raw)
        self._n_obs += 1
        return obs

    async def _async_step(self, act):
        dt = self._pending_dt
        comms = self._driver.comms
        decoded = action.decode_action(act, self.stage)
        zone_on = action.zone_on_flags(act)
        unoccupied = self._last_zone_occupants <= 0
        self._unoccupied_zone_steps += int(unoccupied.sum())
        if self.m5_enabled:
            self._m5_zone_steps += int(self._no_cooling.sum())
        if self.m9_enabled:
            self._m9_tenant_steps += int(self._idle_tenants.sum())
        self._cool_while_unoccupied += int((unoccupied & (act[:constants.N_ZONES] == constants.ZONE_STATE.index("cool"))).sum())

        occupants_at_decision = self._last_zone_occupants.copy()
        await self._write_action(decoded)
        seconds = timing.zone_step_seconds(dt)
        dt_after = await self._driver.advance(seconds)
        is_final = self._driver.final_dtime is not None and dt_after >= self._driver.final_dtime

        snap = await comm.read_step_snapshot(comms)
        # 電力はステップ終端の 1 点で読む（P-8。V5-S と同じ）
        p_outdoor = await comm.read_outdoor_unit_power(comms)
        p_indoor = await comm.read_indoor_unit_power(comms) if self.include_indoor_unit_power else None
        readback = await comm.read_setpoint_readback(comms)
        tenant_thermal = await comm.read_tenant_thermal_sensation(comms)
        self._remote_status = await self._read_remote_status(comms)
        onoff_status = await comm.read_onoff_status(comms)

        cur_energy_mj = snap["building"]["total_energy_consumption_mj"]
        delta_E_gj = (cur_energy_mj - self._prev_building_energy_mj) / 1000.0
        self._prev_building_energy_mj = cur_energy_mj

        p_hex = hex_power_kw(decoded["tenants"])
        heatload = [snap["zones"][l]["heatload_kw"] for l in constants.ZONE_LABELS]
        split = reward_mod.zone_energy_split(delta_E_gj, p_outdoor, p_indoor, p_hex, heatload, zone_on, seconds)
        zone_energy = split["e_gj"]

        zone_dr = np.zeros(constants.N_ZONES)
        zone_n = np.zeros(constants.N_ZONES)
        for k, (ou, iu, tenant, zn, label, area) in enumerate(constants.ZONE_DEFS):
            z = snap["zones"][label]
            zone_dr[k] = reward_mod.combine_dr_components(z["dissatisfied_thermal"], z["dissatisfied_draft"],
                                                          z["dissatisfied_vtmp"], reward_mod.dr_co2(snap["co2_ppm"][tenant]))
            zone_n[k] = reward_mod.dissatisfaction_increment(z["occupant_number"], zone_dr[k], seconds, self.W)
            if decoded["zones"][label]["state"] != "off":
                self._override_opportunities += 1
                if abs(readback[k] - decoded["zones"][label]["setpoint"]) > constants.SETPOINT_OVERRIDE_TOL_C:
                    self._override_count += 1
            if bool(onoff_status[k]) != bool(zone_on[k]):
                self._onoff_mismatch += 1

        for j, tenant in enumerate(action.VENT_TENANT_ORDER):
            self._vent_level_steps[j, decoded["tenants"][tenant]] += 1
            self._vent_level_seconds[j, decoded["tenants"][tenant]] += seconds
            self._remote_permit_steps[j] += int(decoded["remote"][tenant])
        self._hex_energy_gj += float(p_hex.sum()) * seconds / 3600.0 * KWH_TO_GJ

        if self._reward_calc is not None:
            r_t = self._reward_calc.step_reward(delta_E_gj, float(zone_n.sum()))
            zone_rewards = self._reward_calc.zone_rewards(zone_energy, zone_n)
            system_reward = self._reward_calc.system_reward(delta_E_gj, zone_energy)
        else:
            r_t, zone_rewards, system_reward = 0.0, np.zeros(constants.N_ZONES), 0.0

        self._sum_delta_E += delta_E_gj
        self._sum_zone_energy += float(zone_energy.sum())
        self._sum_header_energy += float(split["e_header_gj"])
        self._sum_raw_resid += float(split["raw_resid_gj"])
        self._reward_sum += r_t
        averaged_rate = float(snap["building"]["averaged_dissatisfaction_rate"])
        self._r8p_starts.append(dt)
        self._r8p_reads.append(dt_after)
        self._r8p_A.append(averaged_rate)
        self._r8p_e.append(np.asarray(zone_energy, dtype=np.float64))
        self._r8p_n_end.append(zone_n.copy())
        if is_final:
            await self._driver.park()

        self._step_index += 1
        self._prev_action = act
        self._pending_dt = dt_after
        self._pending_boundary = timing.is_system_boundary(dt_after)
        self._update_masks(dt_after, snap)
        obs = self._build_obs(dt_after, snap, decoded, zone_energy, zone_dr, tenant_thermal, readback)

        info = {
            "action_mask": self.action_masks(), "sim_datetime": dt_after, "stage": self.stage,
            "zone_rewards": np.asarray(zone_rewards, dtype=np.float64), "system_reward": float(system_reward),
            "step_seconds": seconds, "delta_E_gj": delta_E_gj, "zone_energy_gj": zone_energy,
            "zone_dissatisfaction_increment": zone_n, "outdoor_power_kw": p_outdoor,
            "standby_power_kw": split["standby_kw"], "header_energy_gj": float(split["e_header_gj"]),
            "raw_resid_gj": float(split["raw_resid_gj"]), "m5_enabled": self.m5_enabled,
            # R-8'（V6.4-S）: ステップの終わりに BACnet で読んだ建物の平均不満足率 A_t（成績の DR と同じ量）
            "averaged_dissatisfaction_rate": averaged_rate,
            # V6.2-S 第 5 章の記録（在室ゾーン・ステップで停止を選んだ割合）: 行動を選んだ時点のゾーンの在室者数
            "zone_occupants_at_decision": occupants_at_decision,
        }
        if self.diag_info:   # 診断・試験用の内訳（学習では付けない）
            info["diag"] = {
                "hex_power_kw": p_hex, "vent_levels": dict(decoded["tenants"]), "remote": dict(decoded["remote"]),
                "remote_status": dict(self._remote_status), "onoff_status": list(onoff_status),
                "setpoint_readback": list(readback),
                "indoor_power_kw": (np.asarray(p_indoor, dtype=np.float64) if p_indoor is not None else None),
                "outdoor_power_kw": np.asarray(p_outdoor, dtype=np.float64),
                "q_kw": split["q_kw"], "q_header_kw": split["q_header_kw"],
                "zone_dr": zone_dr, "co2_ppm": dict(snap["co2_ppm"]),
                "dissatisfied_thermal": np.array([snap["zones"][l]["dissatisfied_thermal"] for l in constants.ZONE_LABELS]),
                "dissatisfied_draft": np.array([snap["zones"][l]["dissatisfied_draft"] for l in constants.ZONE_LABELS]),
                "dissatisfied_vtmp": np.array([snap["zones"][l]["dissatisfied_vtmp"] for l in constants.ZONE_LABELS]),
                "occupants": np.array([snap["zones"][l]["occupant_number"] for l in constants.ZONE_LABELS]),
                "drybulb_c": np.array([snap["zones"][l]["drybulb_c"] for l in constants.ZONE_LABELS]),
            }
        if is_final:
            info.update(self._final_info(snap))
        return obs, float(r_t), bool(is_final), False, info

    def _final_info(self, snap):
        E_rep = snap["building"]["total_energy_consumption_mj"] / 1000.0
        DR_rep = snap["building"]["averaged_dissatisfaction_rate"]
        # 比率法（指示書 V2-S 第 2-2 節）では Σ_k e_k + e_hdr = ΔE が構成上成立する。記録するのは
        # (1) ステップ積算 ΣΔE と成績の E の相対差、(2) 瞬時値の標本化誤差 raw_resid の週間相対値。
        delta_rel = abs(self._sum_delta_E - E_rep) / E_rep if E_rep > 0 else 0.0
        clip_rate = self._clip_counts / max(self._n_obs, 1)
        obs25 = np.array([clip_rate[observation.zone_base(k) + 25] for k in range(constants.N_ZONES)])
        # 指示書 V3-S 第 2-4 節: 正規化を改訂した 4 種も +25 と同様に個別に記録する（門にするのは +25 だけ）
        obs29 = np.array([clip_rate[observation.zone_base(k) + 29] for k in range(constants.N_ZONES)])
        co2 = clip_rate[observation.IDX_CO2:observation.IDX_CO2 + 2]   # CO2 ÷ 4000 ppm（O-6 は別に記録）
        load = np.array([clip_rate[observation.IDX_SYSTEM + 5 * s + 3] for s in range(4)])
        n_steps = max(self._step_index, 1)
        vent_frac = {t: {constants.VENT_LEVELS[l]: float(self._vent_level_steps[j, l] / n_steps)
                         for l in range(len(constants.VENT_LEVELS))}
                     for j, t in enumerate(action.VENT_TENANT_ORDER)}
        vent_hours = {t: {constants.VENT_LEVELS[l]: float(self._vent_level_seconds[j, l] / 3600.0)
                          for l in range(len(constants.VENT_LEVELS))}
                      for j, t in enumerate(action.VENT_TENANT_ORDER)}
        return {
            "E_rep": E_rep, "DR_rep": DR_rep,
            "eci_official": eci_mod.eci(E_rep, DR_rep), "eci_smooth": eci_mod.eci_smooth(E_rep, DR_rep),
            # V6.1-S: ECI は零点条件なしの √(P_E·P_DR)（= eci_smooth）。R-5 の J は計算しない（NOT-2）
            "ECI": eci_mod.eci_smooth(E_rep, DR_rep), "P_E": eci_mod.percentile_energy(E_rep),
            "P_DR": eci_mod.percentile_dissatisfaction(DR_rep), "J": None,
            "sum_delta_E_gj": self._sum_delta_E, "delta_E_vs_E_rep_rel": delta_rel,
            "sum_zone_energy_gj": self._sum_zone_energy, "sum_header_energy_gj": self._sum_header_energy,
            "header_energy_fraction": self._sum_header_energy / self._sum_delta_E if self._sum_delta_E > 0 else 0.0,
            "raw_resid_sum_gj": self._sum_raw_resid,
            "raw_resid_rel": self._sum_raw_resid / self._sum_delta_E if self._sum_delta_E > 0 else 0.0,
            "observation_clip_rate": clip_rate,
            "obs25_clip_rate_mean": float(obs25.mean()), "obs25_clip_rate_max": float(obs25.max()),
            "obs29_clip_rate_mean": float(obs29.mean()), "obs29_clip_rate_max": float(obs29.max()),
            "co2_clip_rate_mean": float(co2.mean()), "co2_clip_rate_max": float(co2.max()),
            "load_clip_rate_mean": float(load.mean()), "load_clip_rate_max": float(load.max()),
            "progress_clip_rate": float(clip_rate[10]),
            # V6-S 第 3-4 節: 新しい項目のクリップ率（O-1・O-3）
            "vent_obs_clip_rate": float(clip_rate[observation.IDX_VENT_LEVEL:observation.IDX_VENT_LEVEL + 2].max()),
            "remote_obs_clip_rate": (float(clip_rate[observation.IDX_REMOTE:observation.IDX_REMOTE + 2].max())
                                     if observation.IDX_REMOTE is not None else None),     # C-5 で O-3 を廃止
            "co2_delta_obs_clip_rate": float(clip_rate[observation.IDX_CO2_DELTA:observation.IDX_CO2_DELTA + 2].max()),
            "setpoint_override_rate": self._override_count / max(self._override_opportunities, 1),
            "cooling_fraction_zone_unoccupied": self._cool_while_unoccupied / max(self._unoccupied_zone_steps, 1),
            "reward_sum": self._reward_sum, "n_steps": self._step_index, "m5_enabled": self.m5_enabled,
            "m9_enabled": self.m9_enabled,
            # V6-S 第 3-8 節
            "m5_zone_steps": self._m5_zone_steps, "m9_tenant_steps": self._m9_tenant_steps,
            "vent_level_fraction": vent_frac, "vent_level_hours": vent_hours, "hex_energy_gj": self._hex_energy_gj,
            "remote_permit_fraction": {t: float(self._remote_permit_steps[j] / n_steps)
                                       for j, t in enumerate(action.VENT_TENANT_ORDER)},
            "onoff_mismatch": self._onoff_mismatch,
            "setpoint_override_count": self._override_count, "setpoint_rewrite_count": self._setpoint_rewrites,
        }

    async def _write_action(self, decoded):
        """行動と固定値の書き込み（現行 env._write_action の夏季分の手順・キーに、V6-S A-5・A-6 を加えた）。"""
        vrc, vsc = _szk()
        comms = self._driver.comms
        dw = self._dw
        for ou in range(1, 5):
            await dw.write(("sys_refrig_ctrl", ou), True, comms.vrf.enable_refrigerant_temperatureControl(ou))
            evap = decoded["systems"][ou]["evap_c"]
            await dw.write(("sys_evap", ou), evap, comms.vrf.change_evaporating_temperature(ou, evap))
            assert constants.SUMMER_COND_TEMP_C < constants.CONDENSING_TEMP_MAX_C
            await dw.write(("sys_cond_fixed", ou), constants.SUMMER_COND_TEMP_C,
                           comms.vrf.change_condensing_temperature(ou, constants.SUMMER_COND_TEMP_C))

        sp_readback = await comm.read_setpoint_readback(comms) if self.setpoint_rewrite_on_mismatch else None
        for k, (label, (ou, iu, tenant, zn, area)) in enumerate(ZONE_INFO.items()):
            z = decoded["zones"][label]
            # A-6: V5 までは許可（permit_local_control = 1113 へ 1）を固定値で書いていた。同じキー・同じ符号で
            # 許可は permit_local_control、禁止は prohibit_local_control（1113 へ 0）を書く。
            # C-5 の後は decode_action が常に許可を返すため、V5 と同じく許可を固定で書く
            permit = bool(decoded["remote"][tenant])
            await dw.write(("zone_permit", ou, iu), permit,
                           comms.vrf.permit_local_control(ou, iu) if permit else comms.vrf.prohibit_local_control(ou, iu))
            if z["state"] == "off":
                await dw.write(("zone_on", ou, iu), False, comms.vrf.turn_off(ou, iu))
                continue
            mode = vrc.Mode.Cooling if z["state"] == "cool" else vrc.Mode.ThermoOff
            await dw.write(("zone_on", ou, iu), True, comms.vrf.turn_on(ou, iu))
            await dw.write(("zone_mode", ou, iu), z["state"], comms.vrf.change_mode(ou, iu, mode))
            if z["setpoint"] is not None and sp_readback is not None:
                # K-13 (a): 差分書き込みの比較の相手を「前回書いた値」から「読み戻し値」に替える
                if abs(sp_readback[k] - z["setpoint"]) > constants.SETPOINT_OVERRIDE_TOL_C:
                    await comm.call_write(comms.vrf.change_setpoint_temperature(ou, iu, z["setpoint"]), str(("zone_sp", ou, iu)))
                    self._setpoint_rewrites += 1
                dw._last[("zone_sp", ou, iu)] = z["setpoint"]
            elif z["setpoint"] is not None:
                await dw.write(("zone_sp", ou, iu), z["setpoint"], comms.vrf.change_setpoint_temperature(ou, iu, z["setpoint"]))
            await dw.write(("zone_fan", ou, iu), z["fan_speed"],
                           comms.vrf.change_fan_speed(ou, iu, getattr(vrc.FanSpeed, z["fan_speed"])))
            await dw.write(("zone_dir_fixed", ou, iu), constants.ZONE_DIRECTION_FIXED,
                           comms.vrf.change_direction(ou, iu, getattr(vrc.Direction, constants.ZONE_DIRECTION_FIXED)))

        # A-5: 停止は発停を Off（風量は書かない）。Low・Middle・High は風量 1・2・3 を書いてから On。バイパスは無効のまま
        for tenant_name, level in decoded["tenants"].items():
            for label in [l for l in constants.ZONE_LABELS if ZONE_INFO[l][2] == tenant_name]:
                ou, iu, tenant, zn, area = ZONE_INFO[label]
                await dw.write(("vent_bypass", ou, iu), constants.HEX_BYPASS_ENABLED, comms.vent.disable_bypass_control(ou, iu))
                if level == constants.VENT_OFF:
                    await dw.write(("vent_on", ou, iu), False, comms.vent.stop_ventilation(ou, iu))
                else:
                    name = constants.VENT_LEVELS[level]
                    await dw.write(("vent_fan", ou, iu), name, comms.vent.change_fan_speed(ou, iu, getattr(vsc.FanSpeed, name)))
                    await dw.write(("vent_on", ou, iu), True, comms.vent.start_ventilation(ou, iu))
