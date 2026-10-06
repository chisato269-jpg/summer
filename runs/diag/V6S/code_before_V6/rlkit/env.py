"""rlkit.env — 夏季 v2 の Gymnasium 環境（v2 第 5-1〜5-4 節）。

1 エピソード = 1 回のエミュレータ起動（1 週間、378 ステップ）。現行 rlkit/env.py の駆動・書き込み・
読み出しの手順を維持し、次を v2 に合わせて変えた。
  報酬: 実測電力による按分と、基準点で正規化した報酬（reward.py）。終端補正・報酬スケール c は無い
  観測: 638 次元（observation.py）
  行動: 段階（1・2）で設定温度の格子を切り替え、第 1 段階では M-5 を適用する
  書き込み: 差分書き込み（A-5）
各ステップで info["zone_rewards"]（18）と info["system_reward"] を返す（v2 第 5-5-1 節 手順 1）。
基準点は set_reference() で与える。未設定のあいだ（R-1 の初回評価）は報酬を 0 とする。
"""
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box

from rlkit import action, comm, constants, driver, eci as eci_mod, exogenous, observation, timing
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


class SummerEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, stage=1, include_indoor_unit_power=None, zone_step_energy_norm_gj=None, slot=None,
                 m9_enabled=False):
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
        self.observation_space = Box(low=0.0, high=1.0, shape=(constants.OBSERVATION_DIM,), dtype=np.float32)
        self._ob = observation.ObservationBuilder(self.exo, zone_step_energy_norm_gj)
        self.reference = None
        self._reward_calc = None
        self._driver = None
        self._dw = None
        # M-5・M-8 を適用するか（段階ではなく旗で決める。G-3' で両方まとめて解除する）
        self.m5_enabled = True
        self.m8_enabled = True                    # M-8: 在室ゾーンは冷房だけ（指示書 V5-S 第 2-2 節）
        self.m9_enabled = bool(m9_enabled)        # M-9: 非在室テナントの換気停止（K-7 (a) により既定は無効）
        # 診断用の切り替え（指示書 V5-S 第 2-6 節 K-6・K-7）。学習では既定値のまま使う。
        self.hex_fan_speed = constants.HEX_FAN_SPEED_FIXED   # K-7 (b) の風量固定に使う
        self.diag_info = False                               # 真なら info に内訳を足す（採取では重いので既定は偽）
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
        """基準点（reward.ReferencePoint.to_dict()）。次の reset 以降のステップの報酬に使う。"""
        self.reference = reward_mod.ReferencePoint.from_dict(reference_dict) if reference_dict else None
        self._reward_calc = reward_mod.RewardCalculator(self.reference, self.T, self.W) if self.reference else None
        return True

    def set_hex_fan_speed(self, name):
        """全熱交換器の風量を変える（診断 K-7 (b) 専用。学習では使わない）。"""
        if name not in constants.HEX_POWER_KW[26.0]:
            raise ValueError(f"全熱交換器の風量は {sorted(constants.HEX_POWER_KW[26.0])} のいずれか: {name!r}")
        self.hex_fan_speed = name
        return self.hex_fan_speed

    def set_diag_info(self, enabled):
        """info に電力と不満足率の内訳を足すか（診断 K-6・K-7 専用）。"""
        self.diag_info = bool(enabled)
        return self.diag_info

    def set_m5_enabled(self, enabled):
        """M-5（非占有ゾーンの冷房の除外）の適用を切り替える（G-3' の解除・差し戻し）。"""
        self.m5_enabled = bool(enabled)
        return self.m5_enabled

    def set_m8_enabled(self, enabled):
        """M-8（在室ゾーンの非運転の禁止）の適用を切り替える（G-3' で M-5 とともに解除する）。"""
        self.m8_enabled = bool(enabled)
        return self.m8_enabled

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
        self._occupied = np.zeros(constants.N_ZONES, dtype=bool)
        self._idle_tenants = np.zeros(len(constants.TENANTS), dtype=bool)
        self._m8_zone_steps = 0        # M-8 が効いたゾーン×ステップ数（M-5 の記録項目）
        self._m9_tenant_steps = 0      # M-9 が効いたテナント×ステップ数

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

    async def _async_first_obs(self):
        comms = self._driver.comms
        dt = await self._driver.true_sim_datetime()
        self._pending_dt = dt
        self._pending_boundary = timing.is_system_boundary(dt)
        snap = await comm.read_step_snapshot(comms)
        readback = await comm.read_setpoint_readback(comms)
        tenant_thermal = await comm.read_tenant_thermal_sensation(comms)
        self._prev_building_energy_mj = snap["building"]["total_energy_consumption_mj"]
        decoded = action.decode_action(self._prev_action, self.stage)
        self._update_m5(dt, snap)
        return self._build_obs(dt, snap, decoded, None, None, tenant_thermal, readback)

    # ---- マスク ----

    def _update_m5(self, dt, snap):
        """マスクの対象を更新する。M-5（非占有）と M-8（在室）は排他、M-9 はテナント単位。"""
        occupants = np.array([snap["zones"][l]["occupant_number"] for l in constants.ZONE_LABELS], dtype=np.float64)
        self._last_zone_occupants = occupants
        upcoming = self.exo.zone_occupied_within(dt, constants.M5_FORECAST_WINDOW_H)
        self._no_cooling = (occupants <= 0) & (~upcoming)
        self._occupied = occupants >= 1.0
        soon = self.exo.zone_occupied_within(dt, constants.M9_FORECAST_WINDOW_H)
        idle = []
        for tenant in action.VENT_TENANT_ORDER:
            zs = [k for k, z in enumerate(constants.ZONE_DEFS) if z[2] == tenant]
            idle.append(bool(occupants[zs].sum() <= 0.0) and not bool(np.any(soon[zs])))
        self._idle_tenants = np.array(idle, dtype=bool)

    def action_masks(self):
        mask = action.compute_mask(self._prev_action, self._pending_boundary, self.stage, self._no_cooling,
                                   m5_enabled=self.m5_enabled, occupied_zones=self._occupied,
                                   m8_enabled=self.m8_enabled, idle_tenants=self._idle_tenants,
                                   m9_enabled=self.m9_enabled)
        return action.mask_to_flat(mask)

    # ---- ステップ ----

    def step(self, act):
        return driver.run_coro(self._driver, self._async_step(np.asarray(act, dtype=np.int64)))

    def _build_obs(self, dt, snap, decoded, zone_energy, zone_dr, tenant_thermal, readback):
        obs, raw = self._ob.build(dt, self._step_index, self.stage, snap, decoded, zone_energy, zone_dr,
                                  tenant_thermal, readback)
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
        if self.m8_enabled:
            self._m8_zone_steps += int(self._occupied.sum())
        if self.m9_enabled:
            self._m9_tenant_steps += int(self._idle_tenants.sum())
        self._cool_while_unoccupied += int((unoccupied & (act[:constants.N_ZONES] == constants.ZONE_STATE.index("cool"))).sum())

        await self._write_action(decoded)
        seconds = timing.zone_step_seconds(dt)
        dt_after = await self._driver.advance(seconds)
        is_final = self._driver.final_dtime is not None and dt_after >= self._driver.final_dtime

        snap = await comm.read_step_snapshot(comms)
        # 電力はステップ終端の 1 点で読む。指示書 V3-S J-2（ステップ内 4 点の平均）は実現できないため
        # J-5 に従って終端 1 点を維持する。エミュレータの内部時間刻みは 60 秒で、advance(225) は実測 240 秒
        # 進むため、900 秒ステップを 4 等分する 225 秒では区切れず、区切ると 1 ステップが 960 秒になる。
        p_outdoor = await comm.read_outdoor_unit_power(comms)
        p_indoor = await comm.read_indoor_unit_power(comms) if self.include_indoor_unit_power else None
        readback = await comm.read_setpoint_readback(comms)
        tenant_thermal = await comm.read_tenant_thermal_sensation(comms)

        cur_energy_mj = snap["building"]["total_energy_consumption_mj"]
        delta_E_gj = (cur_energy_mj - self._prev_building_energy_mj) / 1000.0
        self._prev_building_energy_mj = cur_energy_mj

        p_hex = [constants.HEX_POWER_KW[area][self.hex_fan_speed] if decoded["tenants"][tenant] else 0.0
                 for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS]
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
        if is_final:
            await self._driver.park()

        self._step_index += 1
        self._prev_action = act
        self._pending_dt = dt_after
        self._pending_boundary = timing.is_system_boundary(dt_after)
        self._update_m5(dt_after, snap)
        obs = self._build_obs(dt_after, snap, decoded, zone_energy, zone_dr, tenant_thermal, readback)

        info = {
            "action_mask": self.action_masks(), "sim_datetime": dt_after, "stage": self.stage,
            "zone_rewards": np.asarray(zone_rewards, dtype=np.float64), "system_reward": float(system_reward),
            "step_seconds": seconds, "delta_E_gj": delta_E_gj, "zone_energy_gj": zone_energy,
            "zone_dissatisfaction_increment": zone_n, "outdoor_power_kw": p_outdoor,
            "standby_power_kw": split["standby_kw"], "header_energy_gj": float(split["e_header_gj"]),
            "raw_resid_gj": float(split["raw_resid_gj"]), "m5_enabled": self.m5_enabled,
        }
        if self.diag_info:   # 診断 K-6・K-7 の内訳（学習では付けない）
            info["diag"] = {
                "hex_power_kw": np.asarray(p_hex, dtype=np.float64),
                "hex_fan_speed": self.hex_fan_speed,
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
        co2 = clip_rate[observation.IDX_CO2:observation.IDX_CO2 + 4]
        load = np.array([clip_rate[observation.IDX_SYSTEM + 5 * s + 3] for s in range(4)])
        return {
            "E_rep": E_rep, "DR_rep": DR_rep,
            "eci_official": eci_mod.eci(E_rep, DR_rep), "eci_smooth": eci_mod.eci_smooth(E_rep, DR_rep),
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
            "setpoint_override_rate": self._override_count / max(self._override_opportunities, 1),
            "cooling_fraction_zone_unoccupied": self._cool_while_unoccupied / max(self._unoccupied_zone_steps, 1),
            "reward_sum": self._reward_sum, "n_steps": self._step_index, "m5_enabled": self.m5_enabled,
            "m8_enabled": self.m8_enabled, "m9_enabled": self.m9_enabled,
            "m8_zone_steps": self._m8_zone_steps, "m9_tenant_steps": self._m9_tenant_steps,
        }

    async def _write_action(self, decoded):
        """行動と固定値の書き込み（現行 env._write_action の夏季分と同じ手順・同じキー）。"""
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

        for label, (ou, iu, tenant, zn, area) in ZONE_INFO.items():
            z = decoded["zones"][label]
            await dw.write(("zone_permit", ou, iu), constants.REMOTE_CONTROL_PERMITTED, comms.vrf.permit_local_control(ou, iu))
            if z["state"] == "off":
                await dw.write(("zone_on", ou, iu), False, comms.vrf.turn_off(ou, iu))
                continue
            mode = vrc.Mode.Cooling if z["state"] == "cool" else vrc.Mode.ThermoOff
            await dw.write(("zone_on", ou, iu), True, comms.vrf.turn_on(ou, iu))
            await dw.write(("zone_mode", ou, iu), z["state"], comms.vrf.change_mode(ou, iu, mode))
            if z["setpoint"] is not None:
                await dw.write(("zone_sp", ou, iu), z["setpoint"], comms.vrf.change_setpoint_temperature(ou, iu, z["setpoint"]))
            await dw.write(("zone_fan", ou, iu), z["fan_speed"],
                           comms.vrf.change_fan_speed(ou, iu, getattr(vrc.FanSpeed, z["fan_speed"])))
            await dw.write(("zone_dir_fixed", ou, iu), constants.ZONE_DIRECTION_FIXED,
                           comms.vrf.change_direction(ou, iu, getattr(vrc.Direction, constants.ZONE_DIRECTION_FIXED)))

        for tenant_name, on in decoded["tenants"].items():
            for label in [l for l in constants.ZONE_LABELS if ZONE_INFO[l][2] == tenant_name]:
                ou, iu, tenant, zn, area = ZONE_INFO[label]
                await dw.write(("vent_bypass", ou, iu), constants.HEX_BYPASS_ENABLED, comms.vent.disable_bypass_control(ou, iu))
                await dw.write(("vent_fan", ou, iu), self.hex_fan_speed,
                               comms.vent.change_fan_speed(ou, iu, getattr(vsc.FanSpeed, self.hex_fan_speed)))
                if on:
                    await dw.write(("vent_on", ou, iu), True, comms.vent.start_ventilation(ou, iu))
                else:
                    await dw.write(("vent_on", ou, iu), False, comms.vent.stop_ventilation(ou, iu))
