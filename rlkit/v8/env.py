"""rlkit.v8.env — V8-S の夏季環境（指示書 第 5 章）。V7 の SummerEnv を継承し、行動の書き込みと 1 刻みの進め方を変える。

1 刻み（在室 900 秒・無人 3,600 秒）を 3 つに分けて進める。どの割合でも、刻みの 1/3 と 2/3 の時点でエミュレータを一時停止し、
室外機・室内機の電力を標本にする（終端と合わせて 3 点）。割合の前詰めの運転は次のとおり。
  刻みの始め: 割合 > 0 のゾーンは運転を書き込み（設定温度・風量・風向は割合 > 0 のときだけ）、割合 0 のゾーンは停止を書き込む
  1/3 の時点: 標本 1 を読み、割合 1/3 のゾーンへ停止を書き込む
  2/3 の時点: 標本 2 を読み、割合 2/3 のゾーンへ停止を書き込む
  終端: 標本 3 を読む
エネルギーの配分の比は、3 点の標本それぞれ（その区間に運転しているゾーン）で V7 と同じ式で q を作り、平均して使う。
ΔE から作るので、エネルギーの側の恒等式（Σ e_k + e_ヘッダ = ΔE）は割合によらず成り立つ。
A-2 のリモコンの許可はゾーンごとに permit_local_control・prohibit_local_control を書く。A-3 のバイパスはテナントの全熱交換器に
enable_bypass_control・disable_bypass_control を書く。蒸発温度は 14〜22 ℃を、設定温度は 0.5 ℃刻みの値を、そのまま書く。
"""
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete

from rlkit import comm, constants, driver, env as env7, reward as reward_mod, timing
from rlkit.v8 import action as act8
from rlkit.v8 import observation as obs8

ZONE_INFO = env7.ZONE_INFO
TENANT_ZONE_INDICES = {t: [k for k, z in enumerate(constants.ZONE_DEFS) if z[2] == t] for t in act8.TENANT_ORDER}
KWH_TO_GJ = env7.KWH_TO_GJ
hex_power_kw = env7.hex_power_kw
N_SEGMENTS = 3


def split_multi(delta_E_gj, p_outdoor_samples, p_indoor_samples, p_hex_kw, heatload_kw, zone_on_samples, step_seconds):
    """3 点の標本から、ゾーンへの配分 e_k,t を作る。標本 j ごとに V7 の zone_energy_split で q（ゾーン別の電力）と Q を作り、
    平均した q・Q で ΔE を配分する（Σ e_k + e_ヘッダ = ΔE は構成上成り立つ）。"""
    parts = [reward_mod.zone_energy_split(delta_E_gj, po, pi, p_hex_kw, heatload_kw, zon, step_seconds)
             for po, pi, zon in zip(p_outdoor_samples, p_indoor_samples, zone_on_samples)]
    q = np.mean([p["q_kw"] for p in parts], axis=0)
    q_header = float(np.mean([p["q_header_kw"] for p in parts]))
    Q = float(q.sum()) + q_header
    if Q > 0.0:
        e_gj = delta_E_gj * q / Q
        e_header = delta_E_gj * q_header / Q
    else:
        e_gj = np.zeros(constants.N_ZONES)
        e_header = delta_E_gj
    raw_resid = delta_E_gj - Q * (step_seconds / 3600.0) * constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0
    return {"e_gj": e_gj, "e_header_gj": e_header, "q_kw": q, "q_header_kw": q_header, "Q_kw": Q, "raw_resid_gj": raw_resid,
            "standby_kw": np.mean([p["standby_kw"] for p in parts], axis=0)}


class SummerEnvV8(env7.SummerEnv):
    def __init__(self, *args, **kwargs):
        zone_norm = kwargs.get("zone_step_energy_norm_gj")
        super().__init__(*args, **kwargs)
        self._ob = obs8.ObservationBuilderV8(self.exo, kwargs.get("zone_step_energy_norm_gj", zone_norm))
        self.observation_space = Box(low=obs8.OBS_LOW.astype(np.float32), high=obs8.OBS_HIGH.astype(np.float32), dtype=np.float32)
        self.action_space = MultiDiscrete(act8.NVEC)
        self.alloc_end_only = False                  # 診断専用: 真なら配分の比を終端の標本だけから作る（一時停止は行う。G-0 の切り分け）
        self.pause_segments = N_SEGMENTS             # 診断専用: 1 にすると途中の一時停止・標本・停止の書き込みを行わない（G-0 の切り分け）
        self._reset_episode_state()

    # ---- V7 からの差し替え ----

    def set_stage(self, stage):
        self.stage = stage
        self.action_space = MultiDiscrete(act8.NVEC)
        return stage

    def _reset_episode_state(self):
        super()._reset_episode_state()
        self._prev_action = act8.initial_action()
        self._clip_counts = np.zeros(obs8.TOTAL_DIM, dtype=np.int64)
        self._remote_permit_steps = np.zeros(act8.N_TENANTS, dtype=np.float64)
        self._remote_zone = [True] * constants.N_ZONES
        # V8 の新しい選択肢の使われ方（報告のみ）
        self._v8_duty_counts = {"occupied": np.zeros(4, dtype=np.int64), "unoccupied": np.zeros(4, dtype=np.int64)}
        self._v8_prohibit_zone_steps = 0
        self._v8_bypass_tenant_steps = 0
        self._v8_setpoint_counts = np.zeros(len(act8.SETPOINT_GRID), dtype=np.int64)
        self._v8_evap_counts = np.zeros(len(act8.EVAP_GRID), dtype=np.int64)
        self._v8_mid_writes = 0

    def action_masks(self):
        mask = act8.compute_mask(self._prev_action, self._pending_boundary, self._no_cooling, self._idle_tenants,
                                 m5_enabled=self.m5_enabled, m9_enabled=self.m9_enabled)
        return act8.mask_to_flat(mask)

    async def _read_remote_status(self, comms):
        st = await comm.read_remote_permission_status(comms)
        self._remote_zone = [bool(v) for v in st]
        return {t: float(np.mean([st[k] for k in TENANT_ZONE_INDICES[t]])) for t in act8.TENANT_ORDER}

    def _build_obs(self, dt, snap, decoded, zone_energy, zone_dr, tenant_thermal, readback):
        obs, raw = self._ob.build(dt, self._step_index, self.stage, snap, decoded, zone_energy, zone_dr, tenant_thermal,
                                  readback, remote_zone=self._remote_zone)
        self._clip_counts += obs8.clipped_entries(raw)
        self._n_obs += 1
        return obs

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
        decoded = act8.decode_action(self._prev_action)
        self._update_masks(dt, snap)
        return self._build_obs(dt, snap, decoded, None, None, tenant_thermal, readback)

    def _final_info(self, snap):
        """V7 の終端の info を、V7 の観測の並びに直したクリップ率で作り、V8 の使われ方を足す。"""
        saved = self._clip_counts
        self._clip_counts = saved[obs8.column_map_v7_to_v8()]
        try:
            info = super()._final_info(snap)
        finally:
            self._clip_counts = saved
        n = max(self._n_obs, 1)
        clip8 = saved / n
        info["v8_new_obs_clip_rate_max"] = float(max(clip8[obs8.IDX_BYPASS:obs8.IDX_BYPASS + 2].max(),
                                                     max(clip8[obs8.zone_base(k) + obs8.ZONE_DIM_V7:obs8.zone_base(k) + obs8.ZONE_DIM].max()
                                                         for k in range(constants.N_ZONES))))
        info["v8_use"] = {
            "duty_counts_occupied": self._v8_duty_counts["occupied"].tolist(),
            "duty_counts_unoccupied": self._v8_duty_counts["unoccupied"].tolist(),
            "prohibit_zone_steps": self._v8_prohibit_zone_steps, "bypass_tenant_steps": self._v8_bypass_tenant_steps,
            "setpoint_counts": self._v8_setpoint_counts.tolist(), "evap_counts": self._v8_evap_counts.tolist(),
            "mid_step_writes": self._v8_mid_writes}
        return info

    # ---- 書き込み ----

    async def _write_start(self, decoded):
        """刻みの始めの書き込み: 蒸発温度、リモコンの許可、ゾーンの運転（割合 > 0）と停止（割合 0）、換気とバイパス。"""
        vrc, vsc = env7._szk()
        comms = self._driver.comms
        dw = self._dw
        for ou in range(1, 5):
            await dw.write(("sys_refrig_ctrl", ou), True, comms.vrf.enable_refrigerant_temperatureControl(ou))
            evap = decoded["systems"][ou]["evap_c"]
            await dw.write(("sys_evap", ou), evap, comms.vrf.change_evaporating_temperature(ou, evap))
            assert constants.SUMMER_COND_TEMP_C < constants.CONDENSING_TEMP_MAX_C
            await dw.write(("sys_cond_fixed", ou), constants.SUMMER_COND_TEMP_C,
                           comms.vrf.change_condensing_temperature(ou, constants.SUMMER_COND_TEMP_C))
        for k, (label, (ou, iu, tenant, zn, area)) in enumerate(ZONE_INFO.items()):
            z = decoded["zones"][label]
            permit = bool(z["permit"])
            await dw.write(("zone_permit", ou, iu), permit,
                           comms.vrf.permit_local_control(ou, iu) if permit else comms.vrf.prohibit_local_control(ou, iu))
            if z["state"] == "off":
                await dw.write(("zone_on", ou, iu), False, comms.vrf.turn_off(ou, iu))
                continue
            await dw.write(("zone_on", ou, iu), True, comms.vrf.turn_on(ou, iu))
            await dw.write(("zone_mode", ou, iu), "cool", comms.vrf.change_mode(ou, iu, vrc.Mode.Cooling))
            await dw.write(("zone_sp", ou, iu), z["setpoint"], comms.vrf.change_setpoint_temperature(ou, iu, z["setpoint"]))
            await dw.write(("zone_fan", ou, iu), z["fan_speed"],
                           comms.vrf.change_fan_speed(ou, iu, getattr(vrc.FanSpeed, z["fan_speed"])))
            await dw.write(("zone_dir_fixed", ou, iu), constants.ZONE_DIRECTION_FIXED,
                           comms.vrf.change_direction(ou, iu, getattr(vrc.Direction, constants.ZONE_DIRECTION_FIXED)))
        for tenant_name, level in decoded["tenants"].items():
            use_bypass = bool(decoded["bypass"][tenant_name])
            for label in [l for l in constants.ZONE_LABELS if ZONE_INFO[l][2] == tenant_name]:
                ou, iu, tenant, zn, area = ZONE_INFO[label]
                await dw.write(("vent_bypass", ou, iu), use_bypass,
                               comms.vent.enable_bypass_control(ou, iu) if use_bypass else comms.vent.disable_bypass_control(ou, iu))
                if level == constants.VENT_OFF:
                    await dw.write(("vent_on", ou, iu), False, comms.vent.stop_ventilation(ou, iu))
                else:
                    name = constants.VENT_LEVELS[level]
                    await dw.write(("vent_fan", ou, iu), name, comms.vent.change_fan_speed(ou, iu, getattr(vsc.FanSpeed, name)))
                    await dw.write(("vent_on", ou, iu), True, comms.vent.start_ventilation(ou, iu))

    async def _write_stops(self, decoded, stop_zone_mask):
        """途中の停止（割合 1/3 のゾーンは 1/3 の時点、2/3 のゾーンは 2/3 の時点）。"""
        comms = self._driver.comms
        for k, (label, (ou, iu, tenant, zn, area)) in enumerate(ZONE_INFO.items()):
            if stop_zone_mask[k]:
                if await self._dw.write(("zone_on", ou, iu), False, comms.vrf.turn_off(ou, iu)):
                    self._v8_mid_writes += 1

    async def _power_sample(self, comms):
        p_out = await comm.read_outdoor_unit_power(comms)
        p_in = await comm.read_indoor_unit_power(comms) if self.include_indoor_unit_power else None
        return p_out, p_in

    async def _async_step(self, act):
        dt = self._pending_dt
        comms = self._driver.comms
        decoded = act8.decode_action(act)
        duty = act8.duty_indices(act)                                          # (18,) 0〜3
        unoccupied = self._last_zone_occupants <= 0
        self._unoccupied_zone_steps += int(unoccupied.sum())
        if self.m5_enabled:
            self._m5_zone_steps += int(self._no_cooling.sum())
        if self.m9_enabled:
            self._m9_tenant_steps += int(self._idle_tenants.sum())
        self._cool_while_unoccupied += int((unoccupied & (duty > 0)).sum())
        for k in range(constants.N_ZONES):
            self._v8_duty_counts["unoccupied" if unoccupied[k] else "occupied"][int(duty[k])] += 1
            if duty[k] > 0:
                self._v8_setpoint_counts[int(act[act8.LAYOUT["zone_setpoint"][0] + k])] += 1
        self._v8_prohibit_zone_steps += int((np.asarray(act[act8.dim_slice("zone_remote")]) == act8.REMOTE_PROHIBIT).sum())
        self._v8_bypass_tenant_steps += int((np.asarray(act[act8.dim_slice("tenant_bypass")]) == act8.BYPASS_ON).sum())
        for s in range(act8.N_SYSTEMS):
            self._v8_evap_counts[int(act[act8.LAYOUT["system_evap"][0] + s])] += 1

        occupants_at_decision = self._last_zone_occupants.copy()
        await self._write_start(decoded)
        seconds = timing.zone_step_seconds(dt)
        samples = []
        if self.pause_segments == N_SEGMENTS:
            seg = seconds // N_SEGMENTS
            await self._driver.advance(seg)
            samples.append((await self._power_sample(comms), duty >= 1))
            await self._write_stops(decoded, duty == 1)
            await self._driver.advance(seg)
            samples.append((await self._power_sample(comms), duty >= 2))
            await self._write_stops(decoded, duty == 2)
            dt_after = await self._driver.advance(seconds - 2 * seg)
            is_final = self._driver.final_dtime is not None and dt_after >= self._driver.final_dtime
            snap = await comm.read_step_snapshot(comms)
            samples.append((await self._power_sample(comms), duty >= 3))
            if self.alloc_end_only:
                samples = [samples[-1]] * 3
        else:                                                                  # 診断専用（V7 と同じ 1 回の進め方、標本は終端の 1 点を 3 回分）
            dt_after = await self._driver.advance(seconds)
            is_final = self._driver.final_dtime is not None and dt_after >= self._driver.final_dtime
            snap = await comm.read_step_snapshot(comms)
            one = (await self._power_sample(comms), duty >= 3)
            samples = [one, one, one]
        readback = await comm.read_setpoint_readback(comms)
        tenant_thermal = await comm.read_tenant_thermal_sensation(comms)
        self._remote_status = await self._read_remote_status(comms)
        onoff_status = await comm.read_onoff_status(comms)

        cur_energy_mj = snap["building"]["total_energy_consumption_mj"]
        delta_E_gj = (cur_energy_mj - self._prev_building_energy_mj) / 1000.0
        self._prev_building_energy_mj = cur_energy_mj

        p_hex = hex_power_kw(decoded["tenants"])
        heatload = [snap["zones"][l]["heatload_kw"] for l in constants.ZONE_LABELS]
        split = split_multi(delta_E_gj, [s[0][0] for s in samples], [s[0][1] for s in samples], p_hex, heatload,
                            [s[1] for s in samples], seconds)
        zone_energy = split["e_gj"]
        p_outdoor = samples[-1][0][0]

        zone_dr = np.zeros(constants.N_ZONES)
        zone_n = np.zeros(constants.N_ZONES)
        for k, (ou, iu, tenant, zn, label, area) in enumerate(constants.ZONE_DEFS):
            z = snap["zones"][label]
            zone_dr[k] = reward_mod.combine_dr_components(z["dissatisfied_thermal"], z["dissatisfied_draft"],
                                                          z["dissatisfied_vtmp"], reward_mod.dr_co2(snap["co2_ppm"][tenant]))
            zone_n[k] = reward_mod.dissatisfaction_increment(z["occupant_number"], zone_dr[k], seconds, self.W)
            dz = decoded["zones"][label]
            if dz["state"] != "off":
                self._override_opportunities += 1
                if abs(readback[k] - dz["setpoint"]) > constants.SETPOINT_OVERRIDE_TOL_C:
                    self._override_count += 1
            if bool(onoff_status[k]) != bool(duty[k] >= 3):                      # 終端の状態は割合 1 のときだけ運転
                self._onoff_mismatch += 1

        for j, tenant in enumerate(act8.TENANT_ORDER):
            self._vent_level_steps[j, decoded["tenants"][tenant]] += 1
            self._vent_level_seconds[j, decoded["tenants"][tenant]] += seconds
            zs = TENANT_ZONE_INDICES[tenant]
            self._remote_permit_steps[j] += float(np.mean([decoded["zones"][constants.ZONE_LABELS[k]]["permit"] for k in zs]))
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
            "averaged_dissatisfaction_rate": averaged_rate, "zone_occupants_at_decision": occupants_at_decision,
        }
        if is_final:
            info.update(self._final_info(snap))
        return obs, float(r_t), bool(is_final), False, info
