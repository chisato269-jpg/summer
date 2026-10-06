"""rlkit.v8.mockenv — V8-S の学習器・学習の入口の試験用の模擬環境（エミュレータを使わない）。

観測 704 次元・行動 80 次元・マスク（M-3・M-5'・M-9）・二段の時間刻み（実際の夏季の暦で 378 ステップ）・終端の info の項目を、
本物と同じ形にする。熱応答の代理モデルではない。成績は、ステップごとに決めた「隠れた目標の行動」との違いの重み付き和で決める
（ステップごとの違いの足し算なので、区間ごとの改善が足し算になる。方策がステップの時刻から目標を覚えられるよう、観測の一部は
ステップ番号から決まる）。reset のたびに同じ系列を返す（本物は乱数種固定で完全に決定論的）。
"""
import datetime

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete

from rlkit import constants, eci as eci_mod, reward as reward_mod, timing
from rlkit.v8 import action as act8
from rlkit.v8 import observation as obs8

SUMMER_START = datetime.datetime(1999, 7, 21, 0, 0)
MOCK_W = 159497.0
T = constants.ZONE_STEPS_PER_WEEK


class MockSummerEnvV8(Env):
    metadata = {"render_modes": []}

    def __init__(self, stage=1, seed=0, include_indoor_unit_power=None, zone_step_energy_norm_gj=None, slot=None,
                 m9_enabled=True, setpoint_grid=None, comm_rpm=False):
        super().__init__()
        self.T = T
        self.W = MOCK_W
        self.stage = stage
        self.observation_space = Box(low=obs8.OBS_LOW.astype(np.float32), high=obs8.OBS_HIGH.astype(np.float32), dtype=np.float32)
        self.action_space = MultiDiscrete(act8.NVEC)
        rng = np.random.default_rng(seed)
        self._obs_table = rng.uniform(0.0, 1.0, size=(T + 1, obs8.TOTAL_DIM)).astype(np.float32)
        self._target = np.stack([[rng.integers(0, n) for n in act8.NVEC] for _ in range(T)]).astype(np.int64)
        self._weight_e = rng.uniform(0.0, 1.0, size=act8.ACTION_DIMS)
        self._weight_d = rng.uniform(0.0, 1.0, size=act8.ACTION_DIMS)
        self._weight_e /= self._weight_e.sum()
        self._weight_d /= self._weight_d.sum()
        self.m5_enabled = True
        self.m9_enabled = bool(m9_enabled)
        self.reference = None
        self._reward_calc = None
        self.diag_info = False

    def set_stage(self, stage):
        self.stage = stage
        return stage

    def set_reference(self, reference_dict):
        self.reference = reward_mod.RewardReference.from_dict(reference_dict) if reference_dict else None
        self._reward_calc = reward_mod.RewardCalculator(self.reference, self.T, self.W) if self.reference else None
        return True

    def set_m5_enabled(self, enabled):
        self.m5_enabled = bool(enabled)
        return self.m5_enabled

    def set_diag_info(self, enabled):
        self.diag_info = bool(enabled)
        return self.diag_info

    def _no_cooling(self):
        unoccupied = self._dt.weekday() >= 5 or not (7 <= self._dt.hour < 21)
        return np.full(constants.N_ZONES, unoccupied)

    def _idle_tenants(self):
        return np.array([bool(self._no_cooling()[0])] * act8.N_TENANTS)

    def _obs(self):
        return self._obs_table[min(self._t, T)]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._dt = SUMMER_START
        self._t = 0
        self._prev_action = act8.initial_action()
        self._E = 0.0
        self._DR = 0.0
        self._cool_unoccupied = 0
        self._unoccupied_steps = 0
        self._m5_zone_steps = 0
        self._m9_tenant_steps = 0
        self._vent_steps = np.zeros((2, len(constants.VENT_LEVELS)))
        self._remote_steps = np.zeros(2)
        self._hex_gj = 0.0
        self._r8p_e, self._r8p_n = [], []
        self._use = {"duty_counts_occupied": np.zeros(4, dtype=np.int64), "duty_counts_unoccupied": np.zeros(4, dtype=np.int64),
                     "prohibit_zone_steps": 0, "bypass_tenant_steps": 0, "mid_step_writes": 0}
        return self._obs(), {"action_mask": self.action_masks(), "sim_datetime": self._dt, "stage": self.stage}

    def action_masks(self):
        return act8.mask_to_flat(act8.compute_mask(
            self._prev_action, timing.is_system_boundary(self._dt), self._no_cooling(),
            self._idle_tenants(), m5_enabled=self.m5_enabled, m9_enabled=self.m9_enabled))

    def step(self, act):
        act = np.asarray(act, dtype=np.int64)
        seconds = timing.zone_step_seconds(self._dt)
        diff = (act != self._target[self._t]).astype(np.float64)
        mis_e = float(diff @ self._weight_e)
        mis_d = float(diff @ self._weight_d)
        delta_E = (0.005 + 0.015 * mis_e) * seconds / 3600.0
        n_total = (0.1 + 0.2 * mis_d) * seconds / (168 * 3600.0)
        self._E += delta_E
        self._DR += n_total
        unoccupied = self._no_cooling()
        self._unoccupied_steps += int(unoccupied.sum())
        if self.m5_enabled:
            self._m5_zone_steps += int(unoccupied.sum())
        if self.m9_enabled:
            self._m9_tenant_steps += int(self._idle_tenants().sum())
        duty = act8.duty_indices(act)
        self._cool_unoccupied += int((unoccupied & (duty > 0)).sum())
        for k in range(constants.N_ZONES):
            self._use["duty_counts_unoccupied" if unoccupied[k] else "duty_counts_occupied"][int(duty[k])] += 1
        self._use["prohibit_zone_steps"] += int((act[act8.dim_slice("zone_remote")] == act8.REMOTE_PROHIBIT).sum())
        self._use["bypass_tenant_steps"] += int((act[act8.dim_slice("tenant_bypass")] == act8.BYPASS_ON).sum())
        d = act8.decode_action(act)
        for j, t in enumerate(act8.TENANT_ORDER):
            self._vent_steps[j, d["tenants"][t]] += 1
            self._remote_steps[j] += 1.0
        self._hex_gj += 0.1 * delta_E
        zone_energy = np.full(constants.N_ZONES, 0.9 * delta_E / constants.N_ZONES)
        header_energy = delta_E - float(zone_energy.sum())
        zone_n = np.full(constants.N_ZONES, n_total / constants.N_ZONES)
        occupants = np.where(unoccupied, 0.0, 1.0)
        self._r8p_e.append(zone_energy)
        self._r8p_n.append(n_total)
        if self._reward_calc is not None:
            r = self._reward_calc.step_reward(delta_E, n_total)
            zone_rewards = self._reward_calc.zone_rewards(zone_energy, zone_n)
            system_reward = self._reward_calc.system_reward(delta_E, zone_energy)
        else:
            r, zone_rewards, system_reward = 0.0, np.zeros(constants.N_ZONES), 0.0
        self._prev_action = act
        self._dt += datetime.timedelta(seconds=seconds)
        self._t += 1
        terminated = self._t >= self.T
        info = {"action_mask": self.action_masks(), "sim_datetime": self._dt, "stage": self.stage,
                "zone_rewards": zone_rewards, "system_reward": system_reward, "step_seconds": seconds,
                "delta_E_gj": delta_E, "zone_energy_gj": zone_energy, "zone_dissatisfaction_increment": zone_n,
                "header_energy_gj": header_energy, "raw_resid_gj": 0.0, "m5_enabled": self.m5_enabled,
                "averaged_dissatisfaction_rate": self._DR, "zone_occupants_at_decision": occupants}
        if terminated:
            n = float(self._t)
            info.update({"E_rep": self._E, "DR_rep": self._DR, "eci_official": eci_mod.eci(self._E, self._DR),
                         "eci_smooth": eci_mod.eci_smooth(self._E, self._DR), "J": None, "ECI": eci_mod.eci_smooth(self._E, self._DR),
                         "P_E": eci_mod.percentile_energy(self._E), "P_DR": eci_mod.percentile_dissatisfaction(self._DR),
                         "sum_delta_E_gj": self._E, "delta_E_vs_E_rep_rel": 0.0,
                         "sum_zone_energy_gj": 0.9 * self._E, "sum_header_energy_gj": 0.1 * self._E,
                         "header_energy_fraction": 0.1, "raw_resid_sum_gj": 0.0, "raw_resid_rel": 0.0,
                         "observation_clip_rate": np.zeros(constants.OBSERVATION_DIM),
                         "obs25_clip_rate_mean": 0.0, "obs25_clip_rate_max": 0.0,
                         "obs29_clip_rate_mean": 0.0, "obs29_clip_rate_max": 0.0,
                         "co2_clip_rate_mean": 0.0, "co2_clip_rate_max": 0.0,
                         "load_clip_rate_mean": 0.0, "load_clip_rate_max": 0.0, "progress_clip_rate": 0.0,
                         "vent_obs_clip_rate": 0.0, "remote_obs_clip_rate": 0.0, "co2_delta_obs_clip_rate": 0.0,
                         "co2_pred_obs_clip_rate": 0.0, "setpoint_override_rate": 0.0,
                         "cooling_fraction_zone_unoccupied": self._cool_unoccupied / max(self._unoccupied_steps, 1),
                         "reward_sum": 0.0, "n_steps": self._t, "m5_enabled": self.m5_enabled, "m9_enabled": self.m9_enabled,
                         "m5_zone_steps": self._m5_zone_steps, "m9_tenant_steps": self._m9_tenant_steps,
                         "vent_level_fraction": {t: {lv: float(self._vent_steps[j, i] / n)
                                                     for i, lv in enumerate(constants.VENT_LEVELS)}
                                                 for j, t in enumerate(act8.TENANT_ORDER)},
                         "vent_level_hours": {t: {lv: 0.0 for lv in constants.VENT_LEVELS} for t in act8.TENANT_ORDER},
                         "hex_energy_gj": self._hex_gj,
                         "remote_permit_fraction": {t: float(self._remote_steps[j] / n) for j, t in enumerate(act8.TENANT_ORDER)},
                         "onoff_mismatch": 0, "v8_new_obs_clip_rate_max": 0.0,
                         "v8_use": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in self._use.items()}})
            share = np.arange(1, constants.N_ZONES + 1, dtype=np.float64)
            share /= share.sum()
            n_t = np.asarray(self._r8p_n)
            n_kt = n_t[:, None] * share[None, :]
            e = np.asarray(self._r8p_e)
            zr = self._reward_calc.zone_rewards(e, n_kt) if self._reward_calc is not None else np.zeros_like(n_kt)
            total = float(n_kt.sum())
            info.update({"r8p_branch": constants.R8P_BRANCH, "r8p_delta_min": constants.R8P_DELTA_MIN,
                         "r8p_zone_n": n_kt, "r8p_building_n": n_t, "r8p_zone_rewards": zr, "r8p_sum_n": total,
                         "r8p_sum_rel": abs(total - self._DR) / self._DR, "r8p_min_n": float(n_t.min()),
                         "r8p_zone_sum_max_abs": 0.0, "r8p_empty_steps": 0, "r8p_empty_max_abs_n": 0.0,
                         "r8p_minute_rows": 0, "r8p_minute_B_week_rel": 0.0, "r8p_seconds": 0.0,
                         "raw_resid_dissatisfaction_rel": (total - self._DR) / self._DR,
                         "raw_resid_dissatisfaction_endpoint_rel": 0.0})
        return self._obs(), float(r), terminated, False, info
