"""rlkit.mockenv — 学習器・並列層・学習の入口の試験用の模擬環境（指示書 V6-S の行動・観測の形）。

エミュレータを使わない。観測・行動の形、マスクの規則（M-3・M-5'・M-9 の旗を含む）、二段の時間刻み
（実際の夏季の暦で 378 ステップ）、info の項目、報酬の式だけを本物と同じにする。熱応答の代理モデルではない。
成績は行動の索引ではなく物理量（運転状態・設定温度・風量・蒸発温度・換気の水準・リモコン許可）から決める。
本物の環境は乱数種固定のもとで完全に決定論的（P-1）なので、模擬環境も reset のたびに同じ系列を返す。
R-8'（V6.4-S）: 終端の info に本物と同じ名前の r8p_* を入れる。建物の値は各ステップの n_t のまま、ゾーンへの配分だけを
ゾーン番号に比例する比に替える（終端 1 点の等分と区別でき、置き換えが効いたことを試験で確かめられるようにする）。
"""
import datetime

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box

from rlkit import action, constants, eci as eci_mod, observation, reward as reward_mod, timing

SUMMER_START = datetime.datetime(1999, 7, 21, 0, 0)
MOCK_W = 159497.0


class MockSummerEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, stage=1, seed=0, include_indoor_unit_power=None, zone_step_energy_norm_gj=None, slot=None,
                 m9_enabled=True, setpoint_grid=None, comm_rpm=False):
        super().__init__()
        self.T = constants.ZONE_STEPS_PER_WEEK
        self.W = MOCK_W
        self.observation_space = Box(low=observation.OBS_LOW.astype(np.float32), high=observation.OBS_HIGH.astype(np.float32),
                                     dtype=np.float32)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        rng = np.random.default_rng(seed)
        target_action = np.array([rng.integers(0, n) for n in action.nvec(1)], dtype=np.int64)
        self.stage = None
        self.set_stage(stage)
        self._target_phys = self._physical(target_action, stage=1)
        self.m5_enabled = True
        self.m9_enabled = bool(m9_enabled)
        self.reference = None
        self._reward_calc = None

    def set_stage(self, stage):
        self.stage = stage
        self.action_space = action.action_space(stage)
        return stage

    def set_reference(self, reference_dict):
        self.reference = reward_mod.RewardReference.from_dict(reference_dict) if reference_dict else None
        self._reward_calc = reward_mod.RewardCalculator(self.reference, self.T, self.W) if self.reference else None
        return True

    def set_m5_enabled(self, enabled):
        self.m5_enabled = bool(enabled)
        return self.m5_enabled

    def _idle_tenants(self):
        return np.array([bool(self._no_cooling()[0])] * len(constants.TENANTS))

    def _physical(self, act, stage=None):
        """行動を物理量の並びに直す（18 ゾーン × 3 ＋ 系統 4 ＋ テナント 2 × 2 = 62。C-5 の後のリモコンは常に許可）。"""
        d = action.decode_action(act, self.stage if stage is None else stage)
        vals = []
        for label in constants.ZONE_LABELS:
            z = d["zones"][label]
            vals.append(float(constants.ZONE_STATE.index(z["state"])))
            vals.append(0.0 if z["setpoint"] is None else float(z["setpoint"]))
            vals.append(float(constants.ZONE_FAN_SPEED.index(z["fan_speed"])))
        vals.extend(float(d["systems"][ou]["evap_c"]) for ou in range(1, 5))
        vals.extend(float(d["tenants"][t]) for t in action.VENT_TENANT_ORDER)
        vals.extend(1.0 if d["remote"][t] else 0.0 for t in action.VENT_TENANT_ORDER)
        return np.asarray(vals, dtype=np.float64)

    def _obs(self):
        return self._rng.uniform(0.0, 1.0, size=constants.OBSERVATION_DIM).astype(np.float32)

    def _no_cooling(self):
        unoccupied = self._dt.weekday() >= 5 or not (7 <= self._dt.hour < 21)
        return np.full(constants.N_ZONES, unoccupied)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(self._seed if seed is None else seed)
        self._dt = SUMMER_START
        self._t = 0
        self._prev_action = action.initial_action(self.stage)
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
        return self._obs(), {"action_mask": self.action_masks(), "sim_datetime": self._dt, "stage": self.stage}

    def action_masks(self):
        return action.mask_to_flat(action.compute_mask(
            self._prev_action, timing.is_system_boundary(self._dt), self.stage, self._no_cooling(),
            m5_enabled=self.m5_enabled, idle_tenants=self._idle_tenants(), m9_enabled=self.m9_enabled))

    def step(self, act):
        act = np.asarray(act, dtype=np.int64)
        seconds = timing.zone_step_seconds(self._dt)
        mismatch = float(np.mean(self._physical(act) != self._target_phys))
        delta_E = (0.005 + 0.015 * mismatch) * seconds / 3600.0
        n_total = (0.1 + 0.2 * mismatch) * seconds / (168 * 3600.0)
        self._E += delta_E
        self._DR += n_total
        unoccupied = self._no_cooling()
        self._unoccupied_steps += int(unoccupied.sum())
        if self.m5_enabled:
            self._m5_zone_steps += int(unoccupied.sum())
        if self.m9_enabled:
            self._m9_tenant_steps += int(self._idle_tenants().sum())
        self._cool_unoccupied += int((unoccupied & (act[:constants.N_ZONES] == constants.ZONE_STATE.index("cool"))).sum())
        d = action.decode_action(act, self.stage)
        for j, t in enumerate(action.VENT_TENANT_ORDER):
            self._vent_steps[j, d["tenants"][t]] += 1
            self._remote_steps[j] += int(d["remote"][t])
        self._hex_gj += 0.1 * delta_E
        zone_energy = np.full(constants.N_ZONES, 0.9 * delta_E / constants.N_ZONES)
        header_energy = delta_E - float(zone_energy.sum())
        zone_n = np.full(constants.N_ZONES, n_total / constants.N_ZONES)
        occupants = np.where(self._no_cooling(), 0.0, 1.0)
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
                         "vent_obs_clip_rate": 0.0, "remote_obs_clip_rate": 0.0, "co2_delta_obs_clip_rate": 0.0, "co2_pred_obs_clip_rate": 0.0,
                         "setpoint_override_rate": 0.0,
                         "cooling_fraction_zone_unoccupied": self._cool_unoccupied / max(self._unoccupied_steps, 1),
                         "reward_sum": 0.0, "n_steps": self._t, "m5_enabled": self.m5_enabled,
                         "m9_enabled": self.m9_enabled,
                         "m5_zone_steps": self._m5_zone_steps, "m9_tenant_steps": self._m9_tenant_steps,
                         "vent_level_fraction": {t: {lv: float(self._vent_steps[j, i] / n)
                                                     for i, lv in enumerate(constants.VENT_LEVELS)}
                                                 for j, t in enumerate(action.VENT_TENANT_ORDER)},
                         "vent_level_hours": {t: {lv: 0.0 for lv in constants.VENT_LEVELS} for t in action.VENT_TENANT_ORDER},
                         "hex_energy_gj": self._hex_gj,
                         "remote_permit_fraction": {t: float(self._remote_steps[j] / n)
                                                    for j, t in enumerate(action.VENT_TENANT_ORDER)},
                         "onoff_mismatch": 0})
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
