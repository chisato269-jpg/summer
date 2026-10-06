"""rlkit.v9.credit — V9-S 段階 A の信用割当（指示書 C9-4）。V6.2-S 以降（trial08）の水準を V8 の行動空間（80 次元）へ当てはめる。

19 出力（ゾーン 18 ＋ 系統 1）の優位度を、行動次元へ割り当てる。
  建物全体  運転の割合・設定温度・風量（trial08 の発停・設定温度・風量と同じ）とリモコンの許可: Σ_k A_k + A_sys
  テナント  換気と、バイパス（テナント内の和）: Σ_{k∈t} A_k
  系統      蒸発温度（系統 s）: Σ_{k∈s} A_k + A_sys · (系統 s の運転ゾーン数 ÷ 全運転ゾーン数)。運転ゾーンが無いステップは A_sys を
            a_s の比で配る（V6-S 以来の補完）。「運転」は割合 > 0 のゾーン
水準ごとの正規化（C-2）は rlkit.advantage.normalize_by_level に dim_levels() を渡して使う。
"""
import numpy as np

from rlkit import constants
from rlkit.v8 import action as act8

N_ZONES = constants.N_ZONES
SYSTEM_OF_ZONE = np.asarray(act8.ZONE_SYSTEM)
TENANT_OF_ZONE = np.asarray(act8.ZONE_TENANT)

LEVEL_OF_GROUP = {"zone_duty": "building", "zone_setpoint": "building", "zone_fan": "building", "zone_remote": "building",
                  "tenant_vent": "tenant", "tenant_bypass": "tenant", "system_evap": "system_evap"}


def dim_levels():
    """各行動次元の水準の名前 (80,)。"""
    out = np.empty(act8.ACTION_DIMS, dtype=object)
    for g in act8.GROUP_ORDER:
        s, n = act8.LAYOUT[g]
        out[s:s + n] = LEVEL_OF_GROUP[g]
    return out


def assign_per_dim_advantages(zone_adv, system_adv, actions):
    """zone_adv (T, 18)、system_adv (T,)、actions (T, 80) → (T, 80)。"""
    zone_adv = np.asarray(zone_adv, dtype=np.float64)
    system_adv = np.asarray(system_adv, dtype=np.float64)
    actions = np.asarray(actions)
    T = zone_adv.shape[0]
    out = np.zeros((T, act8.ACTION_DIMS), dtype=np.float64)
    building = zone_adv.sum(axis=1) + system_adv
    for g in ("zone_duty", "zone_setpoint", "zone_fan", "zone_remote"):
        s, n = act8.LAYOUT[g]
        out[:, s:s + n] = building[:, None]
    tenant = np.stack([zone_adv[:, TENANT_OF_ZONE == t].sum(axis=1) for t in range(act8.N_TENANTS)], axis=1)
    for g in ("tenant_vent", "tenant_bypass"):
        s, n = act8.LAYOUT[g]
        out[:, s:s + n] = tenant
    on = act8.duty_indices(actions) > 0
    n_on_total = on.sum(axis=1)
    a = np.asarray(constants.OUTDOOR_UNIT_A_KW, dtype=np.float64)
    es = act8.LAYOUT["system_evap"][0]
    for s in range(act8.N_SYSTEMS):
        members = SYSTEM_OF_ZONE == s
        n_on_s = on[:, members].sum(axis=1)
        weight = np.where(n_on_total > 0, n_on_s / np.maximum(n_on_total, 1), a[s] / a.sum())
        out[:, es + s] = zone_adv[:, members].sum(axis=1) + system_adv * weight
    return out


def group_layout():
    """エントロピー・KL の群の配置: {群: {start, n, K, lnK}}（start・n は行動次元の単位）。"""
    return {g: {"start": act8.LAYOUT[g][0], "n": act8.LAYOUT[g][1], "K": act8.GROUP_K[g],
                "lnK": float(np.log(act8.GROUP_K[g]))} for g in act8.GROUP_ORDER}
