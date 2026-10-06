"""rlkit.advantage — 時間基準 λ の GAE と、優位度の行動次元への割り当て（v2 第 5-5-2 節・第 5-6 節）。

GAE は報酬・価値について線形であるため、報酬の加法分解 r_t = Σ_k r_k,t + r_sys,t に合わせて
19 出力を独立に計算した優位度の総和は、全体の優位度と一致する（現行 advantage.py の性質を維持）。
"""
import numpy as np

from rlkit import action as action_mod
from rlkit import constants

N_ZONES = constants.N_ZONES
SYSTEM_OF_ZONE = np.array([ou - 1 for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS])
TENANT_OF_ZONE = np.array([action_mod.VENT_TENANT_ORDER.index(tenant)
                           for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS])


def lambda_per_step(step_seconds, lambda_hour=constants.LAMBDA_HOUR):
    """λ_t = λ_hour^(Δt_t / 1 時間)。900 秒で 0.987、3,600 秒で 0.95。"""
    return np.power(lambda_hour, np.asarray(step_seconds, dtype=np.float64) / 3600.0)


def gae_variable_lambda(rewards, values, lambdas, gamma=constants.GAMMA, last_value=0.0):
    """1 本の軌跡の GAE。A_t = δ_t + γ·λ_t·A_{t+1}、δ_t = r_t + γ·V_{t+1} − V_t。

    rewards・values: (T,) または (T, M)。lambdas: (T,)。終端の次状態価値は last_value（エピソード完結は 0）。
    Returns: advantages, returns（returns = advantages + values）
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    running = np.zeros_like(rewards[0])
    next_value = np.full_like(rewards[0], last_value)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lambdas[t] * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values


def episode_boundaries_from_starts(episode_starts):
    """episode_starts（先頭を除き真の位置でエピソードが区切れる）から [0, T) を覆う区間の一覧（現行と同じ）。"""
    T = len(episode_starts)
    starts = [0] + [t for t in range(1, T) if bool(episode_starts[t])]
    return [(s, starts[i + 1] if i + 1 < len(starts) else T) for i, s in enumerate(starts)]


def flatten_env_major(arr):
    """(T, N, ...) → (N·T, ...)。並びは「組0の全ステップ、組1の全ステップ、…」（SB3 の swap_and_flatten と同じ）。"""
    arr = np.asarray(arr)
    return arr.swapaxes(0, 1).reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])


def segmented_gae(rewards, values, step_seconds, segments, lambda_hour=constants.LAMBDA_HOUR,
                  gamma=constants.GAMMA):
    """エピソード境界で区切り、区間ごとに独立して時間基準 λ の GAE を計算する。

    rewards・values: (T, M)、step_seconds: (T,)。Returns: advantages (T, M)、returns (T, M)
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    lambdas = lambda_per_step(step_seconds, lambda_hour)
    adv = np.zeros_like(rewards)
    ret = np.zeros_like(rewards)
    for s, e in segments:
        adv[s:e], ret[s:e] = gae_variable_lambda(rewards[s:e], values[s:e], lambdas[s:e], gamma=gamma)
    return adv, ret


FAMILY_KEYS = {"S": ("zone_state",), "P": ("zone_setpoint", "zone_fan"), "V": ("tenant_vent",)}
SYSTEM_EVAP_LEVEL = "system_evap"


def level_zone_advantages(zone_adv, system_adv, actions, level):
    """ゾーン k（またはテナント t）の次元に与える水準別の優位度（指示書 V6.2-S 第 3 章の式）。

    local    A_k
    system   Σ_{j∈s(k)} A_j + A_sys · (系統 s(k) の運転ゾーン数 ÷ 全運転ゾーン数)（全運転ゾーン数が 0 なら第 2 項は 0）
    tenant   Σ_{j∈t(k)} A_j
    building Σ_j A_j + A_sys
    Returns: (T, 18)（ゾーン k の列）。テナントの次元は tenant_level_advantages を使う
    """
    zone_adv = np.asarray(zone_adv, dtype=np.float64)
    system_adv = np.asarray(system_adv, dtype=np.float64)
    if level == "local":
        return zone_adv.copy()
    if level == "building":
        return np.repeat((zone_adv.sum(axis=1) + system_adv)[:, None], N_ZONES, axis=1)
    if level == "tenant":
        tsum = np.stack([zone_adv[:, TENANT_OF_ZONE == t].sum(axis=1) for t in range(2)], axis=1)
        return tsum[:, TENANT_OF_ZONE]
    if level == "system":
        on = action_mod.zone_on_flags_batch(actions)
        n_on_total = on.sum(axis=1)
        out = np.zeros_like(zone_adv)
        for s in range(4):
            members = SYSTEM_OF_ZONE == s
            n_on_s = on[:, members].sum(axis=1)
            share = np.where(n_on_total > 0, n_on_s / np.maximum(n_on_total, 1), 0.0)
            out[:, members] = (zone_adv[:, members].sum(axis=1) + system_adv * share)[:, None]
        return out
    raise ValueError(f"水準の候補に無い: {level!r}")


def tenant_level_advantages(zone_adv, system_adv, level):
    """テナント t の換気の次元に与える水準別の優位度 (T, 2)。tenant: Σ_{j∈t} A_j、building: Σ_j A_j + A_sys。"""
    zone_adv = np.asarray(zone_adv, dtype=np.float64)
    system_adv = np.asarray(system_adv, dtype=np.float64)
    if level == "tenant":
        return np.stack([zone_adv[:, TENANT_OF_ZONE == t].sum(axis=1) for t in range(2)], axis=1)
    if level == "building":
        return np.repeat((zone_adv.sum(axis=1) + system_adv)[:, None], 2, axis=1)
    raise ValueError(f"家族 V の水準の候補に無い: {level!r}")


def dim_levels(levels=None):
    """各行動次元の水準の名前 (ACTION_DIMS,)。正規化（C-2）と SIL の σ_A の単位になる。"""
    levels = constants.CREDIT_LEVELS if levels is None else levels
    L = action_mod.LAYOUT
    out = np.empty(constants.ACTION_DIMS, dtype=object)
    for fam, keys in FAMILY_KEYS.items():
        for key in keys:
            start, n = L[key]
            out[start:start + n] = levels[fam]
    start, n = L["system_evap"]
    out[start:start + n] = SYSTEM_EVAP_LEVEL
    if "tenant_remote" in L:
        start, n = L["tenant_remote"]
        out[start:start + n] = levels["V"]
    return out


def assign_per_dim_advantages(zone_adv, system_adv, actions, levels=None):
    """19 出力の優位度を行動次元へ割り当てる（V6.2-S C-1。水準は constants.CREDIT_LEVELS、K-14' で決めた）。

    家族 S（運転状態）・P（設定温度・風量）・V（換気）: その家族に採った水準の優位度（level_zone_advantages）
    系統 s の冷媒温度（現行のまま）→ Σ_{k∈s} A_k + A_sys · (系統 s の運転ゾーン数 ÷ 全運転ゾーン数)
    全運転ゾーン数が 0 のステップでは、冷媒温度に限り A_sys を a_s の比で 4 系統へ配る（V6-S までの補完を維持）。

    Args: zone_adv (T, 18)、system_adv (T,)、actions (T, ACTION_DIMS)
    Returns: (T, ACTION_DIMS)
    """
    levels = constants.CREDIT_LEVELS if levels is None else levels
    zone_adv = np.asarray(zone_adv, dtype=np.float64)
    system_adv = np.asarray(system_adv, dtype=np.float64)
    actions = np.asarray(actions)
    T = zone_adv.shape[0]
    out = np.zeros((T, constants.ACTION_DIMS), dtype=np.float64)
    L = action_mod.LAYOUT
    for fam in ("S", "P"):
        vals = level_zone_advantages(zone_adv, system_adv, actions, levels[fam])
        for key in FAMILY_KEYS[fam]:
            start, n = L[key]
            out[:, start:start + n] = vals

    on = actions[:, L["zone_state"][0]:L["zone_state"][0] + N_ZONES] != constants.ZONE_STATE.index("off")
    n_on_total = on.sum(axis=1)
    a = np.asarray(constants.OUTDOOR_UNIT_A_KW, dtype=np.float64)
    evap_start, _ = L["system_evap"]
    for s in range(4):
        members = SYSTEM_OF_ZONE == s
        n_on_s = on[:, members].sum(axis=1)
        weight = np.where(n_on_total > 0, n_on_s / np.maximum(n_on_total, 1), a[s] / a.sum())
        out[:, evap_start + s] = zone_adv[:, members].sum(axis=1) + system_adv * weight

    tv = tenant_level_advantages(zone_adv, system_adv, levels["V"])
    for key in [k for k in ("tenant_vent", "tenant_remote") if k in L]:
        start, _ = L[key]
        out[:, start:start + 2] = tv
    return out


def normalize_by_level(m, include, level_of_dim=None, eps=constants.ADVANTAGE_NORM_EPS):
    """C-2: 水準ごと（局所・系統・テナント・建物・系統の次元）に、GC-1 と 1 択で除かない要素だけの平均・標準偏差で正規化する。

    Returns: (正規化後 (T, D), {水準: {"mean", "std", "n"}}, 要素ごとの σ (T, D)（SIL の σ_A に使う）)
    要素の無い水準は平均 0・標準偏差 1 とする（その水準の要素は更新に寄与しない）。
    """
    m = np.asarray(m, dtype=np.float64)
    inc = np.asarray(include, dtype=bool)
    level_of_dim = dim_levels() if level_of_dim is None else level_of_dim
    out = np.zeros_like(m)
    sigma = np.ones_like(m)
    stats = {}
    for lv in dict.fromkeys(level_of_dim):
        cols = np.flatnonzero(level_of_dim == lv)
        vals = m[:, cols][inc[:, cols]]
        if vals.size == 0:
            mean, std = 0.0, 1.0
        else:
            mean, std = float(vals.mean()), float(vals.std())
        out[:, cols] = (m[:, cols] - mean) / (std + eps)
        sigma[:, cols] = std
        stats[lv] = {"mean": mean, "std": std, "n": int(vals.size)}
    return out, stats, sigma


def normalize_matrix(m, eps=constants.ADVANTAGE_NORM_EPS):
    """割り当て後の行列の全要素で平均 0・標準偏差 1 に正規化する。"""
    m = np.asarray(m, dtype=np.float64)
    return (m - m.mean()) / (m.std() + eps)
