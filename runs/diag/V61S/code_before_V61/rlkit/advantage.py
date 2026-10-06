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


def assign_per_dim_advantages(zone_adv, system_adv, actions):
    """19 出力の優位度を 62 の行動次元へ割り当てる（v2 第 5-5-2 節の信用割当の階層、設計理論書 v3 第 4-5 節）。

    ゾーン k の運転状態・設定温度・風量 → A_k
    系統 s の冷媒温度 → Σ_{k∈s} A_k + A_sys · (系統 s の運転ゾーン数 ÷ 全運転ゾーン数)
    テナント t の換気とリモコン許可（V6-S A-6） → Σ_{k∈t} A_k
    全運転ゾーン数が 0 のステップでは、A_sys を a_s の比で 4 系統へ配る（設計書に規定が無いための補完。
    |Z_s| = 0 の系統の待機電力が系統ヘッダへ入る規則に対応させた）。

    Args: zone_adv (T, 18)、system_adv (T,)、actions (T, 62)
    Returns: (T, 62)
    """
    zone_adv = np.asarray(zone_adv, dtype=np.float64)
    system_adv = np.asarray(system_adv, dtype=np.float64)
    actions = np.asarray(actions)
    T = zone_adv.shape[0]
    out = np.zeros((T, constants.ACTION_DIMS), dtype=np.float64)
    L = action_mod.LAYOUT
    for key in ("zone_state", "zone_setpoint", "zone_fan"):
        start, n = L[key]
        out[:, start:start + n] = zone_adv

    on = actions[:, L["zone_state"][0]:L["zone_state"][0] + N_ZONES] != constants.ZONE_STATE.index("off")
    n_on_total = on.sum(axis=1)
    a = np.asarray(constants.OUTDOOR_UNIT_A_KW, dtype=np.float64)
    evap_start, _ = L["system_evap"]
    for s in range(4):
        members = SYSTEM_OF_ZONE == s
        n_on_s = on[:, members].sum(axis=1)
        weight = np.where(n_on_total > 0, n_on_s / np.maximum(n_on_total, 1), a[s] / a.sum())
        out[:, evap_start + s] = zone_adv[:, members].sum(axis=1) + system_adv * weight

    for key in ("tenant_vent", "tenant_remote"):
        start, _ = L[key]
        for t in range(2):
            out[:, start + t] = zone_adv[:, TENANT_OF_ZONE == t].sum(axis=1)
    return out


def normalize_matrix(m, eps=constants.ADVANTAGE_NORM_EPS):
    """割り当て後の行列の全要素で平均 0・標準偏差 1 に正規化する。"""
    m = np.asarray(m, dtype=np.float64)
    return (m - m.mean()) / (m.std() + eps)
