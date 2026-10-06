"""rlkit.action — 夏季の行動空間・展開・アクションマスク（v2 第 5-3 節・第 5-4 節）。

次元の並びと物理値への展開は現行の _layout()・decode_action() の方式を維持する（v2 第 5-3 節）。
第 1 段階は設定温度 7 択（22〜28 ℃、1.0 ℃刻み、270 ロジット）、第 2 段階は 13 択（0.5 ℃刻み、
378 ロジット）。マスクはロジットへの −∞ 加算で方策側が適用し、環境側は行動を書き換えない。
"""
import math

import numpy as np
from gymnasium.spaces import MultiDiscrete

from rlkit import constants

N_ZONES = constants.N_ZONES
N_SYSTEMS = 4
N_TENANTS = 2

# 次元レイアウト（開始インデックス, 個数）。現行 action._layout("summer") と同一。
LAYOUT = {
    "zone_state": (0, 18),
    "zone_setpoint": (18, 18),
    "zone_fan": (36, 18),
    "system_evap": (54, 4),
    "tenant_vent": (58, 2),
}
# 系統とテナントの行動次元の並び: 系統 1..4、テナント South, North（現行と同じ）
VENT_TENANT_ORDER = ["South", "North"]


def _check_stage(stage):
    if stage not in constants.STAGES:
        raise ValueError(f"段階は 1 または 2: {stage!r}")


def setpoint_grid(stage):
    _check_stage(stage)
    return constants.ZONE_SETPOINT_C[stage]


def zone_states(stage):
    """その段階で選べる運転状態（S-5 により第 1 段階は停止・冷房の 2 択）。"""
    _check_stage(stage)
    return constants.ZONE_STATE_BY_STAGE[stage]


def nvec(stage):
    """MultiDiscrete の各次元の選択肢数。"""
    n_state = len(zone_states(stage))
    n_setpoint = len(setpoint_grid(stage))
    n_fan = len(constants.ZONE_FAN_SPEED)
    n_evap = len(constants.SYSTEM_EVAPORATING_C)
    n_vent = len(constants.TENANT_VENT_ON)
    return [n_state] * N_ZONES + [n_setpoint] * N_ZONES + [n_fan] * N_ZONES + \
           [n_evap] * N_SYSTEMS + [n_vent] * N_TENANTS


def action_space(stage):
    v = nvec(stage)
    assert len(v) == constants.ACTION_DIMS
    assert sum(v) == constants.ACTION_LOGITS[stage]
    return MultiDiscrete(v)


def dim_logit_offsets(stage):
    """各次元の、平坦化したマスク（長さ = ロジット数）内での (開始位置, 幅)。"""
    offsets, pos = [], 0
    for k in nvec(stage):
        offsets.append((pos, k))
        pos += k
    return offsets


def group_layout(stage):
    """エントロピー床の群ごとの {start, n, K, lnK}（群内の次元は同じ選択肢数 K を持つ）。"""
    v = nvec(stage)
    out = {}
    for g in constants.ENTROPY_GROUPS:
        start, n = LAYOUT[g]
        K = int(v[start])
        out[g] = {"start": start, "n": n, "K": K, "lnK": math.log(K)}
    return out


def initial_action(stage):
    """エピソード最初のステップの「前回値」。現行と同じく全次元 0（全ゾーン停止・各格子の先頭）。"""
    return np.zeros(constants.ACTION_DIMS, dtype=np.int64)


def convert_action_stage1_to_stage2(action):
    """第 1 段階の行動索引を第 2 段階の格子へ写す（設定温度の索引 i → 2i。他の次元は同じ）。"""
    out = np.asarray(action, dtype=np.int64).copy()
    start, n = LAYOUT["zone_setpoint"]
    out[start:start + n] *= 2
    return out


def decode_action(action, stage):
    """action: shape=(60,) の整数配列 → 物理値の構造化表現（現行 decode_action と同じ形）。"""
    action = np.asarray(action)
    grid = setpoint_grid(stage)
    state_start, _ = LAYOUT["zone_state"]
    sp_start, _ = LAYOUT["zone_setpoint"]
    fan_start, _ = LAYOUT["zone_fan"]
    states = zone_states(stage)
    zones = {}
    for k, (ou, iu, tenant, zn, label, area) in enumerate(constants.ZONE_DEFS):
        state = states[int(action[state_start + k])]
        setpoint = grid[int(action[sp_start + k])] if state != "off" else None
        fan_speed = constants.ZONE_FAN_SPEED[int(action[fan_start + k])]
        zones[label] = {"state": state, "setpoint": setpoint, "fan_speed": fan_speed}
    evap_start, _ = LAYOUT["system_evap"]
    systems = {i + 1: {"evap_c": constants.SYSTEM_EVAPORATING_C[int(action[evap_start + i])]}
               for i in range(N_SYSTEMS)}
    vent_start, _ = LAYOUT["tenant_vent"]
    tenants = {t: constants.TENANT_VENT_ON[int(action[vent_start + i])] for i, t in enumerate(VENT_TENANT_ORDER)}
    return {"zones": zones, "systems": systems, "tenants": tenants}


def zone_on_flags(action):
    """各ゾーンが運転中（停止以外）か（18,）。停止の索引は段階によらず 0。"""
    start, n = LAYOUT["zone_state"]
    off = constants.ZONE_STATE.index("off")
    return np.asarray(action)[start:start + n] != off


# =====================================================================
# マスク（v2 第 5-4 節）。夏季に適用するのは M-1・M-2・M-3・M-5。
# =====================================================================

def _degenerate_to(dim_mask, value_index):
    dim_mask[:] = False
    dim_mask[int(value_index)] = True


def compute_mask(prev_action, is_boundary, stage, no_cooling_zones=None, m5_enabled=True,
                 occupied_zones=None, m8_enabled=True, idle_tenants=None, m9_enabled=False):
    """各次元について有効な値を True とする真偽配列のリスト（nvec(stage) と同じ形）。

    prev_action: 直前に適用した行動（第 stage 段階の格子の索引）。
    is_boundary: このステップが系統側 3,600 秒境界か。
    no_cooling_zones: M-5 の対象（在室者数 0 かつ 2 時間先までの在室見込み 0）。
    m5_enabled: M-5 を適用するか（G-2・G-3 の解除で切り替える）。
    occupied_zones: M-8 の対象（現在の在室者数が 1 人以上）。
    m8_enabled: M-8 を適用するか（指示書 V5-S 第 2-2 節。第 1 段階のみ真）。
    idle_tenants: M-9 の対象（テナントの在室者数 0 かつ 1 時間先も 0）。並びは VENT_TENANT_ORDER。
    m9_enabled: M-9 を適用するか（診断 K-7 (a) の結果により既定は偽）。
    """
    v = nvec(stage)
    prev_action = np.asarray(prev_action)
    mask = [np.ones(n, dtype=bool) for n in v]
    states = zone_states(stage)
    off_idx = states.index("off")
    cool_idx = states.index("cool")
    fan_idx = states.index("fan") if "fan" in states else None
    state_start, _ = LAYOUT["zone_state"]
    sp_start, _ = LAYOUT["zone_setpoint"]
    fan_start, _ = LAYOUT["zone_fan"]
    occupied = np.zeros(N_ZONES, dtype=bool) if occupied_zones is None else np.asarray(occupied_zones, dtype=bool)
    m8_zones = occupied if m8_enabled else np.zeros(N_ZONES, dtype=bool)

    for k in range(N_ZONES):
        if m8_zones[k]:
            # M-8 の在室ゾーンは常に冷房であるため、M-1・M-2 の退化は働かせない（第 2-2 節）
            continue
        prev_state = int(prev_action[state_start + k])
        if prev_state == off_idx:
            # M-1: 前ステップで停止していたゾーンの設定温度と風量を前回値の 1 点に退化
            _degenerate_to(mask[sp_start + k], prev_action[sp_start + k])
            _degenerate_to(mask[fan_start + k], prev_action[fan_start + k])
        elif fan_idx is not None and prev_state == fan_idx:
            # M-2: 前ステップで送風だったゾーンの設定温度を前回値の 1 点に退化
            _degenerate_to(mask[sp_start + k], prev_action[sp_start + k])

    # M-3: 3,600 秒境界でないステップでは系統側と換気の全次元を前回値の 1 点に退化
    if not is_boundary:
        for key in ("system_evap", "tenant_vent"):
            start, n = LAYOUT[key]
            for i in range(start, start + n):
                _degenerate_to(mask[i], prev_action[i])

    # M-5: 冷房を除外し停止（第 2 段階では送風も）だけを許す。M-8 の対象ゾーンとは排他にする
    if m5_enabled and no_cooling_zones is not None:
        for k in np.flatnonzero(np.asarray(no_cooling_zones, dtype=bool) & ~m8_zones):
            mask[state_start + k][cool_idx] = False

    # M-8: 在室者数が 1 人以上のゾーンは冷房だけを許す（停止と送風を除く。第 1 段階のみ）
    for k in np.flatnonzero(m8_zones):
        _degenerate_to(mask[state_start + k], cool_idx)

    # M-9: 在室者数 0 かつ 1 時間先も 0 のテナントの換気を停止に固定する（採用時のみ）。
    # M-3 の退化より後に置き、境界でないステップでも停止に倒す（選べる値が無くならないようにする）
    if m9_enabled and idle_tenants is not None:
        vent_start, _ = LAYOUT["tenant_vent"]
        off_vent = constants.TENANT_VENT_ON.index(False)
        for j, idle in enumerate(np.asarray(idle_tenants, dtype=bool)):
            if idle:
                _degenerate_to(mask[vent_start + j], off_vent)

    return mask


def mask_to_flat(mask):
    """MaskablePPO が要求する、全次元を連結した 1 本の bool 配列。"""
    return np.concatenate([np.asarray(m, dtype=bool) for m in mask])
