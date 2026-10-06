"""rlkit.action — 夏季の行動空間・展開・アクションマスク（v2 第 5-3 節・第 5-4 節、指示書 V6-S 第 3-1 節・第 3-2 節）。

次元の並びと物理値への展開は現行の _layout()・decode_action() の方式を維持する（v2 第 5-3 節）。
V6-S: 単一段階（A-7）。運転状態 2 択、設定温度 9 択（20〜28 ℃）、風量 3 択、蒸発温度 8 択、
換気 4 択（停止・Low・Middle・High、毎ステップ）、リモコン許可 2 択（K-9 結果 B、3,600 秒境界）。
62 次元・296 ロジット。マスクはロジットへの −∞ 加算で方策側が適用し、環境側は行動を書き換えない。
マスクは M-3（系統とリモコン）、M-5'（12 時間先まで在室の無いゾーンは停止だけ）、M-9（非在室テナントの換気停止）。
M-1 は更新時の勾配遮断 GC-1（gc1_excluded）に置き換え、M-2・M-8 は廃止した。
"""
import math

import numpy as np
from gymnasium.spaces import MultiDiscrete

from rlkit import constants

N_ZONES = constants.N_ZONES
N_SYSTEMS = 4
N_TENANTS = 2

# 次元レイアウト（開始インデックス, 個数）。60 次元までは現行 action._layout("summer") と同一で、
# V6-S でリモコン許可（K-9 結果 B）を末尾に加えた。
LAYOUT = {
    "zone_state": (0, 18),
    "zone_setpoint": (18, 18),
    "zone_fan": (36, 18),
    "system_evap": (54, 4),
    "tenant_vent": (58, 2),
    "tenant_remote": (60, 2),
}
# 系統とテナントの行動次元の並び: 系統 1..4、テナント South, North（現行と同じ。観測 +2 のテナント旗と同じ順）
VENT_TENANT_ORDER = ["South", "North"]
TENANT_OF_ZONE = [VENT_TENANT_ORDER.index(z[2]) for z in constants.ZONE_DEFS]
SYSTEM_OF_ZONE = [z[0] - 1 for z in constants.ZONE_DEFS]


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
    n_vent = len(constants.VENT_LEVELS)
    n_remote = len(constants.TENANT_REMOTE_PERMIT)
    return [n_state] * N_ZONES + [n_setpoint] * N_ZONES + [n_fan] * N_ZONES + \
           [n_evap] * N_SYSTEMS + [n_vent] * N_TENANTS + [n_remote] * N_TENANTS


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
    """群ごとの {start, n, K, lnK}（群内の次元は同じ選択肢数 K を持つ。L-1 の c_g = c0 / ln K_g に使う）。"""
    v = nvec(stage)
    out = {}
    for g in constants.ENTROPY_GROUPS:
        start, n = LAYOUT[g]
        K = int(v[start])
        out[g] = {"start": start, "n": n, "K": K, "lnK": math.log(K)}
    return out


def initial_action(stage):
    """エピソード最初のステップの「前回値」。現行と同じく全次元 0（全ゾーン停止・各格子の先頭・換気停止・
    リモコン禁止）。最初のステップは 0 時 0 分で 3,600 秒境界のため、M-3 の退化は掛からない。"""
    return np.zeros(constants.ACTION_DIMS, dtype=np.int64)


def v5_to_v6_actions(v5_actions):
    """V5（60 次元、設定温度 22〜28 ℃の 7 択、換気は発停 2 択）の行動列を V6 の索引へ写す（段階 B-1）。

    換気の運転 → Middle（2）、停止 → 停止（0）。設定温度は物理値のまま新しい格子の索引へ（22 ℃ → 2）。
    リモコン許可は V5 までの固定値（許可 = 1）。
    """
    v5 = np.asarray(v5_actions, dtype=np.int64)
    single = v5.ndim == 1
    v5 = np.atleast_2d(v5)
    assert v5.shape[1] == 60
    out = np.zeros((v5.shape[0], constants.ACTION_DIMS), dtype=np.int64)
    out[:, :60] = v5
    v5_grid = [22.0 + 1.0 * i for i in range(7)]
    grid = constants.ZONE_SETPOINT_C[1]
    s0 = LAYOUT["zone_setpoint"][0]
    lookup = np.array([grid.index(x) for x in v5_grid], dtype=np.int64)
    out[:, s0:s0 + N_ZONES] = lookup[v5[:, s0:s0 + N_ZONES]]
    v0 = LAYOUT["tenant_vent"][0]
    out[:, v0:v0 + N_TENANTS] = np.where(v5[:, v0:v0 + N_TENANTS] == 1, constants.VENT_LEVELS.index("Middle"),
                                         constants.VENT_OFF)
    r0 = LAYOUT["tenant_remote"][0]
    out[:, r0:r0 + N_TENANTS] = constants.TENANT_REMOTE_PERMIT.index(True)
    return out[0] if single else out


def convert_action_stage1_to_stage2(action):
    """第 1 段階の行動索引を第 2 段階の格子へ写す（設定温度の索引 i → 2i。他の次元は同じ）。"""
    out = np.asarray(action, dtype=np.int64).copy()
    start, n = LAYOUT["zone_setpoint"]
    out[start:start + n] *= 2
    return out


def decode_action(action, stage):
    """action: shape=(62,) の整数配列 → 物理値の構造化表現（現行 decode_action の形に換気の水準とリモコンを加えた）。"""
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
    tenants = {t: int(action[vent_start + i]) for i, t in enumerate(VENT_TENANT_ORDER)}   # 換気の水準 0〜3
    remote_start, _ = LAYOUT["tenant_remote"]
    remote = {t: constants.TENANT_REMOTE_PERMIT[int(action[remote_start + i])] for i, t in enumerate(VENT_TENANT_ORDER)}
    return {"zones": zones, "systems": systems, "tenants": tenants, "remote": remote}


def zone_on_flags(action):
    """各ゾーンが運転中（停止以外）か（18,）。停止の索引は段階によらず 0。"""
    start, n = LAYOUT["zone_state"]
    off = constants.ZONE_STATE.index("off")
    return np.asarray(action)[start:start + n] != off


# =====================================================================
# マスク（V6-S 第 3-2 節）。夏季に適用するのは M-3・M-5'・M-9。M-1 は GC-1 へ置換、M-2・M-4・M-8 は無い。
# =====================================================================

def _degenerate_to(dim_mask, value_index):
    dim_mask[:] = False
    dim_mask[int(value_index)] = True


def compute_mask(prev_action, is_boundary, stage, no_cooling_zones=None, m5_enabled=True,
                 idle_tenants=None, m9_enabled=True):
    """各次元について有効な値を True とする真偽配列のリスト（nvec(stage) と同じ形）。

    prev_action: 直前に適用した行動。
    is_boundary: このステップが系統側 3,600 秒境界か。
    no_cooling_zones: M-5' の対象（在室者数 0 かつ外生表で 12 時間先まで在室 0）。
    m5_enabled: M-5' を適用するか（V6 では常に真。解除の仕組みは廃止）。
    idle_tenants: M-9 の対象（テナントの在室者数 0 かつ 1 時間先も 0）。並びは VENT_TENANT_ORDER。
    m9_enabled: M-9 を適用するか（V6-S で有効）。
    """
    v = nvec(stage)
    prev_action = np.asarray(prev_action)
    mask = [np.ones(n, dtype=bool) for n in v]
    states = zone_states(stage)
    off_idx = states.index("off")
    state_start, _ = LAYOUT["zone_state"]

    # M-3: 3,600 秒境界でないステップでは系統の次元とリモコン許可を前回値の 1 点に退化（換気は対象外）
    if not is_boundary:
        for key in ("system_evap", "tenant_remote"):
            start, n = LAYOUT[key]
            for i in range(start, start + n):
                _degenerate_to(mask[i], prev_action[i])

    # M-5': 運転を除き停止だけを許す
    if m5_enabled and no_cooling_zones is not None:
        for k in np.flatnonzero(np.asarray(no_cooling_zones, dtype=bool)):
            _degenerate_to(mask[state_start + k], off_idx)

    # M-9: 在室者数 0 かつ 1 時間先も 0 のテナントの換気を停止（0）に固定する
    if m9_enabled and idle_tenants is not None:
        vent_start, _ = LAYOUT["tenant_vent"]
        for j, idle in enumerate(np.asarray(idle_tenants, dtype=bool)):
            if idle:
                _degenerate_to(mask[vent_start + j], constants.VENT_OFF)

    return mask


def zone_on_flags_batch(actions):
    """(T, 62) → 各ステップ・各ゾーンが運転中か (T, 18)。"""
    start, n = LAYOUT["zone_state"]
    off = constants.ZONE_STATE.index("off")
    return np.asarray(actions)[:, start:start + n] != off


def gc1_excluded(actions):
    """GC-1（V6-S 第 3-2 節）: 標本化した行動から、更新で除外する要素を真とする (T, 62)。

    停止を標本化したゾーン k の設定温度・風量と、系統内に運転ゾーンが 1 つも無いステップの系統 s の蒸発温度。
    1 択の要素（マスクで選択肢が 1 つ）は呼び出し側がマスクから別に除く。
    """
    actions = np.atleast_2d(np.asarray(actions))
    T = actions.shape[0]
    out = np.zeros((T, constants.ACTION_DIMS), dtype=bool)
    on = zone_on_flags_batch(actions)
    for key in ("zone_setpoint", "zone_fan"):
        start, n = LAYOUT[key]
        out[:, start:start + n] = ~on
    evap_start, _ = LAYOUT["system_evap"]
    sys_idx = np.asarray(SYSTEM_OF_ZONE)
    for s_ in range(N_SYSTEMS):
        out[:, evap_start + s_] = ~on[:, sys_idx == s_].any(axis=1)
    return out


def mask_to_flat(mask):
    """MaskablePPO が要求する、全次元を連結した 1 本の bool 配列。"""
    return np.concatenate([np.asarray(m, dtype=bool) for m in mask])
