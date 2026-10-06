"""rlkit.v8.action — V8-S の行動空間（指示書 第 5 章 A-1〜A-5）。

次元の並び（80 次元、ロジット 372）。V7 の 60 次元の並びを保ち、新しい次元を後ろに足す。
  zone_duty    (0, 18)  A-1 刻みの中の運転の割合 0・1/3・2/3・1（4 択。V7 の発停 2 択を置き換える）
  zone_setpoint(18, 18) A-5 設定温度 22・22.5・23・23.5・24・25・26・27・28 ℃（9 択）
  zone_fan     (36, 18) 風量 Low・Middle・High（3 択。変更なし）
  system_evap  (54, 4)  A-4 蒸発温度 14〜22 ℃の 1 ℃刻み（9 択）
  tenant_vent  (58, 2)  換気の水準 停止・Low・Middle・High（4 択。変更なし）
  zone_remote  (60, 18) A-2 リモコンの許可 許可・禁止（2 択。系統の次元と同じ 1 時間ごとの時機）
  tenant_bypass(78, 2)  A-3 全熱交換器のバイパス 使わない・使う（2 択。換気と同じ時機）
マスクは V7 と同じ M-3（3,600 秒境界でないステップの系統の次元を前回値に固定。A-2 の次元も同じ時機）、M-5'、M-9 だけで、
新しいマスクは置かない（NOT-4）。GC-1 は「割合が 0 のゾーン」の設定温度・風量と、運転ゾーンの無い系統の蒸発温度を除く。
"""
import numpy as np

from rlkit import constants

N_ZONES = constants.N_ZONES
N_SYSTEMS = 4
N_TENANTS = 2
TENANT_ORDER = ["South", "North"]
ZONE_TENANT = [TENANT_ORDER.index(z[2]) for z in constants.ZONE_DEFS]            # ゾーン → テナント（0 南・1 北）
ZONE_SYSTEM = [z[0] - 1 for z in constants.ZONE_DEFS]                             # ゾーン → 系統（0〜3）
SYSTEM_TENANT = []                                                                 # 系統 → テナント（2 つにまたがる系統は無い）
for _s in range(N_SYSTEMS):
    _t = {ZONE_TENANT[_k] for _k in range(N_ZONES) if ZONE_SYSTEM[_k] == _s}
    if len(_t) != 1:
        raise RuntimeError(f"系統 {_s + 1} が 2 つのテナントにまたがる: 指示書 V8-S L-3 により停止して報告する")
    SYSTEM_TENANT.append(_t.pop())

DUTY_FRACTIONS = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
DUTY_OFF, DUTY_FULL = 0, 3
SETPOINT_GRID = [22.0, 22.5, 23.0, 23.5, 24.0, 25.0, 26.0, 27.0, 28.0]
FAN_SPEEDS = list(constants.ZONE_FAN_SPEED)
EVAP_GRID = [14 + i for i in range(9)]
VENT_LEVELS = list(constants.VENT_LEVELS)
VENT_OFF = constants.VENT_OFF
REMOTE_PERMIT, REMOTE_PROHIBIT = 0, 1            # 選択肢の索引（0 が許可）
BYPASS_OFF, BYPASS_ON = 0, 1                     # 選択肢の索引（0 が使わない）

LAYOUT = {
    "zone_duty": (0, 18), "zone_setpoint": (18, 18), "zone_fan": (36, 18), "system_evap": (54, 4),
    "tenant_vent": (58, 2), "zone_remote": (60, 18), "tenant_bypass": (78, 2),
}
GROUP_ORDER = ("zone_duty", "zone_setpoint", "zone_fan", "system_evap", "tenant_vent", "zone_remote", "tenant_bypass")
GROUP_K = {"zone_duty": 4, "zone_setpoint": 9, "zone_fan": 3, "system_evap": 9, "tenant_vent": 4, "zone_remote": 2,
           "tenant_bypass": 2}
ACTION_DIMS = 80
NVEC = np.array([GROUP_K[g] for g in GROUP_ORDER for _ in range(LAYOUT[g][1])], dtype=np.int64)
ACTION_LOGITS = int(NVEC.sum())
assert len(NVEC) == ACTION_DIMS and ACTION_LOGITS == 372
OFFSETS = np.concatenate([[0], np.cumsum(NVEC)[:-1]]).astype(np.int64)             # 各次元のロジットの開始位置
DIM_GROUP = np.array([i for i, g in enumerate(GROUP_ORDER) for _ in range(LAYOUT[g][1])], dtype=np.int64)

# V7 の行動索引（60 次元）→ V8 の索引（中立な初期化、同等性の確認 G-0 の対応表）
V7_SETPOINT_GRID = list(constants.ZONE_SETPOINT_C[1])                                   # 22〜28 ℃、7 択
V7_TO_V8_SETPOINT = np.array([SETPOINT_GRID.index(g) for g in V7_SETPOINT_GRID], dtype=np.int64)
V7_EVAP_GRID = list(constants.SYSTEM_EVAPORATING_C)                                     # 15〜22 ℃、8 択
V7_TO_V8_EVAP = np.array([EVAP_GRID.index(g) for g in V7_EVAP_GRID], dtype=np.int64)


def dim_slice(group):
    start, n = LAYOUT[group]
    return slice(start, start + n)


def v7_to_v8_action(a7):
    """V7 の行動（60 次元）を、同じ物理量を表す V8 の行動（80 次元）に写す。停止 → 割合 0、運転 → 割合 1、リモコンは許可、
    バイパスは使わない。(T, 60) または (60,)。"""
    a7 = np.asarray(a7, dtype=np.int64)
    single = a7.ndim == 1
    a7 = np.atleast_2d(a7)
    out = np.zeros((a7.shape[0], ACTION_DIMS), dtype=np.int64)
    out[:, dim_slice("zone_duty")] = np.where(a7[:, 0:18] == 1, DUTY_FULL, DUTY_OFF)
    out[:, dim_slice("zone_setpoint")] = V7_TO_V8_SETPOINT[a7[:, 18:36]]
    out[:, dim_slice("zone_fan")] = a7[:, 36:54]
    out[:, dim_slice("system_evap")] = V7_TO_V8_EVAP[a7[:, 54:58]]
    out[:, dim_slice("tenant_vent")] = a7[:, 58:60]
    out[:, dim_slice("zone_remote")] = REMOTE_PERMIT
    out[:, dim_slice("tenant_bypass")] = BYPASS_OFF
    return out[0] if single else out


def initial_action():
    """エピソード最初のステップの「前回値」。全次元 0（全ゾーン割合 0・格子の先頭・換気停止・許可・バイパス使わない）。"""
    return np.zeros(ACTION_DIMS, dtype=np.int64)


def decode_action(action):
    """action (80,) → 物理値の構造化表現。環境の書き込みと観測が使う。"state" は V7 と互換の値（割合 > 0 なら cool、0 なら off）。"""
    a = np.asarray(action)
    zones = {}
    for k, (ou, iu, tenant, zn, label, area) in enumerate(constants.ZONE_DEFS):
        duty_idx = int(a[LAYOUT["zone_duty"][0] + k])
        on = duty_idx > 0
        zones[label] = {
            "duty_index": duty_idx, "duty": DUTY_FRACTIONS[duty_idx], "state": "cool" if on else "off",
            "setpoint": SETPOINT_GRID[int(a[LAYOUT["zone_setpoint"][0] + k])] if on else None,
            "fan_speed": FAN_SPEEDS[int(a[LAYOUT["zone_fan"][0] + k])],
            "permit": int(a[LAYOUT["zone_remote"][0] + k]) == REMOTE_PERMIT}
    systems = {s + 1: {"evap_c": EVAP_GRID[int(a[LAYOUT["system_evap"][0] + s])]} for s in range(N_SYSTEMS)}
    tenants = {t: int(a[LAYOUT["tenant_vent"][0] + i]) for i, t in enumerate(TENANT_ORDER)}
    bypass = {t: int(a[LAYOUT["tenant_bypass"][0] + i]) == BYPASS_ON for i, t in enumerate(TENANT_ORDER)}
    return {"zones": zones, "systems": systems, "tenants": tenants, "bypass": bypass}


def duty_indices(action):
    a = np.asarray(action)
    s = LAYOUT["zone_duty"][0]
    return a[..., s:s + N_ZONES]


def compute_mask(prev_action, is_boundary, no_cooling_zones=None, idle_tenants=None, m5_enabled=True, m9_enabled=True):
    """各次元で有効な値を True とする bool 配列のリスト（NVEC と同じ形）。M-3（A-2 の次元を含む）、M-5'、M-9 だけ。"""
    prev_action = np.asarray(prev_action)
    mask = [np.ones(int(n), dtype=bool) for n in NVEC]

    def force(i, value):
        mask[i][:] = False
        mask[i][int(value)] = True

    if not is_boundary:                                            # M-3: 系統の次元と A-2 を前回値に固定（換気・A-3 は対象外）
        for g in ("system_evap", "zone_remote"):
            s = dim_slice(g)
            for i in range(s.start, s.stop):
                force(i, prev_action[i])
    if m5_enabled and no_cooling_zones is not None:                # M-5': 在室の無いゾーンは割合 0 だけ
        for k in np.flatnonzero(np.asarray(no_cooling_zones, dtype=bool)):
            force(LAYOUT["zone_duty"][0] + k, DUTY_OFF)
    if m9_enabled and idle_tenants is not None:                    # M-9: 無人のテナントの換気は停止
        for j, idle in enumerate(np.asarray(idle_tenants, dtype=bool)):
            if idle:
                force(LAYOUT["tenant_vent"][0] + j, VENT_OFF)
    return mask


def mask_to_flat(mask):
    return np.concatenate([np.asarray(m, dtype=bool) for m in mask])


def n_valid_options(flat_masks):
    """(…, 372) のマスク → (…, 80) 各次元の有効な選択肢の数。"""
    m = np.asarray(flat_masks, dtype=bool)
    return np.add.reduceat(m, OFFSETS, axis=-1)


def gc1_excluded(actions, n_options=None):
    """GC-1（V8 での読み替え）: 割合 0 のゾーンの設定温度・風量、運転ゾーンの無い系統の蒸発温度を True とする (T, 80)。
    n_options (T, 80) があれば、選択肢が 1 つしかない次元も True にする。"""
    actions = np.atleast_2d(np.asarray(actions))
    out = np.zeros(actions.shape, dtype=bool)
    on = duty_indices(actions) > 0
    for g in ("zone_setpoint", "zone_fan"):
        out[:, dim_slice(g)] = ~on
    es = LAYOUT["system_evap"][0]
    zone_sys = np.asarray(ZONE_SYSTEM)
    for s in range(N_SYSTEMS):
        out[:, es + s] = ~on[:, zone_sys == s].any(axis=1)
    if n_options is not None:
        out |= np.asarray(n_options) <= 1
    return out
