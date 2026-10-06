"""rlkit.constants — 夏季 v2 の物理定数・校正係数・分布母数・行動格子・学習の固定値。

指示書 V6-S（設計理論書 v3）で変えた値には「V6-S」と記号（A-2、R-5 など）を添える。

コード中に数値を直接書かず、ここに集約する（現行実装の方針を維持）。各定数の出典は直前のコメントに記す。
「v2」は rl_design_theory_v2.md、「現行」は 20_Shizuku2-RL-kit/rlkit/constants.py を指す。
設計書だけでは値が決まらない項目はここに置かず、rlkit/config.py の未確定項目として扱う。
"""

# =====================================================================
# 乱数種（v2 第 5-8 節「種 19800614」）
# =====================================================================
RANDOM_SEED = 19800614


# =====================================================================
# 評価関数（ECI）の Box-Cox 母数と零点条件（夏季）
# 出典: 現行 constants.ECI_PARAMS["summer"]、reference_ja.pdf Fig. 6.1、v2 第 2-1 節
# =====================================================================
ECI_PARAMS = {
    "lamE": 0.8678, "muE": 3.9653, "sigE": 1.5129,
    "lamDR": -0.4356, "muDR": -2.4076, "sigDR": 0.3343,
}
# 公式 ECI が正の値を取る領域は E <= 7.200 GJ かつ DR <= 0.2220（v2 第 2-1 節）。
ZERO_FLOOR_ENERGY_GJ = 7.200
ZERO_FLOOR_DR = 0.2220

# R-5 零点境界の障壁（指示書 V6-S 第 3-3 節・第 12 章）。境界 (E_z, DR_z) は eci.zero_boundary() が
# Box-Cox 母数から P = 0.2 を解いて求める（P-5 の 7.1998 GJ・0.2220 はその 4 桁表記）。
R5_ZERO_PERCENTILE = 0.2
R5_E_Z_GJ_LISTED = 7.1998
R5_DR_Z_LISTED = 0.2220
R5_KAPPA = 1.0
R5_TAU_E_GJ = 0.10
R5_TAU_D = 0.003
# V6.1-S: R-5 は報酬と比の計算から外した（コードは残し、呼ばれないことを UT-21 で確かめる）

# R-7（指示書 V6.1-S 第 3-1 節・第 12 章）: 比は ECI の等高線の傾き β_s/α_s を、走行全体の収穫の最良
# （ECI の大きい順）の (E_ref, DR_ref) で評価する。30〜200 で制限し、前回の比に対して 2 倍の制限を掛ける。
# 報酬は r = −T·(ΔE + ρ·n)/(E_ref + ρ·DR_ref)。更新 1 は ρ₀ = 60、分母は E_ref0 = 6.5 GJ・DR_ref0 = 0.20
R7_RHO0 = 60.0
R7_E_REF0_GJ = 6.5
R7_DR_REF0 = 0.20
R7_RHO_MIN = 30.0
R7_RHO_MAX = 200.0
R7_RATIO_LIMIT = 2.0
R7_TEST_TOLERANCE = 0.005
R7_RECORD_FIXED_RHO = 60.0     # 第 3-6 節: 更新をまたいだ比較用の mean(E + 60·DR)


# =====================================================================
# エネルギー
# =====================================================================
# 一次エネルギー換算係数（v2 第 5-1-2 節の e_k,t の式、現行 ENERGY_CONVERSION_MJ_PER_KWH）
ENERGY_CONVERSION_MJ_PER_KWH = 9.77

# 室外機の固定成分 a_s[kW]。v2 第 5-1-2 節「校正値 1.0236/0.9633/0.8859/0.8742 kW を比率決定にだけ使う」
OUTDOOR_UNIT_A_KW = [1.0236, 0.9633, 0.8859, 0.8742]

# 全熱交換器電力[kW]（床面積・風量段別）。v2 第 5-1-2 節「P_hex,k : 全熱交換器電力（校正モデル。
# 実測 44.000 kWh と差 0 を確認済み）」。値は現行 constants.HEX_POWER_KW をそのまま移植した
# （vent.csv のテナント合計値からの分離校正。V1(s)_report 第 2-4 節で週間 44.000 kWh と一致）。
HEX_POWER_KW = {
    26.0: {"Low": 0.02 / 3, "Middle": 0.11 / 3, "High": 0.23 / 3},
    32.5: {"Low": (0.07 - 0.02) / 6, "Middle": (0.4 - 0.11) / 6, "High": (0.83 - 0.23) / 6},
}

# R-6（指示書 V6-S 第 3-3 節・第 6 章 Z'-4）: 全熱交換器の電力模型を水準ごと（0 停止、1 Low、2 Middle、3 High）に持つ。
# 診断 K-11 の vent.csv はテナント合計しか持たない（南北とも Low 0.07・Middle 0.40・High 0.83 kW）ため、
# Z'-4 の代替規則により、ゾーンの Middle の校正値に全台合計の比（Low 0.14/0.80、High 1.66/0.80）を掛ける。
HEX_TOTAL_KW_MIDDLE = 0.80
HEX_TOTAL_KW_LOW = 0.14
HEX_TOTAL_KW_HIGH_K11D = 1.66   # K-11 (d) の実測（全台合計）
HEX_POWER_KW_BY_LEVEL = {
    area: {0: 0.0,
           1: v["Middle"] * HEX_TOTAL_KW_LOW / HEX_TOTAL_KW_MIDDLE,
           2: v["Middle"],
           3: v["Middle"] * HEX_TOTAL_KW_HIGH_K11D / HEX_TOTAL_KW_MIDDLE}
    for area, v in HEX_POWER_KW.items()
}


# =====================================================================
# 不満足率の合成（現行を維持。current_rl_spec.md 第 1-2 節）
# DR_kt = 1 − (1−DR_温冷感)(1−DR_ドラフト)(1−DR_上下温度分布)(1−DR_CO2)
# DR_CO2 = 1 / (1 + exp((1050 − CO2[ppm]) / 10.8))
# =====================================================================
CO2_DISSATISFACTION_MIDPOINT_PPM = 1050.0
CO2_DISSATISFACTION_SCALE_PPM = 10.8


# =====================================================================
# 系統構成・ゾーン静的属性（現行を維持）
# =====================================================================
I_UNIT_NUM = [5, 4, 5, 4]  # 系統(oUnitIndex 1..4)ごとの室内機台数

SYSTEM_ZONES = {
    1: ["S1", "S2", "S3", "S4", "S5"],
    2: ["S6", "S7", "S8", "S9"],
    3: ["N1", "N2", "N3", "N4", "N5"],
    4: ["N6", "N7", "N8", "N9"],
}

# (oUnitIndex, iUnitIndex, tenant, zone_number, label, floor_area_m2)
# ゾーン索引 k (0-17) はこの並び順（観測のゾーンブロック、行動のゾーン次元、価値のゾーン出力で共通）。
ZONE_DEFS = [
    (1, 1, "South", 1, "S1", 26.0),
    (1, 2, "South", 2, "S2", 26.0),
    (1, 3, "South", 3, "S3", 26.0),
    (1, 4, "South", 4, "S4", 32.5),
    (1, 5, "South", 5, "S5", 32.5),
    (2, 1, "South", 6, "S6", 32.5),
    (2, 2, "South", 7, "S7", 32.5),
    (2, 3, "South", 8, "S8", 32.5),
    (2, 4, "South", 9, "S9", 32.5),
    (3, 1, "North", 1, "N1", 26.0),
    (3, 2, "North", 2, "N2", 26.0),
    (3, 3, "North", 3, "N3", 26.0),
    (3, 4, "North", 4, "N4", 32.5),
    (3, 5, "North", 5, "N5", 32.5),
    (4, 1, "North", 6, "N6", 32.5),
    (4, 2, "North", 7, "N7", 32.5),
    (4, 3, "North", 8, "N8", 32.5),
    (4, 4, "North", 9, "N9", 32.5),
]
ZONE_LABELS = [z[4] for z in ZONE_DEFS]
N_ZONES = 18
assert len(ZONE_DEFS) == N_ZONES

# 室内機容量[kW]。現行 constants.INDOOR_UNIT_CAPACITY_KW（処理熱負荷の週間最大値からの逆算）を維持する。
# 注記: reference_ja.pdf Table 1.3・1.4 の公称値（C71=7.1 kW、C56=5.6 kW）とは値が異なるが、
# ゾーンの 2 群への分かれ方は一致する。v2 第 5-2-2 節が「維持」としたため現行値を使う。
INDOOR_UNIT_CAPACITY_KW = {
    "S1": 5.6, "S2": 5.6, "S3": 5.6, "S4": 5.6, "S5": 5.6,
    "S6": 4.5, "S7": 4.5, "S8": 4.5, "S9": 4.5,
    "N1": 5.6, "N2": 4.5, "N3": 4.5, "N4": 5.6, "N5": 5.6,
    "N6": 4.5, "N7": 4.5, "N8": 4.5, "N9": 4.5,
}

# 系統ごとの室内機容量の合計[kW]。観測の負荷率の正規化母数（指示書 V3-S 第 2-4 節。旧 a_s）
SYSTEM_CAPACITY_KW = [sum(INDOOR_UNIT_CAPACITY_KW[label] for label in SYSTEM_ZONES[ou]) for ou in range(1, 5)]

TENANTS = ["South", "North"]
# テナント別の執務者数。reference_ja.pdf Appendix 2（rseed_oprm=1、Shizuku2/setting.ini と同じ）
# および Appendix 1 の Availability_OC_38（tenant-1）・Availability_OC_48（tenant-2）。
TENANT_OCCUPANT_COUNT = {"South": 38, "North": 48}


# =====================================================================
# 時間刻み（v2 第 2-1 節「二段の時間刻み」、A-6。値は現行を維持）
# 平日 06:00〜20:00 は 900 秒、それ以外は 3,600 秒。平日 14h×4×5 = 280、平日その他 50、週末 48、計 378。
# =====================================================================
ZONE_STEP_OCCUPIED_SEC = 900
ZONE_STEP_UNOCCUPIED_SEC = 3600
SYSTEM_STEP_SEC = 3600
OCCUPIED_WEEKDAY_START_HOUR = 6
OCCUPIED_WEEKDAY_END_HOUR = 20
ZONE_STEPS_PER_WEEK = 378


# =====================================================================
# 行動格子（v2 第 5-3 節の表）
# =====================================================================
STAGES = (1, 2)
ZONE_STATE = ["off", "cool", "fan"]  # 停止/冷房/送風（観測の one-hot はこの 3 種で固定）
# S-5（指示書 V5-S 第 2-4 節）: 診断 K-6 で全停止と全送風の E・DR が 16 進一致したため、
# 第 1 段階の運転状態を 2 択（停止・冷房）に縮める。第 2 段階で送風を戻す。
ZONE_STATE_BY_STAGE = {1: ["off", "cool"], 2: ["off", "cool", "fan"]}
# A-2（V6-S）: 夏の設定温度の下限を 22 → 20 ℃へ広げる（9 択）。A-7: 段階は単一（第 1 段階だけを使う）。
# 第 2 段階（0.5 ℃刻み）の定義とコードは残すが、設定から呼ばれない（UT-13）。
ZONE_SETPOINT_C = {
    1: [20.0 + 1.0 * i for i in range(9)],   # 20〜28 ℃、1.0 ℃刻み（V6-S A-2）
    2: [20.0 + 0.5 * i for i in range(17)],  # 使わない（A-7）。第 1 段階の索引 i を 2i へ写せる形だけ保つ
}
ZONE_FAN_SPEED = ["Low", "Middle", "High"]
SYSTEM_EVAPORATING_C = [15 + i for i in range(8)]  # 15〜22 ℃
# A-5（V6-S）: 換気はテナント別の 4 択（0 停止、1 Low、2 Middle、3 High）、毎ステップ更新（M-3 の対象外）
VENT_LEVELS = ["off", "Low", "Middle", "High"]
VENT_OFF = 0

# K-9 の結果（runs/diag/V6S/K9_decision.json、2026-09-24）。B: O-3（全体 82・83）と A-6（tenant_remote）を加え、
# 観測 +31・+32 と発停の毎ステップ書き込みは加えない（指示書 V6-S 第 6 章の判定表）。
K9_RESULT = "B"
TENANT_REMOTE_ENABLED = K9_RESULT in ("A", "B")
ONOFF_ALWAYS_WRITE = K9_RESULT == "A"
assert K9_RESULT == "B", "結果 A・C の観測・母数は実装していない（K-9 は結果 B で確定）"
# A-6（V6-S）: テナント別のリモコン許可（0 禁止、1 許可）、3,600 秒境界（M-3 の対象）
TENANT_REMOTE_PERMIT = [False, True]

ACTION_DIMS = 62                    # 60 ＋ tenant_remote 2（V6-S 第 3-1 節）
# 第 1 段階: 18×2 + 18×9 + 18×3 + 4×8 + 2×4 + 2×2 = 296（V6-S 第 3-1 節）。第 2 段階は使わない（A-7）
ACTION_LOGITS = {s: N_ZONES * (len(ZONE_STATE_BY_STAGE[s]) + len(ZONE_SETPOINT_C[s]) + len(ZONE_FAN_SPEED))
                 + 4 * len(SYSTEM_EVAPORATING_C) + 2 * len(VENT_LEVELS) + 2 * len(TENANT_REMOTE_PERMIT)
                 for s in STAGES}
assert ACTION_LOGITS[1] == 296

# 行動に含めず固定値で書き込むもの（v2 A-4・A-5、現行を維持）
SUMMER_COND_TEMP_C = 46.0         # 夏季の凝縮温度（現行 driver.SUMMER_COND_TEMP_C）
CONDENSING_TEMP_MAX_C = 52.0      # 凝縮温度 52 ℃以上を書かない（現行の禁止事項）
ZONE_DIRECTION_FIXED = "Horizontal"  # 夏季のゾーン風向（現行）
HEX_BYPASS_ENABLED = False        # 全熱交換器バイパス（A-4。V6-S A-5 でも無効のまま変えない）
USE_DIFF_WRITER = True            # 差分書き込み（A-5「両季節で use_diff_writer = True」。K-9 結果 B で発停も維持）

# M-5'（V6-S 第 3-2 節）: 現在の在室者数 0 かつ外生表で 12 時間先（43,200 秒先）まで在室 0 のゾーンは停止だけ
M5_FORECAST_WINDOW_H = 12.0
# M-9（V6-S 第 3-2 節で有効化）: テナントの在室者数 0 かつ 1 時間先まで 0 なら換気を停止に固定
M9_FORECAST_WINDOW_H = 1.0


# =====================================================================
# 観測（v2 第 5-2 節）
# =====================================================================
# V6-S 第 3-4 節（K-9 結果 B）: 全体 80・81 換気の水準（O-1）、82・83 リモコン許可の状態（O-3）。84 + 31 × 18 = 642
OBSERVATION_DIM = 642
GLOBAL_BLOCK_DIM = 84
ZONE_BLOCK_DIM = 31
assert GLOBAL_BLOCK_DIM + ZONE_BLOCK_DIM * N_ZONES == OBSERVATION_DIM

OUTDOOR_TEMP_NORM_C = 50.0
ABS_HUMIDITY_NORM_GKG = 30.0
RELATIVE_HUMIDITY_NORM_PCT = 100.0
GLOBAL_RADIATION_NORM_WM2 = 1200.0
NOCTURNAL_RADIATION_NORM_WM2 = 150.0
BUILDING_OCCUPANT_NORM = 86.0
CO2_NORM_PPM = 4000.0  # 指示書 V3-S 第 2-4 節（旧 2000。北テナントでクリップ率 0.49 だった）
STEP_LENGTH_NORM_SEC = 3600.0       # 索引 65「900 秒 = 0.25、3,600 秒 = 1.0」
# O-6（V6.1-S 第 3-4 節）: 全体 42・43 は 1 ステップ前からの CO2 の変化量 ÷ 500 ppm（−1〜1 に制限）
CO2_DELTA_NORM_PPM = 500.0
O6_INDICES = (42, 43)
ZONE_AREA_NORM_M2 = 32.5
ZONE_CAPACITY_NORM_KW = 10.0
ZONE_OCCUPANT_NORM = 10.0
THERMAL_SENSATION_MIN = -3.0
THERMAL_SENSATION_MAX = 3.0
THERMAL_SENSATION_LEVELS = [-3, -2, -1, 0, 1, 2, 3]  # Cold〜Hot（索引 66–79）
CLOTHING_INDEX_NORM_CLO = 1.5
TEMP_DELTA_HALF_RANGE_C = 5.0       # +21「(Δ+5)/10」
ZONE_CUM_ENERGY_NORM_GJ = 3.0
ZONE_CUM_DISSATISFIED_OCCUPANT_NORM = 200.0
OCCUPANCY_GAP_NORM_H = 12.0         # +27・+28「÷12 時間、1 でクリップ」
SETPOINT_OVERRIDE_TOL_C = 1e-6      # +30。V1(s)_report 第 2-5 節の検出方式（1e-6 を超える食い違い）
# +29（読み戻した設定温度）の正規化 (sp − 16)/16。指示書 V3-S 第 2-4 節（旧 (sp − 22)/6）
SETPOINT_READBACK_MIN_C = 16.0
SETPOINT_READBACK_SPAN_C = 16.0
OBS29_CLIP_REPORT_THRESHOLD = 0.05

# 先読みの地平（v2 は「維持」。現行 observation.FORECAST_HORIZONS_H）
FORECAST_HORIZONS_H = [0.5, 1, 2, 4, 8, 24]
OCCUPANT_FORECAST_HORIZONS_H = [0.5, 1, 2, 4, 8, 24]
ZONE_OCCUPANT_FORECAST_HORIZONS_H = [0.5, 1, 2]


# =====================================================================
# 学習（v2 第 5-5 節・第 5-6 節・第 5-8 節）
# =====================================================================
GAMMA = 1.0
LAMBDA_HOUR = 0.95

N_TRAIN_SLOTS = 6           # 並列 6 組
EPISODES_PER_SLOT = 2       # × 2 周 = 12 エピソード／更新

POLICY_LR_INIT = 1.0e-4     # 指示書 V4-S 第 2-3 節（旧 2.5e-4）
POLICY_LR_MIN = 1e-5        # 指示書 V5-S S-7（V4-S で 3e-5 にしたが下限に 3 回触れたため戻す）
POLICY_LR_MAX = 1e-3        # V6-S L-2（V4-S〜V5-S は 5e-4）
LR_ADAPT_FACTOR = 1.5
LR_ADAPT_HIGH = 2.0         # approx_kl > 2 × target_kl なら学習率を 1/1.5 倍
LR_ADAPT_LOW = 0.5          # approx_kl < 0.5 × target_kl なら学習率を 1.5 倍
TARGET_KL = 0.05            # 指示書 V4-S 第 2-3 節（旧 0.02。60 次元合計・ステップ平均）
KL_STOP_FACTOR = 1.5        # 打ち切りは 1.5 × target_kl = 0.075
POLICY_EPOCHS = 4
POLICY_MINIBATCH_STEPS = 756  # 2 エピソード相当、1 エポック 6 バッチ
CLIP_RANGE = 0.2

VALUE_LR = 5e-4
VALUE_EPOCHS = 8
VALUE_MINIBATCH_STEPS = 378

ADAM_EPS = 1e-5
MAX_GRAD_NORM = 0.5
ADVANTAGE_NORM_EPS = 1e-8   # SB3 標準の優位度正規化と同じ

ENTROPY_GROUPS = ("zone_state", "zone_setpoint", "zone_fan", "system_evap", "tenant_vent") +     (("tenant_remote",) if TENANT_REMOTE_ENABLED else ())
# L-1（V6-S）: 群別エントロピー床と双対上昇は廃止。群 g の係数 c_g = c0 / ln K_g、c0 = 0.002 を
# 計画更新数の 60%（1,500 更新なら 900）まで固定し、以後 100%（1,500）で 0 まで線形に下げる。
ENTROPY_C0 = 0.002
ENTROPY_CONST_UNTIL_FRACTION = 0.6
ENTROPY_ZERO_AT_FRACTION = 1.0

REFERENCE_RATIO_MAX_CHANGE = 2.0  # R-3''（内外を問わず毎更新適用する。指示書 V4-S 第 2-1 節）
REFERENCE_EMA_COEF = 0.5          # R-2'' 基準点の指数移動平均の係数
INSIDE_RATE_WINDOW = 20           # Q-3 直近の決定論的評価のうち零点条件の内側だった割合を測る窓

# O-5（V6-S 第 3-4 節）: 価値の文脈層へ ρ_n = clip(log10(ρ)/4, 0, 1)、D_n = clip(log10(E* + ρ·DR*)/3, 0, 1)
VALUE_REWARD_CONTEXT_DIM = 2
RHO_N_LOG10_DIVISOR = 4.0
D_N_LOG10_DIVISOR = 3.0

# 自己模倣学習（V6-S 第 3-6 節 SIL-1〜SIL-7、第 12 章）
SIL_START_UPDATE = 301
SIL_BUFFER_SIZE = 8
SIL_C_INIT = 0.1
SIL_C_MIN = 0.0125
SIL_C_MAX = 0.1
SIL_KL_HALVE = 0.075
SIL_KL_REVERT = 0.15
SIL_KL_DOUBLE_BELOW = 0.025
SIL_KL_DOUBLE_STREAK = 10
SIL_VALUE_COEF = 0.01
SIL_MINIBATCHES = 4
SIL_GUARD_UPDATE = 340
SIL_GUARD_RATIO = 0.8
SIL_GUARD_WINDOW = (301, 340)
SIL_GUARD_REFERENCE_WINDOW = (261, 300)

# 判定（V6-S 第 7 章・第 8 章）
H_CHECK_UPDATE = 20
H2_KL_CUTOFF_RATE_MAX = 0.80
H5_UPDATE_SECONDS_MAX = 520.0     # V6.1-S 第 3-3 節（V6-S は 480 秒）
H9_CLIP_STUCK_UPDATES = 10        # V6.1-S H-9（報告のみ）: 使った比が 30 または 200 に 10 更新以上続けて張り付く
S8PRIME_UPDATE = 100
INTERIM_UPDATE = 300
I1_KL_CUTOFF_RATE_MAX = 0.80
I2_RATIO = 0.8
I_BLOCK = 20
I3_RAW_RESID_MAX = 0.03
S7_RAW_RESID_STOP = 0.05
H6_DEGENERATE_STREAK = 10
BD1_EDGE_FRACTION = 0.30
BD1_UPDATES = (300, 1500)
R5_TEST_TOLERANCE = 0.005

# 退行処理（最良重みの再読み込みと学習率の半減）は指示書 V4-S Q-1 により廃止した（D-19）。

ENERGY_IDENTITY_TOL = 1e-3  # 週間エネルギーの一致判定 0.1%


# =====================================================================
# 指示書 V2-S（設計理論書 v2 の夏季補遺。2026-09-16）の決定値
# =====================================================================
# 勾配ノルム比（エントロピー損失／方策損失）。指示書 V3-S E-1・E-2 により判定には使わず、
# 次の更新番号でだけ計測して記録する（V2-S の範囲判定と停止は設計側の誤り D-12 として取り下げ）。
GRAD_NORM_RATIO_MEASURE_AT = (1, 10, 50, 100, 200)

# 第 2-4 節の門
G1_CONSECUTIVE_EVALS = 3               # (a) 公式 ECI > 0 が連続する決定論的評価の回数（G-1'）
G1_MIN_UPDATES = 200                   # (c) 通算更新番号の下限。(b) 非占有冷房率は V4-S D-21 で廃止
G2_ECI_REL_TOLERANCE = 1e-4            # G-2 格子の細分化の前後で公式 ECI が一致すべき相対差
G3_ECI_SMOOTH_RATIO_MIN = 0.98         # G-3 M-5 解除の可否（解除前の eci_smooth に対する比）
G3_RELEASE_JUDGE_UPDATES = 3           # G-3' 解除の可否を判定するまでの更新数（前後 3 更新の中央値を比べる）
G3_RETRY_UPDATES = 50                  # G-3 失敗後に再試行するまでの更新数
G3_MAX_RETRIES = 3                     # G-3 の最大失敗回数

# 第 2-3 節の判定規則
J1_MATCH_TOLERANCE = 0.001             # K-1 一致とみなす相対差（0.1%）
J1_FALLBACK_TOLERANCE = 0.03           # K-1 どちらも一致しないときの続行上限（3%）
J5_MULTIPLIER = 3                      # K-2 1 ステップ最大値に掛ける倍率

# 報告のしきい値（停止はしない）
OBS25_CLIP_RATE_REPORT_THRESHOLD = 0.01
RAW_RESID_REPORT_THRESHOLD = 0.03
EPISODE_TIME_REPORT_THRESHOLD_S = 160

# 数値演算スレッド（v2 第 5-8 節）
REQUIRED_THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "KMP_BLOCKTIME": "0"}
