"""rlkit.config — 走行ごとの設定と、設計書 v2 だけでは値が決まらない項目（未確定項目）の管理。

未確定項目は既定値を持たない（null）。学習の入口は、未確定項目が 1 つでも null なら起動しない。
値を決めるのに必要な情報は rlkit_v2_summer_implementation_report.md にまとめた。
"""
import json
import os

from rlkit import constants

# 未確定項目: キー → (レポートの番号, 内容)
UNRESOLVED = {
    "include_indoor_unit_power": ("J-1", "室内機電力 P_iu,k を按分に別建てで加えるか（true / false）"),
    "zone_step_energy_norm_gj": ("J-5", "観測 +25 の正規化定数 [GJ]（1 ステップのゾーン一次エネルギーの最大値）"),
    "planned_updates": ("J-6", "計画更新数（エントロピー係数の減衰の基準。V6-S L-1）"),
}

DEFAULTS = {
    "run_dir": "runs/summer_v2",
    "seed": constants.RANDOM_SEED,
    "train_slots": [0, 1, 2, 3, 4, 5],
    "eval_slot": 6,
    # 並列数拡張（並列数拡張マニュアル第 7 節）: 訓練の組数 × 1 組あたりの本数 = 12 を保てば、1 更新のエピソード数が同じで学習の
    # 内容（バッチ）は変わらない。既定は従来の 6 組 × 2 本。案 B は 12 組 × 1 本 + 評価 1 組（計 13 組）
    "episodes_per_slot": constants.EPISODES_PER_SLOT,
    # 同時に使う組数の上限（launch.preflight の limit）と、起動前の空きメモリの下限 [GB]。既定は従来の値
    "slot_limit": 7,
    "min_free_gb": 2.0,
    # BACnet の一括読み取り（rlkit/comm_rpm.py）を使うか。値は逐次読み取りと全点一致する（診断 D-5 で確認）
    "comm_rpm": False,
    # model_dir（モデルの置き場所）は既定値に入れない。設定に書かれているか（null を含む）を区別するため。resolve_model_dir を参照
    # 決定論的評価を採取と同時に 7 組目で走らせる（v2 第 5-5-1 節 手順 2「同時に」）。
    # 同時 7 組は PL2 で未検証（N_pass = 6）であり、指示書 V2-S N-5 により判定前の既定値は false とする
    # （false のときは採取の後に 7 組目で直列に評価する。学習の組の子プロセスが組のロックを持つため、
    # 評価はどちらの場合も学習と別の組を使う）。段階 0 の診断 C-2（K-3）の結果で決める。
    "eval_concurrent": False,
    # S-8（V6-S 第 7 章）: 1 更新の中で採取のやり直しが 3 回を超えたら停止する（試行は最大 4 回）
    "max_collection_retries": 4,
    "max_eval_retries": 4,
    # M-9（非在室テナントの換気停止）。V6-S 第 3-2 節で有効にする
    "m9_enabled": True,
    # K-9 の結果（constants.K9_RESULT と一致しなければ起動しない）
    "k9_result": constants.K9_RESULT,
    # 指示書 V7-S C-8: 引き継ぐ保存点（None なら新規初期化）。読むだけ
    "warm_start_from": None,
}
# 設定ファイルに書いてよい任意の項目（既定値には入れない。V7 の設定の項目の集合を変えないため）
#   model_dir … モデルの置き場所（resolve_model_dir）。V8-S（rlkit/v8）の init_weights … 初期の重み（V7 の保存点）、
#   lr・max_epochs・kl_limit … 第 7 章 L-6 の数値（V8 の学習の入口が無ければ 1e-4・8・0.01 を使う）
#   start_E_hex・start_DR_hex … V8.1-S 第 3 章の始点（V8 の更新 0 の基準の軌跡）。あれば始点と C8-2 の確認に使う
#   init_policy … V8.2-S の G-0R が作った初期の方策の重み（無ければ init_weights を V8 に拡張したもの）
#   c83_limit_s … C8-3（更新 1・2 の 1 更新の実時間の平均の上限、既定 600 秒）
OPTIONAL_KEYS = frozenset({"model_dir", "init_weights", "lr", "max_epochs", "kl_limit", "start_E_hex", "start_DR_hex", "c83_limit_s", "init_policy", "l6r_loss", "l6r_lr", "l6r_max_epochs", "v9"})
DEFAULTS.update({k: None for k in UNRESOLVED})


def load(path=None):
    cfg = dict(DEFAULTS)
    if path is not None:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        unknown = sorted(set(user) - set(cfg) - OPTIONAL_KEYS)
        if unknown:
            raise ValueError(f"設定に未知の項目がある: {unknown}")
        cfg.update(user)
    validate(cfg)
    return cfg


def unresolved_items(cfg):
    return [(key, UNRESOLVED[key][0], UNRESOLVED[key][1]) for key in UNRESOLVED if cfg.get(key) is None]


def validate(cfg):
    """与えられた値の形だけを検査する（null の未確定項目は unresolved_items で別に扱う）。"""
    slots = list(cfg["train_slots"])
    per = cfg["episodes_per_slot"]
    if not (isinstance(per, int) and per >= 1):
        raise ValueError("episodes_per_slot は 1 以上の整数")
    episodes = constants.N_TRAIN_SLOTS * constants.EPISODES_PER_SLOT
    if len(slots) * per != episodes or slots != sorted(set(slots)):
        raise ValueError(f"train_slots（重複の無い昇順）の組数 × episodes_per_slot は 1 更新のエピソード数 {episodes} に等しくする"
                         f"（学習の内容を変えないため。設計側の承認なしにバッチを変えない）: {len(slots)} × {per}")
    if cfg["eval_slot"] in slots:
        raise ValueError("eval_slot は train_slots と別の組にする")
    if len(slots) + 1 > cfg["slot_limit"]:
        raise ValueError(f"訓練 {len(slots)} 組 + 評価 1 組が slot_limit {cfg['slot_limit']} を超える")
    if cfg["include_indoor_unit_power"] is not None and not isinstance(cfg["include_indoor_unit_power"], bool):
        raise ValueError("include_indoor_unit_power は true / false")
    if cfg["zone_step_energy_norm_gj"] is not None and not cfg["zone_step_energy_norm_gj"] > 0:
        raise ValueError("zone_step_energy_norm_gj は正の数")
    if cfg["planned_updates"] is not None and not (isinstance(cfg["planned_updates"], int) and cfg["planned_updates"] >= 1):
        raise ValueError("planned_updates は 1 以上の整数")
    if cfg["k9_result"] != constants.K9_RESULT:
        raise ValueError(f"k9_result {cfg['k9_result']!r} が実装（constants.K9_RESULT = {constants.K9_RESULT!r}）と違う")
    for key in ("max_collection_retries", "max_eval_retries"):
        if not (isinstance(cfg[key], int) and cfg[key] >= 1):
            raise ValueError(f"{key} は 1 以上の整数")


def repo_path(path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return path if os.path.isabs(path) else os.path.join(root, path)


def model_dir_of(cfg):
    """モデル（保存点 .pt など）の場所の絶対パス。model_dir が無ければ run_dir。"""
    return repo_path(cfg["model_dir"]) if cfg.get("model_dir") else repo_path(cfg["run_dir"])


MODEL_ROOT = os.path.join(os.path.expanduser("~"), "rl_models")     # OneDrive の外。新しい走行のモデルの置き場所の既定


def resolve_model_dir(cfg, run_dir, resume):
    """モデル（保存点 .pt・SIL の保存庫・best の .pt）の置き場所を決める。レポート・記録・行動列は run_dir のまま。

    設定に "model_dir" があればそれを使う（null なら run_dir）。
    設定に無いとき: 新しい走行（resume でない）は MODEL_ROOT/<run_dir の名前>（OneDrive の外）。
    再開（resume）は、その走行の最初の meta_start_*.json に記録された model_dir（無ければ run_dir）。
    これで、以前の走行（モデルが run_dir にある）はそのまま再開でき、新しい走行からモデルが OneDrive の外に出る。"""
    if "model_dir" in cfg:
        return repo_path(cfg["model_dir"]) if cfg["model_dir"] else run_dir
    if not resume:
        return os.path.join(MODEL_ROOT, os.path.basename(os.path.normpath(run_dir)))
    import glob
    import json
    for path in sorted(glob.glob(os.path.join(run_dir, "meta_start_*.json"))):
        try:
            rec = json.load(open(path, encoding="utf-8")).get("model_dir")
        except Exception:
            continue
        if rec:
            return rec
        break
    return run_dir
