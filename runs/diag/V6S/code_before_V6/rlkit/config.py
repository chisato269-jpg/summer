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
    "entropy_coef_init": ("U-1", "群別エントロピー係数 c_g の初期値（c_min・c_max はその 1/10・100 倍）"),
    "planned_updates": ("J-6", "計画更新数（floor_fraction の減衰と段階の切り替えの基準）"),
    "stage2_start_fraction": ("U-5", "第 1 段階の門（G-1）を通過できなかった場合の上限。"
                                     "通算更新番号がこの割合 × planned_updates に達しても第 1 段階のままなら停止する"),
}

DEFAULTS = {
    "run_dir": "runs/summer_v2",
    "seed": constants.RANDOM_SEED,
    "train_slots": [0, 1, 2, 3, 4, 5],
    "eval_slot": 6,
    # 決定論的評価を採取と同時に 7 組目で走らせる（v2 第 5-5-1 節 手順 2「同時に」）。
    # 同時 7 組は PL2 で未検証（N_pass = 6）であり、指示書 V2-S N-5 により判定前の既定値は false とする
    # （false のときは採取の後に 7 組目で直列に評価する。学習の組の子プロセスが組のロックを持つため、
    # 評価はどちらの場合も学習と別の組を使う）。段階 0 の診断 C-2（K-3）の結果で決める。
    "eval_concurrent": False,
    # G-5 の手動切替。通算更新番号がこの値に達したら G-2・G-3 を実行する（null なら G-1 の門だけで切り替える）
    "manual_stage2_update": None,
    "max_collection_retries": 5,
    "max_eval_retries": 5,
    # M-9（非在室テナントの換気停止）。診断 K-7 (a) が採用条件を満たさなかったため既定は無効（V5-S 第 2-3 節）
    "m9_enabled": False,
}
DEFAULTS.update({k: None for k in UNRESOLVED})


def load(path=None):
    cfg = dict(DEFAULTS)
    if path is not None:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        unknown = sorted(set(user) - set(cfg))
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
    if len(slots) != constants.N_TRAIN_SLOTS or slots != sorted(set(slots)):
        raise ValueError(f"train_slots は重複の無い昇順の {constants.N_TRAIN_SLOTS} 組: {slots}")
    if cfg["eval_slot"] in slots:
        raise ValueError("eval_slot は train_slots と別の組にする")
    if cfg["include_indoor_unit_power"] is not None and not isinstance(cfg["include_indoor_unit_power"], bool):
        raise ValueError("include_indoor_unit_power は true / false")
    if cfg["zone_step_energy_norm_gj"] is not None and not cfg["zone_step_energy_norm_gj"] > 0:
        raise ValueError("zone_step_energy_norm_gj は正の数")
    if cfg["entropy_coef_init"] is not None and not cfg["entropy_coef_init"] > 0:
        raise ValueError("entropy_coef_init は正の数")
    if cfg["planned_updates"] is not None and not (isinstance(cfg["planned_updates"], int) and cfg["planned_updates"] >= 1):
        raise ValueError("planned_updates は 1 以上の整数")
    f = cfg["stage2_start_fraction"]
    if f is not None and not 0.0 < f <= 1.0:
        raise ValueError("stage2_start_fraction は 0 より大きく 1 以下")
    m = cfg["manual_stage2_update"]
    if m is not None and not (isinstance(m, int) and m >= 1):
        raise ValueError("manual_stage2_update は 1 以上の整数または null")


def repo_path(path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return path if os.path.isabs(path) else os.path.join(root, path)
