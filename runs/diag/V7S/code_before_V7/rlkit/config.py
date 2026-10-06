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
