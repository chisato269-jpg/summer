"""rlkit.train_summer — 夏季の学習の入口（v2 第 5-5 節、指示書 V6-S 第 3 章・第 6〜8 章、V6.1-S 第 3 章）。

指示書 V6.1-S（零点条件を目的にしない）での変更:
  R-7 比 ρ は走行全体の収穫の最良（ECI の大きい順）の (E_ref, DR_ref) での ECI の等高線の傾き。30〜200 で制限し、
      前回に対して 2 倍の制限。更新 u の収穫の後に更新し、更新 u+1 の採取に使う。更新 1 は ρ₀ = 60、分母 6.5・0.20
  選抜 収穫・提出候補・SIL の保存庫の順位はすべて ECI（零点条件なし）の大きい順。公式 ECI は参考の記録だけ
  判定 H-4'' に替え、H-5 は 520 秒、H-7・S-8' は廃止、H-9（比が 30・200 に 10 更新張り付き、報告のみ）を追加
  R-5 の J と基準点の指数移動平均は呼ばない（コードは rlkit/eci.py・rlkit/reward.py に残す）

リポジトリルートから（PowerShell の例）:
  $env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1; $env:KMP_BLOCKTIME=0
  python -m rlkit.train_summer --config rlkit/config_summer_v6.json [--updates 20] [--resume]

未確定項目（rlkit/config.py の UNRESOLVED）が 1 つでも null なら起動しない。

1 回の更新:
  1 学習の 6 組 × 2 周で 12 エピソードを採取する（基準点と価値の文脈 (ρ_n, D_n) は採取の前に渡す）
  2 同時に、採取に使う方策の写しで決定論的評価を 1 エピソード行う（診断のみ。判定と選抜には使わない）
  L-4 収穫: 12 エピソードのうち公式 ECI 最大（同点は R-5 の J 最大）の行動列を残し、走行全体の最良を harvest/ に保持する
  SIL-1 保存庫: 毎更新の収穫の最良 1 本を候補とし、走行全体で上位 8 本（L-4 と同じ順位）を sil_buffer.pt に保つ
  3〜7 学習（rlkit.learner.SummerPPO.train）。SIL は更新 1〜300 は記録だけ、301 以降は中間判定に合格した場合だけ適用
  R-2''・R-3''・R-5: 訓練 12 エピソードの (E, DR) の平均で基準点を指数移動平均で動かし、R-5 の比に 2 倍制限を掛ける
  8 記録と保存（第 3-8 節の項目）
判定（第 7 章・第 8 章）: 更新 20 で H-1〜H-6・H-9、更新 300 で I-1〜I-3（合格なら SIL を有効化）、
  更新 340 で SIL-6 の見張り、毎更新 S-6（NaN）・S-7（raw_resid）。更新 300・1,500 で BD-1、終了時に F-1（収穫の再生）。
段階と門（G-1''・G-2・G-3'）は A-7 により呼ばない。門の判定関数と解除の状態は過去の走行の試験のために残す。
"""
import argparse
import copy
import json
import math
import os
import sys
import threading
import time
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch as th

from rlkit import action as action_mod
from rlkit import config as config_mod
from rlkit import constants, eci as eci_mod, runlog
from rlkit.learner import (SummerPPO, entropy_c0_at, policy_weights, replay_action_sequence,
                           run_deterministic_episode)
from rlkit.parallel.batched import SlotVecEnv, SlotVecEnvError
from rlkit.reward import R7Reference

STOP_FILE_NAME = "STOP"
V6_STAGE = 1   # A-7: 単一段階


# =====================================================================
# 門の判定（V5 までの純粋関数。A-7 により V6 では呼ばない）
# =====================================================================

def gate_g1_satisfied(official_positive_streak, update_index):
    """G-1''（指示書 V5-S 第 2-1 節）: 収穫の最良の公式 ECI > 0 が 3 更新連続、かつ通算更新 200 以上。"""
    return (official_positive_streak >= constants.G1_CONSECUTIVE_EVALS
            and update_index >= constants.G1_MIN_UPDATES)


def gate_g2_relative_difference(pre_official, post_official):
    """格子の細分化の前後の公式 ECI の相対差。基準が 0 のときは、一致していれば 0、違えば無限大とする。"""
    if pre_official > 0:
        return abs(post_official - pre_official) / pre_official
    return 0.0 if post_official == pre_official else math.inf


def gate_g3_release_ok(before_smooth, after_smooth):
    """G-3'（指示書 V5-S 第 2-1 節）: 解除後 3 更新の収穫の中央値が解除前の 0.98 倍以上か。"""
    return after_smooth >= constants.G3_ECI_SMOOTH_RATIO_MIN * before_smooth


class M5ReleaseState:
    """G-3 の解除・差し戻し・再試行の管理（V5 まで。A-7 により V6 では使わない）。"""

    def __init__(self, released=False, retries=0, retry_at=None, pending_since=None, before_median=None):
        self.released = released
        self.retries = retries
        self.retry_at = retry_at
        self.pending_since = pending_since
        self.before_median = before_median

    def succeed(self):
        self.released = True
        self.retry_at = None
        self.pending_since = None

    def fail(self, update_index):
        self.retries += 1
        self.retry_at = update_index + constants.G3_RETRY_UPDATES
        self.pending_since = None

    @property
    def exhausted(self):
        return self.retries >= constants.G3_MAX_RETRIES

    def due(self, update_index):
        return (not self.released) and self.retry_at is not None and update_index >= self.retry_at

    def to_dict(self):
        return {"released": self.released, "retries": self.retries, "retry_at": self.retry_at,
                "pending_since": self.pending_since, "before_median": self.before_median}

    @classmethod
    def from_dict(cls, d):
        return cls(d["released"], d["retries"], d["retry_at"], d.get("pending_since"), d.get("before_median"))


# =====================================================================
# 判定の純粋関数（V6-S 第 8 章）
# =====================================================================

def median_of(values):
    return float(np.median(np.asarray(values, dtype=np.float64))) if len(values) else None


def is_degenerate_actions(actions):
    """H-6: 全 378 ステップ・全 18 ゾーンの運転状態が同一か（決定論的評価の退化の検出）。"""
    states = np.asarray(actions)[:, :constants.N_ZONES]
    return bool(states.size > 0 and np.all(states == states.flat[0]))


def harvest_key(info):
    """V6.1-S 第 3-2 節: 収穫・提出候補・SIL の保存庫の順位は ECI（零点条件なし、√(P_E·P_DR)）の大きい順。"""
    return (float(eci_mod.eci_smooth(info["E"], info["DR"])),)


def c_u_values(updates, rho):
    """H-4'': 更新 u の訓練 12 本について C_u = mean(E + ρ·DR)。"""
    return [float(np.mean(np.asarray(u["train_E_list"]) + rho * np.asarray(u["train_DR_list"]))) for u in updates]


def window(values_by_update, lo, hi):
    """更新番号 lo〜hi（両端を含む）の値。values_by_update は更新 1 から順に並んだ列。"""
    return list(values_by_update[lo - 1:hi])


def h_checks(updates, required_keys):
    """H-1〜H-7（第 8 章、更新 20）。updates は更新 1〜20 の updates.jsonl の行。"""
    n = constants.H_CHECK_UPDATE
    rows = [u for u in updates if 1 <= u["update"] <= n]
    smooth = [u["harvest_update_best"]["eci_smooth"] for u in rows]
    half = n // 2
    cut = [bool(u["kl_early_stop"]) for u in rows if n - half < u["update"] <= n]
    secs = [u["update_seconds"] for u in rows]
    rho_n = rows[-1]["reference_used"]["rho"] if rows else None         # ρ₂₀: 更新 20 の採取で使った比
    C = c_u_values(rows, rho_n) if rows else []
    missing = {u["update"]: [k for k in required_keys if k not in u] for u in rows}
    missing = {k: v for k, v in missing.items() if v}
    out = {
        "H-1": {"passed": len(rows) == n and not missing, "n_updates": len(rows), "missing": missing},
        "H-2": {"passed": (sum(cut) / len(cut) if cut else 1.0) < constants.H2_KL_CUTOFF_RATE_MAX,
                "cutoff_rate_11_20": (sum(cut) / len(cut)) if cut else None},
        "H-3": {"passed": bool(smooth) and max(smooth) > smooth[0], "max_1_20": max(smooth) if smooth else None,
                "update1": smooth[0] if smooth else None},
        "H-4''": {"passed": len(C) == n and median_of(C[half:n]) < median_of(C[0:half]),
                  "rho20": rho_n, "C_u": C, "median_1_10": median_of(C[0:half]), "median_11_20": median_of(C[half:n])},
        # V6-S の H-4'（収穫の中央値の比較）は報告のみ
        "H-4'_report": {"harvest_ECI_median_1_10": median_of(smooth[0:half]),
                        "harvest_ECI_median_11_20": median_of(smooth[half:n])},
        "H-5": {"passed": bool(secs) and median_of(secs) < constants.H5_UPDATE_SECONDS_MAX,
                "median_update_seconds": median_of(secs)},
        # 報告だけで止めない事項
        "H-6": {"report": max((u.get("eval_degenerate_streak") or 0) for u in rows) >= constants.H6_DEGENERATE_STREAK if rows else False,
                "max_streak": max((u.get("eval_degenerate_streak") or 0) for u in rows) if rows else None},
        "H-9": h9_check(rows),
    }
    out["all_passed"] = all(out[k]["passed"] for k in ("H-1", "H-2", "H-3", "H-4''", "H-5"))
    return out


def h9_check(updates):
    """H-9（V6.1-S、報告のみ）: 採取に使った比が制限の端（30 または 200）に 10 更新以上続けて張り付いたか。"""
    best = streak = 0
    for u in updates:
        rho = u["reference_used"]["rho"]
        streak = streak + 1 if rho in (constants.R7_RHO_MIN, constants.R7_RHO_MAX) else 0
        best = max(best, streak)
    return {"report": best >= constants.H9_CLIP_STUCK_UPDATES, "max_streak": best}


def s8prime_check(updates):
    """S-8'（V6-S 更新 100）。V6.1-S で廃止（零点条件に基づくため呼ばない。UT-21）。"""
    rows = [u for u in updates if 1 <= u["update"] <= constants.S8PRIME_UPDATE]
    inside = sum(u["train_inside_count"] for u in rows)
    return {"passed": inside >= 1, "inside_episodes_1_100": inside, "n_updates": len(rows)}


def interim_checks(updates, required_keys):
    """I-1〜I-3（第 8 章、更新 300）。"""
    n = constants.INTERIM_UPDATE
    rows = [u for u in updates if 1 <= u["update"] <= n]
    by_u = {u["update"]: u for u in rows}
    smooth = [by_u[k]["harvest_update_best"]["eci_smooth"] for k in range(1, n + 1) if k in by_u]
    blk = constants.I_BLOCK
    cut = [bool(by_u[k]["kl_early_stop"]) for k in range(n - blk + 1, n + 1) if k in by_u]
    block_medians = [median_of(smooth[i:i + blk]) for i in range(0, n - blk, blk)]
    last = median_of(smooth[n - blk:n]) if len(smooth) == n else None
    missing = {u["update"]: [k for k in required_keys if k not in u] for u in rows}
    missing = {k: v for k, v in missing.items() if v}
    raw = by_u[n]["eval_raw_resid_rel"] if n in by_u else None
    out = {
        "I-1": {"passed": (sum(cut) / len(cut) if len(cut) == blk else 1.0) < constants.I1_KL_CUTOFF_RATE_MAX,
                "cutoff_rate_281_300": (sum(cut) / len(cut)) if cut else None},
        "I-2": {"passed": last is not None and bool(block_medians) and last >= constants.I2_RATIO * max(block_medians),
                "median_281_300": last, "block_medians_1_280": block_medians,
                "threshold": constants.I2_RATIO * max(block_medians) if block_medians else None},
        "I-3": {"passed": len(rows) == n and not missing and raw is not None and abs(raw) < constants.I3_RAW_RESID_MAX,
                "n_updates": len(rows), "missing": missing, "eval_raw_resid_rel_300": raw},
    }
    out["all_passed"] = all(out[k]["passed"] for k in ("I-1", "I-2", "I-3"))
    return out


def sil_guard_check(harvest_smooth_by_update):
    """SIL-6（更新 340）: 更新 301〜340 の収穫の最良 eci_smooth の中央値が、更新 261〜300 の中央値の 0.8 倍を下回るか。"""
    after = median_of(window(harvest_smooth_by_update, *constants.SIL_GUARD_WINDOW))
    before = median_of(window(harvest_smooth_by_update, *constants.SIL_GUARD_REFERENCE_WINDOW))
    triggered = after is not None and before is not None and after < constants.SIL_GUARD_RATIO * before
    return {"triggered": bool(triggered), "median_301_340": after, "median_261_300": before,
            "threshold": constants.SIL_GUARD_RATIO * before if before is not None else None}


def bd1_edges(actions, masks):
    """BD-1（報告のみ）: 行動列で、格子の端の選択肢（先頭・末尾）が有効な選択（GC-1 で除かず 2 択以上）に占める割合。"""
    actions = np.asarray(actions)
    masks = np.asarray(masks, dtype=bool)
    offsets = action_mod.dim_logit_offsets(V6_STAGE)
    n_opt = np.stack([masks[:, s:s + w].sum(axis=1) for s, w in offsets], axis=1)
    include = (~action_mod.gc1_excluded(actions)) & (n_opt >= 2)
    out = {}
    for g, info in action_mod.group_layout(V6_STAGE).items():
        s, n, K = info["start"], info["n"], info["K"]
        inc = include[:, s:s + n]
        a = actions[:, s:s + n]
        total = int(inc.sum())
        low = int(((a == 0) & inc).sum())
        high = int(((a == K - 1) & inc).sum())
        out[g] = {"K": K, "valid": total, "low_edge_frac": low / total if total else None,
                  "high_edge_frac": high / total if total else None}
        out[g]["flag"] = bool(K >= 3 and total and max(low, high) / total > constants.BD1_EDGE_FRACTION)
    return out


def _sparse_rates(rate_array):
    rate_array = np.asarray(rate_array, dtype=np.float64)
    return {int(i): float(rate_array[i]) for i in np.flatnonzero(rate_array > 0)}


# 第 3-8 節の記録項目（H-1・I-3 で欠落を確かめる、updates.jsonl のキー）
REQUIRED_UPDATE_KEYS = (
    # 換気
    "train_vent_level_fraction_mean", "train_hex_energy_gj_mean", "eval_vent_level_fraction", "eval_hex_energy_gj",
    # マスクと遮断
    "train_m5_zone_steps_mean", "train_m9_tenant_steps_mean", "eval_m5_zone_steps", "eval_m9_tenant_steps",
    "gc1_excluded_by_group", "one_option_excluded_by_group",
    # 報酬
    "reference_used", "reference_update",
    # 探索と学習率
    "c_g", "H_g", "policy_lr_after", "approx_kl_after", "kl_early_stop", "epochs_applied",
    # SIL
    "sil", "sil_buffer",
    # 選抜
    "harvest_update_best", "harvest_best", "train_E_mean", "train_DR_mean", "train_E_min", "train_DR_min",
    "train_inside_count", "train_E_list", "train_DR_list", "train_C60_mean",
    # R-7・収穫・観測（V6.1-S 第 3-6 節）
    "harvest_update_ECI", "harvest_best_P_E", "harvest_best_P_DR", "train_co2_delta_obs_clip_rate_max",
    # K-9 の結果 A・B
    "train_remote_permit_fraction_mean", "train_onoff_mismatch_sum", "eval_remote_permit_fraction", "eval_onoff_mismatch",
    # 従来の項目と判定に使う項目
    "eval_raw_resid_rel", "update_seconds",
)


class Trainer:
    def __init__(self, cfg, run_dir, factory="real", max_updates=None):
        """factory: "real"（エミュレータ）または "mock"（試験用の模擬環境。組の複製を作らない）。
        max_updates: 通算の更新数の上限（--updates は通算の意味。None なら計画更新数まで）。"""
        self.cfg = cfg
        self.run_dir = run_dir
        self.factory = factory
        self.max_updates = max_updates
        self.T = constants.ZONE_STEPS_PER_WEEK
        self.planned = int(cfg["planned_updates"])
        self.model = None
        self.train_vec = None
        self.eval_vec = None
        self.reference = None
        self.W = None
        self.best_smooth = {"eci_smooth": -math.inf}
        self.best_official = {"eci_official": -math.inf}
        self.harvest_smooth_history = []   # 毎更新の収穫の最良 eci_smooth（H-3・H-4'・I-2・SIL-6）
        self.harvest_official_streak = 0   # 収穫の最良の公式 ECI > 0 が続いた更新数（V5 からの記録項目）
        self.inside_history = []           # 決定論的評価が零点条件の内側だったか（V5 の Q-3。記録のみ）
        self.official_positive_streak = 0  # 決定論的評価の公式 ECI > 0 の連続回数（記録のみ）
        self.degenerate_streak = 0
        self.best_harvest = None           # L-4: 走行全体の収穫の最良（提出候補）
        self.harvest_replay = None         # F-1
        self.sil_buffer = []               # SIL-1
        self.sil_enabled = False           # SIL-4: 更新 300 の中間判定に合格したら真
        self.sil_guard = None              # SIL-6 の結果
        self.judgments = {}
        self.stop_reason = None
        self.nonfinite = False
        self._log_lock = threading.Lock()

    # ---- 記録 ----

    def _path(self, name):
        return os.path.join(self.run_dir, name)

    def log_event(self, **obj):
        with self._log_lock:
            runlog.append_jsonl(self._path("events.jsonl"), {"time": runlog.now_iso(), **obj})
        print(f"[train_summer] {obj}", flush=True)

    # ---- 環境と学習器 ----

    def env_kwargs(self, stage):
        return {"stage": stage, "include_indoor_unit_power": self.cfg["include_indoor_unit_power"],
                "zone_step_energy_norm_gj": self.cfg["zone_step_energy_norm_gj"],
                "m9_enabled": self.cfg["m9_enabled"]}

    def make_train_vec(self, stage):
        return SlotVecEnv(self.cfg["train_slots"], self.env_kwargs(stage),
                          steps_per_collection=self.T * constants.EPISODES_PER_SLOT, factory=self.factory)

    def make_eval_vec(self, stage):
        return SlotVecEnv([self.cfg["eval_slot"]], self.env_kwargs(stage), steps_per_collection=self.T,
                          factory=self.factory)

    def make_model(self, stage):
        return SummerPPO(self.train_vec, stage, self.planned, seed=self.cfg["seed"],
                         n_steps_per_env=self.T * constants.EPISODES_PER_SLOT)

    def apply_reference(self):
        """基準点を学習の組へ、(ρ_n, D_n) を価値ネットワークへ渡す（次の採取から使う。R-4''・O-5）。"""
        self.train_vec.env_method("set_reference", self.reference.to_dict())
        self.model.policy.set_reward_context(*self.reference.value_context())

    def collect_with_retries(self):
        """S-8: 1 更新の中で採取のやり直しが max_collection_retries − 1 回（既定 3 回）を超えたら例外で止める。"""
        for attempt in range(1, self.cfg["max_collection_retries"] + 1):
            try:
                return self.model.collect_one()
            except SlotVecEnvError as exc:
                self.log_event(event="collection_failed_discarded", attempt=attempt, error=str(exc)[-2000:])
                self.train_vec.close()
                if attempt == self.cfg["max_collection_retries"]:
                    raise
                self.train_vec = self.make_train_vec(self.model.stage)
                self.train_vec.env_method("set_reference", self.reference.to_dict())
                self.model.set_env(self.train_vec, force_reset=True)

    def evaluate_with_retries(self, policy):
        for attempt in range(1, self.cfg["max_eval_retries"] + 1):
            try:
                self.eval_vec.env_method("set_reference", self.reference.to_dict() if self.reference else None)
                info, actions = run_deterministic_episode(policy, self.eval_vec, self.T)
                return {"info": info, "actions": actions}
            except SlotVecEnvError as exc:
                self.log_event(event="eval_failed", attempt=attempt, error=str(exc)[-2000:])
                self.eval_vec.close()
                if attempt == self.cfg["max_eval_retries"]:
                    raise
                self.eval_vec = self.make_eval_vec(policy.stage)

    def _eval_thread(self, policy, out):
        try:
            out.update(self.evaluate_with_retries(policy))
        except Exception:
            out["error"] = traceback.format_exc()

    # ---- 決定論的評価の記録（診断のみ） ----

    def record_eval(self, update, info, actions, policy, tag="update"):
        rec = {
            "update": update, "tag": tag, "stage": policy.stage,
            "E": info["E_rep"], "DR": info["DR_rep"], "E_hex": float.hex(info["E_rep"]), "DR_hex": float.hex(info["DR_rep"]),
            "ECI": info["ECI"], "P_E": info["P_E"], "P_DR": info["P_DR"],
            "eci_official": info["eci_official"], "eci_smooth": info["eci_smooth"], "J": info["J"],
            "delta_E_vs_E_rep_rel": info["delta_E_vs_E_rep_rel"], "raw_resid_rel": info["raw_resid_rel"],
            "header_energy_fraction": info["header_energy_fraction"],
            "setpoint_override_rate": info["setpoint_override_rate"],
            "cooling_fraction_zone_unoccupied": info["cooling_fraction_zone_unoccupied"],
            "obs25_clip_rate_mean": info["obs25_clip_rate_mean"], "obs25_clip_rate_max": info["obs25_clip_rate_max"],
            "obs29_clip_rate_mean": info["obs29_clip_rate_mean"], "obs29_clip_rate_max": info["obs29_clip_rate_max"],
            "co2_clip_rate_mean": info["co2_clip_rate_mean"], "co2_clip_rate_max": info["co2_clip_rate_max"],
            "load_clip_rate_mean": info["load_clip_rate_mean"], "load_clip_rate_max": info["load_clip_rate_max"],
            "progress_clip_rate": info["progress_clip_rate"],
            "vent_obs_clip_rate": info["vent_obs_clip_rate"], "remote_obs_clip_rate": info["remote_obs_clip_rate"],
            "observation_clip_rate": _sparse_rates(info["observation_clip_rate"]),
            "vent_level_fraction": info["vent_level_fraction"], "vent_level_hours": info["vent_level_hours"],
            "hex_energy_gj": info["hex_energy_gj"], "m5_zone_steps": info["m5_zone_steps"],
            "m9_tenant_steps": info["m9_tenant_steps"], "remote_permit_fraction": info["remote_permit_fraction"],
            "onoff_mismatch": info["onoff_mismatch"],
        }
        rec["improved_smooth"] = info["eci_smooth"] > self.best_smooth["eci_smooth"]
        rec["improved_official"] = info["eci_official"] > self.best_official["eci_official"]
        if rec["improved_smooth"] or rec["improved_official"]:
            weights = policy_weights(policy)
            entry = {"update": update, "stage": policy.stage, "E": info["E_rep"], "DR": info["DR_rep"],
                     "E_hex": float.hex(info["E_rep"]), "DR_hex": float.hex(info["DR_rep"]),
                     "eci_smooth": info["eci_smooth"], "eci_official": info["eci_official"], "J": info["J"],
                     "weights": weights}
            os.makedirs(self._path("best"), exist_ok=True)
            if rec["improved_smooth"]:
                self.best_smooth = entry
                th.save(entry, self._path(os.path.join("best", "best_smooth.pt")))
                np.savez_compressed(self._path(os.path.join("best", "best_smooth_actions.npz")),
                                    actions=actions.astype(np.int8))
            if rec["improved_official"]:
                self.best_official = entry
                th.save(entry, self._path(os.path.join("best", "best_official.pt")))
                np.savez_compressed(self._path(os.path.join("best", "best_official_actions.npz")),
                                    actions=actions.astype(np.int8))
        rec["m5_enabled"] = True     # V5 からの記録項目（M-5' は常に有効）
        if tag == "update":
            self.official_positive_streak = self.official_positive_streak + 1 if info["eci_official"] > 0 else 0
            self.inside_history.append(bool(info["eci_official"] > 0))
            degenerate = is_degenerate_actions(actions)                   # H-6
            self.degenerate_streak = self.degenerate_streak + 1 if degenerate else 0
            rec["degenerate_actions"] = degenerate
            rec["degenerate_streak"] = self.degenerate_streak
        runlog.append_jsonl(self._path("evals.jsonl"), rec)
        return rec

    def record_train_episodes(self, update, terminal_infos):
        rows = []
        for step, i, info in sorted(terminal_infos, key=lambda x: (x[1], x[0])):
            row = {"update": update, "slot": self.cfg["train_slots"][i], "step_in_collection": step,
                   "E": info["E_rep"], "DR": info["DR_rep"], "eci_official": info["eci_official"],
                   "eci_smooth": info["eci_smooth"], "J": info["J"], "ECI": info["ECI"],
                   "P_E": info["P_E"], "P_DR": info["P_DR"],
                   "delta_E_vs_E_rep_rel": info["delta_E_vs_E_rep_rel"],
                   "raw_resid_rel": info["raw_resid_rel"], "header_energy_fraction": info["header_energy_fraction"],
                   "setpoint_override_rate": info["setpoint_override_rate"],
                   "cooling_fraction_zone_unoccupied": info["cooling_fraction_zone_unoccupied"],
                   "vent_level_fraction": info["vent_level_fraction"], "hex_energy_gj": info["hex_energy_gj"],
                   "m5_zone_steps": info["m5_zone_steps"], "m9_tenant_steps": info["m9_tenant_steps"],
                   "remote_permit_fraction": info["remote_permit_fraction"], "onoff_mismatch": info["onoff_mismatch"],
                   "vent_obs_clip_rate": info["vent_obs_clip_rate"], "remote_obs_clip_rate": info["remote_obs_clip_rate"],
                   "co2_delta_obs_clip_rate": info["co2_delta_obs_clip_rate"]}
            runlog.append_jsonl(self._path("episodes.jsonl"), row)
            rows.append((row, info["observation_clip_rate"]))
        expected = constants.N_TRAIN_SLOTS * constants.EPISODES_PER_SLOT
        if len(rows) != expected:
            raise RuntimeError(f"1 回の採取で終端したエピソードが {len(rows)} 本（{expected} 本であるべき）")
        R = [r for r, _ in rows]
        tenants = action_mod.VENT_TENANT_ORDER
        vent_mean = {t: {lv: float(np.mean([r["vent_level_fraction"][t][lv] for r in R])) for lv in constants.VENT_LEVELS}
                     for t in tenants}
        return {"train_episodes": len(rows),
                "train_eci_smooth_mean": float(np.mean([r["eci_smooth"] for r in R])),
                "train_eci_smooth_max": float(np.max([r["eci_smooth"] for r in R])),
                "train_E_mean": float(np.mean([r["E"] for r in R])),      # R-2'' の入力 Ē_u
                "train_DR_mean": float(np.mean([r["DR"] for r in R])),    # R-2'' の入力 DR̄_u
                "train_E_min": float(np.min([r["E"] for r in R])), "train_DR_min": float(np.min([r["DR"] for r in R])),
                "train_E_median": float(np.median([r["E"] for r in R])),
                "train_DR_median": float(np.median([r["DR"] for r in R])),
                "train_inside_count": None,          # V6.1-S で廃止（零点条件）。項目は None で残す
                "train_E_list": [r["E"] for r in R], "train_DR_list": [r["DR"] for r in R],   # H-4'' の C_u 用
                "train_C60_mean": float(np.mean([r["E"] + constants.R7_RECORD_FIXED_RHO * r["DR"] for r in R])),
                "train_ECI_mean": float(np.mean([r["ECI"] for r in R])),
                "train_co2_delta_obs_clip_rate_max": float(max(r["co2_delta_obs_clip_rate"] for r in R)),
                "train_raw_resid_rel_median": float(np.median([r["raw_resid_rel"] for r in R])),
                "train_vent_level_fraction_mean": vent_mean,
                "train_hex_energy_gj_mean": float(np.mean([r["hex_energy_gj"] for r in R])),
                "train_m5_zone_steps_mean": float(np.mean([r["m5_zone_steps"] for r in R])),
                "train_m9_tenant_steps_mean": float(np.mean([r["m9_tenant_steps"] for r in R])),
                "train_remote_permit_fraction_mean": {t: float(np.mean([r["remote_permit_fraction"][t] for r in R]))
                                                      for t in tenants},
                "train_onoff_mismatch_sum": int(sum(r["onoff_mismatch"] for r in R)),
                "train_vent_obs_clip_rate_max": float(max(r["vent_obs_clip_rate"] for r in R)),
                "train_remote_obs_clip_rate_max": float(max(r["remote_obs_clip_rate"] for r in R)),
                "train_observation_clip_rate": _sparse_rates(np.mean([c for _, c in rows], axis=0))}

    # ---- 収穫（L-4）と SIL の保存庫（SIL-1） ----

    def harvest_best(self, update, terminal_infos, rollout):
        """L-4: 訓練 12 エピソードのうち公式 ECI 最大（同点は J 最大）の行動列を残し、SIL の保存庫の候補にする。

        行動列は採取済みの rollout から切り出す。組 i の行は env-major で [i·n, (i+1)·n) に並び、
        終端ステップ step のエピソードはその末尾 T 行である。
        """
        n_steps = self.T * constants.EPISODES_PER_SLOT
        best = None
        for step, i, info in terminal_infos:
            meta = {"E": info["E_rep"], "DR": info["DR_rep"]}
            key = harvest_key(meta)
            if best is None or key > best["key"]:
                end = i * n_steps + int(step) + 1
                best = {"key": key, "end": end, "update": update, "slot": self.cfg["train_slots"][i],
                        "step_in_collection": int(step), "stage": self.model.stage,
                        "E": info["E_rep"], "DR": info["DR_rep"], "eci_official": info["eci_official"],
                        "eci_smooth": info["eci_smooth"], "ECI": info["ECI"], "P_E": info["P_E"],
                        "P_DR": info["P_DR"], "J": None,
                        "E_hex": float.hex(info["E_rep"]), "DR_hex": float.hex(info["DR_rep"])}
        sl = slice(best["end"] - self.T, best["end"])
        entry = {"update": update, "E": best["E"], "DR": best["DR"], "eci_official": best["eci_official"],
                 "eci_smooth": best["eci_smooth"], "ECI": best["ECI"], "J": None, "W": self.W,
                 "actions": np.asarray(rollout["actions"][sl], dtype=np.int64),
                 "obs": np.asarray(rollout["observations"][sl], dtype=np.float32),
                 "masks": np.asarray(rollout["action_masks"][sl], dtype=bool),
                 "step_seconds": np.asarray(rollout["step_seconds"][sl], dtype=np.float64),
                 "delta_E": np.asarray(rollout["delta_E"][sl], dtype=np.float64),
                 "zone_energy": np.asarray(rollout["zone_energy"][sl], dtype=np.float64),
                 "zone_n": np.asarray(rollout["zone_n"][sl], dtype=np.float64)}
        summary = {k: v for k, v in best.items() if k not in ("key", "end")}
        out = {"harvest_update_best": summary}
        prev = self.best_harvest
        if prev is None or best["key"] > harvest_key(prev):
            os.makedirs(self._path("harvest"), exist_ok=True)
            np.savez_compressed(self._path(os.path.join("harvest", "best_harvest.npz")),
                                actions=entry["actions"].astype(np.int8), masks=entry["masks"])
            self.best_harvest = dict(summary)
            runlog.write_json(self._path(os.path.join("harvest", "best_harvest.json")), self.best_harvest)
        out["harvest_best"] = dict(self.best_harvest)
        self.harvest_smooth_history.append(best["eci_smooth"])
        self.harvest_official_streak = (self.harvest_official_streak + 1) if best["eci_official"] > 0 else 0
        out["harvest_best_official"] = self.best_harvest["eci_official"]      # 参考（V6.1-S では判定・選抜に使わない）
        out["harvest_update_ECI"] = best["ECI"]
        out["harvest_best_P_E"] = self.best_harvest["P_E"]
        out["harvest_best_P_DR"] = self.best_harvest["P_DR"]
        out["harvest_update_official"] = best["eci_official"]
        out["harvest_official_streak"] = self.harvest_official_streak
        self.add_to_sil_buffer(entry)
        out["sil_buffer"] = [{"update": e["update"], "ECI": e["ECI"], "eci_official": e["eci_official"]}
                             for e in self.sil_buffer]
        return out

    def add_to_sil_buffer(self, entry):
        """SIL-1: 走行全体で上位 8 本（L-4 と同じ順位）を保つ。"""
        self.sil_buffer.append(entry)
        self.sil_buffer.sort(key=lambda e: harvest_key(e), reverse=True)
        del self.sil_buffer[constants.SIL_BUFFER_SIZE:]

    def sil_mode(self, u):
        """SIL-4・SIL-7: 更新 1〜300 は記録だけ、301 以降は中間判定に合格した場合だけ適用。"""
        if u < constants.SIL_START_UPDATE:
            return "stats"
        return "apply" if self.sil_enabled else "stats"

    def verify_harvest_replay(self):
        """F-1: 収穫の最良の行動列を評価用の組で 1 エピソード再生し、E・DR の 16 進一致を確かめる。"""
        path = self._path(os.path.join("harvest", "best_harvest.npz"))
        if self.best_harvest is None or self.eval_vec is None or not os.path.exists(path):
            return None
        actions = np.load(path)["actions"].astype(np.int64)
        self.eval_vec.env_method("set_reference", self.reference.to_dict() if self.reference else None)
        info = replay_action_sequence(self.eval_vec, actions)
        rec = {"time": runlog.now_iso(), "harvested_at_update": self.best_harvest["update"],
               "slot_harvested": self.best_harvest["slot"], "replay_slot": self.cfg["eval_slot"],
               "saved_E": self.best_harvest["E"], "saved_DR": self.best_harvest["DR"],
               "saved_E_hex": self.best_harvest["E_hex"], "saved_DR_hex": self.best_harvest["DR_hex"],
               "replay_E": info["E_rep"], "replay_DR": info["DR_rep"],
               "replay_E_hex": float.hex(info["E_rep"]), "replay_DR_hex": float.hex(info["DR_rep"]),
               "replay_eci_official": info["eci_official"], "replay_eci_smooth": info["eci_smooth"],
               "replay_ECI": info["ECI"], "replay_P_E": info["P_E"], "replay_P_DR": info["P_DR"]}
        rec["E_hex_match"] = rec["replay_E_hex"] == rec["saved_E_hex"]
        rec["DR_hex_match"] = rec["replay_DR_hex"] == rec["saved_DR_hex"]
        rec["F-1_passed"] = rec["E_hex_match"] and rec["DR_hex_match"]
        runlog.write_json(self._path(os.path.join("harvest", "replay_check.json")), rec)
        self.log_event(event="harvest_replay", **rec)
        return rec

    def bd1(self, update):
        path = self._path(os.path.join("harvest", "best_harvest.npz"))
        if not os.path.exists(path):
            return None
        d = np.load(path)
        rec = {"update": update, "harvest_best_update": self.best_harvest["update"],
               "groups": bd1_edges(d["actions"].astype(np.int64), d["masks"])}
        rec["flagged"] = [g for g, v in rec["groups"].items() if v["flag"]]
        runlog.write_json(self._path(f"bd1_u{update}.json"), rec)
        return rec

    # ---- 保存と再開 ----

    def save_checkpoint(self):
        state = {"model": self.model.training_state(), "reference": self.reference.to_dict(),
                 "best_smooth": self.best_smooth, "best_official": self.best_official,
                 "best_harvest": self.best_harvest, "harvest_smooth_history": self.harvest_smooth_history,
                 "degenerate_streak": self.degenerate_streak, "sil_enabled": self.sil_enabled,
                 "harvest_official_streak": self.harvest_official_streak, "inside_history": self.inside_history,
                 "official_positive_streak": self.official_positive_streak,
                 "sil_guard": self.sil_guard, "judgments": self.judgments,
                 "torch_rng": th.get_rng_state(), "numpy_rng": np.random.get_state()}
        path = self._path("checkpoint_latest.pt")
        th.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)
        sil_path = self._path("sil_buffer.pt")
        th.save(self.sil_buffer, sil_path + ".tmp")
        os.replace(sil_path + ".tmp", sil_path)

    def load_checkpoint(self):
        state = th.load(self._path("checkpoint_latest.pt"), weights_only=False)
        stage = state["model"]["stage"]
        self.train_vec = self.make_train_vec(stage)
        self.eval_vec = self.make_eval_vec(stage)
        self.model = self.make_model(stage)
        self.model.load_training_state(state["model"])
        self.reference = R7Reference.from_dict(state["reference"])
        self.best_smooth = state["best_smooth"]
        self.best_official = state["best_official"]
        self.best_harvest = state["best_harvest"]
        self.harvest_smooth_history = list(state["harvest_smooth_history"])
        self.degenerate_streak = state["degenerate_streak"]
        self.harvest_official_streak = state["harvest_official_streak"]
        self.inside_history = list(state["inside_history"])
        self.official_positive_streak = state["official_positive_streak"]
        self.sil_enabled = state["sil_enabled"]
        self.sil_guard = state["sil_guard"]
        self.judgments = dict(state["judgments"])
        self.sil_buffer = th.load(self._path("sil_buffer.pt"), weights_only=False)
        th.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])

    def resume_snapshot(self):
        """MUST-9: 再開時に引き継いだ状態（更新 21 で直前の更新の記録と照合する）。"""
        return {"update_index": self.model.update_index,
                "reference": self.reference.summary(),
                "rho": self.reference.rho, "policy_lr": self.model.policy_lr,
                "best_harvest": self.best_harvest,
                "sil_buffer": [{"update": e["update"], "ECI": e["ECI"]} for e in self.sil_buffer],
                "c_sil": self.model.c_sil,
                "entropy_c0_next_update": entropy_c0_at(self.model.update_index + 1, self.planned),
                "planned_updates": self.planned, "sil_enabled": self.sil_enabled}

    # ---- 判定 ----

    def read_updates(self):
        path = self._path("updates.jsonl")
        if not os.path.exists(path):
            return []
        return [json.loads(l) for l in open(path, encoding="utf-8")]

    def judge_after_update(self, u, record):
        """第 7 章・第 8 章の判定。止める場合は stop_reason を入れる。"""
        if abs(record["eval_raw_resid_rel"]) > constants.S7_RAW_RESID_STOP:
            self.stop_reason = f"S-7: 更新 {u} の決定論的評価の raw_resid {record['eval_raw_resid_rel']:.4f} が 5% を超えた"
        if u == constants.H_CHECK_UPDATE:
            res = h_checks(self.read_updates(), REQUIRED_UPDATE_KEYS)
            self.judgments["H"] = res
            runlog.write_json(self._path(f"judgment_u{u}.json"), res)
            self.log_event(event="H_checks", update=u, all_passed=res["all_passed"])
            if not res["all_passed"]:
                self.stop_reason = f"S-3: 更新 {u} の H-1〜H-5 に不合格がある（judgment_u{u}.json）"
        # S-8'（V6-S）は V6.1-S で廃止（零点条件に基づくため）
        h9 = h9_check(self.read_updates())
        if h9["report"] and not self.judgments.get("H-9_reported"):
            self.judgments["H-9_reported"] = u
            self.log_event(event="H-9_ratio_stuck_at_clip", update=u, **h9)     # 報告のみ
        if u == constants.INTERIM_UPDATE:
            res = interim_checks(self.read_updates(), REQUIRED_UPDATE_KEYS)
            res["bd1"] = self.bd1(u)
            self.judgments["I"] = res
            runlog.write_json(self._path(f"judgment_u{u}.json"), res)
            self.log_event(event="interim_checks", update=u, all_passed=res["all_passed"])
            if res["all_passed"]:
                self.sil_enabled = True                      # SIL-4: 更新 301 から有効
                self.log_event(event="SIL_enabled_from", update=constants.SIL_START_UPDATE)
            else:
                self.stop_reason = f"S-5: 更新 {u} の I-1〜I-3 に不合格がある（judgment_u{u}.json）"
        if u == constants.SIL_GUARD_UPDATE and self.sil_enabled:
            res = sil_guard_check(self.harvest_smooth_history)
            self.sil_guard = res
            runlog.write_json(self._path(f"judgment_u{u}.json"), res)
            self.log_event(event="SIL-6_guard", update=u, **res)
            if res["triggered"]:
                self.model.c_sil = 0.0                       # 以後 c_sil = 0 にして学習を続ける（報告事項）
        if u == self.planned and u in constants.BD1_UPDATES:
            self.judgments[f"BD-1_u{u}"] = self.bd1(u)

    # ---- 1 回の更新 ----

    def one_update(self):
        model = self.model
        u = model.update_index + 1
        t0 = time.time()
        self.apply_reference()
        # R-7: この更新の採取に使う比と評価点（第 3-6 節の記録）。R-5 の項目は廃止のため None で残す
        ref_used = self.reference.summary()
        ref_used.update({"rho_raw": (R7Reference.raw_ratio(self.reference.E_ref, self.reference.DR_ref)
                                     if self.reference.ref_update is not None else None),   # 現在の評価点での ρ_raw
                         "E_star": self.reference.E_ref, "DR_star": self.reference.DR_ref,
                         "alpha_s": None, "beta_s": None, "sigma_E": None, "sigma_D": None, "rho_r5": None})
        eval_policy = copy.deepcopy(model.policy)
        eval_out = {}
        thread = None
        if self.cfg["eval_concurrent"]:
            thread = threading.Thread(target=self._eval_thread, args=(eval_policy, eval_out), daemon=True)
            thread.start()
        try:
            terminal = self.collect_with_retries()
        finally:
            if thread is not None:
                thread.join()
        collect_s = time.time() - t0
        if thread is None:
            self._eval_thread(eval_policy, eval_out)
        if "error" in eval_out:
            raise RuntimeError(f"決定論的評価に失敗した:\n{eval_out['error']}")
        info = eval_out["info"]
        episode_summary = self.record_train_episodes(u, terminal)
        eval_rec = self.record_eval(u, info, eval_out["actions"], eval_policy)
        record = {"update": u, "stage": model.stage, "collect_and_eval_s": collect_s,
                  "eval_E": info["E_rep"], "eval_DR": info["DR_rep"], "eval_eci_official": info["eci_official"],
                  "eval_ECI": info["ECI"], "eval_P_E": info["P_E"], "eval_P_DR": info["P_DR"],
                  "eval_co2_delta_obs_clip_rate": info["co2_delta_obs_clip_rate"],
                  "eval_eci_smooth": info["eci_smooth"], "eval_J": info["J"],
                  "eval_delta_E_vs_E_rep_rel": info["delta_E_vs_E_rep_rel"],
                  "eval_raw_resid_rel": info["raw_resid_rel"],
                  "eval_header_energy_fraction": info["header_energy_fraction"],
                  "eval_cooling_fraction_zone_unoccupied": info["cooling_fraction_zone_unoccupied"],
                  "eval_obs25_clip_rate_max": info["obs25_clip_rate_max"],
                  "eval_obs29_clip_rate_max": info["obs29_clip_rate_max"],
                  "eval_co2_clip_rate_max": info["co2_clip_rate_max"],
                  "eval_load_clip_rate_max": info["load_clip_rate_max"],
                  "eval_vent_obs_clip_rate": info["vent_obs_clip_rate"],
                  "eval_remote_obs_clip_rate": info["remote_obs_clip_rate"],
                  "eval_observation_clip_rate": eval_rec["observation_clip_rate"],
                  "eval_vent_level_fraction": info["vent_level_fraction"], "eval_hex_energy_gj": info["hex_energy_gj"],
                  "eval_m5_zone_steps": info["m5_zone_steps"], "eval_m9_tenant_steps": info["m9_tenant_steps"],
                  "eval_remote_permit_fraction": info["remote_permit_fraction"],
                  "eval_onoff_mismatch": info["onoff_mismatch"],
                  "eval_obs25_clip_rate_mean": info["obs25_clip_rate_mean"],
                  "eval_obs29_clip_rate_mean": info["obs29_clip_rate_mean"],
                  "eval_progress_clip_rate": info["progress_clip_rate"],
                  "eval_inside_zero_floor": bool(info["eci_official"] > 0),
                  "official_positive_streak": self.official_positive_streak,
                  "inside_rate_recent20": (float(np.mean(self.inside_history[-constants.INSIDE_RATE_WINDOW:]))
                                           if self.inside_history else None),
                  # V5 の記録項目のうち廃止した仕組みのもの（M-8 と解除の状態）。M-5' は常に有効
                  "m5_enabled": True, "m5_state": None, "m8_enabled": False, "eval_m8_zone_steps": None,
                  "eval_degenerate_actions": eval_rec.get("degenerate_actions"),
                  "eval_degenerate_streak": eval_rec.get("degenerate_streak"),
                  "m9_enabled": bool(self.cfg["m9_enabled"]), **episode_summary}
        # L-4・SIL-1: 学習の前に、採取した行動列から最良の 1 本を収穫する（train() が rollout を捨てるため）
        record.update(self.harvest_best(u, terminal, model._rollout))
        if self.degenerate_streak >= constants.H6_DEGENERATE_STREAK:      # H-6（報告事項。停止はしない）
            self.log_event(event="H-6_degenerate_argmax", update=u, streak=self.degenerate_streak)

        t1 = time.time()
        mode = self.sil_mode(u)
        record.update(model.train(sil_buffer=self.sil_buffer, sil_mode=mode, reference=self.reference))
        record["train_s"] = time.time() - t1
        record["reference_used"] = ref_used
        # R-7: 更新 u の収穫の後に評価点を更新し、その比を更新 u+1 の採取に使う（評価点が変わらなければ比も変えない）
        best = dict(self.best_harvest, ECI=self.best_harvest["ECI"])
        record["reference_update"] = self.reference.update_after_harvest(best, u)
        record["reference"] = self.reference.summary()
        record["sil_enabled"] = self.sil_enabled
        if record["nonfinite"]:
            self.nonfinite = True
            self.stop_reason = f"S-6: 更新 {u} で損失またはパラメータに NaN・無限大が出た（直前の保存点を残す）"
            record["update_seconds"] = time.time() - t0
            runlog.append_jsonl(self._path("updates.jsonl"), record)
            return
        record["update_seconds"] = time.time() - t0
        runlog.append_jsonl(self._path("updates.jsonl"), record)
        self.save_checkpoint()
        self.judge_after_update(u, record)
        if self.stop_reason is None or not self.stop_reason.startswith("S-6"):
            self.save_checkpoint()   # 判定で変わった状態（SIL の有効化など）を保存する

    # ---- 走行 ----

    def run(self, resume=False):
        os.makedirs(self.run_dir, exist_ok=True)
        meta = {"started_at": runlog.now_iso(), "argv": sys.argv, "config": self.cfg, "factory": self.factory,
                "max_updates_total": self.max_updates, "resume": resume, "k9_result": constants.K9_RESULT}
        if self.factory == "real":
            from rlkit.parallel import launch as launch_mod
            from rlkit.parallel import slots
            slot_indices = list(self.cfg["train_slots"]) + [self.cfg["eval_slot"]]
            meta["preflight"] = launch_mod.preflight(self.run_dir, slot_indices)
            os.environ[launch_mod.LAUNCH_LOG_ENV] = meta["preflight"]["log_dir"]
            meta["replica_checks"] = [slots.prepare_replica(slots.make_slot(k)) for k in slot_indices]
        runlog.write_json(self._path(f"meta_{'resume' if resume else 'start'}_{int(time.time())}.json"), meta)

        status = "error"
        done = 0
        try:
            if resume:
                self.load_checkpoint()
                self.W = self.train_vec.get_attr("W")[0]
                self.log_event(event="resumed", **self.resume_snapshot())
            else:
                self.train_vec = self.make_train_vec(V6_STAGE)
                self.eval_vec = self.make_eval_vec(V6_STAGE)
                self.model = self.make_model(V6_STAGE)
                self.W = self.train_vec.get_attr("W")[0]
                # R-7: 更新 1 は ρ₀ = 60、分母は E_ref0 = 6.5 GJ・DR_ref0 = 0.20。新規初期化の決定論的評価は診断として記録だけ
                self.reference = R7Reference()
                out = self.evaluate_with_retries(self.model.policy)
                info = out["info"]
                self.record_eval(0, info, out["actions"], self.model.policy, tag="initial")
                self.log_event(event="initial_reference", **self.reference.summary(),
                               initial_eval_E=info["E_rep"], initial_eval_DR=info["DR_rep"], initial_eval_ECI=info["ECI"])
                self.save_checkpoint()

            while self.model.update_index < self.planned:
                if self.max_updates is not None and self.model.update_index >= self.max_updates:
                    self.stop_reason = f"通算 {self.max_updates} 更新に達した（--updates は通算）"
                    break
                if os.path.exists(self._path(STOP_FILE_NAME)):
                    self.stop_reason = "STOP ファイルが置かれた"
                    break
                self.one_update()
                done += 1
                if self.stop_reason:
                    break
            if self.stop_reason is None:
                self.stop_reason = f"計画更新数 {self.planned} に到達した"
                if self.planned in constants.BD1_UPDATES and f"BD-1_u{self.planned}" not in self.judgments:
                    self.judgments[f"BD-1_u{self.planned}"] = self.bd1(self.planned)
                try:
                    self.harvest_replay = self.verify_harvest_replay()   # F-1
                except Exception:
                    self.log_event(event="harvest_replay_failed", traceback=traceback.format_exc())
            status = "stopped"
        except Exception:
            self.stop_reason = f"例外: {traceback.format_exc()}"
            self.log_event(event="exception", traceback=traceback.format_exc())
        finally:
            if self.model is not None and self.reference is not None and not self.nonfinite:
                self.save_checkpoint()
            for vec in (self.train_vec, self.eval_vec):
                if vec is not None:
                    vec.close()
            runlog.write_json(self._path("status.json"), {
                "status": status, "stop_reason": self.stop_reason, "finished_at": runlog.now_iso(),
                "updates_this_run": done,
                "update_index": self.model.update_index if self.model is not None else None,
                "stage": self.model.stage if self.model is not None else None,
                "best_smooth": {k: v for k, v in self.best_smooth.items() if k != "weights"},
                "best_official": {k: v for k, v in self.best_official.items() if k != "weights"},
                "best_harvest": self.best_harvest, "harvest_replay": self.harvest_replay,
                "sil_enabled": self.sil_enabled, "sil_guard": self.sil_guard,
                "judgments": {k: (v.get("all_passed", v.get("passed")) if isinstance(v, dict) else v)
                              for k, v in self.judgments.items()}})
        return self.stop_reason


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description="夏季の学習（指示書 V6-S）")
    p.add_argument("--config", required=True, help="走行の設定 JSON（rlkit/config_summer_v6.json）")
    p.add_argument("--resume", action="store_true", help="run_dir の checkpoint_latest.pt から再開する")
    p.add_argument("--updates", type=int, default=None, help="通算の更新数の上限（試走は 20）")
    a = p.parse_args(argv)
    cfg = config_mod.load(a.config)
    missing = config_mod.unresolved_items(cfg)
    if missing:
        lines = "\n".join(f"  {key}（{item}）: {desc}" for key, item, desc in missing)
        raise SystemExit("未確定項目が設定されていないため起動しない:\n" + lines)
    runlog.check_thread_env()
    th.set_num_threads(1)
    trainer = Trainer(cfg, config_mod.repo_path(cfg["run_dir"]), max_updates=a.updates)
    reason = trainer.run(resume=a.resume)
    print(f"[train_summer] 停止: {reason}", flush=True)


if __name__ == "__main__":
    main()
