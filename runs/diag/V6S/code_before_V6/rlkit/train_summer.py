"""rlkit.train_summer — 夏季 v2 の学習の入口（v2 第 5-5 節。指示書 V2-S 第 2 章の決定を反映）。

リポジトリルートから（PowerShell の例）:
  $env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1; $env:KMP_BLOCKTIME=0
  python -m rlkit.train_summer --config rlkit/config_summer_v2.json [--updates 20] [--resume]

未確定項目（rlkit/config.py の UNRESOLVED）が 1 つでも null なら起動しない。

1 回の更新:
  1 学習の 6 組 × 2 周で 12 エピソードを採取する（基準点は採取の前に全組へ渡す）
  2 同時に（eval_concurrent=false なら採取の後に）、採取に使う方策の写しで決定論的評価を 1 エピソード行う
  H-1（収穫）: 12 エピソードのうち公式 ECI 最大の行動列を残し、走行全体の最良を harvest/ に保持する
  3〜7 学習（rlkit.learner.SummerPPO.train）。勾配ノルム比は更新 1・10・50・100・200 で計測して記録する（E-2）
  R-2''・R-3''（指示書 V4-S 第 2-1 節）: 訓練 12 エピソードの (E, DR) の平均で基準点を指数移動平均で動かす
    （学習後。次の採取から使う）。β/α の 2 倍制限は毎更新かける。決定論的評価の点は基準点に使わない
  退行処理（最良重みの再読み込みと学習率の半減）は指示書 V4-S Q-1 により廃止した
  8 記録と保存
段階（指示書 V2-S 第 2-4 節）: 更新の前に門を評価する。
  G-1' 第 1 段階の合格（公式 ECI > 0 が 3 回連続、通算更新 200 以上。非占有冷房率は V4-S D-21 で廃止）
  G-2 格子の細分化（M-5 は残す）。変換の前後で公式 ECI の相対差が 1e-4 を超えたら停止
  G-3 M-5 の解除。解除後の eci_smooth が解除前の 0.98 倍以上なら確定、下回れば M-5 を戻して 50 更新後に再試行（3 回まで）
  G-4 更新番号は段階をまたいで通算する
  G-5 G-1 を満たしたら自動で切り替える。設定 manual_stage2_update による手動切替も使える
  U-5 通算更新番号が planned_updates × stage2_start_fraction に達しても第 1 段階のままなら停止する
"""
import argparse
import copy
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
from rlkit import constants, runlog
from rlkit.learner import (SummerPPO, policy_weights, replay_action_sequence, run_deterministic_episode,
                           transfer_to_stage2)
from rlkit.parallel.batched import SlotVecEnv, SlotVecEnvError
from rlkit.reward import ReferencePoint

STOP_FILE_NAME = "STOP"


# =====================================================================
# 門の判定（純粋関数。指示書 V2-S 第 2-4 節）
# =====================================================================

def gate_g1_satisfied(official_positive_streak, update_index):
    """G-1''（指示書 V5-S 第 2-1 節）: 収穫の最良の公式 ECI > 0 が 3 更新連続、かつ通算更新 200 以上。

    V4-S の G-1' は (a) を決定論的評価で測っていたが、argmax が退化するため収穫に置き換えた（D-24）。
    V2-S の (b) 非占有冷房率 10% 未満は D-21 により廃止済み。
    """
    return (official_positive_streak >= constants.G1_CONSECUTIVE_EVALS
            and update_index >= constants.G1_MIN_UPDATES)


def gate_g2_relative_difference(pre_official, post_official):
    """格子の細分化の前後の公式 ECI の相対差。基準が 0 のときは、一致していれば 0、違えば無限大とする。"""
    if pre_official > 0:
        return abs(post_official - pre_official) / pre_official
    return 0.0 if post_official == pre_official else math.inf


def gate_g3_release_ok(before_smooth, after_smooth):
    """G-3'（指示書 V5-S 第 2-1 節）: M-5・M-8 の解除の可否。

    解除後 3 更新の収穫の最良 eci_smooth の中央値が、解除前 3 更新の中央値の 0.98 倍以上か。
    """
    return after_smooth >= constants.G3_ECI_SMOOTH_RATIO_MIN * before_smooth


def median_of(values):
    return float(np.median(np.asarray(values, dtype=np.float64))) if len(values) else None


def is_degenerate_actions(actions):
    """S-3: 全 378 ステップ・全 18 ゾーンの運転状態が同一か（決定論的評価の退化の検出）。"""
    states = np.asarray(actions)[:, :constants.N_ZONES]
    return bool(states.size > 0 and np.all(states == states.flat[0]))


class M5ReleaseState:
    """G-3 の解除・差し戻し・再試行の管理。"""

    def __init__(self, released=False, retries=0, retry_at=None, pending_since=None, before_median=None):
        self.released = released
        self.retries = retries
        self.retry_at = retry_at
        self.pending_since = pending_since     # G-3': 解除した更新番号（3 更新後に判定する）
        self.before_median = before_median     # G-3': 解除前 3 更新の収穫の中央値

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


def _sparse_rates(rate_array):
    rate_array = np.asarray(rate_array, dtype=np.float64)
    return {int(i): float(rate_array[i]) for i in np.flatnonzero(rate_array > 0)}


class Trainer:
    def __init__(self, cfg, run_dir, factory="real", max_updates=None):
        """factory: "real"（エミュレータ）または "mock"（試験用の模擬環境。組の複製を作らない）。
        max_updates: 通算の更新数の上限（指示書 V4-S W-3 により --updates は通算の意味。None なら計画更新数まで）。"""
        self.cfg = cfg
        self.run_dir = run_dir
        self.factory = factory
        self.max_updates = max_updates
        self.T = constants.ZONE_STEPS_PER_WEEK
        self.planned = int(cfg["planned_updates"])
        self.stage1_ceiling_update = max(1, int(round(cfg["stage2_start_fraction"] * self.planned)))
        self.model = None
        self.train_vec = None
        self.eval_vec = None
        self.reference = None
        self.best_smooth = {"eci_smooth": -math.inf}
        self.best_official = {"eci_official": -math.inf}
        self.inside_history = []      # Q-3: 決定論的評価が零点条件の内側だったか（古い順）
        self.harvest_smooth_history = []   # S-1: 毎更新の収穫の最良 eci_smooth（H-4'・G-3' に使う）
        self.harvest_official_streak = 0   # G-1'': 収穫の最良の公式 ECI > 0 が続いた更新数
        self.degenerate_streak = 0         # S-3: 決定論的評価が全ゾーン同一だった連続更新数
        self.best_harvest = None      # H-1: 走行全体の収穫の最良
        self.harvest_replay = None    # H-2: 収穫の再生確認
        self.official_positive_streak = 0
        self.last_eval = None
        self.m5_enabled = True
        self.m8_enabled = True        # M-8（在室ゾーンは冷房だけ）。G-3' で M-5 とともに解除する
        self.m5_state = M5ReleaseState()
        self.stage2_reference_smooth = None
        self.stop_reason = None
        self._log_lock = threading.Lock()

    # ---- 記録 ----

    def _path(self, name):
        return os.path.join(self.run_dir, name)

    def log_event(self, **obj):
        with self._log_lock:
            runlog.append_jsonl(self._path("events.jsonl"), {"time": runlog.now_iso(), **obj})
        print(f"[train_summer] {obj}", flush=True)

    def log_gate(self, **obj):
        runlog.append_jsonl(self._path("gates.jsonl"), {"time": runlog.now_iso(), **obj})
        self.log_event(**obj)

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
        return SummerPPO(self.train_vec, stage, self.cfg["entropy_coef_init"], self.planned, seed=self.cfg["seed"],
                         n_steps_per_env=self.T * constants.EPISODES_PER_SLOT)

    def apply_m5(self):
        """M-5 と M-8 の旗を全組へ配る（G-3' で両方まとめて解除する）。"""
        for vec in (self.train_vec, self.eval_vec):
            if vec is not None:
                vec.env_method("set_m5_enabled", self.m5_enabled)
                vec.env_method("set_m8_enabled", self.m8_enabled)

    def collect_with_retries(self):
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
                self.train_vec.env_method("set_m5_enabled", self.m5_enabled)
                self.model.set_env(self.train_vec, force_reset=True)

    def evaluate_with_retries(self, policy):
        for attempt in range(1, self.cfg["max_eval_retries"] + 1):
            try:
                self.eval_vec.env_method("set_reference", self.reference.to_dict() if self.reference else None)
                self.eval_vec.env_method("set_m5_enabled", self.m5_enabled)
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

    # ---- 評価の記録と最良の管理 ----

    def record_eval(self, update, info, actions, policy, tag="update"):
        rec = {
            "update": update, "tag": tag, "stage": policy.stage, "m5_enabled": self.m5_enabled,
            "E": info["E_rep"], "DR": info["DR_rep"],
            "eci_official": info["eci_official"], "eci_smooth": info["eci_smooth"],
            "delta_E_vs_E_rep_rel": info["delta_E_vs_E_rep_rel"], "raw_resid_rel": info["raw_resid_rel"],
            "header_energy_fraction": info["header_energy_fraction"],
            "setpoint_override_rate": info["setpoint_override_rate"],
            "cooling_fraction_zone_unoccupied": info["cooling_fraction_zone_unoccupied"],
            "obs25_clip_rate_mean": info["obs25_clip_rate_mean"], "obs25_clip_rate_max": info["obs25_clip_rate_max"],
            # 指示書 V3-S 第 2-4 節の 4 種（報告事項。門は +25 だけ）
            "obs29_clip_rate_mean": info["obs29_clip_rate_mean"], "obs29_clip_rate_max": info["obs29_clip_rate_max"],
            "co2_clip_rate_mean": info["co2_clip_rate_mean"], "co2_clip_rate_max": info["co2_clip_rate_max"],
            "load_clip_rate_mean": info["load_clip_rate_mean"], "load_clip_rate_max": info["load_clip_rate_max"],
            "progress_clip_rate": info["progress_clip_rate"],
            "observation_clip_rate": _sparse_rates(info["observation_clip_rate"]),
        }
        rec["improved_smooth"] = info["eci_smooth"] > self.best_smooth["eci_smooth"]
        rec["improved_official"] = info["eci_official"] > self.best_official["eci_official"]
        if rec["improved_smooth"] or rec["improved_official"]:
            weights = policy_weights(policy)
            entry = {"update": update, "stage": policy.stage, "E": info["E_rep"], "DR": info["DR_rep"],
                     "eci_smooth": info["eci_smooth"], "eci_official": info["eci_official"], "weights": weights}
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
        runlog.append_jsonl(self._path("evals.jsonl"), rec)
        if tag == "update":
            self.official_positive_streak = self.official_positive_streak + 1 if info["eci_official"] > 0 else 0
            self.last_eval = {"eci_smooth": info["eci_smooth"], "eci_official": info["eci_official"],
                              "cooling_fraction_zone_unoccupied": info["cooling_fraction_zone_unoccupied"]}
            self.inside_history.append(bool(info["eci_official"] > 0))   # Q-3
            degenerate = is_degenerate_actions(actions)                   # S-3
            self.degenerate_streak = self.degenerate_streak + 1 if degenerate else 0
            rec["degenerate_actions"] = degenerate
            rec["degenerate_streak"] = self.degenerate_streak
        return rec

    def record_train_episodes(self, update, terminal_infos):
        rows = []
        for step, i, info in sorted(terminal_infos, key=lambda x: (x[1], x[0])):
            row = {"update": update, "slot": self.cfg["train_slots"][i], "step_in_collection": step,
                   "E": info["E_rep"], "DR": info["DR_rep"], "eci_official": info["eci_official"],
                   "eci_smooth": info["eci_smooth"], "delta_E_vs_E_rep_rel": info["delta_E_vs_E_rep_rel"],
                   "raw_resid_rel": info["raw_resid_rel"], "header_energy_fraction": info["header_energy_fraction"],
                   "setpoint_override_rate": info["setpoint_override_rate"],
                   "cooling_fraction_zone_unoccupied": info["cooling_fraction_zone_unoccupied"]}
            runlog.append_jsonl(self._path("episodes.jsonl"), row)
            rows.append((row, info["observation_clip_rate"]))
        expected = constants.N_TRAIN_SLOTS * constants.EPISODES_PER_SLOT
        if len(rows) != expected:
            raise RuntimeError(f"1 回の採取で終端したエピソードが {len(rows)} 本（{expected} 本であるべき）")
        smooth = [r["eci_smooth"] for r, _ in rows]
        return {"train_episodes": len(rows), "train_eci_smooth_mean": float(np.mean(smooth)),
                "train_eci_smooth_max": float(np.max(smooth)),
                "train_E_mean": float(np.mean([r["E"] for r, _ in rows])),      # R-2'' の入力 Ē_u
                "train_DR_mean": float(np.mean([r["DR"] for r, _ in rows])),    # R-2'' の入力 DR̄_u
                "train_E_median": float(np.median([r["E"] for r, _ in rows])),
                "train_DR_median": float(np.median([r["DR"] for r, _ in rows])),
                "train_raw_resid_rel_median": float(np.median([r["raw_resid_rel"] for r, _ in rows])),
                "train_observation_clip_rate": _sparse_rates(np.mean([c for _, c in rows], axis=0))}

    # ---- 収穫（指示書 V4-S 第 2-5 節 H-1・H-2） ----

    def inside_rate(self, window=constants.INSIDE_RATE_WINDOW):
        """Q-3: 直近 window 回の決定論的評価のうち、零点条件の内側だった割合。"""
        recent = self.inside_history[-window:]
        return float(np.mean(recent)) if recent else None

    def harvest_best(self, update, terminal_infos, rollout):
        """H-1: 訓練 12 エピソードのうち公式 ECI 最大（同点なら eci_smooth 最大）の行動列を残す。

        行動列は採取済みの rollout から切り出す。組 i の行は env-major で [i·n, (i+1)·n) に並び、
        終端ステップ step のエピソードはその末尾 T 行である。
        """
        if rollout is None or not terminal_infos:
            return {}
        actions_all = np.asarray(rollout["actions"], dtype=np.int64)
        n_steps = self.T * constants.EPISODES_PER_SLOT
        best = None
        for step, i, info in terminal_infos:
            key = (info["eci_official"], info["eci_smooth"])
            if best is None or key > best["key"]:
                end = i * n_steps + int(step) + 1
                best = {"key": key, "update": update, "slot": self.cfg["train_slots"][i],
                        "step_in_collection": int(step), "stage": self.model.stage,
                        "E": info["E_rep"], "DR": info["DR_rep"],
                        "eci_official": info["eci_official"], "eci_smooth": info["eci_smooth"],
                        "E_hex": float.hex(info["E_rep"]), "DR_hex": float.hex(info["DR_rep"]),
                        "actions": actions_all[end - self.T:end]}
        out = {"harvest_update_best": {k: v for k, v in best.items() if k not in ("key", "actions")}}
        prev = self.best_harvest
        if prev is None or best["key"] > (prev["eci_official"], prev["eci_smooth"]):
            os.makedirs(self._path("harvest"), exist_ok=True)
            np.savez_compressed(self._path(os.path.join("harvest", "best_harvest.npz")),
                                actions=best["actions"].astype(np.int8))
            self.best_harvest = {k: v for k, v in best.items() if k not in ("key", "actions")}
            runlog.write_json(self._path(os.path.join("harvest", "best_harvest.json")), self.best_harvest)
        out["harvest_best"] = dict(self.best_harvest)
        out["harvest_best_official"] = self.best_harvest["eci_official"]       # M-5 の記録項目
        self.harvest_smooth_history.append(best["eci_smooth"])                  # S-1（H-4'・G-3' に使う）
        self.harvest_official_streak = (self.harvest_official_streak + 1) if best["eci_official"] > 0 else 0
        out["harvest_official_streak"] = self.harvest_official_streak           # G-1''
        out["harvest_update_official"] = best["eci_official"]
        return out

    def verify_harvest_replay(self):
        """H-2: 収穫の最良の行動列を評価用の組で 1 エピソード再生し、E・DR の 16 進一致を確かめる。"""
        path = self._path(os.path.join("harvest", "best_harvest.npz"))
        if self.best_harvest is None or self.eval_vec is None or not os.path.exists(path):
            return None
        actions = np.load(path)["actions"].astype(np.int64)
        saved_stage = self.best_harvest.get("stage", 1)
        converted = False
        if saved_stage == 1 and self.model.stage == 2:
            # 段階が進んでいる場合、同じ物理量になるよう設定温度の索引を写す（行動そのものは変えない）
            actions = np.stack([action_mod.convert_action_stage1_to_stage2(a) for a in actions])
            converted = True
        elif saved_stage != self.model.stage:
            return {"skipped": f"収穫時の段階 {saved_stage} と現在の段階 {self.model.stage} を対応づけられない"}
        self.eval_vec.env_method("set_reference", self.reference.to_dict() if self.reference else None)
        self.eval_vec.env_method("set_m5_enabled", self.m5_enabled)
        info = replay_action_sequence(self.eval_vec, actions)
        rec = {"time": runlog.now_iso(), "harvested_at_update": self.best_harvest["update"],
               "slot_harvested": self.best_harvest["slot"], "replay_slot": self.cfg["eval_slot"],
               "stage_converted_1_to_2": converted,
               "saved_E": self.best_harvest["E"], "saved_DR": self.best_harvest["DR"],
               "saved_E_hex": self.best_harvest["E_hex"], "saved_DR_hex": self.best_harvest["DR_hex"],
               "replay_E": info["E_rep"], "replay_DR": info["DR_rep"],
               "replay_E_hex": float.hex(info["E_rep"]), "replay_DR_hex": float.hex(info["DR_rep"]),
               "replay_eci_official": info["eci_official"], "replay_eci_smooth": info["eci_smooth"]}
        rec["E_hex_match"] = rec["replay_E_hex"] == rec["saved_E_hex"]
        rec["DR_hex_match"] = rec["replay_DR_hex"] == rec["saved_DR_hex"]
        runlog.write_json(self._path(os.path.join("harvest", "replay_check.json")), rec)
        self.log_event(event="harvest_replay", **rec)
        return rec

    # ---- 保存と再開 ----

    def save_checkpoint(self):
        state = {"model": self.model.training_state(), "reference": self.reference.to_dict(),
                 "best_smooth": self.best_smooth, "best_official": self.best_official,
                 "inside_history": self.inside_history, "best_harvest": self.best_harvest,
                 "harvest_smooth_history": self.harvest_smooth_history,
                 "harvest_official_streak": self.harvest_official_streak,
                 "degenerate_streak": self.degenerate_streak, "m8_enabled": self.m8_enabled,
                 "official_positive_streak": self.official_positive_streak, "last_eval": self.last_eval,
                 "m5_enabled": self.m5_enabled, "m5_state": self.m5_state.to_dict(),
                 "stage2_reference_smooth": self.stage2_reference_smooth,
                 "torch_rng": th.get_rng_state(), "numpy_rng": np.random.get_state()}
        path = self._path("checkpoint_latest.pt")
        th.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load_checkpoint(self):
        state = th.load(self._path("checkpoint_latest.pt"), weights_only=False)
        stage = state["model"]["stage"]
        self.m5_enabled = state["m5_enabled"]
        self.m8_enabled = state.get("m8_enabled", True)
        self.train_vec = self.make_train_vec(stage)
        self.eval_vec = self.make_eval_vec(stage)
        self.model = self.make_model(stage)
        self.model.load_training_state(state["model"])
        self.reference = ReferencePoint.from_dict(state["reference"])
        self.best_smooth = state["best_smooth"]
        self.best_official = state["best_official"]
        self.inside_history = list(state.get("inside_history", []))
        self.harvest_smooth_history = list(state.get("harvest_smooth_history", []))
        self.harvest_official_streak = state.get("harvest_official_streak", 0)
        self.degenerate_streak = state.get("degenerate_streak", 0)
        self.best_harvest = state.get("best_harvest")
        self.official_positive_streak = state["official_positive_streak"]
        self.last_eval = state["last_eval"]
        self.m5_state = M5ReleaseState.from_dict(state["m5_state"])
        self.stage2_reference_smooth = state["stage2_reference_smooth"]
        th.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        self.apply_m5()

    # ---- 段階の門（G-1〜G-5） ----

    def check_stage1_ceiling(self):
        """U-5: 通算更新番号が上限に達しても第 1 段階のままなら停止する。"""
        if self.model.stage == 1 and self.model.update_index >= self.stage1_ceiling_update:
            self.stop_reason = (f"U-5: 通算更新 {self.model.update_index} が第 1 段階の上限 "
                                f"{self.stage1_ceiling_update}（planned_updates × stage2_start_fraction）に達したが "
                                f"G-1 を通過していない")
            return True
        return False

    def maybe_switch_stage(self):
        """G-5: G-1 を満たすか手動指定に達したら、G-2（格子の細分化）と G-3（M-5 の解除）を実行する。"""
        if self.model.stage != 1:
            return
        manual = self.cfg["manual_stage2_update"]
        if manual is not None and self.model.update_index >= manual:
            trigger = "manual"
        elif gate_g1_satisfied(self.harvest_official_streak, self.model.update_index):
            trigger = "G-1''"
        else:
            return
        self.log_gate(gate="G-1''", trigger=trigger, update=self.model.update_index, passed=True,
                      harvest_official_streak=self.harvest_official_streak,
                      det_official_positive_streak=self.official_positive_streak,
                      inside_rate_recent=self.inside_rate())
        self.convert_to_stage2(trigger)

    def _rebuild_for_stage(self, stage):
        old = self.model
        self.train_vec.close()
        self.eval_vec.close()
        self.train_vec = self.make_train_vec(stage)
        self.eval_vec = self.make_eval_vec(stage)
        new = self.make_model(stage)
        transfer_to_stage2(old, new)
        self.model = new
        self.apply_m5()

    def convert_to_stage2(self, trigger):
        """G-2: M-5 を残したまま設定温度の格子を 0.5 ℃刻みへ細分化し、前後で公式 ECI が一致することを確かめる。"""
        pre = self.evaluate_with_retries(self.model.policy)
        self.record_eval(self.model.update_index, pre["info"], pre["actions"], self.model.policy, tag="g2_pre")
        self._rebuild_for_stage(2)
        post = self.evaluate_with_retries(self.model.policy)
        self.record_eval(self.model.update_index, post["info"], post["actions"], self.model.policy, tag="g2_post")
        rel = gate_g2_relative_difference(pre["info"]["eci_official"], post["info"]["eci_official"])
        passed = rel <= constants.G2_ECI_REL_TOLERANCE
        self.log_gate(gate="G-2", trigger=trigger, update=self.model.update_index, passed=passed,
                      pre_eci_official=pre["info"]["eci_official"], post_eci_official=post["info"]["eci_official"],
                      relative_difference=rel, tolerance=constants.G2_ECI_REL_TOLERANCE)
        if not passed:
            self.stop_reason = f"G-2: 格子の細分化の前後で公式 ECI の相対差 {rel} が {constants.G2_ECI_REL_TOLERANCE} を超えた"
            return
        self.stage2_reference_smooth = post["info"]["eci_smooth"]   # 記録用（判定は収穫で行う）
        self.attempt_m5_release()

    def attempt_m5_release(self):
        """G-3'（指示書 V5-S 第 2-1 節）: M-5 と M-8 を解除する。可否は 3 更新後に収穫の中央値で判定する。"""
        before = median_of(self.harvest_smooth_history[-constants.G3_RELEASE_JUDGE_UPDATES:])
        self.m5_enabled = False
        self.m8_enabled = False
        self.apply_m5()
        self.m5_state.pending_since = self.model.update_index
        self.m5_state.before_median = before
        self.m5_state.retry_at = None      # 判定が済むまで再試行の予約を外す（外さないと毎更新解除してしまう）
        self.log_gate(gate="G-3'", phase="released", update=self.model.update_index, before_median=before,
                      judge_at=self.model.update_index + constants.G3_RELEASE_JUDGE_UPDATES,
                      retries=self.m5_state.retries)

    def check_m5_release_result(self):
        """G-3' の判定（解除から 3 更新後）。下回れば M-5・M-8 を戻して 50 更新後に再試行する。"""
        st = self.m5_state
        n = constants.G3_RELEASE_JUDGE_UPDATES
        if st.released or st.pending_since is None or self.model.update_index < st.pending_since + n:
            return
        after = median_of(self.harvest_smooth_history[-n:])
        ok = True if st.before_median is None else gate_g3_release_ok(st.before_median, after)
        if ok:
            st.succeed()
        else:
            self.m5_enabled = True
            self.m8_enabled = True
            self.apply_m5()
            st.fail(self.model.update_index)
        self.log_gate(gate="G-3'", phase="judged", update=self.model.update_index, passed=ok,
                      before_median=st.before_median, after_median=after,
                      ratio_min=constants.G3_ECI_SMOOTH_RATIO_MIN, retries=st.retries, retry_at=st.retry_at)
        if not ok and st.exhausted:
            self.stop_reason = f"G-3': M-5・M-8 の解除に {st.retries} 回失敗した"

    def maybe_retry_m5_release(self):
        """G-3' の再試行（差し戻しから 50 更新後）。"""
        if self.model.stage != 2 or not self.m5_state.due(self.model.update_index):
            return
        self.attempt_m5_release()

    # ---- 1 回の更新 ----

    def one_update(self):
        model = self.model
        u = model.update_index + 1
        t0 = time.time()
        self.train_vec.env_method("set_reference", self.reference.to_dict())
        self.train_vec.env_method("set_m5_enabled", self.m5_enabled)
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
        record = {"update": u, "stage": model.stage, "m5_enabled": self.m5_enabled,
                  "m5_state": self.m5_state.to_dict(), "collect_and_eval_s": collect_s,
                  "eval_E": info["E_rep"], "eval_DR": info["DR_rep"], "eval_eci_official": info["eci_official"],
                  "eval_eci_smooth": info["eci_smooth"],
                  "eval_delta_E_vs_E_rep_rel": info["delta_E_vs_E_rep_rel"],
                  "eval_raw_resid_rel": info["raw_resid_rel"],
                  "eval_header_energy_fraction": info["header_energy_fraction"],
                  "eval_cooling_fraction_zone_unoccupied": info["cooling_fraction_zone_unoccupied"],
                  "eval_obs25_clip_rate_mean": info["obs25_clip_rate_mean"],
                  "eval_obs25_clip_rate_max": info["obs25_clip_rate_max"],
                  "eval_obs29_clip_rate_mean": info["obs29_clip_rate_mean"],
                  "eval_obs29_clip_rate_max": info["obs29_clip_rate_max"],
                  "eval_co2_clip_rate_max": info["co2_clip_rate_max"],
                  "eval_load_clip_rate_max": info["load_clip_rate_max"],
                  "eval_progress_clip_rate": info["progress_clip_rate"],
                  "eval_observation_clip_rate": eval_rec["observation_clip_rate"],
                  "official_positive_streak": self.official_positive_streak, **episode_summary}
        # H-1: 学習の前に、採取した行動列から最良の 1 本を収穫する（train() が rollout を捨てるため）
        record.update(self.harvest_best(u, terminal, model._rollout))
        self.check_m5_release_result()                                # G-3'（解除から 3 更新後の判定）
        record["inside_rate_recent20"] = self.inside_rate()           # Q-3
        record["eval_inside_zero_floor"] = bool(info["eci_official"] > 0)
        record["eval_degenerate_actions"] = eval_rec.get("degenerate_actions")      # S-3
        record["eval_degenerate_streak"] = eval_rec.get("degenerate_streak")
        record["m8_enabled"] = self.m8_enabled
        record["m9_enabled"] = bool(self.cfg["m9_enabled"])
        record["eval_m8_zone_steps"] = info.get("m8_zone_steps")      # M-8 の適用回数（M-5 の記録項目）
        record["eval_m9_tenant_steps"] = info.get("m9_tenant_steps")
        if self.degenerate_streak >= 10:                              # H-6（報告事項。停止はしない）
            self.log_event(event="H-6_degenerate_argmax", update=u, streak=self.degenerate_streak)

        t1 = time.time()
        record.update(model.train())
        record["train_s"] = time.time() - t1
        # R-2''・R-3''（指示書 V4-S 第 2-1 節）: 訓練 12 エピソードの (E, DR) の平均で基準点を指数移動平均で
        # 動かす。決定論的評価の点は使わず、零点条件の内外も区別しない。β/α の 2 倍制限は毎更新かける。
        record["reference_update"] = self.reference.update(
            episode_summary["train_E_mean"], episode_summary["train_DR_mean"], update_index=u, first=(u == 1))
        record["reference"] = {k: v for k, v in self.reference.to_dict().items() if k != "history"}
        record["reference"]["rho"] = self.reference.rho                                   # R-5''
        record["reference"]["rho_clipped"] = record["reference_update"]["rho_clipped"]    # R-5''
        runlog.append_jsonl(self._path("updates.jsonl"), record)
        self.save_checkpoint()

    # ---- 走行 ----

    def run(self, resume=False):
        os.makedirs(self.run_dir, exist_ok=True)
        meta = {"started_at": runlog.now_iso(), "argv": sys.argv, "config": self.cfg, "factory": self.factory,
                "max_updates_total": self.max_updates, "stage1_ceiling_update": self.stage1_ceiling_update,
                "resume": resume}
        if self.factory == "real":
            from rlkit.parallel import launch as launch_mod
            from rlkit.parallel import slots
            slot_indices = list(self.cfg["train_slots"]) + [self.cfg["eval_slot"]]
            # 指示書 V4-S 第 2-7 節: 組数の上限・空きメモリの下限・孤児プロセスの掃除を起動前に行う
            meta["preflight"] = launch_mod.preflight(self.run_dir, slot_indices)
            os.environ[launch_mod.LAUNCH_LOG_ENV] = meta["preflight"]["log_dir"]
            meta["replica_checks"] = [slots.prepare_replica(slots.make_slot(k)) for k in slot_indices]
        runlog.write_json(self._path(f"meta_{'resume' if resume else 'start'}_{int(time.time())}.json"), meta)

        status = "error"
        done = 0
        try:
            if resume:
                self.load_checkpoint()
                self.log_event(event="resumed", update_index=self.model.update_index, stage=self.model.stage,
                               planned_updates=self.model.planned_updates, m5_enabled=self.m5_enabled)
            else:
                self.train_vec = self.make_train_vec(1)
                self.eval_vec = self.make_eval_vec(1)
                self.model = self.make_model(1)
                self.apply_m5()
                # R-1: 新規初期化直後の方策の決定論的評価 1 回で初期の基準点を決める
                out = self.evaluate_with_retries(self.model.policy)
                info = out["info"]
                self.reference = ReferencePoint(info["E_rep"], info["DR_rep"])
                self.record_eval(0, info, out["actions"], self.model.policy, tag="initial")
                self.log_event(event="initial_reference", E_star=self.reference.E_star, DR_star=self.reference.DR_star,
                               alpha=self.reference.alpha, beta=self.reference.beta,
                               eci_smooth=info["eci_smooth"], eci_official=info["eci_official"])
                self.save_checkpoint()

            while self.model.update_index < self.planned:
                if self.max_updates is not None and self.model.update_index >= self.max_updates:
                    self.stop_reason = f"通算 {self.max_updates} 更新に達した（--updates は通算。指示書 V4-S W-3）"
                    break
                if os.path.exists(self._path(STOP_FILE_NAME)):
                    self.stop_reason = "STOP ファイルが置かれた"
                    break
                if self.check_stage1_ceiling():
                    break
                self.maybe_switch_stage()
                if self.stop_reason:
                    break
                self.maybe_retry_m5_release()
                if self.stop_reason:
                    break
                self.one_update()
                done += 1
                if self.stop_reason:
                    break
            if self.stop_reason is None:
                self.stop_reason = f"計画更新数 {self.planned} に到達した"
            try:
                self.harvest_replay = self.verify_harvest_replay()   # H-2
            except Exception:
                self.log_event(event="harvest_replay_failed", traceback=traceback.format_exc())
            status = "stopped"
        except Exception:
            self.stop_reason = f"例外: {traceback.format_exc()}"
            self.log_event(event="exception", traceback=traceback.format_exc())
        finally:
            if self.model is not None and self.reference is not None:
                self.save_checkpoint()
            for vec in (self.train_vec, self.eval_vec):
                if vec is not None:
                    vec.close()
            runlog.write_json(self._path("status.json"), {
                "status": status, "stop_reason": self.stop_reason, "finished_at": runlog.now_iso(),
                "updates_this_run": done,
                "update_index": self.model.update_index if self.model is not None else None,
                "stage": self.model.stage if self.model is not None else None,
                "m5_enabled": self.m5_enabled, "m5_state": self.m5_state.to_dict(),
                "best_smooth": {k: v for k, v in self.best_smooth.items() if k != "weights"},
                "best_official": {k: v for k, v in self.best_official.items() if k != "weights"},
                "best_harvest": self.best_harvest, "harvest_replay": self.harvest_replay,
                "inside_rate_recent20": self.inside_rate()})
        return self.stop_reason


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description="夏季 v2 の学習")
    p.add_argument("--config", required=True, help="走行の設定 JSON（雛形: rlkit/config_summer_v2.json）")
    p.add_argument("--resume", action="store_true", help="run_dir の checkpoint_latest.pt から再開する")
    p.add_argument("--updates", type=int, default=None,
                   help="通算の更新数の上限（指示書 V4-S: 試走は 20、継続は 300）")
    a = p.parse_args(argv)
    cfg = config_mod.load(a.config)
    missing = config_mod.unresolved_items(cfg)
    if missing:
        lines = "\n".join(f"  {key}（{item}）: {desc}" for key, item, desc in missing)
        raise SystemExit("未確定項目が設定されていないため起動しない。rlkit_v2_summer_implementation_report.md を参照:\n" + lines)
    runlog.check_thread_env()
    th.set_num_threads(1)
    trainer = Trainer(cfg, config_mod.repo_path(cfg["run_dir"]), max_updates=a.updates)
    reason = trainer.run(resume=a.resume)
    print(f"[train_summer] 停止: {reason}", flush=True)


if __name__ == "__main__":
    main()
