"""rlkit.v9.phase_a — V9-S 段階 A（PPO）の 1 更新（指示書 C9-1・C9-4・C9-5・C9-7）。

trial08 の学習の入口 rlkit.train_summer.Trainer を継承し、1 更新の流れ（基準点の受け渡し・決定論的評価の同時実行・採取・記録・R-7・保存）を
そのまま使う。変えたのは次の点だけ。
  環境        V8 の環境（SlotVecEnv の factory が real_v8 / mock_v8）と、V9 の PPO（rlkit.v9.learner9.SummerPPOV9）
  収穫        走行全体の最良は、訓練 12 本と決定論的評価 1 本の全軌跡のうち ECI 最大（C9-5）。R-7 の評価点にも同じ最良を使う（V9S-k）
  別名の保存  最良が更新された更新だけ、採取に使った重みを best/best_harvest_weights_u{更新}.pt に、行動列を harvest/best_harvest_u{更新}.npz に
              残す（C9-7。既存のファイルは上書きしない）
  自己模倣学習 使わない。判定は C-11 の停止規則だけ（崩れの見張り・更新 20/300 の判定・H-9 などは記録だけ。C9-10）
段階 A の 1 更新の記録（updates.jsonl の列）は trial08 と同じ項目に、段階の名前（phase）と、新しい選択肢の使用の集計を加える。
"""
import copy
import os
import threading
import time

import numpy as np
import torch as th

from rlkit import constants, eci as eci_mod, runlog, train_summer as ts
from rlkit.parallel.batched import SlotVecEnv
from rlkit.reward import R7Reference
from rlkit.v9.learner9 import SummerPPOV9

PHASE_NAME = "A"


def v8_use_total(infos):
    """終端の info の v8_use（新しい選択肢の使われ方）を、複数のエピソードで合計する。"""
    keys = ("duty_counts_occupied", "duty_counts_unoccupied", "setpoint_counts", "evap_counts")
    out = {}
    for k in keys:
        arrs = [np.asarray(i["v8_use"][k], dtype=np.int64) for i in infos if i.get("v8_use") and k in i["v8_use"]]
        out[k] = np.sum(arrs, axis=0).tolist() if arrs else None
    for k in ("prohibit_zone_steps", "bypass_tenant_steps", "mid_step_writes"):
        out[k] = int(sum(int(i["v8_use"][k]) for i in infos if i.get("v8_use") and k in i["v8_use"]))
    out["episodes"] = len(infos)
    return out


def new_option_fractions(use):
    """段階 A の報告用: 新しい選択肢の使用の割合（合計の使用から）。
    prohibit: リモコン禁止のゾーン・時点の割合、duty_partial_occupied: 在室の運転の判断のうち割合 1/3・2/3 の割合、
    setpoint_new: 運転ゾーンの設定温度のうち 22.5・23.5 ℃の割合、bypass: テナント・時点のうちバイパスを使う割合。"""
    n_steps = constants.ZONE_STEPS_PER_WEEK * use["episodes"]
    occ = use["duty_counts_occupied"]
    occ_on = (sum(occ[1:]) if occ else 0)
    sp = use["setpoint_counts"]
    return {
        "prohibit_fraction": use["prohibit_zone_steps"] / max(n_steps * constants.N_ZONES, 1),
        "duty_partial_occupied_fraction": ((occ[1] + occ[2]) / occ_on) if occ_on else None,
        "duty_counts_occupied": occ, "duty_counts_unoccupied": use["duty_counts_unoccupied"],
        "setpoint_new_fraction": ((sp[1] + sp[3]) / max(sum(sp), 1)) if sp else None,
        "bypass_fraction": use["bypass_tenant_steps"] / max(n_steps * 2, 1),
        "evap_14_fraction": (use["evap_counts"][0] / max(sum(use["evap_counts"]), 1)) if use["evap_counts"] else None}


class PhaseA(ts.Trainer):
    def __init__(self, cfg, run_dir, factory="real_v8"):
        super().__init__(cfg, run_dir, factory=factory)
        self.best_run = None                  # 走行全体の最良（訓練 12 本と決定論的評価 1 本のうち ECI 最大）。段階 B へ引き継ぐ
        self.best_eci_history = []            # 更新 u の終わりの走行全体の最良の ECI（B(u)。C9-5）
        self.eval_eci_history = []
        self.nonfinite = False

    # ---- 環境と学習器（V8・V9 に差し替え） ----

    def make_train_vec(self, stage=1):
        return SlotVecEnv(self.cfg["train_slots"], self.env_kwargs(stage),
                          steps_per_collection=self.T * self.eps_per_slot, factory=self.factory)

    def make_eval_vec(self, stage=1):
        return SlotVecEnv([self.cfg["eval_slot"]], self.env_kwargs(stage), steps_per_collection=self.T, factory=self.factory)

    def make_model(self, stage=1):
        return SummerPPOV9(self.train_vec, self.planned, seed=self.cfg["seed"], n_steps_per_env=self.T * self.eps_per_slot)

    def sil_mode(self, u):
        return "off"

    def add_to_sil_buffer(self, entry):
        pass

    # ---- 開始 ----

    def start(self):
        """新規の初期化（S-1）。環境・学習器を作り、R-7 の初期値（trial07 と同じ ρ₀=60、分母 6.5・0.20）を置く。"""
        v9 = self.cfg["v9"]
        self.train_vec = self.make_train_vec()
        self.eval_vec = self.make_eval_vec()
        self.model = self.make_model()
        self.W = self.train_vec.get_attr("W")[0]
        r0 = v9["reference0"]
        self.reference = R7Reference(rho=r0["rho"], E_ref=r0["E_ref"], DR_ref=r0["DR_ref"])
        self.log_event(event="phase_a_start", reference0=self.reference.summary(), seed=self.cfg["seed"],
                       policy_params=int(sum(p.numel() for p in self.model.policy.policy_network.parameters())),
                       value_params=int(sum(p.numel() for p in self.model.policy.value_network.parameters())))

    # ---- 収穫（C9-5・C9-7。走行全体の最良は 13 本の全軌跡のうち ECI 最大） ----

    def harvest_run_best(self, update, terminal_infos, rollout, eval_info, eval_actions, eval_policy):
        n_steps = self.T * self.eps_per_slot
        cands = []
        for step, i, info in terminal_infos:
            cands.append({"role": "train", "ECI": float(info["ECI"]), "info": info, "step": int(step), "i": int(i)})
        cands.append({"role": "eval", "ECI": float(eval_info["ECI"]), "info": eval_info})
        best = max(cands, key=lambda c: c["ECI"])          # 同点は先に並べたもの（訓練）を残す

        def summary(c):
            info = c["info"]
            slot = (self.cfg["train_slots"][c["i"]] if c["role"] == "train" else self.cfg["eval_slot"])
            return {"update": update, "slot": int(slot), "role": c["role"], "E": float(info["E_rep"]), "DR": float(info["DR_rep"]),
                    "ECI": float(info["ECI"]), "P_E": float(info["P_E"]), "P_DR": float(info["P_DR"]),
                    "eci_official": float(info["eci_official"]), "eci_smooth": float(info["eci_smooth"]),
                    "E_hex": float.hex(float(info["E_rep"])), "DR_hex": float.hex(float(info["DR_rep"])),
                    "stage": 1, "J": None}
        upd = summary(best)
        out = {"harvest_update_best": upd}
        improved = self.best_run is None or best["ECI"] > self.best_run["ECI"]
        if improved:
            if best["role"] == "train":
                end = best["i"] * n_steps + best["step"] + 1
                sl = slice(end - self.T, end)
                actions = np.asarray(rollout["actions"][sl], dtype=np.int64)
                masks = np.asarray(rollout["action_masks"][sl], dtype=bool)
            else:
                actions = np.asarray(eval_actions, dtype=np.int64)
                masks = None
            os.makedirs(self._path("harvest"), exist_ok=True)
            if masks is not None:
                np.savez_compressed(self._path(os.path.join("harvest", "best_harvest.npz")), actions=actions.astype(np.int8), masks=masks)
            else:
                np.savez_compressed(self._path(os.path.join("harvest", "best_harvest.npz")), actions=actions.astype(np.int8))
            alias = self._path(os.path.join("harvest", f"best_harvest_u{update}.npz"))
            if os.path.exists(alias):
                self.log_event(event="harvest_alias_exists_not_overwritten", path=alias, update=update)
            else:
                np.savez_compressed(alias, actions=actions.astype(np.int8))
            os.makedirs(self._model_path("best"), exist_ok=True)
            wpath = self._model_path(os.path.join("best", f"best_harvest_weights_u{update}.pt"))
            if os.path.exists(wpath):
                self.log_event(event="harvest_weights_exist_not_overwritten", path=wpath, update=update)
            else:
                th.save({"update": update, "weights": {k: v.detach().clone() for k, v in eval_policy.policy_network.state_dict().items()},
                         "harvest": upd}, wpath)
            self.best_run = dict(upd)
            runlog.write_json(self._path(os.path.join("harvest", "best_harvest.json")), self.best_run)
            upd = dict(upd, improved_best=True)
            out["harvest_update_best"] = upd
        self.best_harvest = self.best_run              # R-7 の評価点・段階 B への引き継ぎも走行全体の最良（V9S-k）
        out["harvest_best"] = dict(self.best_run)
        self.best_eci_history.append(float(self.best_run["ECI"]))
        out["harvest_update_ECI"] = best["ECI"]
        out["harvest_best_official"] = self.best_run["eci_official"]
        out["harvest_best_P_E"] = self.best_run["P_E"]
        out["harvest_best_P_DR"] = self.best_run["P_DR"]
        out["harvest_update_official"] = float(best["info"]["eci_official"])
        out["harvest_official_streak"] = None
        out["sil_buffer"] = []
        out["B_run_best_ECI"] = float(self.best_run["ECI"])
        out["harvest_role"] = best["role"]
        return out

    # ---- 1 更新 ----

    def one_update(self):
        model = self.model
        u = model.update_index + 1
        t0 = time.time()
        self._mem = ts.MemoryWatch().start()
        self.apply_reference()
        ref_used = self.reference.summary()
        ref_used.update({"rho_raw": (R7Reference.raw_ratio(self.reference.E_ref, self.reference.DR_ref)
                                     if self.reference.ref_update is not None else None),
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
        record = {"update": u, "stage": model.stage, "phase": PHASE_NAME, "collect_and_eval_s": collect_s,
                  "eval_E": info["E_rep"], "eval_DR": info["DR_rep"], "eval_eci_official": info["eci_official"],
                  "eval_E_hex": float.hex(info["E_rep"]), "eval_DR_hex": float.hex(info["DR_rep"]),
                  "eval_ECI": info["ECI"], "eval_P_E": info["P_E"], "eval_P_DR": info["P_DR"],
                  "eval_co2_delta_obs_clip_rate": info["co2_delta_obs_clip_rate"],
                  "eval_co2_pred_obs_clip_rate": info["co2_pred_obs_clip_rate"],
                  "eval_eci_smooth": info["eci_smooth"], "eval_J": info["J"],
                  "eval_delta_E_vs_E_rep_rel": info["delta_E_vs_E_rep_rel"], "eval_raw_resid_rel": info["raw_resid_rel"],
                  "eval_c11": ts.c11_metrics(info), "eval_raw_resid_sum_gj": info["raw_resid_sum_gj"],
                  "eval_raw_resid_energy_rel": info["raw_resid_rel"],
                  "eval_raw_resid_dissatisfaction_rel": info["raw_resid_dissatisfaction_rel"],
                  "eval_raw_resid_dissatisfaction_endpoint_rel": info["raw_resid_dissatisfaction_endpoint_rel"],
                  "eval_r8p_sum_rel": info["r8p_sum_rel"], "eval_r8p_seconds": info["r8p_seconds"],
                  "eval_header_energy_fraction": info["header_energy_fraction"],
                  "eval_cooling_fraction_zone_unoccupied": info["cooling_fraction_zone_unoccupied"],
                  "eval_obs25_clip_rate_max": info["obs25_clip_rate_max"], "eval_obs29_clip_rate_max": info["obs29_clip_rate_max"],
                  "eval_co2_clip_rate_max": info["co2_clip_rate_max"], "eval_load_clip_rate_max": info["load_clip_rate_max"],
                  "eval_vent_obs_clip_rate": info["vent_obs_clip_rate"], "eval_remote_obs_clip_rate": eval_rec["remote_obs_clip_rate"],
                  "eval_observation_clip_rate": eval_rec["observation_clip_rate"],
                  "eval_vent_level_fraction": info["vent_level_fraction"], "eval_hex_energy_gj": info["hex_energy_gj"],
                  "eval_m5_zone_steps": info["m5_zone_steps"], "eval_m9_tenant_steps": info["m9_tenant_steps"],
                  "eval_remote_permit_fraction": eval_rec["remote_permit_fraction"], "eval_onoff_mismatch": info["onoff_mismatch"],
                  "eval_obs25_clip_rate_mean": info["obs25_clip_rate_mean"], "eval_obs29_clip_rate_mean": info["obs29_clip_rate_mean"],
                  "eval_progress_clip_rate": info["progress_clip_rate"],
                  "eval_inside_zero_floor": bool(info["eci_official"] > 0),
                  "official_positive_streak": self.official_positive_streak,
                  "m5_enabled": True, "m5_state": None, "m8_enabled": False, "eval_m8_zone_steps": None,
                  "eval_degenerate_actions": eval_rec.get("degenerate_actions"),
                  "eval_degenerate_streak": eval_rec.get("degenerate_streak"),
                  "m9_enabled": bool(self.cfg["m9_enabled"]), **episode_summary}
        # 新しい選択肢の使用（報告用）: 訓練 12 本の合計と、決定論的評価
        train_use = v8_use_total([i for _, _, i in terminal])
        record["v8_use_train"] = train_use
        record["new_option_use_train"] = new_option_fractions(train_use)
        record["v8_use_eval"] = info.get("v8_use")
        record["new_option_use_eval"] = new_option_fractions(v8_use_total([info])) if info.get("v8_use") else None
        self.eval_eci_history.append(float(info["ECI"]))
        record.update(self.harvest_run_best(u, terminal, model._rollout, info, eval_out["actions"], eval_policy))

        t1 = time.time()
        record.update(model.train())
        record["train_s"] = time.time() - t1
        record["reference_used"] = ref_used
        best = dict(self.best_harvest, ECI=self.best_harvest["ECI"])
        record["reference_update"] = self.reference.update_after_harvest(best, u)
        record["reference"] = self.reference.summary()
        if record["nonfinite"]:
            self.nonfinite = True
            self.stop_reason = f"数値: 更新 {u} で損失またはパラメータに NaN・無限大が出た（直前の保存点を残す）"
            self.finish_record(record, t0)
            runlog.append_jsonl(self._path("updates.jsonl"), record)
            return
        self.finish_record(record, t0)
        runlog.append_jsonl(self._path("updates.jsonl"), record)
        self.save_checkpoint()
        if u % constants.CHECKPOINT_EVERY == 0:
            import shutil
            runlog.retrying(lambda: shutil.copy2(self._model_path("checkpoint_latest.pt"), self._model_path(f"checkpoint_u{u}.pt")))
        self.judge_after_update(u, record)

    # ---- 判定（C-11 の停止規則だけ。ほかは記録のみ。C9-10） ----

    def judge_after_update(self, u, record):
        c11 = ts.c11_stop_reasons(ts.c11_entries_from_record(record))
        if c11:
            self.stop_reason = f"更新 {u}: " + "；".join(c11)
        h = self.eval_eci_history
        if len(h) >= 20 and u % 20 == 0:                                   # 崩れの見張り（記録だけ。止めない）
            recent = float(np.median(h[-20:]))
            prev_best = float(np.max(h))
            runlog.append_jsonl(self._path("collapse_watch.jsonl"), {"update": u, "median_recent20_eval_ECI": recent,
                                                                      "max_eval_ECI": prev_best, "ratio": recent / prev_best if prev_best else None,
                                                                      "recorded_only": True})

    def save_checkpoint(self):
        state = {"model": self.model.training_state(), "reference": self.reference.to_dict(),
                 "best_run": self.best_run, "best_eci_history": self.best_eci_history, "eval_eci_history": self.eval_eci_history,
                 "judgments": self.judgments, "torch_rng": th.get_rng_state(), "numpy_rng": np.random.get_state()}
        os.makedirs(self.model_dir, exist_ok=True)
        path = self._model_path("checkpoint_latest.pt")
        th.save(state, path + ".tmp")
        retries = runlog.replace_with_retries(path + ".tmp", path)
        if retries:
            self.log_event(event="checkpoint_replace_retried", retries=retries, update_index=self.model.update_index)
