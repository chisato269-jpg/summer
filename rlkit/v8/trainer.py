"""rlkit.v8.trainer — V8-S の学習の入口（指示書 第 4・7・11・12 章）。

構成: 13 組。組 0〜11 が兄弟の軌跡、組 12 が基準の軌跡（決定論的評価を兼ねる）。1 更新 = 13 本を同時に 1 週間走らせる。
基盤は V7.1-S のもの（C-9 50 更新ごとの保存点、C-11 停止規則、C-12 終了時の再生、C-13 記録、C-15 起動の保護、model_dir）。
"""
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
import traceback

import numpy as np
import torch as th

from rlkit import config as config_mod
from rlkit import constants, eci as eci_mod, runlog, train_summer as ts
from rlkit.parallel.batched import SlotVecEnv, SlotVecEnvError
from rlkit.reward import R7Reference
from rlkit.v8 import action as act8
from rlkit.v8 import blocks as blk
from rlkit.v8 import improver as imp
from rlkit.v8 import observation as obs8
from rlkit.v8 import policy as pol8

STOP_FILE_NAME = "STOP"
INSTRUCTION_VERSION = "V8-S"
T = constants.ZONE_STEPS_PER_WEEK
N_SIB = blk.N_SIBLINGS
N_ENV = N_SIB + 1
BASE = N_SIB                                     # 基準の軌跡の組（評価用の組）
COLLAPSE_WINDOW = 10
COLLAPSE_DROP = 0.01
G20_UPDATE, G50_UPDATE = 20, 50
G20_SECONDS_MAX = 400.0
G20_MEDIAN_DROP_MAX = 0.003
G50_RISE_MIN = 0.002
MAX_COLLECTION_RETRIES = 3
C83_LIMIT_SECONDS = 600.0


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def collect(vec, policy, perturbers, n_steps=T):
    """13 本を同時に走らせる。基準（組 BASE）はマスク付きの確率最大、兄弟は区間の中だけ perturbers に従って揺らす（MUST-4）。
    Returns: (軌跡 dict（obs (13,T,704)、masks (13,T,372)、actions (13,T,80)）、終端の info のリスト）"""
    obs = vec.reset()
    traj = {"obs": np.zeros((N_ENV, n_steps, obs8.TOTAL_DIM), dtype=np.float32),
            "masks": np.zeros((N_ENV, n_steps, act8.ACTION_LOGITS), dtype=bool),
            "actions": np.zeros((N_ENV, n_steps, act8.ACTION_DIMS), dtype=np.int64)}
    finals = [None] * N_ENV
    for t in range(n_steps):
        masks = np.stack(vec.env_method("action_masks"))
        traj["obs"][:, t] = obs
        traj["masks"][:, t] = masks
        with th.no_grad():
            logits = policy(th.as_tensor(np.asarray(obs), dtype=th.float32)).numpy()
        base_act = pol8.masked_argmax(logits, masks)
        acts = base_act.copy()
        for i, p in perturbers.items():
            acts[i] = p.apply(t, base_act[i], masks[i])
        traj["actions"][:, t] = acts
        obs, _r, dones, infos = vec.step(acts)
        for i, d in enumerate(dones):
            if d:
                finals[i] = infos[i]
    if any(f is None for f in finals):
        raise RuntimeError("1 週間の採取で終端しなかった軌跡がある")
    return traj, finals


def collapse_check(base_eci_history, u):
    """崩れの見張り（第 12 章）: 基準の軌跡の ECI の直近 10 更新の中央値が、それまでの基準の ECI の最大より 0.01 以上低い。
    base_eci_history は更新 0（始点）から更新 u までの基準の ECI。"""
    if u < COLLAPSE_WINDOW:
        return {"update": u, "triggered": False, "median_recent": None, "running_max": None}
    recent = float(np.median(base_eci_history[u - COLLAPSE_WINDOW + 1:u + 1]))
    best = float(np.max(base_eci_history[:u + 1]))
    return {"update": u, "triggered": bool(recent < best - COLLAPSE_DROP), "median_recent": recent, "running_max": best,
            "threshold": best - COLLAPSE_DROP}


class V8Trainer:
    VERSION = INSTRUCTION_VERSION

    def __init__(self, cfg, run_dir, factory="real_v8", max_updates=None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.model_dir = config_mod.repo_path(cfg["model_dir"]) if cfg.get("model_dir") else run_dir
        self.factory = factory
        self.max_updates = max_updates
        self.planned = int(cfg["planned_updates"])
        self.seed = int(cfg["seed"])
        self.slots = list(cfg["train_slots"]) + [cfg["eval_slot"]]
        self.order = blk.block_order(self.seed)
        self.block_steps = blk.block_steps_table()
        self.policy = None
        self.value = None
        self.improver = None
        self.vec = None
        self.reference = None
        self.best_harvest = None
        self.base_eci_history = []          # 更新 0（始点）から
        self.prev_base = None               # L-8: 前の更新の基準の (E, DR)
        self.prev_sum_a = None
        self.update_index = 0
        self.stop_reason = None
        self.nonfinite = False
        self.judgments = {}
        self.harvest_replay = None
        self._rpm_prev = {}
        self._mem = None
        self._log_lock = threading.Lock()
        self.init_meta = None

    # ---- 記録 ----

    def _path(self, name):
        return os.path.join(self.run_dir, name)

    def _model_path(self, name):
        return os.path.join(self.model_dir, name)

    def log_event(self, **obj):
        with self._log_lock:
            runlog.append_jsonl(self._path("events.jsonl"), {"time": runlog.now_iso(), **obj})
        print(f"[v8.train] {obj}", flush=True)

    def env_kwargs(self):
        return {"stage": 1, "include_indoor_unit_power": self.cfg["include_indoor_unit_power"],
                "zone_step_energy_norm_gj": self.cfg["zone_step_energy_norm_gj"], "m9_enabled": self.cfg["m9_enabled"],
                "comm_rpm": bool(self.cfg.get("comm_rpm", False))}

    def make_vec(self, slots=None):
        return SlotVecEnv(slots or self.slots, self.env_kwargs(), steps_per_collection=T, factory=self.factory)

    # ---- 初期化・保存・読み込み ----

    def init_from_weights(self):
        path = config_mod.repo_path(self.cfg["init_weights"])
        entry = th.load(path, weights_only=False)
        self.policy, self.value = pol8.build_from_v7(entry)
        for p in self.value.parameters():
            p.requires_grad_(False)                                       # L-11: 価値ネットワークは更新しない
        self.improver = imp.PairedImprover(self.policy, lr=float(self.cfg.get("lr", imp.LEARNING_RATE)),
                                           max_epochs=int(self.cfg.get("max_epochs", imp.MAX_EPOCHS)),
                                           kl_limit=float(self.cfg.get("kl_limit", imp.KL_LIMIT)))
        E0, DR0 = float(entry["E"]), float(entry["DR"])
        eci0 = float(entry["eci_smooth"])
        sh, sd = self.cfg.get("start_E_hex"), self.cfg.get("start_DR_hex")
        if sh and sd:                                  # V8.1-S 第 3 章: 始点は V8 の更新 0 の基準の軌跡（C8-1）
            E0, DR0 = float.fromhex(sh), float.fromhex(sd)
            eci0 = float(eci_mod.eci_smooth(E0, DR0))
        rho_raw = R7Reference.raw_ratio(E0, DR0)
        rho0, _ = R7Reference.clip_ratio(rho_raw, rho_raw)
        self.reference = R7Reference(rho=rho0, E_ref=E0, DR_ref=DR0, ref_update=0, ref_eci=eci0)
        self.best_harvest = {"update": 0, "E": E0, "DR": DR0, "ECI": eci0, "E_hex": float.hex(E0), "DR_hex": float.hex(DR0),
                             "source": "始点（V8 の更新 0 の基準の軌跡）" if (sh and sd) else "初期の重みの決定論的評価"}
        self.base_eci_history = [eci0]
        self.prev_base = (E0, DR0)
        self.init_meta = {"init_weights": path, "init_sha256": file_sha256(path), "init_update": int(entry["update"]),
                          "E_hex": entry["E_hex"], "DR_hex": entry["DR_hex"], "eci_of_init_record": float(entry["eci_smooth"]),
                          "start_point": {"E_hex": float.hex(E0), "DR_hex": float.hex(DR0), "ECI": eci0,
                                          "from_config": bool(sh and sd)},
                          "eci": eci0, "rho0": rho0, "rho_raw0": rho_raw}

    def save_checkpoint(self):
        state = {"policy": self.policy.state_dict(), "value": self.value.state_dict(), "optimizer": self.improver.state(),
                 "reference": self.reference.to_dict(), "best_harvest": self.best_harvest,
                 "base_eci_history": self.base_eci_history, "prev_base": self.prev_base, "prev_sum_a": self.prev_sum_a,
                 "update_index": self.update_index, "judgments": self.judgments, "init_meta": self.init_meta,
                 "rpm_prev": self._rpm_prev}
        os.makedirs(self.model_dir, exist_ok=True)
        path = self._model_path("checkpoint_latest.pt")
        th.save(state, path + ".tmp")
        runlog.replace_with_retries(path + ".tmp", path)

    def load_checkpoint(self):
        state = th.load(self._model_path("checkpoint_latest.pt"), weights_only=False)
        self.policy = pol8.PolicyNetworkV8()
        self.policy.load_state_dict(state["policy"])
        self.value = pol8.ValueNetworkV8()
        self.value.load_state_dict(state["value"], strict=False)
        for p in self.value.parameters():
            p.requires_grad_(False)
        self.improver = imp.PairedImprover(self.policy, lr=float(self.cfg.get("lr", imp.LEARNING_RATE)),
                                           max_epochs=int(self.cfg.get("max_epochs", imp.MAX_EPOCHS)),
                                           kl_limit=float(self.cfg.get("kl_limit", imp.KL_LIMIT)))
        self.improver.load_state(state["optimizer"])
        self.reference = R7Reference.from_dict(state["reference"])
        self.best_harvest = state["best_harvest"]
        self.base_eci_history = list(state["base_eci_history"])
        self.prev_base = tuple(state["prev_base"]) if state["prev_base"] is not None else None
        self.prev_sum_a = state["prev_sum_a"]
        self.update_index = int(state["update_index"])
        self.judgments = dict(state["judgments"])
        self.init_meta = state["init_meta"]

    # ---- 1 回の更新 ----

    def _rpm_delta(self, vec):
        out = {"requests": 0, "points": 0, "fallback_chunks": 0, "fallback_points": 0}
        try:
            cur = vec.rpm_stats()
        except Exception:
            return out
        prev = self._rpm_prev.get("train", {})
        for k in out:
            c, p = cur.get(k, 0), prev.get(k, 0)
            out[k] = c - p if c >= p else c
        self._rpm_prev["train"] = dict(cur)
        return out

    def collect_with_retries(self, perturbers_factory):
        for attempt in range(1, MAX_COLLECTION_RETRIES + 2):
            try:
                return collect(self.vec, self.policy, perturbers_factory()), attempt - 1
            except SlotVecEnvError as exc:
                self.log_event(event="collection_failed_discarded", attempt=attempt, error=str(exc)[-2000:])
                self.vec.close()
                if attempt > MAX_COLLECTION_RETRIES:
                    raise
                self.vec = self.make_vec()

    def update_policy(self, base, sibs, adv):
        """方策の更新（V8-S の L-6。V8.2-S の学習の入口は、これを L-6R で置き換える）。"""
        return self.improver.update(base, sibs, adv)

    def one_update(self):
        u = self.update_index + 1
        t0 = time.time()
        self._mem = ts.MemoryWatch().start()
        assign = blk.assignment(u, self.order)

        def make_perturbers():
            ps = {}
            for j in range(N_SIB):
                ps[j] = blk.Perturber(assign[j], blk.sibling_kind(u, j), blk.sibling_seed(self.seed, u, j),
                                      self.block_steps[assign[j]])
            return ps
        holder = {}

        def factory():
            holder["p"] = make_perturbers()
            return holder["p"]
        (traj, finals), retries = self.collect_with_retries(factory)
        perturbers = holder["p"]
        collect_s = time.time() - t0

        rho = self.reference.rho
        den = self.reference.E_ref + rho * self.reference.DR_ref
        E = np.array([f["E_rep"] for f in finals])
        DR = np.array([f["DR_rep"] for f in finals])
        ECI = np.array([f["ECI"] for f in finals])
        C = E + rho * DR
        adv = imp.advantages(C[BASE], C[:N_SIB], den)
        sibs = [{"obs": traj["obs"][j], "masks": traj["masks"][j], "actions": traj["actions"][j],
                 "steps": self.block_steps[assign[j]], "dims": perturbers[j].dims} for j in range(N_SIB)]
        base = {"obs": traj["obs"][BASE], "masks": traj["masks"][BASE]}

        # 収穫（L-9）: 13 本のうち ECI が最大の軌跡。最良が更新されたら、採取に使った重みと行動列を残す
        pre_update_state = {k: v.clone() for k, v in self.policy.state_dict().items()}
        i_best = int(np.argmax(ECI))
        harvest_update = {"update": u, "slot": int(self.slots[i_best]), "role": "base" if i_best == BASE else "sibling",
                          "E": float(E[i_best]), "DR": float(DR[i_best]), "ECI": float(ECI[i_best]),
                          "P_E": float(finals[i_best]["P_E"]), "P_DR": float(finals[i_best]["P_DR"]),
                          "E_hex": float.hex(float(E[i_best])), "DR_hex": float.hex(float(DR[i_best]))}
        if ECI[i_best] > self.best_harvest["ECI"]:
            self.best_harvest = dict(harvest_update)
            os.makedirs(self._path("harvest"), exist_ok=True)
            np.savez_compressed(self._path(os.path.join("harvest", "best_harvest.npz")),
                                actions=traj["actions"][i_best].astype(np.int8), masks=traj["masks"][i_best])
            alias = self._path(os.path.join("harvest", f"best_harvest_u{u}.npz"))
            if os.path.exists(alias):                                                  # C8-4: 既存のファイルは上書きしない
                self.log_event(event="harvest_alias_exists_not_overwritten", path=alias, update=u)
            else:
                np.savez_compressed(alias, actions=traj["actions"][i_best].astype(np.int8))
            runlog.write_json(self._path(os.path.join("harvest", "best_harvest.json")), self.best_harvest)
            os.makedirs(self._model_path("best"), exist_ok=True)
            th.save({"update": u, "weights": pre_update_state, "harvest": self.best_harvest}, self._model_path(os.path.join("best", "best_harvest.pt")))
            harvest_update["improved_best"] = True

        # 更新（L-6）
        t1 = time.time()
        self._cur = {"E": E, "DR": DR}
        upd = self.update_policy(base, sibs, adv)
        train_s = time.time() - t1
        finite = all(bool(th.isfinite(p).all()) for p in self.policy.parameters())
        if not finite or upd["nonfinite"]:
            self.nonfinite = True
            self.stop_reason = f"数値の発散: 更新 {u} で損失またはパラメータに NaN・無限大が出た（直前の保存点を残す）"

        # 記録: 軌跡・兄弟・C-11・C-13
        c11 = [ts.c11_metrics(f) for f in finals]
        for j in range(N_SIB):
            runlog.append_jsonl(self._path("episodes.jsonl"), {
                "update": u, "slot": int(self.slots[j]), "role": "sibling", "E": float(E[j]), "DR": float(DR[j]),
                "ECI": float(ECI[j]), "C": float(C[j]), "A": float(adv[j]), "improved": bool(adv[j] > 0), "c11": c11[j],
                "block": int(assign[j]), "kind": perturbers[j].kind})
        runlog.append_jsonl(self._path("episodes.jsonl"), {
            "update": u, "slot": int(self.slots[BASE]), "role": "base", "E": float(E[BASE]), "DR": float(DR[BASE]),
            "ECI": float(ECI[BASE]), "C": float(C[BASE]), "c11": c11[BASE], "v8_use": finals[BASE].get("v8_use")})
        runlog.append_jsonl(self._path("siblings.jsonl"), {                       # MUST-5
            "update": u, "siblings": [dict(perturbers[j].record(), slot=int(self.slots[j]), A=float(adv[j]),
                                           improved=bool(adv[j] > 0)) for j in range(N_SIB)]})
        base_eci = float(ECI[BASE])
        self.base_eci_history.append(base_eci)
        improved = adv > 0
        sum_a = float(adv[improved].sum()) if improved.any() else 0.0
        dC = None
        if self.prev_base is not None:
            dC = float(((E[BASE] + rho * DR[BASE]) - (self.prev_base[0] + rho * self.prev_base[1])) / den)    # L-8
        mem = self._mem.stop()
        waits = np.asarray(self.vec.pop_wait_seconds(), dtype=np.float64)
        rpm = self._rpm_delta(self.vec)
        record = {
            "update": u, "update_seconds": time.time() - t0, "collect_s": collect_s, "train_s": train_s,
            "collection_retries": retries, "rho": rho, "E_ref": self.reference.E_ref, "DR_ref": self.reference.DR_ref,
            "denominator": den,
            "eval_E": float(E[BASE]), "eval_DR": float(DR[BASE]), "eval_ECI": base_eci, "eval_C": float(C[BASE]),
            "eval_c11": c11[BASE], "train_c11": [dict(c11[j], slot=int(self.slots[j])) for j in range(N_SIB)],
            "sibling_ECI": ECI[:N_SIB].tolist(), "sibling_A": adv.tolist(),
            "n_improved": int(improved.sum()), "frac_improved": float(improved.mean()),
            "sum_A_improved": sum_a, "prev_sum_A_improved": self.prev_sum_a, "delta_C_base_norm": dC,
            "adoption_rate": upd["adoption_rate"], "updated": upd["updated"], "epochs_applied": upd["epochs_applied"],
            "kl_cutoff": upd["kl_cutoff"], "kl_per_epoch": upd["kl_per_epoch"], "loss_per_epoch": upd["loss_per_epoch"],
            "kl_last": (upd["kl_per_epoch"][-1] if upd["kl_per_epoch"] else None),            # C8-4
            "n_decisions": upd["n_decisions"], "weights": upd["weights"],
            "by_kind": {k: {"n": int(sum(1 for j in range(N_SIB) if perturbers[j].kind == k)),
                            "n_improved": int(sum(1 for j in range(N_SIB) if perturbers[j].kind == k and improved[j]))}
                        for k in ("sparse", "coherent")},
            "by_window": {str(w): {"n": int(sum(1 for j in range(N_SIB) if perturbers[j].window == w)),
                                   "n_improved": int(sum(1 for j in range(N_SIB) if perturbers[j].window == w and improved[j])),
                                   "sum_A": float(sum(adv[j] for j in range(N_SIB) if perturbers[j].window == w))}
                          for w in range(blk.N_WINDOWS)},
            "harvest_update_best": harvest_update, "harvest_update_ECI": float(ECI[i_best]), "harvest_best": self.best_harvest,
            "v8_use_base": finals[BASE].get("v8_use"), "setpoint_override_rate_base": finals[BASE].get("setpoint_override_rate"),
            "c13": {"update_wall_s": time.time() - t0, "rpm_fallback_chunks": rpm["fallback_chunks"],
                    "rpm_fallback_points": rpm["fallback_points"], "rpm_requests": rpm["requests"],
                    "parent_wait_median_s": float(np.median(waits)) if waits.size else None,
                    "parent_wait_p99_s": float(np.percentile(waits, 99)) if waits.size else None,
                    "parent_wait_steps": int(waits.size), "min_free_memory_gb": mem,
                    "eval_raw_resid_rel": c11[BASE]["r"], "eval_raw_resid_a_gj": c11[BASE]["a_gj"],
                    "train_raw_resid_rel": [c["r"] for c in c11[:N_SIB]], "train_raw_resid_a_gj": [c["a_gj"] for c in c11[:N_SIB]]},
            "reference": self.reference.summary(), "nonfinite": self.nonfinite}
        record.update(upd.get("extra", {}))
        self.prev_base = (float(E[BASE]), float(DR[BASE]))
        self.prev_sum_a = sum_a
        record["reference_update"] = self.reference.update_after_harvest(self.best_harvest, u)
        record["reference_next"] = self.reference.summary()
        record["update_seconds"] = time.time() - t0
        self.update_index = u
        runlog.append_jsonl(self._path("updates.jsonl"), record)
        if not self.nonfinite:
            self.save_checkpoint()
            if u % constants.CHECKPOINT_EVERY == 0:
                runlog.retrying(lambda: shutil.copy2(self._model_path("checkpoint_latest.pt"), self._model_path(f"checkpoint_u{u}.pt")))
        self.judge_after_update(u, record, c11)

    # ---- 判定 ----

    def read_updates(self):
        path = self._path("updates.jsonl")
        return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []

    def judge_after_update(self, u, record, c11):
        reasons = ts.c11_stop_reasons(
            [("基準の軌跡", c11[BASE])] + [(f"兄弟（組 {self.slots[j]}）", c11[j]) for j in range(N_SIB)])
        if reasons:
            self.stop_reason = f"更新 {u}: " + "；".join(reasons)
        res = collapse_check(self.base_eci_history, u)
        runlog.append_jsonl(self._path("collapse_watch.jsonl"), res)
        if res["triggered"]:
            self.stop_reason = (f"崩れの見張り: 更新 {u} の基準の ECI の直近 {COLLAPSE_WINDOW} 更新の中央値 {res['median_recent']:.4f} が、"
                                f"それまでの最大 {res['running_max']:.4f} より 0.01 以上低い")
        if u == 1 and self.cfg.get("start_E_hex") and self.cfg.get("start_DR_hex"):            # C8-2
            g = {"base_E_hex": float.hex(record["eval_E"]), "base_DR_hex": float.hex(record["eval_DR"]),
                 "start_E_hex": self.cfg["start_E_hex"], "start_DR_hex": self.cfg["start_DR_hex"]}
            g["passed"] = bool(g["base_E_hex"] == g["start_E_hex"] and g["base_DR_hex"] == g["start_DR_hex"])
            self.judgments["C8-2"] = g
            runlog.write_json(self._path("judgment_c82.json"), g)
            self.log_event(event="C8-2", **g)
            if not g["passed"]:
                self.stop_reason = (f"C8-2（更新 1）: 基準の軌跡の E・DR が始点と 16 進で一致しない"
                                    f"（E {g['base_E_hex']}、DR {g['base_DR_hex']}）")
        if u == 2:                                                                             # C8-3
            ups = self.read_updates()
            mean_s = float(np.mean([r["update_seconds"] for r in ups[:2]]))
            limit = float(self.cfg.get("c83_limit_s", C83_LIMIT_SECONDS))
            g = {"mean_update_seconds_1_2": mean_s, "limit_s": limit, "passed": bool(mean_s <= limit)}
            self.judgments["C8-3"] = g
            runlog.write_json(self._path("judgment_c83.json"), g)
            self.log_event(event="C8-3", **g)
            if not g["passed"]:
                self.stop_reason = f"C8-3（更新 2）: 更新 1・2 の 1 更新の実時間の平均 {mean_s:.0f} 秒が {limit:g} 秒を超えた"
        if u == G20_UPDATE:
            ups = self.read_updates()
            secs = float(np.median([r["update_seconds"] for r in ups[:G20_UPDATE]]))
            start = self.base_eci_history[0]
            med = float(np.median(self.base_eci_history[11:21]))
            g = {"records_complete": all(k in record for k in ("sibling_A", "adoption_rate", "c13", "eval_c11")),
                 "median_update_seconds": secs, "time_ok": secs < G20_SECONDS_MAX, "start_eci": start,
                 "median_base_eci_11_20": med, "eci_ok": med >= start - G20_MEDIAN_DROP_MAX}
            g["passed"] = bool(g["records_complete"] and g["time_ok"] and g["eci_ok"])
            self.judgments["G-20"] = g
            runlog.write_json(self._path("judgment_g20.json"), g)
            self.log_event(event="G-20", **g)
            if not g["passed"]:
                self.stop_reason = f"G-20（更新 {u}）に不合格がある（judgment_g20.json）"
        if u == G50_UPDATE:
            start = self.base_eci_history[0]
            mx = float(np.max(self.base_eci_history[1:51]))
            g = {"start_eci": start, "max_base_eci_1_50": mx, "passed": bool(mx >= start + G50_RISE_MIN)}
            self.judgments["G-50"] = g
            runlog.write_json(self._path("judgment_g50.json"), g)
            self.log_event(event="G-50", **g)
            if not g["passed"]:
                self.stop_reason = f"G-50（更新 {u}）に不合格: 基準の ECI の最大 {mx:.4f} が始点 {start:.4f} + 0.002 に届かない"

    # ---- 終了時の再生（C-12） ----

    def final_replay(self):
        path = self._path(os.path.join("harvest", "best_harvest.npz"))
        if self.best_harvest is None or self.best_harvest.get("update", 0) == 0 or not os.path.exists(path):
            return None
        from rlkit.learner import replay_action_sequence
        actions = np.load(path)["actions"].astype(np.int64)
        attempts = []
        for attempt in (1, 2):
            vec = None
            try:
                vec = self.make_vec([self.slots[BASE]])
                info = replay_action_sequence(vec, actions)
                rec = {"time": runlog.now_iso(), "harvested_at_update": self.best_harvest["update"],
                       "saved_E_hex": self.best_harvest["E_hex"], "saved_DR_hex": self.best_harvest["DR_hex"],
                       "replay_E_hex": float.hex(info["E_rep"]), "replay_DR_hex": float.hex(info["DR_rep"]),
                       "replay_ECI": info["ECI"], "replay_P_E": info["P_E"], "replay_P_DR": info["P_DR"]}
                rec["E_hex_match"] = rec["replay_E_hex"] == rec["saved_E_hex"]
                rec["DR_hex_match"] = rec["replay_DR_hex"] == rec["saved_DR_hex"]
                rec["F-1_passed"] = rec["E_hex_match"] and rec["DR_hex_match"]
                rec["attempts"] = attempts + [{"attempt": attempt, "result": "ok"}]
                runlog.write_json(self._path(os.path.join("harvest", "replay_check.json")), rec)
                self.log_event(event="harvest_replay", **rec)
                return rec
            except Exception:
                tb = traceback.format_exc()
                attempts.append({"attempt": attempt, "result": "failed", "traceback": tb[-2000:]})
                self.log_event(event="harvest_replay_failed", attempt=attempt, traceback=tb[-2000:])
            finally:
                if vec is not None:
                    try:
                        vec.close()
                    except Exception:
                        pass
        rec = {"time": runlog.now_iso(), "F-1_passed": None, "failed": True, "attempts": attempts,
               "harvested_at_update": self.best_harvest["update"]}
        runlog.write_json(self._path(os.path.join("harvest", "replay_check.json")), rec)
        return rec

    # ---- 走行 ----

    def run(self, resume=False):
        if not resume and os.path.exists(self._path("updates.jsonl")):               # C-15 (2)
            raise SystemExit(f"起動しない: {self._path('updates.jsonl')} が既にある。続きから動かすときは --resume を付ける")
        os.makedirs(self.run_dir, exist_ok=True)
        meta = {"started_at": runlog.now_iso(), "argv": sys.argv, "config": self.cfg, "factory": self.factory,
                "resume": resume, "instruction_version": self.VERSION, "train_slots": list(self.cfg["train_slots"]),
                "eval_slot": self.cfg["eval_slot"], "comm_rpm": bool(self.cfg.get("comm_rpm", False)),
                "model_dir": self.model_dir}
        if self.factory == "real_v8":
            from rlkit.parallel import launch as launch_mod
            from rlkit.parallel import slots
            meta["preflight"] = launch_mod.preflight(self.run_dir, self.slots, limit=self.cfg.get("slot_limit", launch_mod.SLOT_LIMIT),
                                                     min_free_gb=self.cfg.get("min_free_gb", launch_mod.MIN_FREE_MEM_GB))
            os.environ[launch_mod.LAUNCH_LOG_ENV] = meta["preflight"]["log_dir"]
            meta["replica_checks"] = [slots.prepare_replica(slots.make_slot(k)) for k in self.slots]
        runlog.write_json(self._path(f"meta_{'resume' if resume else 'start'}_{int(time.time())}.json"), meta)
        status = "error"
        done = 0
        try:
            if resume:
                self.load_checkpoint()
                self.log_event(event="resumed", update_index=self.update_index, rho=self.reference.rho,
                               best_harvest=self.best_harvest, instruction_version=self.VERSION)
            else:
                self.init_from_weights()
                self.save_checkpoint()
                self.log_event(event="run_start", instruction_version=self.VERSION, init=self.init_meta,   # MUST-12
                               policy_params=pol8.parameter_count(self.policy), value_params=pol8.parameter_count(self.value),
                               observation_dim=obs8.TOTAL_DIM, action_dims=act8.ACTION_DIMS, logits=act8.ACTION_LOGITS,
                               model_dir=self.model_dir, slots=self.slots)
            self.vec = self.make_vec()
            while self.update_index < self.planned:
                if self.max_updates is not None and self.update_index >= self.max_updates:
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
            status = "stopped"
        except Exception:
            self.stop_reason = f"例外: {traceback.format_exc()}"
            self.log_event(event="exception", traceback=traceback.format_exc())
        finally:
            if self._mem is not None:
                try:
                    self._mem.stop()
                except Exception:
                    pass
            if self.vec is not None:
                try:
                    self.vec.close()
                except Exception:
                    pass
                self.vec = None
            self.harvest_replay = self.final_replay()                                # C-12
            if self.policy is not None and self.reference is not None and not self.nonfinite:
                try:
                    self.save_checkpoint()
                except Exception:
                    self.log_event(event="final_checkpoint_failed", traceback=traceback.format_exc())
            runlog.write_json(self._path("status.json"), {
                "status": status, "stop_reason": self.stop_reason, "finished_at": runlog.now_iso(), "updates_this_run": done,
                "update_index": self.update_index, "best_harvest": self.best_harvest, "harvest_replay": self.harvest_replay,
                "judgments": self.judgments, "instruction_version": self.VERSION})
        return self.stop_reason


def main(argv=None):
    import argparse
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(repo_root)
    p = argparse.ArgumentParser(description="夏季の学習（指示書 V8-S）")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--updates", type=int, default=None, help="通算の更新数の上限")
    a = p.parse_args(argv)
    cfg = config_mod.load(a.config)
    missing = config_mod.unresolved_items(cfg)
    if missing:
        raise SystemExit("未確定項目が設定されていないため起動しない:\n" + "\n".join(f"  {k}（{i}）: {d}" for k, i, d in missing))
    runlog.check_thread_env()
    th.set_num_threads(1)
    run_dir = config_mod.repo_path(cfg["run_dir"])
    cfg["model_dir"] = config_mod.resolve_model_dir(cfg, run_dir, a.resume)
    print(f"[v8.train] モデルの置き場所: {cfg['model_dir']}", flush=True)
    trainer = V8Trainer(cfg, run_dir, max_updates=a.updates)
    print(f"[v8.train] 停止: {trainer.run(resume=a.resume)}", flush=True)


if __name__ == "__main__":
    main()
