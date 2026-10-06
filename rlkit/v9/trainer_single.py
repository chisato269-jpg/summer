"""rlkit.v9.trainer_single — V9-S の入口（指示書 C9-1・C9-5〜C9-9、第 7 章）。

新規の初期化から、段階 A（PPO。rlkit/v9/phase_a.py）と段階 B（対の比較と採用。rlkit/v9/phase_b.py）を、1 つのプロセスの中で順に行う。
切り替えは設定に書いた規則（rlkit/v9/switch.py）による自動の動作だけ。再開（--resume）と STOP ファイルは受け付けない（V9S-d）:
  --resume が付いたら、何も書かずに拒否して終了する。起動の時点で走行ディレクトリに STOP ファイルがあっても同じ。
  走行中に STOP ファイルが置かれても、読まず、止まらない（置かれたことを events.jsonl に 1 回だけ記録する）。
  記録のある走行ディレクトリ（updates.jsonl）では起動しない（C-15 (2)）。

  python -m rlkit.launch_detached --log <log> --pidfile <json> -- python -m rlkit.v9.trainer_single --config rlkit/config_summer_v9.json
"""
import argparse
import copy
import json
import os
import platform
import sys
import threading
import time
import traceback

import numpy as np
import torch as th

from rlkit import config as config_mod
from rlkit import constants, runlog
from rlkit.v9 import phase_a as pa
from rlkit.v9 import phase_b as pb
from rlkit.v9 import switch as sw

STOP_FILE_NAME = "STOP"
INSTRUCTION_VERSION = "V9-S"


def refuse(msg):
    """何も書かずに拒否する（標準エラー出力と終了コードだけ）。"""
    sys.stderr.write(f"[v9] 起動しない（拒否）: {msg}\n")
    sys.stderr.flush()
    raise SystemExit(f"起動しない（拒否）: {msg}")


def library_versions():
    out = {"python": platform.python_version(), "platform": platform.platform(), "torch": th.__version__, "numpy": np.__version__}
    for name in ("stable_baselines3", "sb3_contrib", "gymnasium", "psutil"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def read_env_seeds(setting_ini):
    """エミュレータの設定ファイルから環境の種 3 つの値を読む（S-5）。"""
    import re
    seeds = {}
    with open(setting_ini, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*(rseed_o|rseed_w|rseed_oprm)\s*=\s*(\d+)", line)
            if m:
                seeds[m.group(1)] = int(m.group(2))
    return seeds


class V9Run:
    def __init__(self, cfg, run_dir, factory="real_v8"):
        self.cfg = cfg
        self.run_dir = run_dir
        self.factory = factory
        self.v9 = cfg["v9"]
        self.planned = int(cfg["planned_updates"])
        self.model_dir = config_mod.repo_path(cfg["model_dir"])
        self.A = None
        self.B = None
        self.stop_reason = None
        self.phase = "A"
        self.switch_update = None
        self.switch_record = None
        self._stop_seen = False
        self._log_lock = threading.Lock()

    def _path(self, name):
        return os.path.join(self.run_dir, name)

    def log_event(self, **obj):
        with self._log_lock:
            runlog.append_jsonl(self._path("events.jsonl"), {"time": runlog.now_iso(), **obj})
        print(f"[v9] {obj}", flush=True)

    def note_stop_file(self):
        """走行中に STOP ファイルが置かれても読まず、止まらない。置かれたことを 1 回だけ記録する（V9S-d）。"""
        if not self._stop_seen and os.path.exists(self._path(STOP_FILE_NAME)):
            self._stop_seen = True
            self.log_event(event="stop_file_ignored", note="V9-S は STOP ファイルを受け付けない（R-1）。走行は続ける")

    # ---- 段階の切り替え（C9-5・C9-6） ----

    def do_switch(self, u, reason, detail):
        A = self.A
        t0 = time.time()
        for vec in (A.train_vec, A.eval_vec):
            vec.close()
        best = dict(A.best_run)
        policy_net = A.model.policy.policy_network
        value_net = A.model.policy.value_network
        cfg = self.cfg
        c1 = sw.candidate1(cfg, policy_net, self.factory)
        actions_path = A._path(os.path.join("harvest", f"best_harvest_u{best['update']}.npz"))
        weights_path = A._model_path(os.path.join("best", f"best_harvest_weights_u{best['update']}.pt"))
        c2, c2_policy = sw.candidate2(cfg, best, actions_path, weights_path, self.factory)
        choice = sw.select_start(c1, c2 if c2.get("ok") else None) if c1.get("ok") else None
        if choice is None:
            raise RuntimeError("切り替え: 候補 1 の基準の軌跡が走らなかった（停止規則）: " + str(c1.get("error")))
        if choice["chosen"] == 2:
            start_policy = c2_policy
            start = {"E_hex": c2["base_E_hex"], "DR_hex": c2["base_DR_hex"], "ECI": c2["ECI"], "candidate": 2}
        else:
            start_policy = policy_net
            start = {"E_hex": c1["E_hex"], "DR_hex": c1["DR_hex"], "ECI": c1["ECI"], "candidate": 1}
            start_policy.eval()
        c1_rec = {k: v for k, v in c1.items() if k != "actions"}
        self.switch_record = {"update": u, "reason": reason, "detail": detail, "B_run_best": best, "candidate1": c1_rec,
                              "candidate2": c2, "choice": choice, "start": start, "seconds": time.time() - t0}
        runlog.write_json(self._path("switch.json"), self.switch_record)
        self.log_event(event="phase_switch", **{k: v for k, v in self.switch_record.items()})
        l6r = {"loss": self.v9["phase_b"]["l6r_loss"], "lr": self.v9["phase_b"]["l6r_lr"], "max_epochs": self.v9["phase_b"]["l6r_max_epochs"]}
        self.B = pb.PhaseB.from_phase_a(cfg, self.run_dir, self.factory, start_policy, value_net, A.reference, A.best_run,
                                        A.eval_eci_history, u, start, l6r)
        self.switch_update = u
        self.phase = "B"

    # ---- 走行 ----

    def run(self):
        os.makedirs(self.run_dir, exist_ok=True)
        meta = {"started_at": runlog.now_iso(), "pid": os.getpid(), "argv": sys.argv, "config": self.cfg, "factory": self.factory,
                "instruction_version": INSTRUCTION_VERSION, "learner_seed": self.cfg["seed"],
                "libraries": library_versions(), "model_dir": self.model_dir,
                "train_slots": list(self.cfg["train_slots"]), "eval_slot": self.cfg["eval_slot"],
                "comm_rpm": bool(self.cfg.get("comm_rpm", False))}
        if self.factory == "real_v8":
            from rlkit.parallel import launch as launch_mod
            from rlkit.parallel import slots
            slot_indices = list(self.cfg["train_slots"]) + [self.cfg["eval_slot"]]
            meta["preflight"] = launch_mod.preflight(self.run_dir, slot_indices, limit=self.cfg.get("slot_limit", launch_mod.SLOT_LIMIT),
                                                     min_free_gb=self.cfg.get("min_free_gb", launch_mod.MIN_FREE_MEM_GB))
            os.environ[launch_mod.LAUNCH_LOG_ENV] = meta["preflight"]["log_dir"]
            meta["replica_checks"] = [slots.prepare_replica(slots.make_slot(k)) for k in slot_indices]
            meta["env_seeds"] = read_env_seeds(os.path.join(slots.ORIGINAL_DIR, "setting.ini"))
            meta["env_seeds_replicas"] = {k: read_env_seeds(os.path.join(slots.replica_dir_for(k), "setting.ini")) for k in slot_indices}
            want = self.v9["environment_seeds"]
            bad = {k: v for k, v in meta["env_seeds_replicas"].items() if v != want}
            if meta["env_seeds"] != want or bad:
                raise SystemExit(f"環境の種が設定（{want}）と違う: 原本 {meta['env_seeds']}、複製 {bad}")
        runlog.write_json(self._path(f"meta_start_{int(time.time())}.json"), meta)

        status = "error"
        done = 0
        try:
            self.A = pa.PhaseA(self.cfg, self.run_dir, factory=self.factory)
            self.A.start()
            force = self.v9.get("force_switch_at")                                   # 試走だけの設定
            while True:
                self.note_stop_file()
                self.A.one_update()
                done += 1
                u = self.A.model.update_index
                if self.A.stop_reason:
                    self.stop_reason = self.A.stop_reason
                    break
                if force is not None:
                    due, reason, detail = (u >= int(force)), "forced_for_smoke_test", {"update": u}
                else:
                    due, reason, detail = sw.switch_due(self.A.best_eci_history, u, cap=int(self.v9["switch"]["cap_update"]),
                                                        min_update=int(self.v9["switch"]["min_update"]),
                                                        window=int(self.v9["switch"]["window"]), rise=float(self.v9["switch"]["rise"]))
                if due:
                    self.do_switch(u, reason, detail)
                    break
                if u >= self.planned:
                    self.stop_reason = f"計画更新数 {self.planned} に到達した（段階 A のまま）"
                    break
            if self.phase == "B" and self.stop_reason is None:
                while self.B.update_index < self.planned:
                    self.note_stop_file()
                    self.B.one_update()
                    done += 1
                    if self.B.stop_reason:
                        self.stop_reason = self.B.stop_reason
                        break
                if self.stop_reason is None:
                    self.stop_reason = f"計画更新数 {self.planned} に到達した"
            status = "stopped"
        except Exception:
            self.stop_reason = f"例外: {traceback.format_exc()}"
            self.log_event(event="exception", traceback=traceback.format_exc())
        finally:
            self.finalize(status, done)
        return self.stop_reason

    def finalize(self, status, done):
        """C9-9: 終わる前に、最良の軌跡の再生（C-12）と、最終の方策で基準の軌跡を 1 本走らせる確認を行う。例外は投げない。"""
        run = self.B if self.phase == "B" else self.A
        replay, final_check = None, None
        try:
            if self.phase == "A" and self.A is not None:
                for vec in (self.A.train_vec, self.A.eval_vec):
                    if vec is not None:
                        try:
                            vec.close()
                        except Exception:
                            pass
                self.A.eval_vec = None
                replay = self.A.final_replay() if self.A.best_harvest else None
            elif self.B is not None:
                try:
                    self.B.vec.close()
                except Exception:
                    pass
                replay = self.B.final_replay()
        except Exception:
            self.log_event(event="final_replay_failed", traceback=traceback.format_exc())
        try:
            if run is not None and not getattr(run, "nonfinite", False):
                net = run.policy if self.phase == "B" else self.A.model.policy.policy_network
                info, _acts = sw.run_base_episode(self.cfg, net, self.factory)
                final_check = {"E_hex": float.hex(float(info["E_rep"])), "DR_hex": float.hex(float(info["DR_rep"])),
                               "ECI": float(info["ECI"]), "P_E": float(info["P_E"]), "P_DR": float(info["P_DR"])}
                if self.phase == "B":
                    exp = self.B.expected_base
                    final_check["expected_E_hex"], final_check["expected_DR_hex"] = float.hex(exp[0]), float.hex(exp[1])
                    final_check["match_expected_base"] = bool(final_check["E_hex"] == final_check["expected_E_hex"]
                                                              and final_check["DR_hex"] == final_check["expected_DR_hex"])
                best = run.best_harvest if self.phase == "B" else self.A.best_run
                final_check["best_run_ECI"] = best["ECI"] if best else None
                final_check["final_policy_base_equals_best_run"] = bool(
                    best and final_check["E_hex"] == best["E_hex"] and final_check["DR_hex"] == best["DR_hex"])
                runlog.write_json(self._path("final_policy_check.json"), final_check)
        except Exception:
            self.log_event(event="final_policy_check_failed", traceback=traceback.format_exc())
        try:
            if self.phase == "A" and self.A is not None and not self.A.nonfinite:
                self.A.save_checkpoint()
        except Exception:
            self.log_event(event="final_checkpoint_failed", traceback=traceback.format_exc())
        best_run = (self.B.best_harvest if self.phase == "B" and self.B is not None else (self.A.best_run if self.A is not None else None))
        runlog.write_json(self._path("status.json"), {
            "status": status, "stop_reason": self.stop_reason, "finished_at": runlog.now_iso(), "updates_this_run": done,
            "update_index": (self.B.update_index if self.phase == "B" and self.B is not None
                             else (self.A.model.update_index if self.A is not None and self.A.model is not None else None)),
            "phase": self.phase, "switch_update": self.switch_update,
            "switch_choice": (self.switch_record or {}).get("choice"), "best_run": best_run,
            "harvest_replay": replay, "final_policy_check": final_check, "instruction_version": INSTRUCTION_VERSION})


def main(argv=None):
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(repo_root)
    p = argparse.ArgumentParser(description="夏季の学習（指示書 V9-S。一度も止めない 1 回の走行）")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true", help="受け付けない（拒否する）")
    a = p.parse_args(argv)
    if a.resume:
        refuse("V9-S は再開（--resume）を受け付けない（S-2・R-1）")
    cfg = config_mod.load(a.config)
    missing = config_mod.unresolved_items(cfg)
    if missing:
        raise SystemExit("未確定項目が設定されていないため起動しない:\n" + "\n".join(f"  {k}（{i}）: {d}" for k, i, d in missing))
    run_dir = config_mod.repo_path(cfg["run_dir"])
    if os.path.exists(os.path.join(run_dir, STOP_FILE_NAME)):
        refuse(f"走行ディレクトリに STOP ファイルがある: {run_dir}")
    if os.path.exists(os.path.join(run_dir, "updates.jsonl")):
        refuse(f"{os.path.join(run_dir, 'updates.jsonl')} が既にある（新しい走行は別の run_dir で始める）")
    runlog.check_thread_env()
    th.set_num_threads(1)
    if not cfg.get("model_dir"):
        refuse("設定の model_dir が無い（モデルは OneDrive の外の絶対パスに置く）")
    print(f"[v9] 走行 {run_dir} / モデル {cfg['model_dir']}", flush=True)
    run = V9Run(cfg, run_dir)
    print(f"[v9] 停止: {run.run()}", flush=True)


if __name__ == "__main__":
    main()
