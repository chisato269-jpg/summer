"""rlkit.v8.trainer2 — V8.2-S の学習の入口（指示書 V8.2-S 第 3〜7 章）。V8Trainer を継承し、方策の更新と判定だけを替える。

  L-6R 方策の更新: 最も改善した兄弟（最大 3 本まで）の軌跡を、方策の確率最大の選択として正確に再現する（rlkit/v8/adopt.py）
  L-7R 再現の確認: 採用した更新の次の基準の軌跡の E・DR が採用した兄弟の E・DR と 16 進で一致すること、採用が無かった更新の次の基準の軌跡が
       前の基準と 16 進で一致すること。一致しなければ停止して報告する（V8-S の崩れの見張りを置き換える）
  L-8R 記録: 毎更新、試した兄弟の番号と A_s、採否、要したエポック数、学習の前に一致していなかった有効な判断の数、最小のロジットの差
  始点: 設定の start_E_hex・start_DR_hex（G-0R で決めた始点）。初期の方策は設定の init_policy（G-0R が作った重み）、無ければ init_weights（V7 の保存点）
"""
import json
import os
import sys

import numpy as np
import torch as th

from rlkit import config as config_mod
from rlkit import constants, eci as eci_mod, runlog
from rlkit.reward import R7Reference
from rlkit.v8 import adopt
from rlkit.v8 import policy as pol8
from rlkit.v8 import trainer as tr1

INSTRUCTION_VERSION = "V8.2-S"
BASE = tr1.BASE
N_SIB = tr1.N_SIB


def l7r_check(expected, actual):
    """L-7R: (E, DR) の 16 進が一致するか。"""
    ok = float.hex(expected[0]) == float.hex(actual[0]) and float.hex(expected[1]) == float.hex(actual[1])
    return {"passed": bool(ok), "expected_E_hex": float.hex(expected[0]), "expected_DR_hex": float.hex(expected[1]),
            "actual_E_hex": float.hex(actual[0]), "actual_DR_hex": float.hex(actual[1])}


class V82Trainer(tr1.V8Trainer):
    VERSION = INSTRUCTION_VERSION
    G20_SECONDS_MAX = tr1.G20_SECONDS_MAX              # V8.3-S の C83-4 で 450 秒に改める（V83Trainer）

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_base = None           # 次の更新の基準の軌跡の (E, DR) の期待値（L-7R）
        self._check_base = None
        self._l7r = None

    # ---- 初期化・保存・読み込み ----

    def init_from_weights(self):
        super().init_from_weights()                                     # 価値ネットワーク・始点・比 ρ（設定の start_*_hex から）
        pol_path = self.cfg.get("init_policy")
        if pol_path:
            path = config_mod.repo_path(pol_path)
            entry = th.load(path, weights_only=False)
            state = entry["weights"] if "weights" in entry else entry
            self.policy.load_state_dict(state)
            self.init_meta["init_policy"] = {"path": path, "sha256": tr1.file_sha256(path)}
        self.improver = None
        sh, sd = self.cfg.get("start_E_hex"), self.cfg.get("start_DR_hex")
        if not (sh and sd):
            raise SystemExit("V8.2-S: 始点（start_E_hex・start_DR_hex）が設定に無い")
        self.expected_base = (float.fromhex(sh), float.fromhex(sd))
        self.init_meta["instruction_version"] = INSTRUCTION_VERSION

    def save_checkpoint(self):
        state = {"policy": self.policy.state_dict(), "value": self.value.state_dict(),
                 "reference": self.reference.to_dict(), "best_harvest": self.best_harvest,
                 "base_eci_history": self.base_eci_history, "prev_base": self.prev_base, "prev_sum_a": self.prev_sum_a,
                 "update_index": self.update_index, "judgments": self.judgments, "init_meta": self.init_meta,
                 "rpm_prev": self._rpm_prev, "expected_base": self.expected_base}
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
        self.improver = None
        self.reference = R7Reference.from_dict(state["reference"])
        self.best_harvest = state["best_harvest"]
        self.base_eci_history = list(state["base_eci_history"])
        self.prev_base = tuple(state["prev_base"]) if state["prev_base"] is not None else None
        self.prev_sum_a = state["prev_sum_a"]
        self.update_index = int(state["update_index"])
        self.judgments = dict(state["judgments"])
        self.init_meta = state["init_meta"]
        self.expected_base = tuple(state["expected_base"])

    # ---- 方策の更新（L-6R）と L-7R の準備 ----

    def try_adopt(self, sibs, adv):
        """L-6R の 1 更新ぶん（V8.2-S: 平均の交差エントロピー・学習率 3e-4・最大 300 エポック。V8.3-S で差し替える）。"""
        return adopt.try_adopt(self.policy, sibs, adv)

    def update_policy(self, base, sibs, adv):
        E, DR = self._cur["E"], self._cur["DR"]
        self._check_base = self.expected_base                          # この更新の基準の軌跡が一致すべき (E, DR)
        self._l7r = l7r_check(self._check_base, (float(E[BASE]), float(DR[BASE])))
        r = self.try_adopt(sibs, adv)
        s = r["adopted"]
        # 次の更新の基準の期待値: 採用したら兄弟 s の (E, DR)、採用が無ければ今の基準の (E, DR)（方策が変わらないので同じ軌跡）
        self.expected_base = (float(E[s]), float(DR[s])) if s is not None else (float(E[BASE]), float(DR[BASE]))
        last = r["tried"][-1] if r["tried"] else None
        finite = all(bool(th.isfinite(p).all()) for p in self.policy.parameters())
        extra = {"adopt": {"adopted_sibling": s, "tried": r["tried"], "n_tried": len(r["tried"]),
                           "epochs_to_adopt": (last["epochs"] if s is not None else None),
                           "n_mismatch_before": (last["n_mismatch_before"] if last else None),
                           "min_margin_after": (last["min_margin_after"] if last else None),
                           "expected_next_base": {"E_hex": float.hex(self.expected_base[0]), "DR_hex": float.hex(self.expected_base[1])}},
                 "l7r": self._l7r}
        return {"n_improved": int((np.asarray(adv) > 0).sum()), "updated": s is not None,
                "epochs_applied": (last["epochs"] if s is not None else 0), "kl_cutoff": False, "kl_per_epoch": [],
                "loss_per_epoch": [], "adoption_rate": None, "n_decisions": (last["n_effective"] if last else 0), "weights": {},
                "nonfinite": not finite, "extra": extra}

    # ---- 判定 ----

    def judge_after_update(self, u, record, c11):
        from rlkit import train_summer as ts
        reasons = ts.c11_stop_reasons(
            [("基準の軌跡", c11[BASE])] + [(f"兄弟（組 {self.slots[j]}）", c11[j]) for j in range(N_SIB)])
        if reasons:
            self.stop_reason = f"更新 {u}: " + "；".join(reasons)
        if self._l7r is not None and not self._l7r["passed"]:                    # L-7R
            self.stop_reason = (f"L-7R（更新 {u}）: 基準の軌跡の E・DR が期待と 16 進で一致しない"
                                f"（期待 E {self._l7r['expected_E_hex']}・DR {self._l7r['expected_DR_hex']}、"
                                f"実際 E {self._l7r['actual_E_hex']}・DR {self._l7r['actual_DR_hex']}）")
        runlog.append_jsonl(self._path("l7r.jsonl"), {"update": u, **(self._l7r or {})})
        if u == 2:                                                               # C8-3
            ups = self.read_updates()
            mean_s = float(np.mean([r["update_seconds"] for r in ups[:2]]))
            limit = float(self.cfg.get("c83_limit_s", tr1.C83_LIMIT_SECONDS))
            g = {"mean_update_seconds_1_2": mean_s, "limit_s": limit, "passed": bool(mean_s <= limit)}
            self.judgments["C8-3"] = g
            runlog.write_json(self._path("judgment_c83.json"), g)
            self.log_event(event="C8-3", **g)
            if not g["passed"]:
                self.stop_reason = f"C8-3（更新 2）: 更新 1・2 の 1 更新の実時間の平均 {mean_s:.0f} 秒が {limit:g} 秒を超えた"
        if u == tr1.G20_UPDATE:                                                  # G-20
            ups = self.read_updates()[:tr1.G20_UPDATE]
            secs = float(np.median([r["update_seconds"] for r in ups]))
            n_adopt = int(sum(1 for r in ups if r.get("adopt", {}).get("adopted_sibling") is not None))
            g = {"records_complete": all(k in record for k in ("sibling_A", "adopt", "l7r", "c13", "eval_c11")),
                 "median_update_seconds": secs, "time_ok": secs < self.G20_SECONDS_MAX, "n_adoptions_1_20": n_adopt,
                 "adoption_ok": n_adopt >= 1}
            g["passed"] = bool(g["records_complete"] and g["time_ok"] and g["adoption_ok"])
            self.judgments["G-20"] = g
            runlog.write_json(self._path("judgment_g20.json"), g)
            self.log_event(event="G-20", **g)
            if not g["passed"]:
                self.stop_reason = f"G-20（更新 {u}）に不合格がある（judgment_g20.json）"
        if u == tr1.G50_UPDATE:                                                  # G-50: 更新 50 の基準の ECI
            start = self.base_eci_history[0]
            e50 = float(self.base_eci_history[u])
            g = {"start_eci": start, "base_eci_u50": e50, "threshold": start + 0.002, "passed": bool(e50 >= start + 0.002)}
            self.judgments["G-50"] = g
            runlog.write_json(self._path("judgment_g50.json"), g)
            self.log_event(event="G-50", **g)
            if not g["passed"]:
                self.stop_reason = f"G-50（更新 {u}）に不合格: 基準の ECI {e50:.4f} が始点 {start:.4f} + 0.002 に届かない"


def main(argv=None):
    import argparse
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(repo_root)
    p = argparse.ArgumentParser(description="夏季の学習（指示書 V8.2-S）")
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
    print(f"[v8.2.train] モデルの置き場所: {cfg['model_dir']}", flush=True)
    trainer = V82Trainer(cfg, run_dir, max_updates=a.updates)
    print(f"[v8.2.train] 停止: {trainer.run(resume=a.resume)}", flush=True)


if __name__ == "__main__":
    main()
