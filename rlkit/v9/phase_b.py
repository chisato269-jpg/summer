"""rlkit.v9.phase_b — V9-S 段階 B（対の比較と採用）の 1 更新（指示書 C9-8・C9-10、第 9 章）。

trial03 の学習の入口 rlkit.v8.trainer3.V83Trainer（V8.3-S: 兄弟 12 本と基準 1 本、L-6R の採用 K-3・学習率 3e-3・最大 1,000 エポック、L-7R）を
そのまま使い、段階 A の状態から作る（from_phase_a）。1 つのプロセスの中で、段階 A の終わりの更新番号の次から続ける。
  引き継ぎ  比 ρ（R7Reference）と走行全体の最良（best_harvest）は段階 A のものを、そのまま R-7 で更新し続ける（C9-8）
  判定      C-11 の停止規則と、L-7R の不一致（採用した更新の次の基準の軌跡が採用した兄弟と 16 進で一致しない）だけで止める。
            崩れの見張りは記録だけ（C9-10）。V8 の更新 2・20・50 の判定（C8-3・G-20・G-50）は使わない
"""
import numpy as np

from rlkit import runlog, train_summer as ts
from rlkit.v8 import trainer3 as tr3
from rlkit.v8 import trainer as tr1

BASE = tr1.BASE
N_SIB = tr1.N_SIB


class PhaseB(tr3.V83Trainer):
    VERSION = "V9-S"

    @classmethod
    def from_phase_a(cls, cfg, run_dir, factory, policy_net, value_net, reference, best_run, eval_eci_history, update_index,
                     start, l6r):
        """start: {"E_hex", "DR_hex", "ECI"}（段階 B の始点の基準の軌跡）、l6r: {"loss", "lr", "max_epochs"}。"""
        t = cls(cfg, run_dir, factory=factory)
        t.policy = policy_net
        t.value = value_net
        for p in t.value.parameters():
            p.requires_grad_(False)
        t.improver = None
        t.reference = reference
        t.best_harvest = dict(best_run)
        t.base_eci_history = [float(start["ECI"])] + [float(x) for x in eval_eci_history]
        t.prev_base = (float.fromhex(start["E_hex"]), float.fromhex(start["DR_hex"]))
        t.prev_sum_a = None
        t.update_index = int(update_index)
        t.expected_base = t.prev_base
        t.judgments = {}
        t._rpm_prev = {}
        t.init_meta = {"instruction_version": "V9-S", "phase": "B", "l6r": dict(l6r), "start_point": dict(start),
                       "start_update": int(update_index)}
        t.vec = t.make_vec()
        return t

    def update_policy(self, base, sibs, adv):
        upd = super().update_policy(base, sibs, adv)
        upd["extra"]["phase"] = "B"
        return upd

    def judge_after_update(self, u, record, c11):
        reasons = ts.c11_stop_reasons(
            [("基準の軌跡", c11[BASE])] + [(f"兄弟（組 {self.slots[j]}）", c11[j]) for j in range(N_SIB)])
        if reasons:
            self.stop_reason = f"更新 {u}: " + "；".join(reasons)
        if self._l7r is not None and not self._l7r["passed"]:                    # L-7R
            self.stop_reason = (f"L-7R（更新 {u}）: 基準の軌跡の E・DR が期待と 16 進で一致しない"
                                f"（期待 E {self._l7r['expected_E_hex']}・DR {self._l7r['expected_DR_hex']}、"
                                f"実際 E {self._l7r['actual_E_hex']}・DR {self._l7r['actual_DR_hex']}）")
        runlog.append_jsonl(self._path("l7r.jsonl"), {"update": u, **(self._l7r or {})})
        h = self.base_eci_history[1:]                                           # 崩れの見張り（記録だけ。止めない）
        if len(h) >= 20 and u % 20 == 0:
            recent = float(np.median(h[-20:]))
            best = float(np.max(h))
            runlog.append_jsonl(self._path("collapse_watch.jsonl"), {"update": u, "median_recent20_base_ECI": recent,
                                                                      "max_base_ECI": best, "ratio": recent / best if best else None,
                                                                      "recorded_only": True})
