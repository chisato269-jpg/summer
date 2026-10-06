"""指示書 V8.3-S 手順 3（第 3-1 節）: 教師の再生成。

trial02 の更新 1〜20 で L-6R が試した兄弟（updates.jsonl の adopt.tried）を、記録された（更新番号、組番号）の乱数の種と、trial02 の方策
（~/rl_models/v8s_trial02/g0r_policy.pt。20 更新で変わっていない）で作り直し、観測・マスク・行動を記録する。13 本（兄弟 12 本と基準）を
元の走行と同じ形で同時に走らせる。試した兄弟の E・DR が trial02 の記録（episodes.jsonl）と 16 進で一致するかを確かめ、一致しなければ
較正から外す。外した本数が 5 を超えたら停止して報告する（この script は結果を書くだけで、停止の判断は呼び出し側）。
結果: runs/diag/V83S/calib/teachers/*.npz（obs・masks・actions）、runs/diag/V83S/calib/regen.jsonl（更新ごとの照合）、regen_summary.json。
trial02 の記録と保存点は読むだけ（MUST-20）。再開できる（作り終えた更新は飛ばす）。

  python -m rlkit.diag.v83s_regen
"""
import json
import os
import sys
import time

import numpy as np
import torch as th

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rlkit import config as config_mod
from rlkit import constants, runlog
from rlkit.parallel.batched import SlotVecEnv
from rlkit.v8 import blocks as blk
from rlkit.v8 import policy as pol8
from rlkit.v8 import trainer as tr1

CALIB_DIR = os.path.join("runs", "diag", "V83S", "calib")
TEACHER_DIR = os.path.join(CALIB_DIR, "teachers")
CONFIG = os.path.join("rlkit", "config_summer_v82.json")
SRC_RUN = os.path.join("runs", "v8s_trial02")
POLICY_PATH = os.path.join(os.path.expanduser("~"), "rl_models", "v8s_trial02", "g0r_policy.pt")
G0R_TEACHER = os.path.join("runs", "diag", "V82S", "g0r", "teacher.npz")
MAX_DROPPED = 5


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tried_by_update():
    """trial02 の更新ごとの、試した兄弟の番号のリスト（updates.jsonl の adopt.tried）。"""
    out = {}
    for row in read_jsonl(os.path.join(SRC_RUN, "updates.jsonl")):
        tried = [int(t["sibling"]) for t in row.get("adopt", {}).get("tried", [])]
        if tried:
            out[int(row["update"])] = tried
    return out


def recorded_hex():
    """trial02 の episodes.jsonl の兄弟の (E, DR) の 16 進: {(update, slot): (E_hex, DR_hex)}。"""
    out = {}
    for r in read_jsonl(os.path.join(SRC_RUN, "episodes.jsonl")):
        if r.get("role") == "sibling":
            out[(int(r["update"]), int(r["slot"]))] = (float.hex(float(r["E"])), float.hex(float(r["DR"])))
    return out


def teacher_path(update, slot):
    return os.path.join(TEACHER_DIR, f"trial02_u{update}_s{slot}.npz")


def main():
    os.chdir(_REPO_ROOT)
    from rlkit.parallel import launch, slots
    runlog.check_thread_env()
    cfg = config_mod.load(CONFIG)
    os.makedirs(TEACHER_DIR, exist_ok=True)
    log_path = os.path.join(CALIB_DIR, "regen.jsonl")
    plan = tried_by_update()
    rec = recorded_hex()
    done_updates = {int(r["update"]) for r in read_jsonl(log_path)} if os.path.exists(log_path) else set()
    todo = [u for u in sorted(plan) if u not in done_updates]
    print(f"[regen] 試みのある更新 {len(plan)}、試みた兄弟の合計 {sum(len(v) for v in plan.values())}、未処理の更新 {todo}", flush=True)
    if todo:
        all_slots = list(cfg["train_slots"]) + [cfg["eval_slot"]]
        report = launch.preflight(os.path.abspath(CALIB_DIR), all_slots, limit=cfg["slot_limit"], min_free_gb=cfg["min_free_gb"])
        os.environ[launch.LAUNCH_LOG_ENV] = report["log_dir"]
        for k in all_slots:
            slots.prepare_replica(slots.make_slot(k))
        entry = th.load(POLICY_PATH, weights_only=False)
        policy = pol8.PolicyNetworkV8()
        policy.load_state_dict(entry["weights"] if "weights" in entry else entry)
        policy.eval()
        order = blk.block_order(int(cfg["seed"]))
        block_steps = blk.block_steps_table()
        kw = {"stage": 1, "include_indoor_unit_power": cfg["include_indoor_unit_power"],
              "zone_step_energy_norm_gj": cfg["zone_step_energy_norm_gj"], "m9_enabled": cfg["m9_enabled"],
              "comm_rpm": bool(cfg.get("comm_rpm", False))}
        vec = SlotVecEnv(all_slots, kw, steps_per_collection=tr1.T, factory="real_v8")
        try:
            for u in todo:
                t0 = time.time()
                assign = blk.assignment(u, order)
                perturbers = {j: blk.Perturber(assign[j], blk.sibling_kind(u, j), blk.sibling_seed(int(cfg["seed"]), u, j),
                                               block_steps[assign[j]]) for j in range(tr1.N_SIB)}
                traj, finals = tr1.collect(vec, policy, perturbers)
                row = {"update": u, "seconds": time.time() - t0, "tried": plan[u], "siblings": [], "base_E_hex": float.hex(float(finals[tr1.BASE]["E_rep"])),
                       "base_DR_hex": float.hex(float(finals[tr1.BASE]["DR_rep"]))}
                for j in range(tr1.N_SIB):
                    slot = int(all_slots[j])
                    got = (float.hex(float(finals[j]["E_rep"])), float.hex(float(finals[j]["DR_rep"])))
                    want = rec.get((u, slot))
                    match = want is not None and got == want
                    row["siblings"].append({"sibling": j, "slot": slot, "match": bool(match), "got": list(got), "recorded": list(want) if want else None})
                    if j in plan[u] and match:
                        np.savez_compressed(teacher_path(u, slot), obs=traj["obs"][j], masks=traj["masks"][j],
                                            actions=traj["actions"][j].astype(np.int8))
                runlog.append_jsonl(log_path, row)
                print(f"[regen] 更新 {u}: 試した {plan[u]}、全 12 本の一致 {sum(s['match'] for s in row['siblings'])}/12、{row['seconds']:.0f} 秒", flush=True)
        finally:
            vec.close()
    # 集計
    rows = read_jsonl(log_path)
    tried_total, matched, dropped = 0, [], []
    for r in rows:
        for s in r["siblings"]:
            if s["sibling"] in r["tried"]:
                tried_total += 1
                (matched if s["match"] else dropped).append({"update": r["update"], "sibling": s["sibling"], "slot": s["slot"]})
    all12 = [s["match"] for r in rows for s in r["siblings"]]
    summary = {"tried_total": tried_total, "matched": len(matched), "dropped": dropped, "n_dropped": len(dropped),
               "all_siblings_checked": len(all12), "all_siblings_matched": int(sum(all12)), "max_dropped": MAX_DROPPED,
               "over_limit": len(dropped) > MAX_DROPPED, "updates": [r["update"] for r in rows]}
    runlog.write_json(os.path.join(CALIB_DIR, "regen_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
