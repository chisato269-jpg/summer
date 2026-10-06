"""指示書 V8.3-S 手順 4（第 3-2・3-3 節）: 較正。

教師（runs/diag/V83S/calib/teachers/*.npz と、G-0R の教師）のそれぞれで、候補 K-1・K-3・K-4（学習率 3e-3、最大 1,000 エポック）を、
trial02 の方策（g0r_policy.pt）から独立に学習させ、成功・エポック数・実時間・崩れた判断の最大数・最小のロジット差を記録する。
候補・測る量・選び方は指示書第 3 章のとおりで、結果を見て足したり変えたりしない（MUST-19）。
結果は runs/diag/V83S/calib/calib_results.jsonl（1 回の学習ごとに 1 行。再開できる）。選択は v83s_calib_select.py。

  python -m rlkit.diag.v83s_calib [--workers 6]
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CALIB_DIR = os.path.join("runs", "diag", "V83S", "calib")
TEACHER_DIR = os.path.join(CALIB_DIR, "teachers")
RESULTS = os.path.join(CALIB_DIR, "calib_results.jsonl")
POLICY_PATH = os.path.join(os.path.expanduser("~"), "rl_models", "v8s_trial02", "g0r_policy.pt")
G0R_TEACHER = os.path.join("runs", "diag", "V82S", "g0r", "teacher.npz")
CANDIDATES = {"K-1": "k1", "K-3": "k3", "K-4": "k4"}
LR = 3e-3
MAX_EPOCHS = 1000


def teacher_files():
    files = sorted(glob.glob(os.path.join(TEACHER_DIR, "trial02_u*_s*.npz")))
    out = [(os.path.splitext(os.path.basename(f))[0], f) for f in files]
    out.append(("g0r_trial01_u22", G0R_TEACHER))
    return out


def fit_one(job):
    """1 本の教師 × 1 候補（子プロセスで走る）。"""
    import torch as th
    from rlkit.v8 import adopt
    from rlkit.v8 import policy as pol8
    th.set_num_threads(1)
    name, path, cand = job
    z = np.load(os.path.join(_REPO_ROOT, path))
    teacher = adopt.make_teacher(z["obs"], z["masks"], z["actions"].astype(np.int64))
    entry = th.load(os.path.join(_REPO_ROOT, POLICY_PATH) if not os.path.isabs(POLICY_PATH) else POLICY_PATH, weights_only=False)
    policy = pol8.PolicyNetworkV8()
    policy.load_state_dict(entry["weights"] if "weights" in entry else entry)
    policy.eval()
    r = adopt.fit_to_teacher(policy, teacher, lr=LR, max_epochs=MAX_EPOCHS, loss=CANDIDATES[cand])
    b, a = r["before"], r["after"]
    return {"teacher": name, "candidate": cand, "success": bool(r["adopted"]), "epochs": int(r["epochs"]), "seconds": float(r["seconds"]),
            "broken_max": int(r["broken_max"]), "n_effective": b["n_effective"], "n_mismatch_before": b["n_mismatch"],
            "n_margin_fail_before": b["n_margin_fail"], "min_margin_before": b["min_margin"], "n_mismatch_after": a["n_mismatch"],
            "n_margin_fail_after": a["n_margin_fail"], "min_margin_after": a["min_margin"]}


def main():
    os.chdir(_REPO_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(CALIB_DIR, exist_ok=True)
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            done = {(r["teacher"], r["candidate"]) for r in map(json.loads, (l for l in f if l.strip()))}
    jobs = [(n, p, c) for n, p in teacher_files() for c in CANDIDATES if (n, c) not in done]
    print(f"[calib] 教師 {len(teacher_files())} 本、未処理の学習 {len(jobs)} 回、並列 {args.workers}", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(fit_one, jobs), 1):
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"[calib] {i}/{len(jobs)} {res['teacher']} {res['candidate']} 成功={res['success']} エポック={res['epochs']} "
                  f"{res['seconds']:.1f} 秒 経過 {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
