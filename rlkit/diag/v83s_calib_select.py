"""指示書 V8.3-S 手順 4（第 3-3 節）: 較正の結果の集計と候補の選択。選び方は事前登録のとおり（結果を見てから変えない。MUST-19）。

  python -m rlkit.diag.v83s_calib_select

選び方: 成功の割合 = 成功した教師の本数 ÷ 較正に使った教師の本数。最も高い候補との差が 2 本以内の候補を同程度とし、同程度の候補の中で、
成功した教師の実時間の中央値が最小の候補を選ぶ。選んだ候補の成功の割合が 0.8 未満なら走行を始めない（stop = true）。
結果: runs/diag/V83S/calib/calib_selection.json
"""
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALIB_DIR = os.path.join("runs", "diag", "V83S", "calib")
RESULTS = os.path.join(CALIB_DIR, "calib_results.jsonl")
OUT = os.path.join(CALIB_DIR, "calib_selection.json")
CANDIDATES = ("K-1", "K-3", "K-4")
TIE_COUNT = 2
MIN_SUCCESS_RATE = 0.8
LOSS_OF = {"K-1": "k1", "K-3": "k3", "K-4": "k4"}


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else None


def summarize(rows):
    out = {}
    for c in CANDIDATES:
        r = [x for x in rows if x["candidate"] == c]
        ok = [x for x in r if x["success"]]
        out[c] = {"n": len(r), "n_success": len(ok), "success_rate": (len(ok) / len(r)) if r else None,
                  "epochs_success": {"median": q([x["epochs"] for x in ok], 50), "p10": q([x["epochs"] for x in ok], 10),
                                     "p90": q([x["epochs"] for x in ok], 90), "max": q([x["epochs"] for x in ok], 100)},
                  "seconds_success": {"median": q([x["seconds"] for x in ok], 50), "p10": q([x["seconds"] for x in ok], 10),
                                      "p90": q([x["seconds"] for x in ok], 90), "max": q([x["seconds"] for x in ok], 100)},
                  "broken_max": {"median": q([x["broken_max"] for x in r], 50), "mean": float(np.mean([x["broken_max"] for x in r])) if r else None,
                                 "max": q([x["broken_max"] for x in r], 100)},
                  "min_margin_after_success": {"min": q([x["min_margin_after"] for x in ok], 0), "median": q([x["min_margin_after"] for x in ok], 50)},
                  "failed_teachers": [x["teacher"] for x in r if not x["success"]]}
    return out


def select(summary, n_teachers):
    top = max(summary[c]["n_success"] for c in CANDIDATES)
    similar = [c for c in CANDIDATES if top - summary[c]["n_success"] <= TIE_COUNT]
    if len(similar) == 1:
        chosen = similar[0]
        reason = "成功の本数が最大で、差が 2 本以内の候補が他に無い"
    else:
        chosen = min(similar, key=lambda c: (summary[c]["seconds_success"]["median"] if summary[c]["seconds_success"]["median"] is not None else 1e18))
        reason = f"成功の本数の差が 2 本以内の候補 {similar} の中で、成功した教師の実時間の中央値が最小"
    rate = summary[chosen]["n_success"] / n_teachers
    return {"chosen": chosen, "loss": LOSS_OF[chosen], "reason": reason, "similar": similar, "success_rate": rate,
            "stop": bool(rate < MIN_SUCCESS_RATE)}


def main():
    os.chdir(_REPO_ROOT)
    rows = [json.loads(l) for l in open(RESULTS, encoding="utf-8") if l.strip()]
    teachers = sorted({r["teacher"] for r in rows})
    complete = [t for t in teachers if all(any(x["teacher"] == t and x["candidate"] == c for x in rows) for c in CANDIDATES)]
    rows = [r for r in rows if r["teacher"] in complete]
    summary = summarize(rows)
    sel = select(summary, len(complete))
    sel.update({"n_teachers": len(complete), "summary": summary, "lr": 3e-3, "max_epochs": 1000})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: sel[k] for k in ("chosen", "loss", "reason", "success_rate", "stop", "n_teachers")}, ensure_ascii=False))
    for c in CANDIDATES:
        s = summary[c]
        print(c, s["n_success"], "/", s["n"], "epochs median", s["epochs_success"]["median"], "sec median", s["seconds_success"]["median"],
              "broken max", s["broken_max"]["max"])


if __name__ == "__main__":
    main()
