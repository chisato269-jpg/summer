"""指示書 V7.1-S 手順 8: 更新 685 の終わりの MUST-9 の確認。runs/diag/V71S/must9_check_u685.json に書く。

読むだけ（走行ディレクトリには書かない）。更新 684 の値は保存点 checkpoint_u684.pt（複写）から、更新 685 の値は updates.jsonl の
685 行目から取る。期待値は R-7・L-1・L-2 の規則から独立に計算し、記録と比べる。
"""
import json
import os
import sys

import torch as th

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rlkit import config as config_mod
from rlkit import constants
from rlkit.learner import adapt_learning_rate, entropy_c0_at
from rlkit.reward import R7Reference

RUN_DIR = os.path.join("runs", "v2s_trial08")
OUT = os.path.join("runs", "diag", "V71S", "must9_check_u685.json")
EXPECT_E_HEX = "0x1.266872b020c4ap+2"
EXPECT_DR_HEX = "0x1.73ac600000000p-3"
EXPECT_ECI_BEST = 0.6713


def main():
    os.chdir(_REPO_ROOT)
    rows = [json.loads(l) for l in open(os.path.join(RUN_DIR, "updates.jsonl"), encoding="utf-8")]
    model_dir = config_mod.model_dir_of(config_mod.load(os.path.join("rlkit", "config_summer_v7p.json")))
    st = th.load(os.path.join(model_dir, "checkpoint_u684.pt"), weights_only=False)
    r684 = st["reference"]
    lr684 = float(st["model"]["policy_lr"])
    rec = rows[684] if len(rows) > 684 else None
    items = {}
    items["1_update_number"] = {"passed": rec is not None and rec["update"] == 685 and
                                [r["update"] for r in rows[:685]] == list(range(1, 686)),
                                "last_update": rows[-1]["update"], "rows": len(rows)}
    if rec is None:
        out = {"passed": False, "items": items, "note": "更新 685 の行が無い"}
    else:
        used, after = rec["reference_used"], rec["reference"]
        rho_prev = float(r684["rho"])
        rho_raw = R7Reference.raw_ratio(float(r684["E_ref"]), float(r684["DR_ref"]))
        _c, rho_expect = R7Reference.clip_ratio(rho_raw, rho_prev)
        items["2_ratio_rule"] = {"passed": used["rho"] == rho_prev and after["rho"] == rho_expect and
                                 rec["reference_update"]["rho_prev"] == rho_prev,
                                 "rho_u684": rho_prev, "rho_used_u685": used["rho"], "rho_after_u685": after["rho"],
                                 "rho_expected_by_R7": rho_expect, "rho_raw_at_reference": rho_raw,
                                 "reference_changed": rec["reference_update"]["changed"]}
        items["3_reference_point_hex"] = {
            "passed": (float.hex(used["E_ref"]) == EXPECT_E_HEX and float.hex(used["DR_ref"]) == EXPECT_DR_HEX and
                       float.hex(after["E_ref"]) == EXPECT_E_HEX and float.hex(after["DR_ref"]) == EXPECT_DR_HEX),
            "E_ref_used_hex": float.hex(used["E_ref"]), "DR_ref_used_hex": float.hex(used["DR_ref"]),
            "E_ref_after_hex": float.hex(after["E_ref"]), "DR_ref_after_hex": float.hex(after["DR_ref"]),
            "expected": [EXPECT_E_HEX, EXPECT_DR_HEX]}
        lr_expect = adapt_learning_rate(lr684, rec["approx_kl_after"])
        items["4_policy_lr_rule"] = {"passed": rec["policy_lr_before"] == lr684 and rec["policy_lr_after"] == lr_expect,
                                     "lr_u684": lr684, "lr_before_u685": rec["policy_lr_before"],
                                     "approx_kl_after": rec["approx_kl_after"], "lr_expected_by_L2": lr_expect,
                                     "lr_after_u685": rec["policy_lr_after"]}
        hb = rec["harvest_best"]
        items["5_harvest_best"] = {"passed": (hb["update"] == 330 and round(hb["ECI"], 4) == EXPECT_ECI_BEST and
                                              hb["E_hex"] == EXPECT_E_HEX and hb["DR_hex"] == EXPECT_DR_HEX),
                                   "update": hb["update"], "ECI": hb["ECI"], "E_hex": hb["E_hex"], "DR_hex": hb["DR_hex"],
                                   "this_update_harvest_ECI": rec.get("harvest_update_ECI")}
        c0 = entropy_c0_at(685, int(rec.get("planned_updates", 1500)) if "planned_updates" in rec else 1500)
        items["6_entropy_coefficient"] = {"passed": rec["entropy_c0"] == c0 == constants.ENTROPY_C0 and
                                          685 <= constants.ENTROPY_CONST_UNTIL_FRACTION * 1500,
                                          "entropy_c0_recorded": rec["entropy_c0"], "expected_constant_section": c0,
                                          "constant_until_update": constants.ENTROPY_CONST_UNTIL_FRACTION * 1500}
        out = {"passed": all(v["passed"] for v in items.values()), "items": items,
               "update_seconds": rec["update_seconds"], "c13": rec.get("c13", {}).get("rpm_fallback_chunks")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: (v["passed"] if isinstance(v, dict) else v) for k, v in out["items"].items()}, ensure_ascii=False))
    print("ALL PASSED" if out["passed"] else "FAILED")
    sys.exit(0 if out["passed"] else 2)


if __name__ == "__main__":
    main()
