"""指示書 V8.3-S 手順 5（第 5 章）: 始点の再現（G-0R3）。

始点の軌跡は trial02 の収穫の最良（更新 5 の兄弟。harvest/best_harvest.npz の行動列）。その行動列を V8 の環境で再生して教師（観測・マスク・行動）を作り
（再生の E・DR が trial02 の記録と 16 進で一致するかも確かめる）、学習前の重み（~/rl_models/v8s_trial02/g0r_policy.pt）に、較正で選んだ C83-1 の
学習（calib_selection.json の損失・学習率・最大エポック）を行う。採用の基準を満たしたら、その方策の基準の軌跡を 1 本走らせ、E・DR が始点と
16 進で一致することを確かめる。満たさない・一致しない場合は、学習前の重みとその基準の軌跡（trial02 の始点）で走行を始める（分岐）。
結果: runs/diag/V83S/calib/g0r3_result.json、方策の重み ~/rl_models/v8s_trial03/g0r3_policy.pt。trial02 の記録と保存点は読むだけ（MUST-20）。

  python -m rlkit.diag.v83s_g0r3
"""
import json
import os
import sys

import numpy as np
import torch as th

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rlkit import config as config_mod
from rlkit import eci as eci_mod, runlog
from rlkit.diag import v82s_g0r as g0r
from rlkit.v8 import adopt
from rlkit.v8 import policy as pol8

CALIB_DIR = os.path.join("runs", "diag", "V83S", "calib")
CONFIG = os.path.join("rlkit", "config_summer_v82.json")
HARVEST_NPZ = os.path.join("runs", "v8s_trial02", "harvest", "best_harvest.npz")
HARVEST_JSON = os.path.join("runs", "v8s_trial02", "harvest", "best_harvest.json")
PRE_WEIGHTS = os.path.join(os.path.expanduser("~"), "rl_models", "v8s_trial02", "g0r_policy.pt")
FALLBACK_START = ("0x1.1a204189374bcp+2", "0x1.6cd5d80000000p-3")        # trial02 の始点（学習前の重みの基準の軌跡）
OUT_POLICY = os.path.join(os.path.expanduser("~"), "rl_models", "v8s_trial03", "g0r3_policy.pt")


def main():
    os.chdir(_REPO_ROOT)
    from rlkit.parallel import launch, slots
    runlog.check_thread_env()
    cfg = config_mod.load(CONFIG)
    sel = json.load(open(os.path.join(CALIB_DIR, "calib_selection.json"), encoding="utf-8"))
    if sel["stop"]:
        raise SystemExit("較正の選択が停止の条件（成功の割合 < 0.8）。G-0R3 は行わない")
    harvest = json.load(open(HARVEST_JSON, encoding="utf-8"))
    start_hex = (harvest["E_hex"], harvest["DR_hex"])
    report = launch.preflight(os.path.abspath(CALIB_DIR), [cfg["eval_slot"]], limit=cfg["slot_limit"], min_free_gb=cfg["min_free_gb"])
    os.environ[launch.LAUNCH_LOG_ENV] = report["log_dir"]
    slots.prepare_replica(slots.make_slot(cfg["eval_slot"]))
    actions = np.load(HARVEST_NPZ)["actions"].astype(np.int64)
    entry = th.load(PRE_WEIGHTS, weights_only=False)
    policy = pol8.PolicyNetworkV8()
    policy.load_state_dict(entry["weights"])
    policy.eval()
    pre_state = {k: v.clone() for k, v in policy.state_dict().items()}
    rec = {"start_expected_E_hex": start_hex[0], "start_expected_DR_hex": start_hex[1], "start_harvest": harvest,
           "selected": {k: sel[k] for k in ("chosen", "loss", "lr", "max_epochs")}, "pre_weights": PRE_WEIGHTS}
    # 1. 始点の軌跡を再生して教師を作る
    obs, masks, info = g0r.replay_and_record(cfg, actions)
    rec["replay_E_hex"], rec["replay_DR_hex"] = float.hex(info["E_rep"]), float.hex(info["DR_rep"])
    rec["replay_matches_start"] = rec["replay_E_hex"] == start_hex[0] and rec["replay_DR_hex"] == start_hex[1]
    teacher = adopt.make_teacher(obs, masks, actions)
    np.savez_compressed(os.path.join(CALIB_DIR, "g0r3_teacher.npz"), obs=obs, masks=masks, actions=actions.astype(np.int8))
    # 2. C83-1 の学習
    fit = adopt.fit_to_teacher(policy, teacher, lr=sel["lr"], max_epochs=sel["max_epochs"], loss=sel["loss"])
    rec["fit"] = {"adopted": fit["adopted"], "epochs": fit["epochs"], "seconds": fit["seconds"], "broken_max": fit["broken_max"],
                  "n_effective": fit["before"]["n_effective"], "n_mismatch_before": fit["before"]["n_mismatch"],
                  "n_margin_fail_before": fit["before"]["n_margin_fail"], "min_margin_before": fit["before"]["min_margin"],
                  "n_mismatch_after": fit["after"]["n_mismatch"], "n_margin_fail_after": fit["after"]["n_margin_fail"],
                  "min_margin_after": fit["after"]["min_margin"]}
    use_trained = False
    if fit["adopted"] and rec["replay_matches_start"]:
        # 3. その方策の基準の軌跡が始点と 16 進で一致するか
        binfo, bacts = g0r.run_base(cfg, policy)
        rec["base_E_hex"], rec["base_DR_hex"] = float.hex(binfo["E_rep"]), float.hex(binfo["DR_rep"])
        rec["base_matches_start"] = rec["base_E_hex"] == start_hex[0] and rec["base_DR_hex"] == start_hex[1]
        rec["base_actions_equal_teacher_on_effective"] = bool(((bacts == actions) | ~teacher["eff"]).all())
        rec["base_n_action_diffs_all_dims"] = int((bacts != actions).sum())
        use_trained = bool(rec["base_matches_start"])
    if use_trained:
        chosen = policy
        start = {"E_hex": rec["base_E_hex"], "DR_hex": rec["base_DR_hex"], "source": "G-0R3 で再現した始点（trial02 更新 5 の兄弟の軌跡）"}
    else:
        policy.load_state_dict(pre_state)
        policy.eval()
        chosen = policy
        start = {"E_hex": FALLBACK_START[0], "DR_hex": FALLBACK_START[1], "source": "G-0R3 の分岐: 学習前の重みの基準の軌跡（trial02 の始点）"}
        binfo, _ = g0r.run_base(cfg, policy)
        rec["fallback_base_E_hex"], rec["fallback_base_DR_hex"] = float.hex(binfo["E_rep"]), float.hex(binfo["DR_rep"])
        rec["fallback_base_matches_trial02_start"] = (rec["fallback_base_E_hex"], rec["fallback_base_DR_hex"]) == FALLBACK_START
    rec["start_point"] = dict(start, ECI=float(eci_mod.eci_smooth(float.fromhex(start["E_hex"]), float.fromhex(start["DR_hex"]))))
    rec["G0R3_reproduced"] = use_trained
    os.makedirs(os.path.dirname(OUT_POLICY), exist_ok=True)
    th.save({"weights": chosen.state_dict(), "start_point": rec["start_point"], "reproduced": use_trained}, OUT_POLICY)
    rec["policy_path"] = OUT_POLICY
    runlog.write_json(os.path.join(CALIB_DIR, "g0r3_result.json"), rec)
    print(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
