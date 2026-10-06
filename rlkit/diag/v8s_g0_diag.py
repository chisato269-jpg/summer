"""V8-S の G-0 の不一致の切り分け（診断。報告に使う）。D-1 の V7 の行動列を V8 の行動に写して、V8 の環境で再生する。

  python -m rlkit.diag.v8s_g0_diag --segments 3    # 通常（刻みを 3 分割して一時停止）
  python -m rlkit.diag.v8s_g0_diag --segments 1    # 一時停止なし（V7 と同じ 1 回の進め方）
どちらも方策を使わないので、方策の行動の違いの影響は入らない。記録の E・DR と 16 進で比べる。
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rlkit import config as config_mod
from rlkit import constants, runlog
from rlkit.parallel.batched import SlotVecEnv
from rlkit.v8 import action as act8

OUT_DIR = os.path.join("runs", "diag", "V8S", "g0")


def main():
    os.chdir(_REPO_ROOT)
    from rlkit.parallel import launch, slots
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=int, default=3, choices=[1, 3])
    a = ap.parse_args()
    runlog.check_thread_env()
    cfg = config_mod.load(os.path.join("rlkit", "config_summer_v8.json"))
    d1 = json.load(open(os.path.join("runs", "diag", "V8S", "d1", "d1_result.json"), encoding="utf-8"))
    actions = act8.v7_to_v8_action(np.load(os.path.join("runs", "diag", "V8S", "d1", "d1_actions.npz"))["actions"].astype(np.int64))
    slot_index = cfg["eval_slot"]
    report = launch.preflight(os.path.abspath(OUT_DIR), [slot_index], limit=cfg["slot_limit"], min_free_gb=cfg["min_free_gb"])
    os.environ[launch.LAUNCH_LOG_ENV] = report["log_dir"]
    slots.prepare_replica(slots.make_slot(slot_index))
    env_kwargs = {"stage": 1, "include_indoor_unit_power": cfg["include_indoor_unit_power"],
                  "zone_step_energy_norm_gj": cfg["zone_step_energy_norm_gj"], "m9_enabled": cfg["m9_enabled"],
                  "comm_rpm": bool(cfg.get("comm_rpm", False))}
    vec = SlotVecEnv([slot_index], env_kwargs, constants.ZONE_STEPS_PER_WEEK, factory="real_v8")
    try:
        vec.set_attr("pause_segments", a.segments)
        vec.reset()
        for t in range(actions.shape[0]):
            _o, _r, dones, infos = vec.step(actions[t][None, :])
            if dones[0]:
                info = infos[0]
                break
    finally:
        vec.close()
    rec = {"segments": a.segments, "recorded_E_hex": d1["recorded_E_hex"], "recorded_DR_hex": d1["recorded_DR_hex"],
           "replay_E_hex": float.hex(info["E_rep"]), "replay_DR_hex": float.hex(info["DR_rep"]), "replay_ECI": info["ECI"],
           "dE": info["E_rep"] - float.fromhex(d1["recorded_E_hex"]), "dDR": info["DR_rep"] - float.fromhex(d1["recorded_DR_hex"])}
    rec["matched"] = rec["replay_E_hex"] == rec["recorded_E_hex"] and rec["replay_DR_hex"] == rec["recorded_DR_hex"]
    runlog.write_json(os.path.join(OUT_DIR, f"g0_diag_segments{a.segments}.json"), rec)
    print(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
