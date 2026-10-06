"""指示書 V8-S 手順 7（第 8 章 N-4・G-0）: 初期化した方策の基準の軌跡を 1 本走らせ、E・DR が、初期の重みの決定論的評価の記録と
16 進で一致することを確かめる。あわせて、V7 の方策の行動列（D-1 の記録）を V8 の行動に写したものと、V8 の方策の確率最大の行動が
全ステップ・全次元で一致するか、ステップの開始時刻が区間の表（blocks.step_datetimes）と一致するかを確かめる。
一致しなければ停止して報告する（回避策を自分で講じない）。結果は runs/diag/V8S/g0/ に書く。

  python -m rlkit.diag.v8s_g0                  # 通常（刻みを 3 分割し、配分の比は 3 点の標本から作る）
  python -m rlkit.diag.v8s_g0 --segments 1     # 診断: 一時停止なし・配分の比は終端の標本だけ（V7 と同じ）。不一致の切り分け用
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
from rlkit import constants, runlog
from rlkit.parallel.batched import SlotVecEnv
from rlkit.v8 import action as act8
from rlkit.v8 import blocks as blk
from rlkit.v8 import policy as pol8

OUT_DIR = os.path.join("runs", "diag", "V8S", "g0")
CONFIG = os.path.join("rlkit", "config_summer_v8.json")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=int, default=3, choices=[1, 3])
    ap.add_argument("--alloc-end", action="store_true", help="診断: 一時停止は行うが、配分の比は終端の標本だけから作る")
    args = ap.parse_args()
    os.chdir(_REPO_ROOT)
    from rlkit.parallel import launch, slots
    runlog.check_thread_env()
    cfg = config_mod.load(CONFIG)
    entry = th.load(config_mod.repo_path(cfg["init_weights"]), weights_only=False)
    policy, _value = pol8.build_from_v7(entry)
    policy.eval()
    os.makedirs(OUT_DIR, exist_ok=True)
    slot_index = cfg["eval_slot"]
    report = launch.preflight(os.path.abspath(OUT_DIR), [slot_index], limit=cfg["slot_limit"], min_free_gb=cfg["min_free_gb"])
    os.environ[launch.LAUNCH_LOG_ENV] = report["log_dir"]
    slots.prepare_replica(slots.make_slot(slot_index))
    env_kwargs = {"stage": 1, "include_indoor_unit_power": cfg["include_indoor_unit_power"],
                  "zone_step_energy_norm_gj": cfg["zone_step_energy_norm_gj"], "m9_enabled": cfg["m9_enabled"],
                  "comm_rpm": bool(cfg.get("comm_rpm", False))}
    vec = SlotVecEnv([slot_index], env_kwargs, constants.ZONE_STEPS_PER_WEEK, factory="real_v8")
    expected_dts = blk.step_datetimes()
    actions, dts = [], []
    try:
        vec.set_attr("pause_segments", args.segments)
        vec.set_attr("alloc_end_only", bool(args.alloc_end))
        obs = vec.reset()
        for t in range(constants.ZONE_STEPS_PER_WEEK):
            masks = np.stack(vec.env_method("action_masks"))
            dts.append(vec.get_attr("_pending_dt")[0])
            with th.no_grad():
                logits = policy(th.as_tensor(np.asarray(obs), dtype=th.float32)).numpy()
            act = pol8.masked_argmax(logits, masks)
            obs, _r, dones, infos = vec.step(act)
            actions.append(act[0].copy())
            if dones[0]:
                info = infos[0]
                break
        else:
            raise RuntimeError("1 週間で終端しなかった")
    finally:
        vec.close()
    actions = np.stack(actions)
    d1 = np.load(os.path.join("runs", "diag", "V8S", "d1", "d1_actions.npz"))["actions"].astype(np.int64)
    mapped = act8.v7_to_v8_action(d1)
    mismatch = np.argwhere(mapped != actions)
    rec = {"recorded_E_hex": entry["E_hex"], "recorded_DR_hex": entry["DR_hex"],
           "replay_E_hex": float.hex(info["E_rep"]), "replay_DR_hex": float.hex(info["DR_rep"]), "replay_ECI": info["ECI"],
           "recorded_ECI": entry["eci_smooth"], "n_steps": int(actions.shape[0]),
           "actions_equal_to_v7_mapped": bool(mismatch.size == 0), "n_action_mismatches": int(len(mismatch)),
           "first_mismatches": mismatch[:10].tolist(), "step_datetimes_match_table": [str(x) for x in dts] == [str(x) for x in expected_dts],
           "v8_use": info.get("v8_use"), "raw_resid_rel": info["raw_resid_rel"], "raw_resid_sum_gj": info["raw_resid_sum_gj"],
           "energy_identity_rel": abs(info["sum_zone_energy_gj"] + info["sum_header_energy_gj"] - info["sum_delta_E_gj"]) / info["sum_delta_E_gj"],
           "diss_identity_rel": info["r8p_sum_rel"], "v8_new_obs_clip_rate_max": info.get("v8_new_obs_clip_rate_max"),
           "setpoint_override_rate": info["setpoint_override_rate"]}
    rec["E_hex_match"] = rec["replay_E_hex"] == rec["recorded_E_hex"]
    rec["DR_hex_match"] = rec["replay_DR_hex"] == rec["recorded_DR_hex"]
    rec["G0_passed"] = bool(rec["E_hex_match"] and rec["DR_hex_match"])
    rec["segments"] = args.segments
    rec["alloc_end_only"] = bool(args.alloc_end)
    suffix = ("" if args.segments == 3 else f"_segments{args.segments}") + ("_allocend" if args.alloc_end else "")
    runlog.write_json(os.path.join(OUT_DIR, f"g0_result{suffix}.json"), rec)
    np.savez_compressed(os.path.join(OUT_DIR, f"g0_actions{suffix}.npz"), actions=actions.astype(np.int8))
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    sys.exit(0 if rec["G0_passed"] else 2)


if __name__ == "__main__":
    main()
