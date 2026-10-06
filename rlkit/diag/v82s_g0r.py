"""指示書 V8.2-S 手順 3（G-0R、第 4 章）: 始点の再現。

始点の軌跡（trial01 の収穫の最良。更新 22 の兄弟）の行動列を V8 の環境で再生して、観測・マスク・行動を記録する（1 エピソード）。
その記録を教師として、学習の前の重み（trial01 の best/best_harvest.pt）を L-6R と同じ方法（最大 300 エポック）で学習させる。
一致したら、その方策の基準の軌跡を 1 本走らせ、E・DR が始点と 16 進で一致することを確かめる。
300 エポックで一致しない、または基準の軌跡が始点と一致しない場合は、学習の前の重みに戻し、その方策の基準の軌跡を始点にする（第 4 章の分岐）。
結果は runs/diag/V82S/g0r/ に、採った方策の重みは ~/rl_models/v8s_trial02/g0r_policy.pt に書く。trial01 の記録と保存点は読むだけ（MUST-17）。

  python -m rlkit.diag.v82s_g0r
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
from rlkit import constants, eci as eci_mod, runlog
from rlkit.parallel.batched import SlotVecEnv
from rlkit.v8 import action as act8
from rlkit.v8 import adopt
from rlkit.v8 import policy as pol8

OUT_DIR = os.path.join("runs", "diag", "V82S", "g0r")
CONFIG = os.path.join("rlkit", "config_summer_v82.json")
START_NPZ = os.path.join("runs", "v8s_trial01", "harvest", "best_harvest.npz")
START_HEX = ("0x1.1aea1a9fbe76dp+2", "0x1.6c1d5c0000000p-3")
PRE_WEIGHTS = os.path.join(os.path.expanduser("~"), "rl_models", "v8s_trial01", "best", "best_harvest.pt")


def _vec(cfg):
    kw = {"stage": 1, "include_indoor_unit_power": cfg["include_indoor_unit_power"],
          "zone_step_energy_norm_gj": cfg["zone_step_energy_norm_gj"], "m9_enabled": cfg["m9_enabled"],
          "comm_rpm": bool(cfg.get("comm_rpm", False))}
    return SlotVecEnv([cfg["eval_slot"]], kw, constants.ZONE_STEPS_PER_WEEK, factory="real_v8")


def replay_and_record(cfg, actions):
    """行動列を再生し、各ステップの観測・マスクと終端の info を返す。"""
    vec = _vec(cfg)
    obs_l, mask_l = [], []
    try:
        obs = vec.reset()
        for t in range(actions.shape[0]):
            mask_l.append(np.stack(vec.env_method("action_masks"))[0].copy())
            obs_l.append(np.asarray(obs)[0].copy())
            obs, _r, dones, infos = vec.step(actions[t][None, :])
            if dones[0]:
                info = infos[0]
                break
    finally:
        vec.close()
    return np.stack(obs_l), np.stack(mask_l), info


def run_base(cfg, policy):
    vec = _vec(cfg)
    acts = []
    try:
        obs = vec.reset()
        for t in range(constants.ZONE_STEPS_PER_WEEK):
            masks = np.stack(vec.env_method("action_masks"))
            with th.no_grad():
                logits = policy(th.as_tensor(np.asarray(obs), dtype=th.float32)).numpy()
            act = pol8.masked_argmax(logits, masks)
            acts.append(act[0].copy())
            obs, _r, dones, infos = vec.step(act)
            if dones[0]:
                return infos[0], np.stack(acts)
    finally:
        vec.close()
    raise RuntimeError("1 週間で終端しなかった")


def main():
    os.chdir(_REPO_ROOT)
    from rlkit.parallel import launch, slots
    runlog.check_thread_env()
    cfg = config_mod.load(CONFIG)
    os.makedirs(OUT_DIR, exist_ok=True)
    report = launch.preflight(os.path.abspath(OUT_DIR), [cfg["eval_slot"]], limit=cfg["slot_limit"], min_free_gb=cfg["min_free_gb"])
    os.environ[launch.LAUNCH_LOG_ENV] = report["log_dir"]
    slots.prepare_replica(slots.make_slot(cfg["eval_slot"]))
    actions = np.load(START_NPZ)["actions"].astype(np.int64)
    entry = th.load(PRE_WEIGHTS, weights_only=False)
    policy = pol8.PolicyNetworkV8()
    policy.load_state_dict(entry["weights"])
    policy.eval()
    pre_state = {k: v.clone() for k, v in policy.state_dict().items()}
    rec = {"start_expected_E_hex": START_HEX[0], "start_expected_DR_hex": START_HEX[1],
           "pre_weights": PRE_WEIGHTS, "pre_weights_update": int(entry["update"])}
    # 1. 始点の軌跡を再生して教師を作る（再生の E・DR が始点と 16 進で一致するかも確かめる）
    obs, masks, info = replay_and_record(cfg, actions)
    rec["replay_E_hex"], rec["replay_DR_hex"] = float.hex(info["E_rep"]), float.hex(info["DR_rep"])
    rec["replay_matches_start"] = rec["replay_E_hex"] == START_HEX[0] and rec["replay_DR_hex"] == START_HEX[1]
    teacher = adopt.make_teacher(obs, masks, actions)
    np.savez_compressed(os.path.join(OUT_DIR, "teacher.npz"), obs=obs, masks=masks, actions=actions.astype(np.int8))
    # 2. L-6R と同じ方法で学習させる
    fit = adopt.fit_to_teacher(policy, teacher)
    rec["fit"] = {"adopted": fit["adopted"], "epochs": fit["epochs"], "n_effective": fit["before"]["n_effective"],
                  "n_mismatch_before": fit["before"]["n_mismatch"], "n_margin_fail_before": fit["before"]["n_margin_fail"],
                  "min_margin_before": fit["before"]["min_margin"], "n_mismatch_after": fit["after"]["n_mismatch"],
                  "n_margin_fail_after": fit["after"]["n_margin_fail"], "min_margin_after": fit["after"]["min_margin"]}
    use_trained = False
    if fit["adopted"]:
        # 3. その方策の基準の軌跡が始点と 16 進で一致するか
        binfo, bacts = run_base(cfg, policy)
        rec["base_E_hex"], rec["base_DR_hex"] = float.hex(binfo["E_rep"]), float.hex(binfo["DR_rep"])
        rec["base_matches_start"] = rec["base_E_hex"] == START_HEX[0] and rec["base_DR_hex"] == START_HEX[1]
        rec["base_actions_equal_teacher_on_effective"] = bool(((bacts == actions) | ~teacher["eff"]).all())
        rec["base_n_action_diffs_all_dims"] = int((bacts != actions).sum())
        use_trained = bool(rec["base_matches_start"])
    if use_trained:
        chosen, start_info = policy, {"E_hex": rec["base_E_hex"], "DR_hex": rec["base_DR_hex"], "source": "G-0R で再現した始点（trial01 更新 22 の兄弟の軌跡）"}
    else:
        # 分岐: 学習前の重みに戻し、その方策の基準の軌跡を始点にする
        policy.load_state_dict(pre_state)
        policy.eval()
        binfo, _bacts = run_base(cfg, policy)
        chosen = policy
        start_info = {"E_hex": float.hex(binfo["E_rep"]), "DR_hex": float.hex(binfo["DR_rep"]),
                      "source": "G-0R の分岐: 学習前の重み（trial01 更新 22 の採取に使った重み）の基準の軌跡"}
        rec["fallback_base_E_hex"], rec["fallback_base_DR_hex"] = start_info["E_hex"], start_info["DR_hex"]
    rec["start_point"] = dict(start_info, ECI=float(eci_mod.eci_smooth(float.fromhex(start_info["E_hex"]), float.fromhex(start_info["DR_hex"]))))
    rec["G0R_reproduced"] = use_trained
    model_dir = os.path.join(os.path.expanduser("~"), "rl_models", "v8s_trial02")
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "g0r_policy.pt")
    th.save({"weights": chosen.state_dict(), "start_point": rec["start_point"], "reproduced": use_trained}, out_path)
    rec["policy_path"] = out_path
    runlog.write_json(os.path.join(OUT_DIR, "g0r_result.json"), rec)
    print(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
