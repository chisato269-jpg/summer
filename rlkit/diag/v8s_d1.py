"""指示書 V8-S 手順 2（V7.2-S の D-1 を含む）: 初期の重みの候補を決定論的に再生し、記録と 16 進で比べる。

候補は trial08 の best/best_official.pt（更新 1461 の決定論的評価の重み）。記録の E・DR は同じファイルの entry
（E_hex・DR_hex）にある。読むだけ（runs/v2s_trial08 には書かない。MUST-11）。結果は runs/diag/V8S/d1/ に書く。

  python -m rlkit.diag.v8s_d1
"""
import json
import os
import sys

import numpy as np
import torch as th

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rlkit import action as action_mod
from rlkit import config as config_mod
from rlkit import constants, policy as policy_mod, runlog
from rlkit.parallel.batched import SlotVecEnv

OUT_DIR = os.path.join("runs", "diag", "V8S", "d1")
CANDIDATE = os.path.join("runs", "v2s_trial08", "best", "best_official.pt")
CONFIG = os.path.join("rlkit", "config_summer_v7p.json")


class ArgmaxPolicy:
    """方策網の重みから、マスクを掛けた次元ごとの確率最大の選択（決定論的評価と同じ規則）を返す最小の方策。"""

    def __init__(self, net, nvec):
        self.net = net.eval()
        self.nvec = list(nvec)

    def predict(self, obs, deterministic=True, action_masks=None):
        with th.no_grad():
            logits = self.net(th.as_tensor(np.asarray(obs), dtype=th.float32)).numpy()
        out, pos = [], 0
        masks = np.asarray(action_masks, dtype=bool)
        for k in self.nvec:
            seg = np.where(masks[:, pos:pos + k], logits[:, pos:pos + k], -np.inf)
            out.append(seg.argmax(axis=1))
            pos += k
        return np.stack(out, axis=1).astype(np.int64), None


def run_deterministic(policy, vec, n_steps=constants.ZONE_STEPS_PER_WEEK):
    obs = vec.reset()
    actions = []
    for _ in range(n_steps):
        masks = np.stack(vec.env_method("action_masks"))
        act, _ = policy.predict(obs, deterministic=True, action_masks=masks)
        obs, _r, dones, infos = vec.step(act)
        actions.append(np.asarray(act[0], dtype=np.int64))
        if dones[0]:
            return infos[0], np.stack(actions)
    raise RuntimeError("決定論的評価が終端しなかった")


def main():
    os.chdir(_REPO_ROOT)
    from rlkit.parallel import launch, slots
    runlog.check_thread_env()
    cfg = config_mod.load(CONFIG)
    entry = th.load(CANDIDATE, weights_only=False)
    net = policy_mod.PolicyNetwork(entry["stage"])
    net.load_state_dict(entry["weights"]["policy_net"])
    pol = ArgmaxPolicy(net, action_mod.nvec(entry["stage"]).tolist() if hasattr(action_mod.nvec(entry["stage"]), "tolist")
                       else action_mod.nvec(entry["stage"]))
    os.makedirs(OUT_DIR, exist_ok=True)
    slot_index = 12
    report = launch.preflight(os.path.abspath(OUT_DIR), [slot_index], limit=cfg.get("slot_limit", launch.SLOT_LIMIT),
                              min_free_gb=cfg.get("min_free_gb", launch.MIN_FREE_MEM_GB))
    os.environ[launch.LAUNCH_LOG_ENV] = report["log_dir"]
    slots.prepare_replica(slots.make_slot(slot_index))
    env_kwargs = {"stage": entry["stage"], "include_indoor_unit_power": cfg["include_indoor_unit_power"],
                  "zone_step_energy_norm_gj": cfg["zone_step_energy_norm_gj"], "m9_enabled": cfg["m9_enabled"],
                  "comm_rpm": bool(cfg.get("comm_rpm", False))}
    vec = SlotVecEnv([slot_index], env_kwargs, constants.ZONE_STEPS_PER_WEEK, factory="real")
    try:
        vec.env_method("set_reference", None)
        info, actions = run_deterministic(pol, vec)
    finally:
        vec.close()
    rec = {"candidate": CANDIDATE, "update_of_weights": entry["update"],
           "recorded_E_hex": entry["E_hex"], "recorded_DR_hex": entry["DR_hex"],
           "replay_E": info["E_rep"], "replay_DR": info["DR_rep"],
           "replay_E_hex": float.hex(info["E_rep"]), "replay_DR_hex": float.hex(info["DR_rep"]),
           "replay_ECI": info["ECI"], "recorded_eci_smooth": entry["eci_smooth"]}
    rec["E_hex_match"] = rec["replay_E_hex"] == rec["recorded_E_hex"]
    rec["DR_hex_match"] = rec["replay_DR_hex"] == rec["recorded_DR_hex"]
    rec["matched"] = rec["E_hex_match"] and rec["DR_hex_match"]
    runlog.write_json(os.path.join(OUT_DIR, "d1_result.json"), rec)
    np.savez_compressed(os.path.join(OUT_DIR, "d1_actions.npz"), actions=actions.astype(np.int8))
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    sys.exit(0 if rec["matched"] else 2)


if __name__ == "__main__":
    main()
