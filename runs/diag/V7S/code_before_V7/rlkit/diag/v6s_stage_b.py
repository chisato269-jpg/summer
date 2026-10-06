"""段階 B: 基準の再生と並列の確認（指示書 V6-S 第 6 章）。学習は行わない。

  B-1 V6 の環境の書き込み層で基準の軌跡を再生する。V5 の指令を V6 の索引に写す（換気の運転 → Middle、
      停止 → 停止、設定温度は物理値のまま新しい格子の索引へ、リモコンは許可）。発停は差分書き込みのまま
  B-2 K-9 の結果 A のときだけ（結果 B のため行わない）
  B-3 全組で B-1 を同時に行い、全組が同じ E・DR（16 進）になることを確かめる

使い方（リポジトリルートから）:
  python -m rlkit.diag.v6s_stage_b --step b1   # 組 6 で 1 本
  python -m rlkit.diag.v6s_stage_b --step b3   # 7 組同時
  指示書 V6.1-S: --out runs/diag/V61S/stageB --config rlkit/config_summer_v61.json
  指示書 V6.4-S: --out runs/diag/V64S/stageB --config rlkit/config_summer_v64.json
    （C-5 でリモコンは許可に固定。毎ステップの 1114 の読み戻しが全台許可であることと、T-3: R8'-A の
      第 2-2 節の (1)〜(3) を、環境が記録した A_t と組の分単位の出力で確かめる）
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json

import numpy as np

from rlkit import action, config as config_mod, constants, r8p, runlog
from rlkit.parallel import launch as launch_mod
from rlkit.parallel import slots

OUT_DIR = os.path.join("runs", "diag", "V6S", "stageB")
BASELINE = os.path.join("runs", "v2s_trial04", "harvest")
BASELINE_E_HEX = "0x1.a228e147ae148p+2"
BASELINE_DR_HEX = "0x1.863c140000000p-3"


def baseline_v6_actions():
    v5 = np.load(os.path.join(BASELINE, "best_harvest.npz"))["actions"].astype(np.int64)
    return action.v5_to_v6_actions(v5)


def replay_on_slot(k, launch_dir, q, config_path=os.path.join("rlkit", "config_summer_v6.json")):
    try:
        os.environ[launch_mod.LAUNCH_LOG_ENV] = launch_dir
        from rlkit.env import SummerEnv
        cfg = config_mod.load(config_path)
        slot = slots.make_slot(k)
        slots.prepare_replica(slot)
        env = SummerEnv(include_indoor_unit_power=cfg["include_indoor_unit_power"],
                        zone_step_energy_norm_gj=cfg["zone_step_energy_norm_gj"], slot=slot,
                        m9_enabled=cfg["m9_enabled"])
        acts = baseline_v6_actions()
        env.set_diag_info(True)
        remote_all_permit = True
        t0 = time.time()
        try:
            env.reset()
            for t in range(acts.shape[0]):
                _o, _r, done, _tr, info = env.step(acts[t])
                remote_all_permit &= info["diag"]["remote_status"] == {"South": 1.0, "North": 1.0}
                if done:
                    break
        finally:
            env.close()
        rec = {"slot": k, "E": info["E_rep"], "DR": info["DR_rep"], "E_hex": float.hex(info["E_rep"]),
               "DR_hex": float.hex(info["DR_rep"]), "eci_official": info["eci_official"],
               "ECI": info.get("ECI"), "co2_delta_obs_clip_rate": info.get("co2_delta_obs_clip_rate"),
               "raw_resid_rel": info["raw_resid_rel"], "hex_energy_gj_model": info["hex_energy_gj"],
               "vent_level_fraction": info["vent_level_fraction"], "onoff_mismatch": info["onoff_mismatch"],
               "remote_permit_fraction": info["remote_permit_fraction"], "elapsed_s": time.time() - t0}
        rec["match"] = rec["E_hex"] == BASELINE_E_HEX and rec["DR_hex"] == BASELINE_DR_HEX
        if "r8p_sum_rel" in info:     # V6.4-S: T-3 と C-5 の確認
            ev = r8p.evaluate(env._r8p_A, env._r8p_reads, env._r8p_starts, r8p.load_minute_data(slot.data_dir), env.exo)
            b = ev["by_delta"][str(constants.R8P_DELTA_MIN)]
            t3 = {"week_rel": b["week_rel"], "L1": b["L1"], "min_n": b["min_n"],
                  "cond1": b["week_rel"] <= r8p.A_WEEK_REL_TOL, "cond2": b["L1"] <= r8p.A_L1_MAX,
                  "cond3": b["min_n"] >= r8p.A_MIN_N, "branch_by_evaluate": ev["branch"], "delta_by_evaluate": ev["delta"],
                  "n_B_week_rel": ev["n_B_week_rel"]}
            t3["passed"] = t3["cond1"] and t3["cond2"] and t3["cond3"]
            rec.update({"T-3": t3, "r8p_sum_rel": info["r8p_sum_rel"], "r8p_zone_sum_max_abs": info["r8p_zone_sum_max_abs"],
                        "r8p_empty_max_abs_n": info["r8p_empty_max_abs_n"], "r8p_min_n": info["r8p_min_n"],
                        "r8p_minute_B_week_rel": info["r8p_minute_B_week_rel"], "r8p_seconds": info["r8p_seconds"],
                        "raw_resid_dissatisfaction_rel": info["raw_resid_dissatisfaction_rel"],
                        "raw_resid_dissatisfaction_endpoint_rel": info["raw_resid_dissatisfaction_endpoint_rel"],
                        "remote_status_all_permit": bool(remote_all_permit), "action_dims": int(acts.shape[1])})
        q.put(("ok", k, rec))
    except Exception:
        q.put(("error", k, traceback.format_exc()))


def run(slot_ids, name, out_dir=OUT_DIR, config_path=os.path.join("rlkit", "config_summer_v6.json")):
    os.makedirs(out_dir, exist_ok=True)
    report = launch_mod.preflight(out_dir, slot_ids)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=replay_on_slot, args=(k, report["log_dir"], q, config_path)) for k in slot_ids]
    for p in procs:
        p.start()
    results, errors = {}, {}
    for _ in slot_ids:
        status, k, payload = q.get()
        (results if status == "ok" else errors)[k] = payload
    for p in procs:
        p.join(timeout=60)
    out = {"time": runlog.now_iso(), "step": name, "baseline_E_hex": BASELINE_E_HEX, "baseline_DR_hex": BASELINE_DR_HEX,
           "results": results, "errors": errors,
           "all_match": not errors and all(r["match"] for r in results.values()),
           "all_T3": (not errors and all(r["T-3"]["passed"] for r in results.values())
                      if all("T-3" in r for r in results.values()) else None),
           "all_same": not errors and len({(r["E_hex"], r["DR_hex"]) for r in results.values()}) == 1}
    runlog.write_json(os.path.join(out_dir, f"{name}.json"), out)
    for k, r in sorted(results.items()):
        print(f"組{k}: E {r['E_hex']} DR {r['DR_hex']} 一致 {r['match']} ({r['elapsed_s']:.0f} 秒)")
    print(name, "全一致:", out["all_match"], "全組同じ:", out["all_same"], "T-3:", out["all_T3"], "失敗:", sorted(errors))
    return out


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", required=True, choices=["b1", "b3"])
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--config", default=os.path.join("rlkit", "config_summer_v6.json"))
    a = p.parse_args(argv)
    runlog.check_thread_env()
    if a.step == "b1":
        run([6], "B1", a.out, a.config)
    else:
        run([0, 1, 2, 3, 4, 5, 6], "B3", a.out, a.config)


if __name__ == "__main__":
    main()
