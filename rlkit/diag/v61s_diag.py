"""指示書 V6.1-S の Z'-1 と診断 K-13（第 4 章・第 7 章）。学習は行わない。組 6 で順に再生する。

  Z'-1  基準の軌跡（runs/v2s_trial04 更新 16、V6 の索引へ写したもの）をそのまま再生し、P-5 と 16 進で照合する
  K-13 (a) リモコンは許可のまま。毎ステップの書き込みの前に設定温度（1106）を読み戻し、冷房を指令しているゾーンで
           読み戻しが指令値と違えば書き直す（env.set_setpoint_rewrite_on_mismatch）
  K-13 (b) 全室内機のリモコンを 1 週間を通して禁止にする（行動の tenant_remote を 0。1113 へ 0 を書く）

使い方（リポジトリルートから）:
  python -m rlkit.diag.v61s_diag
"""
import os
import shutil
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from rlkit import action, config as config_mod, constants, eci, localdata, reward as reward_mod, runlog
from rlkit.parallel import launch as launch_mod
from rlkit.parallel import slots

OUT_DIR = os.path.join("runs", "diag", "V61S")
BASELINE = os.path.join("runs", "v2s_trial04", "harvest")
BASELINE_E_HEX = "0x1.a228e147ae148p+2"
BASELINE_DR_HEX = "0x1.863c140000000p-3"
SLOT = 6
KWH_TO_GJ = constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0


def baseline_actions():
    v5 = np.load(os.path.join(BASELINE, "best_harvest.npz"))["actions"].astype(np.int64)
    return action.v5_to_v6_actions(v5)


def replay(label, actions, rewrite=False):
    from rlkit.env import SummerEnv
    cfg = config_mod.load(os.path.join("rlkit", "config_summer_v6.json"))
    slot = slots.make_slot(SLOT)
    slots.prepare_replica(slot)
    env = SummerEnv(include_indoor_unit_power=cfg["include_indoor_unit_power"],
                    zone_step_energy_norm_gj=cfg["zone_step_energy_norm_gj"], slot=slot, m9_enabled=cfg["m9_enabled"])
    env.set_diag_info(True)
    env.set_setpoint_rewrite_on_mismatch(rewrite)
    W = env.W
    comp = {k: 0.0 for k in ("thermal", "draft", "vtmp", "co2", "combined")}
    vrf_gj = hex_gj = 0.0
    remote_off_steps = 0
    t0 = time.time()
    try:
        env.reset()
        for t in range(actions.shape[0]):
            _o, _r, done, _tr, info = env.step(actions[t])
            d = info["diag"]
            m = info["step_seconds"] / 60.0
            occ = d["occupants"]
            co2 = np.array([reward_mod.dr_co2(d["co2_ppm"][z[2]]) for z in constants.ZONE_DEFS])
            for key, arr in (("thermal", d["dissatisfied_thermal"]), ("draft", d["dissatisfied_draft"]),
                             ("vtmp", d["dissatisfied_vtmp"]), ("co2", co2), ("combined", d["zone_dr"])):
                comp[key] += float(np.sum(occ * arr) * m / W)
            h = float(np.sum(d["hex_power_kw"])) * info["step_seconds"] / 3600.0 * KWH_TO_GJ
            hex_gj += h
            vrf_gj += info["delta_E_gj"] - h
            remote_off_steps += sum(1 for v in d["remote_status"].values() if v < 1.0)
            if done:
                break
    finally:
        env.close()
    E, DR = info["E_rep"], info["DR_rep"]
    rec = {"label": label, "E": E, "DR": DR, "E_hex": float.hex(E), "DR_hex": float.hex(DR),
           "ECI": eci.eci_smooth(E, DR), "P_E": eci.percentile_energy(E), "P_DR": eci.percentile_dissatisfaction(DR),
           "official_eci_reference": info["eci_official"], "dr_components": comp,
           "setpoint_mismatch_zone_steps": info["setpoint_override_count"],
           "setpoint_mismatch_rate_of_cooling": info["setpoint_override_rate"],
           "setpoint_rewrite_count": info["setpoint_rewrite_count"],
           "vrf_energy_gj": vrf_gj, "hex_energy_gj_model": hex_gj, "raw_resid_rel": info["raw_resid_rel"],
           "remote_status_prohibited_tenant_steps": remote_off_steps,
           "remote_permit_fraction": info["remote_permit_fraction"], "onoff_mismatch": info["onoff_mismatch"],
           "elapsed_s": time.time() - t0}
    dst = localdata.diag_csv_dir(os.path.join(OUT_DIR, "slot_output", label))      # OneDrive の外
    os.makedirs(dst, exist_ok=True)
    for name in ("vent.csv", "vrf.csv", "result.txt"):
        src = os.path.join(slot.data_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, name))
    runlog.write_json(os.path.join(OUT_DIR, f"{label}.json"), rec)
    print(f"{label}: E {E:.6f} ({rec['E_hex']}) DR {DR:.6f} ({rec['DR_hex']}) ECI {rec['ECI']:.4f} "
          f"不一致 {rec['setpoint_mismatch_zone_steps']} 書き直し {rec['setpoint_rewrite_count']} ({rec['elapsed_s']:.0f} 秒)",
          flush=True)
    return rec


def main():
    os.chdir(_REPO_ROOT)
    runlog.check_thread_env()
    os.makedirs(OUT_DIR, exist_ok=True)
    report = launch_mod.preflight(OUT_DIR, [SLOT])
    os.environ[launch_mod.LAUNCH_LOG_ENV] = report["log_dir"]
    base = baseline_actions()
    z1 = replay("Z1_baseline", base)
    z1["match"] = z1["E_hex"] == BASELINE_E_HEX and z1["DR_hex"] == BASELINE_DR_HEX
    runlog.write_json(os.path.join(OUT_DIR, "Z1_baseline.json"), z1)
    if not z1["match"]:
        print("Z'-1 が 16 進で一致しない（S-2）。K-13 は行わない", flush=True)
        return
    a = replay("K13_a_rewrite", base, rewrite=True)
    prohibited = base.copy()
    r0 = action.LAYOUT["tenant_remote"][0]
    prohibited[:, r0:r0 + 2] = constants.TENANT_REMOTE_PERMIT.index(False)
    b = replay("K13_b_remote_prohibited", prohibited)
    out = {"baseline": z1, "conditions": {}}
    for rec in (a, b):
        c = dict(rec)
        c["dE"] = rec["E"] - z1["E"]
        c["dDR"] = rec["DR"] - z1["DR"]
        c["dECI"] = rec["ECI"] - z1["ECI"]
        c["dDR_components"] = {k: rec["dr_components"][k] - z1["dr_components"][k] for k in z1["dr_components"]}
        c["dVRF_gj"] = rec["vrf_energy_gj"] - z1["vrf_energy_gj"]
        out["conditions"][rec["label"]] = c
    runlog.write_json(os.path.join(OUT_DIR, "K13.json"), out)
    for k, c in out["conditions"].items():
        print(f"{k}: dE {c['dE']:+.4f} dDR {c['dDR']:+.5f} dECI {c['dECI']:+.4f} dVRF {c['dVRF_gj']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
