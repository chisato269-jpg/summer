"""段階 Z の診断 K-6・K-7・K-8（指示書 V5-S 第 2-6 節・第 6-1 節）。学習は行わない。

K-6: 全停止の軌跡を再生してエネルギーの内訳を取り、運転状態を送風に置き換えた 1 本と 16 進照合する。
K-7: 収穫の最良の軌跡（runs/v2s_trial03/harvest/best_harvest.npz）を基準に、(a) 非在室テナントの換気停止、
     (b) 全熱交換器の風量 Low 固定、(c) 両方、の 3 本を再生し、E・DR と DR の成分を基準と比べる。
K-8: 設定温度 18 ℃・30 ℃を 1 ステップだけ書き込み、読み戻して受理されるかを見る。

エミュレータ実行は合計 6 エピソード相当（K-6 が 2、K-7 が基準 1 ＋ 3、K-8 は 1 ステップ）。
基準の再生は K-7 の 3 本と比べるために要る。組は S-12 により評価用の組 6 を使う。

使い方（リポジトリルートから）:
  python -m rlkit.diag.v5s_diag --step k6
  python -m rlkit.diag.v5s_diag --step k7
  python -m rlkit.diag.v5s_diag --step k8
"""
import argparse
import datetime
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from rlkit import action as action_mod
from rlkit import comm, config as config_mod, constants, driver, env as env_mod, exogenous, runlog, timing
from rlkit import reward as reward_mod
from rlkit.parallel import launch as launch_mod
from rlkit.parallel import slots

DIAG_SLOT = 6                      # S-12: 診断は評価用の組を使う
OUT_ROOT = os.path.join("runs", "diag", "V5S")
TRIAL03 = os.path.join("runs", "v2s_trial03")
KWH_TO_GJ = constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0
TENANT_ZONES = {t: [k for k, z in enumerate(constants.ZONE_DEFS) if z[2] == t] for t in constants.TENANTS}


# =====================================================================
# 行動列の組み立て
# =====================================================================

def all_stop_actions(evap_index=6, vent_on=True):
    """全ゾーン停止の行動列（378 × 60）。保存済みの決定論的評価の列と同じ蒸発温度・換気にそろえる。"""
    a = np.zeros((constants.ZONE_STEPS_PER_WEEK, constants.ACTION_DIMS), dtype=np.int64)
    start = action_mod.LAYOUT["system_evap"][0]
    a[:, start:start + 4] = evap_index
    if vent_on:
        a[:, action_mod.LAYOUT["tenant_vent"][0]:] = constants.TENANT_VENT_ON.index(True)
    return a


def to_fan(actions):
    """運転状態をすべて送風に置き換える（K-6 の後半）。"""
    out = np.asarray(actions, dtype=np.int64).copy()
    out[:, :constants.N_ZONES] = constants.ZONE_STATE.index("fan")
    return out


def step_datetimes():
    """エピソードの各ステップの開始日時（二段の時間刻み）。"""
    dt = exogenous.load().t0
    out = []
    for _ in range(constants.ZONE_STEPS_PER_WEEK):
        out.append(dt)
        dt = dt + datetime.timedelta(seconds=timing.zone_step_seconds(dt))
    return out


def tenant_idle_mask(hours_ahead=1.0):
    """テナントごとに「現在の在室者数が 0 かつ hours_ahead 先も 0」のステップ（M-9 の条件）。"""
    exo = exogenous.load()
    out = {t: np.zeros(constants.ZONE_STEPS_PER_WEEK, dtype=bool) for t in constants.TENANTS}
    for i, dt in enumerate(step_datetimes()):
        now = exo.at_horizon(dt, 0.0)["zone_occupants"]
        upcoming = exo.zone_occupied_within(dt, hours_ahead)
        for t, zs in TENANT_ZONES.items():
            out[t][i] = (float(np.sum(now[zs])) <= 0.0) and (not bool(np.any(upcoming[zs])))
    return out


def stop_vent_when_idle(actions, hours_ahead=1.0):
    """K-7 (a): 非在室のテナントの換気を停止に置き換える。"""
    out = np.asarray(actions, dtype=np.int64).copy()
    start = action_mod.LAYOUT["tenant_vent"][0]
    idle = tenant_idle_mask(hours_ahead)
    off = constants.TENANT_VENT_ON.index(False)
    changed = 0
    for j, t in enumerate(action_mod.VENT_TENANT_ORDER):
        col = start + j
        target = idle[t] & (out[:, col] != off)
        changed += int(target.sum())
        out[target, col] = off
    return out, changed, {t: int(idle[t].sum()) for t in constants.TENANTS}


# =====================================================================
# 再生
# =====================================================================

def replay(actions, hex_fan=None, label=""):
    """1 エピソード再生して、終端の成績とステップごとの内訳を返す。"""
    cfg = config_mod.load(os.path.join("rlkit", "config_summer_v2.json"))
    slot = slots.make_slot(DIAG_SLOT)
    slots.prepare_replica(slot)
    env = env_mod.SummerEnv(stage=1, include_indoor_unit_power=cfg["include_indoor_unit_power"],
                            zone_step_energy_norm_gj=cfg["zone_step_energy_norm_gj"], slot=slot)
    env.set_diag_info(True)
    if hex_fan is not None:
        env.set_hex_fan_speed(hex_fan)
    rec = {"label": label, "hex_fan": env.hex_fan_speed, "steps": []}
    try:
        env.reset()
        for t in range(actions.shape[0]):
            _obs, _r, done, _trunc, info = env.step(actions[t])
            d = info["diag"]
            rec["steps"].append({
                "step": t, "seconds": info["step_seconds"], "delta_E_gj": info["delta_E_gj"],
                "hex_kw": float(np.sum(d["hex_power_kw"])),
                "outdoor_kw": float(np.sum(d["outdoor_power_kw"])),
                "indoor_kw": (float(np.sum(d["indoor_power_kw"])) if d["indoor_power_kw"] is not None else None),
                "q_header_kw": float(d["q_header_kw"]),
                "dr_with_co2": [float(x) for x in d["zone_dr"]],
                "dr_wo_co2": [float(reward_mod.combine_dr_components(
                    d["dissatisfied_thermal"][k], d["dissatisfied_draft"][k], d["dissatisfied_vtmp"][k], 0.0))
                    for k in range(constants.N_ZONES)],
                "occ_zone": [float(x) for x in d["occupants"]],
                "co2_ppm": d["co2_ppm"],
            })
            if done:
                rec["final"] = {k: info[k] for k in ("E_rep", "DR_rep", "eci_official", "eci_smooth",
                                                     "sum_delta_E_gj", "raw_resid_rel", "header_energy_fraction",
                                                     "setpoint_override_rate", "n_steps")}
                break
    finally:
        env.close()
    if "final" not in rec:
        raise RuntimeError(f"{label}: 再生が終端しなかった")
    rec["final"]["E_hex"] = float.hex(rec["final"]["E_rep"])
    rec["final"]["DR_hex"] = float.hex(rec["final"]["DR_rep"])
    rec["summary"] = summarize(rec["steps"])
    return rec


def summarize(steps):
    """週間の内訳（エネルギーは kWh → GJ、不満足は在室者数と経過分で重み付け）。"""
    W = exogenous.load().total_occupant_minutes()
    hours = np.array([s["seconds"] for s in steps], dtype=np.float64) / 3600.0
    minutes = np.array([s["seconds"] for s in steps], dtype=np.float64) / 60.0
    hex_gj = float(np.sum(np.array([s["hex_kw"] for s in steps]) * hours) * KWH_TO_GJ)
    out_gj = float(np.sum(np.array([s["outdoor_kw"] for s in steps]) * hours) * KWH_TO_GJ)
    ind = [s["indoor_kw"] for s in steps]
    ind_gj = float(np.sum(np.array(ind, dtype=np.float64) * hours) * KWH_TO_GJ) if ind[0] is not None else None
    n_with = n_wo = 0.0
    for s, m in zip(steps, minutes):
        occ = np.array(s["occ_zone"])
        n_with += float(np.sum(occ * np.array(s["dr_with_co2"])) * m / W)
        n_wo += float(np.sum(occ * np.array(s["dr_wo_co2"])) * m / W)
    return {"sum_delta_E_gj": float(np.sum([s["delta_E_gj"] for s in steps])),
            "hex_energy_gj": hex_gj, "outdoor_energy_gj": out_gj, "indoor_energy_gj": ind_gj,
            "modelled_total_gj": hex_gj + out_gj + (ind_gj or 0.0),
            "dr_weighted_with_co2": n_with, "dr_weighted_without_co2": n_wo,
            "dr_co2_component": n_with - n_wo}


# =====================================================================
# 手順
# =====================================================================

def run_k6(out_dir):
    base = all_stop_actions()
    stop = replay(base, label="all_stop")
    fan = replay(to_fan(base), label="all_fan")
    ref = {"E_rep": 1.3255498046875, "DR_rep": 0.6349297165870667}
    out = {
        "note": ("全停止の行動列は runs/v2s_trial03 に保存されていない（決定論的評価は最良のときだけ行動列を"
                 "保存する仕様のため）。全ゾーン停止・蒸発温度の索引 6・換気は両テナント運転で組み立て直した。"
                 "保存済みの決定論的評価の行動列がこの換気・蒸発温度であることを確認して合わせた"),
        "reference_point": ref,
        "reference_E_hex": float.hex(ref["E_rep"]), "reference_DR_hex": float.hex(ref["DR_rep"]),
        "all_stop": {"final": stop["final"], "summary": stop["summary"]},
        "all_fan": {"final": fan["final"], "summary": fan["summary"]},
    }
    out["reconstruction_matches_reference"] = (stop["final"]["E_hex"] == out["reference_E_hex"]
                                               and stop["final"]["DR_hex"] == out["reference_DR_hex"])
    out["fan_equals_stop"] = (fan["final"]["E_hex"] == stop["final"]["E_hex"]
                              and fan["final"]["DR_hex"] == stop["final"]["DR_hex"])
    s = stop["summary"]
    total = s["sum_delta_E_gj"]
    out["breakdown_share"] = {"hex": s["hex_energy_gj"] / total, "outdoor": s["outdoor_energy_gj"] / total,
                              "indoor": (s["indoor_energy_gj"] / total if s["indoor_energy_gj"] is not None else None),
                              "unexplained": (total - s["modelled_total_gj"]) / total}
    os.makedirs(out_dir, exist_ok=True)
    runlog.write_json(os.path.join(out_dir, "breakdown.json"), out)
    runlog.write_json(os.path.join(out_dir, "all_stop_steps.json"), {"steps": stop["steps"][:8]})
    for k in ("reconstruction_matches_reference", "fan_equals_stop", "breakdown_share"):
        print(k, "=", out[k])
    print("all_stop E =", stop["final"]["E_rep"], stop["final"]["E_hex"], "DR =", stop["final"]["DR_rep"])
    print("all_fan  E =", fan["final"]["E_rep"], fan["final"]["E_hex"], "DR =", fan["final"]["DR_rep"])
    return out


def run_k7(out_dir):
    actions = np.load(os.path.join(TRIAL03, "harvest", "best_harvest.npz"))["actions"].astype(np.int64)
    meta = json.load(open(os.path.join(TRIAL03, "harvest", "best_harvest.json"), encoding="utf-8"))
    vent_off, changed, idle_steps = stop_vent_when_idle(actions)
    runs = {
        "baseline": replay(actions, label="baseline"),
        "a_vent_stop": replay(vent_off, label="a_vent_stop"),
        "b_hex_low": replay(actions, hex_fan="Low", label="b_hex_low"),
        "c_both": replay(vent_off, hex_fan="Low", label="c_both"),
    }
    base = runs["baseline"]["final"]
    out = {"harvest_meta": meta, "vent_steps_changed": changed, "tenant_idle_steps": idle_steps,
           "baseline_reproduces_saved": (base["E_hex"] == meta["E_hex"] and base["DR_hex"] == meta["DR_hex"]),
           "conditions": {}}
    for name, r in runs.items():
        f, s = r["final"], r["summary"]
        out["conditions"][name] = {
            "E": f["E_rep"], "DR": f["DR_rep"], "E_hex": f["E_hex"], "DR_hex": f["DR_hex"],
            "eci_official": f["eci_official"], "eci_smooth": f["eci_smooth"],
            "dE": f["E_rep"] - base["E_rep"], "dDR": f["DR_rep"] - base["DR_rep"],
            "hex_energy_gj": s["hex_energy_gj"], "outdoor_energy_gj": s["outdoor_energy_gj"],
            "indoor_energy_gj": s["indoor_energy_gj"],
            "dr_weighted_with_co2": s["dr_weighted_with_co2"],
            "dr_weighted_without_co2": s["dr_weighted_without_co2"],
            "dr_co2_component": s["dr_co2_component"]}
    a = out["conditions"]["a_vent_stop"]
    out["M9_adopt"] = bool(a["dE"] <= -0.30 and a["dDR"] <= 0.002)
    out["M9_thresholds"] = {"dE_max": -0.30, "dDR_max": 0.002}
    for name in ("b_hex_low", "c_both"):
        c = out["conditions"][name]
        out.setdefault("hex_fan_report_candidate", {})[name] = bool(c["dE"] <= -0.15 and c["dDR"] <= 0.002)
    os.makedirs(out_dir, exist_ok=True)
    runlog.write_json(os.path.join(out_dir, "counterfactual.json"), out)
    print("baseline が保存値を再現:", out["baseline_reproduces_saved"])
    for name, c in out["conditions"].items():
        print("%-12s E=%.6f (dE %+.4f)  DR=%.6f (dDR %+.5f)  CO2成分 %.6f" % (
            name, c["E"], c["dE"], c["DR"], c["dDR"], c["dr_co2_component"]))
    print("M-9 採用:", out["M9_adopt"], " 風量の報告候補:", out.get("hex_fan_report_candidate"))
    return out


def run_k8(out_dir):
    """K-8: 設定温度 18 ℃・30 ℃を 1 ステップ書き込み、読み戻す。"""
    from szklib.VRFSystemCommunicator import VRFSystemCommunicator as vrc
    slot = slots.make_slot(DIAG_SLOT)
    slots.prepare_replica(slot)
    drv = driver.EpisodeDriver(slot=slot)
    drv.start()
    run = lambda coro: driver.run_coro(drv, coro, timeout=300)
    results = []
    try:
        comms = drv.comms
        for value in (18.0, 30.0):
            run(comm.call_write(comms.vrf.turn_on(1, 1), "turn_on"))
            run(comm.call_write(comms.vrf.change_mode(1, 1, vrc.Mode.Cooling), "mode"))
            ok, error = True, None
            try:
                run(comm.call_write(comms.vrf.change_setpoint_temperature(1, 1, value), f"setpoint {value}"))
            except Exception as exc:                      # 書き込みが拒否される場合も記録する
                ok, error = False, repr(exc)
            run(drv.advance(constants.ZONE_STEP_UNOCCUPIED_SEC))
            readback = run(comm.read_setpoint_readback(comms))
            results.append({"written": value, "write_ok": ok, "error": error,
                            "readback_zone_S1": float(readback[0]),
                            "accepted": abs(float(readback[0]) - value) <= constants.SETPOINT_OVERRIDE_TOL_C})
    finally:
        drv.close()
    out = {"grid_stage1": constants.ZONE_SETPOINT_C[1], "results": results,
           "note": "夏季の格子は変えない。冬季機の格子拡張の可否を判断する材料として記録する（K-8）"}
    os.makedirs(out_dir, exist_ok=True)
    runlog.write_json(os.path.join(out_dir, "setpoint_range.json"), out)
    for r in results:
        print("書き込み %.1f ℃ → 読み戻し %.3f ℃  受理 %s" % (r["written"], r["readback_zone_S1"], r["accepted"]))
    return out


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", required=True, choices=["k6", "k7", "k8"])
    a = p.parse_args(argv)
    runlog.check_thread_env()
    out_dir = os.path.join(OUT_ROOT, a.step.upper())
    os.makedirs(out_dir, exist_ok=True)
    launch_mod.preflight(out_dir, [DIAG_SLOT])            # S-12・第 2-7 節の安全装置
    os.environ[launch_mod.LAUNCH_LOG_ENV] = os.path.join(out_dir, launch_mod.LAUNCH_DIR_NAME)
    {"k6": run_k6, "k7": run_k7, "k8": run_k8}[a.step](out_dir)


if __name__ == "__main__":
    main()
