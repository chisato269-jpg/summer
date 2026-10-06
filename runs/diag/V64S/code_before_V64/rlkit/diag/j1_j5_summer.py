"""段階 0 の診断 C-1・C-3・C-4・C-5（指示書 V2-S 第 6-3 節）。学習は行わない。

規則ベース最良軌跡（verify/raw/H1/best_summer_trajectory_v2.json、夏季）を原本の Shizuku2 で 1 本再生し、
各ステップの終わりに v2 の読み出し一式（建物・ゾーンの 172 点、室外機電力 4、室内機電力 18、設定温度の
読み戻し 18、テナント別の在室判定と温冷感）を行い、所要時間と点数も測る。

C-1（J-1・K-1）: Σ(P_ou + P_hex)·Δt·9.77/1000 と Σ(P_ou + P_iu + P_hex)·Δt·9.77/1000 を ΣΔE と比べ、
    相対差 0.1% 以内の方を採る。両方 0.1% 以内なら「加えない」。どちらも超えるなら小さい方を採り、
    その相対差が 3% を超えるなら判定しない（停止して報告）。
C-1（J-5・K-2）: 採用した側の 1 ステップのゾーン一次エネルギー最大値 × 3 を有効数字 2 桁に切り上げる。
C-3（K-4）: 1 エピソードの実時間、1 ステップの読み出し点数と所要時間。
C-4: 比率法の raw_resid の週間相対値。
C-5（R-5）: setting.ini の rseed_oprm、テナント別在室者数の最大値（Appendix 2 の表との照合材料）。

使い方（リポジトリルートから）: python -m rlkit.diag.j1_j5_summer [--out runs/diag/V2S/C1]
"""
import argparse
import math
import os
import re
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from rlkit import comm, constants, driver, eci as eci_mod, runlog, timing
from rlkit import reward as reward_mod

SNAPSHOT_READ_POINTS = 172  # comm.read_step_snapshot の点数（現行と同じ）


def ceil_significant(x, digits=2):
    """有効数字 digits 桁への切り上げ（K-2）。例: 0.0369 → 0.037"""
    if x <= 0:
        return 0.0
    exp = math.floor(math.log10(x)) - (digits - 1)
    return math.ceil(x / 10.0 ** exp) * 10.0 ** exp


def decide_j1(rel_without_indoor, rel_with_indoor,
              tolerance=constants.J1_MATCH_TOLERANCE, fallback=constants.J1_FALLBACK_TOLERANCE):
    """K-1 の判定。戻り値: (include_indoor_unit_power または None, 理由)"""
    ok_without = rel_without_indoor <= tolerance
    ok_with = rel_with_indoor <= tolerance
    if ok_without and ok_with:
        return False, "both_within_tolerance_choose_without"
    if ok_without:
        return False, "without_indoor_within_tolerance"
    if ok_with:
        return True, "with_indoor_within_tolerance"
    best = min(rel_without_indoor, rel_with_indoor)
    if best > fallback:
        return None, "both_exceed_fallback_tolerance"
    return (rel_with_indoor < rel_without_indoor), "fallback_smaller_relative_difference"


def analyze(delta_E_gj, step_seconds, p_outdoor_kw, p_indoor_kw, p_hex_kw, heatload_kw, zone_on):
    """再生の記録から J-1 の判定、J-5 の値、比率法の raw_resid をまとめる（エミュレータに依存しない計算部分）。"""
    delta_E_gj = np.asarray(delta_E_gj, dtype=np.float64)
    hours = np.asarray(step_seconds, dtype=np.float64) / 3600.0
    kwh_to_gj = constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0
    total_dE = float(delta_E_gj.sum())
    e_outdoor = float((np.asarray(p_outdoor_kw).sum(axis=1) * hours).sum() * kwh_to_gj)
    e_indoor = float((np.asarray(p_indoor_kw).sum(axis=1) * hours).sum() * kwh_to_gj)
    e_hex = float((np.asarray(p_hex_kw).sum(axis=1) * hours).sum() * kwh_to_gj)
    without_iu = e_outdoor + e_hex
    with_iu = without_iu + e_indoor
    rel_without = abs(without_iu - total_dE) / total_dE
    rel_with = abs(with_iu - total_dE) / total_dE
    include, reason = decide_j1(rel_without, rel_with)

    out = {"sum_delta_E_gj": total_dE, "outdoor_gj": e_outdoor, "indoor_gj": e_indoor, "hex_gj": e_hex,
           "sum_without_indoor_gj": without_iu, "sum_with_indoor_gj": with_iu,
           "rel_error_without_indoor": rel_without, "rel_error_with_indoor": rel_with,
           "J1_include_indoor_unit_power": include, "J1_reason": reason,
           "J1_match_tolerance": constants.J1_MATCH_TOLERANCE, "J1_fallback_tolerance": constants.J1_FALLBACK_TOLERANCE}

    for label, use_indoor in (("without_indoor", False), ("with_indoor", True)):
        e_max, raw_resid, header = 0.0, 0.0, 0.0
        for t in range(len(delta_E_gj)):
            split = reward_mod.zone_energy_split(delta_E_gj[t], p_outdoor_kw[t],
                                                 p_indoor_kw[t] if use_indoor else None, p_hex_kw[t],
                                                 heatload_kw[t], zone_on[t], step_seconds[t])
            e_max = max(e_max, float(split["e_gj"].max()))
            raw_resid += float(split["raw_resid_gj"])
            header += float(split["e_header_gj"])
        out[f"J5_zone_step_energy_max_gj_{label}"] = e_max
        out[f"J5_zone_step_energy_norm_gj_{label}"] = ceil_significant(e_max * constants.J5_MULTIPLIER)
        out[f"raw_resid_sum_gj_{label}"] = raw_resid
        out[f"raw_resid_rel_{label}"] = raw_resid / total_dE if total_dE else None
        out[f"header_energy_fraction_{label}"] = header / total_dE if total_dE else None

    if include is not None:
        label = "with_indoor" if include else "without_indoor"
        out["adopted_label"] = label
        out["zone_step_energy_norm_gj"] = out[f"J5_zone_step_energy_norm_gj_{label}"]
        out["raw_resid_rel"] = out[f"raw_resid_rel_{label}"]
    return out


def read_rseed_oprm(path=None):
    path = path or os.path.join(_REPO_ROOT, "Shizuku2", "setting.ini")
    with open(path, encoding="utf-8", errors="replace") as f:
        m = re.search(r"^\s*rseed_oprm\s*=\s*([0-9]+)", f.read(), re.MULTILINE)
    return int(m.group(1)) if m else None


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join("runs", "diag", "V2S", "C1"))
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    trajectory = driver.load_summer_best_trajectory()
    if set(trajectory["perturbations"]) != {"p2", "p4", "p5", "p10"}:
        raise SystemExit("保存済み夏季軌跡の摂動集合が想定（p2, p4, p5, p10）と異なる")
    p2 = trajectory["p2_offset_min"]
    schedule = driver.make_summer_best_schedule(p2)

    keys = ("delta_E_gj", "step_seconds", "p_outdoor_kw", "p_indoor_kw", "p_hex_kw", "heatload_kw", "zone_on",
            "t_snapshot_s", "t_extra_s", "t_step_s", "n_reads", "n_present_south", "n_present_north",
            "n_tenant_south", "n_tenant_north")
    rec = {k: [] for k in keys}
    drv = driver.EpisodeDriver()
    episode_t0 = time.time()
    drv.start()
    boot_s = time.time() - episode_t0
    run = lambda coro: driver.run_coro(drv, coro, timeout=300)
    try:
        comms = drv.comms
        dt = run(drv.true_sim_datetime())
        prev_mj = run(comm.read_step_snapshot(comms))["building"]["total_energy_consumption_mj"]
        while True:
            step_t0 = time.time()
            run(schedule(drv, dt, timing.is_system_boundary(dt)))
            seconds = timing.zone_step_seconds(dt)
            dt_after = run(drv.advance(seconds))
            t0 = time.time()
            snap = run(comm.read_step_snapshot(comms))
            t_snapshot = time.time() - t0
            t0 = time.time()
            p_ou = run(comm.read_outdoor_unit_power(comms))
            p_iu = run(comm.read_indoor_unit_power(comms))
            readback = run(comm.read_setpoint_readback(comms))
            tenant = run(comm.read_tenant_thermal_sensation(comms))
            t_extra = time.time() - t0

            vent_on = driver._is_occupied_weekday_hour_8_19(dt)
            cur_mj = snap["building"]["total_energy_consumption_mj"]
            rec["delta_E_gj"].append((cur_mj - prev_mj) / 1000.0)
            prev_mj = cur_mj
            rec["step_seconds"].append(seconds)
            rec["p_outdoor_kw"].append(p_ou)
            rec["p_indoor_kw"].append(p_iu)
            rec["p_hex_kw"].append([constants.HEX_POWER_KW[area][constants.HEX_FAN_SPEED_FIXED] if vent_on else 0.0
                                    for ou, iu, tenant_name, zn, label, area in constants.ZONE_DEFS])
            rec["heatload_kw"].append([snap["zones"][l]["heatload_kw"] for l in constants.ZONE_LABELS])
            rec["zone_on"].append([driver.summer_best_zone_state(dt, (ou, iu), p2)["on"]
                                   for ou, iu, tenant_name, zn, label, area in constants.ZONE_DEFS])
            n_present = {t: tenant[t]["n_present"] for t in constants.TENANTS}
            n_tenant = {t: (constants.TENANT_OCCUPANT_COUNT[t] if n_present[t] > 0 else 0) for t in constants.TENANTS}
            reads = SNAPSHOT_READ_POINTS + 4 + 18 + 18
            for t in constants.TENANTS:
                reads += 1 + (constants.TENANT_OCCUPANT_COUNT[t] + n_present[t] if n_present[t] > 0 else 0)
            rec["t_snapshot_s"].append(t_snapshot)
            rec["t_extra_s"].append(t_extra)
            rec["t_step_s"].append(time.time() - step_t0)
            rec["n_reads"].append(reads)
            rec["n_present_south"].append(n_present["South"])
            rec["n_present_north"].append(n_present["North"])
            rec["n_tenant_south"].append(n_tenant["South"])
            rec["n_tenant_north"].append(n_tenant["North"])

            dt = dt_after
            if drv.final_dtime is not None and dt_after >= drv.final_dtime:
                run(drv.park())
                E_rep = cur_mj / 1000.0
                DR_rep = snap["building"]["averaged_dissatisfaction_rate"]
                break
    finally:
        drv.close()
    episode_s = time.time() - episode_t0

    arrays = {k: np.asarray(rec[k]) for k in ("delta_E_gj", "step_seconds", "p_outdoor_kw", "p_indoor_kw",
                                              "p_hex_kw", "heatload_kw", "zone_on")}
    np.savez(os.path.join(a.out, "steps.npz"), **arrays,
             **{k: np.asarray(rec[k]) for k in ("t_snapshot_s", "t_extra_s", "t_step_s", "n_reads",
                                                "n_present_south", "n_present_north")})
    summary = analyze(**arrays)
    summary.update({
        "n_steps": int(len(arrays["delta_E_gj"])), "E_rep": E_rep, "DR_rep": DR_rep,
        "E_rep_hex": float.hex(E_rep), "DR_rep_hex": float.hex(DR_rep),
        "eci_smooth": eci_mod.eci_smooth(E_rep, DR_rep), "eci_official": eci_mod.eci(E_rep, DR_rep),
        "recorded_trajectory_E_GJ": trajectory.get("E_GJ"), "recorded_trajectory_DR": trajectory.get("DR"),
        "rel_error_sum_delta_E_vs_E_rep": abs(summary["sum_delta_E_gj"] - E_rep) / E_rep,
        "C3_episode_wall_s": episode_s, "C3_boot_s": boot_s,
        "C3_step_wall_s_median": float(np.median(rec["t_step_s"])), "C3_step_wall_s_max": float(np.max(rec["t_step_s"])),
        "C3_snapshot_s_median": float(np.median(rec["t_snapshot_s"])),
        "C3_extra_reads_s_median": float(np.median(rec["t_extra_s"])),
        "C3_reads_per_step_median": float(np.median(rec["n_reads"])), "C3_reads_per_step_max": int(np.max(rec["n_reads"])),
        "C3_report_threshold_s": constants.EPISODE_TIME_REPORT_THRESHOLD_S,
        "C5_rseed_oprm": read_rseed_oprm(),
        "C5_tenant_occupant_count_constant": constants.TENANT_OCCUPANT_COUNT,
        "C5_n_present_max_south": int(np.max(rec["n_present_south"])),
        "C5_n_present_max_north": int(np.max(rec["n_present_north"])),
        "note_J5": "規則ベース最良軌跡のみ（現行方策の軌跡は本機に無い）。K-2 により最大値 × 3 を有効数字 2 桁へ切り上げ",
        "finished_at": runlog.now_iso(),
    })
    runlog.write_json(os.path.join(a.out, "summary.json"), summary)
    for k in ("n_steps", "E_rep", "DR_rep", "rel_error_without_indoor", "rel_error_with_indoor",
              "J1_include_indoor_unit_power", "J1_reason", "zone_step_energy_norm_gj", "raw_resid_rel",
              "C3_episode_wall_s", "C3_reads_per_step_median", "C5_rseed_oprm",
              "C5_n_present_max_south", "C5_n_present_max_north"):
        print(f"{k} = {summary.get(k)}")


if __name__ == "__main__":
    main()
