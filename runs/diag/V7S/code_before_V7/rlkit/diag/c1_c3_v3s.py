"""段階 C の診断 C-1・C-2・C-3（指示書 V3-S 第 6-3 節）。学習は行わない。

規則ベース最良軌跡（verify/raw/H1/best_summer_trajectory_v2.json、夏季）を原本の Shizuku2 で 1 本再生し、
第 2-4 節で改訂した観測を毎ステップ組み立てる。J-2（ステップ内 4 点標本化）は実現できないため、電力は
J-5 に従い従来どおりステップ終端の 1 点で読む。

C-1（J-3）: E・DR の 16 進表記が P-2（V2-S C-1 の E = 4.58734814453125、DR = 0.1848166584968567）と
    一致するか。観測の正規化を変えても環境の状態遷移は変わらないため、一致するはずである。
C-2（J-4）: 比率法の raw_resid の週間相対値。終端 1 点のままなので V2-S C-4 の −0.700% と一致するはずである。
C-3: 改訂後の観測クリップ率 5 種（+25、+29、CO2、負荷率、進行率）。

使い方（リポジトリルートから）: python -m rlkit.diag.c1_c3_v3s [--out runs/diag/V3S/C1]
"""
import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from rlkit import comm, config as config_mod, constants, driver, eci as eci_mod, observation, runlog, timing
from rlkit import reward as reward_mod

# P-2: V2-S C-1 の再生値（決定論性の回帰確認の基準）
P2_E_HEX = "0x1.25971cac08312p+2"
P2_DR_HEX = "0x1.7a81280000000p-3"
P2_E_GJ = 4.58734814453125
P2_DR = 0.1848166584968567
# V2-S C-4 の raw_resid 週間相対値（終端 1 点標本化、室内機電力を含む側）
V2S_RAW_RESID_REL = -0.007004229323533928


def decoded_from_rule(dt, p2_offset_min):
    """規則ベース最良軌跡のその時刻の運転内容を、action.decode_action と同じ形に整える。"""
    zones = {}
    for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS:
        st = driver.summer_best_zone_state(dt, (ou, iu), p2_offset_min)
        zones[label] = {"state": "cool" if st["on"] else "off",
                        "setpoint": st["setpoint"] if st["on"] else None,
                        "fan_speed": st["fan_speed"] if st["on"] else constants.ZONE_FAN_SPEED[0]}
    systems = {ou: {"evap_c": driver.SUMMER_EVAP_BASE_C} for ou in range(1, 5)}
    vent_on = driver._is_occupied_weekday_hour_8_19(dt)
    return {"zones": zones, "systems": systems, "tenants": {t: vent_on for t in constants.TENANTS}}


def zone_dr_and_hex(snap, dt):
    """ゾーン別の合成不満足率と、全熱交換器電力（再生スケジュールは平日 08:00–19:00 に中風量で運転）。"""
    vent_on = driver._is_occupied_weekday_hour_8_19(dt)
    dr, hex_kw = [], []
    for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS:
        z = snap["zones"][label]
        dr.append(reward_mod.combine_dr_components(z["dissatisfied_thermal"], z["dissatisfied_draft"],
                                                   z["dissatisfied_vtmp"], reward_mod.dr_co2(snap["co2_ppm"][tenant])))
        hex_kw.append(constants.HEX_POWER_KW[area][constants.HEX_FAN_SPEED_FIXED] if vent_on else 0.0)
    return np.asarray(dr), hex_kw


def clip_rate_summary(clip_counts, n_obs):
    """5 種のクリップ率（指示書 V3-S 第 2-4 節）と、率が正の全項目。"""
    rate = np.asarray(clip_counts, dtype=np.float64) / max(n_obs, 1)
    obs25 = np.array([rate[observation.zone_base(k) + 25] for k in range(constants.N_ZONES)])
    obs29 = np.array([rate[observation.zone_base(k) + 29] for k in range(constants.N_ZONES)])
    co2 = rate[observation.IDX_CO2:observation.IDX_CO2 + 4]
    load = np.array([rate[observation.IDX_SYSTEM + 5 * s + 3] for s in range(4)])
    return {
        "n_observations": int(n_obs),
        "obs25_zone_step_energy": {"mean": float(obs25.mean()), "max": float(obs25.max()),
                                   "per_zone": [float(x) for x in obs25]},
        "obs29_setpoint_readback": {"mean": float(obs29.mean()), "max": float(obs29.max()),
                                    "per_zone": [float(x) for x in obs29]},
        "co2_40_43": {"mean": float(co2.mean()), "max": float(co2.max()), "per_index": [float(x) for x in co2]},
        "system_load_47_52_57_62": {"mean": float(load.mean()), "max": float(load.max()),
                                    "per_system": [float(x) for x in load]},
        "progress_10": float(rate[10]),
        "obs25_report_threshold": constants.OBS25_CLIP_RATE_REPORT_THRESHOLD,
        "obs29_report_threshold": constants.OBS29_CLIP_REPORT_THRESHOLD,
        "all_nonzero": {int(i): float(rate[i]) for i in np.flatnonzero(rate > 0)},
    }


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join("runs", "diag", "V3S", "C1"))
    p.add_argument("--config", default=os.path.join("rlkit", "config_summer_v2.json"))
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cfg = config_mod.load(a.config)
    include_indoor = cfg["include_indoor_unit_power"]
    energy_norm = cfg["zone_step_energy_norm_gj"]

    trajectory = driver.load_summer_best_trajectory()
    p2 = trajectory["p2_offset_min"]
    schedule = driver.make_summer_best_schedule(p2)

    from rlkit import exogenous
    ob = observation.ObservationBuilder(exogenous.load(), energy_norm)
    clip_counts = np.zeros(constants.OBSERVATION_DIM, dtype=np.int64)
    n_obs = 0
    sum_delta_E = sum_raw_resid = sum_header = 0.0
    sum_zone_energy = 0.0
    step_seconds_list, raw_resid_list = [], []
    zone_step_energy_max = 0.0

    drv = driver.EpisodeDriver()
    episode_t0 = time.time()
    drv.start()
    run = lambda coro: driver.run_coro(drv, coro, timeout=300)
    try:
        comms = drv.comms
        dt = run(drv.true_sim_datetime())
        snap = run(comm.read_step_snapshot(comms))
        readback = run(comm.read_setpoint_readback(comms))
        tenant_thermal = run(comm.read_tenant_thermal_sensation(comms))
        prev_mj = snap["building"]["total_energy_consumption_mj"]
        # エピソード最初の観測（環境の reset と同じ手順）
        _, raw = ob.build(dt, 0, 1, snap, decoded_from_rule(dt, p2), None, None, tenant_thermal, readback)
        clip_counts += observation.clipped_entries(raw)
        n_obs += 1
        step_index = 0
        while True:
            decoded = decoded_from_rule(dt, p2)
            run(schedule(drv, dt, timing.is_system_boundary(dt)))
            seconds = timing.zone_step_seconds(dt)
            dt_after = run(drv.advance(seconds))

            snap = run(comm.read_step_snapshot(comms))
            p_ou = run(comm.read_outdoor_unit_power(comms))
            p_iu = run(comm.read_indoor_unit_power(comms)) if include_indoor else None
            readback = run(comm.read_setpoint_readback(comms))
            tenant_thermal = run(comm.read_tenant_thermal_sensation(comms))

            cur_mj = snap["building"]["total_energy_consumption_mj"]
            delta_E = (cur_mj - prev_mj) / 1000.0
            prev_mj = cur_mj
            zone_dr, p_hex = zone_dr_and_hex(snap, dt)
            heatload = [snap["zones"][l]["heatload_kw"] for l in constants.ZONE_LABELS]
            zone_on = np.array([decoded["zones"][l]["state"] != "off" for l in constants.ZONE_LABELS])
            split = reward_mod.zone_energy_split(delta_E, p_ou, p_iu, p_hex, heatload, zone_on, seconds)

            sum_delta_E += delta_E
            sum_zone_energy += float(split["e_gj"].sum())
            sum_header += float(split["e_header_gj"])
            sum_raw_resid += float(split["raw_resid_gj"])
            raw_resid_list.append(float(split["raw_resid_gj"]))
            step_seconds_list.append(seconds)
            zone_step_energy_max = max(zone_step_energy_max, float(split["e_gj"].max()))

            step_index += 1
            _, raw = ob.build(dt_after, step_index, 1, snap, decoded, split["e_gj"], zone_dr,
                              tenant_thermal, readback)
            clip_counts += observation.clipped_entries(raw)
            n_obs += 1

            dt = dt_after
            if drv.final_dtime is not None and dt_after >= drv.final_dtime:
                run(drv.park())
                E_rep = cur_mj / 1000.0
                DR_rep = snap["building"]["averaged_dissatisfaction_rate"]
                break
    finally:
        drv.close()
    episode_s = time.time() - episode_t0

    determinism = {
        "n_steps": step_index, "E_rep": E_rep, "DR_rep": DR_rep,
        "E_rep_hex": float.hex(E_rep), "DR_rep_hex": float.hex(DR_rep),
        "P2_E_hex": P2_E_HEX, "P2_DR_hex": P2_DR_HEX, "P2_E_GJ": P2_E_GJ, "P2_DR": P2_DR,
        "E_hex_match": float.hex(E_rep) == P2_E_HEX, "DR_hex_match": float.hex(DR_rep) == P2_DR_HEX,
        "eci_official": eci_mod.eci(E_rep, DR_rep), "eci_smooth": eci_mod.eci_smooth(E_rep, DR_rep),
        "power_sampling": "end_of_step_single_point",
        "J2_implemented": False,
        "J2_reason": ("エミュレータの内部時間刻みが 60 秒のため、900 秒ステップを 4 等分する 225 秒で区切れない"
                      "（advance(225) は実測 240 秒進み、4 区切りで 960 秒になる）。指示書 V3-S J-5 に従い終端 1 点を維持した"),
        "episode_wall_s": episode_s, "include_indoor_unit_power": include_indoor,
        "finished_at": runlog.now_iso(),
    }
    raw_resid = {
        "sum_delta_E_gj": sum_delta_E, "raw_resid_sum_gj": sum_raw_resid,
        "raw_resid_rel": sum_raw_resid / sum_delta_E if sum_delta_E else None,
        "raw_resid_rel_v2s_C4_end_of_step": V2S_RAW_RESID_REL,
        "raw_resid_rel_900s_steps": float(np.sum([r for r, s in zip(raw_resid_list, step_seconds_list) if s == 900])
                                          / sum_delta_E) if sum_delta_E else None,
        "raw_resid_rel_3600s_steps": float(np.sum([r for r, s in zip(raw_resid_list, step_seconds_list) if s != 900])
                                           / sum_delta_E) if sum_delta_E else None,
        "sum_zone_energy_gj": sum_zone_energy, "sum_header_energy_gj": sum_header,
        "header_energy_fraction": sum_header / sum_delta_E if sum_delta_E else None,
        "identity_residual_gj": sum_delta_E - (sum_zone_energy + sum_header),
        "delta_E_vs_E_rep_rel": abs(sum_delta_E - E_rep) / E_rep,
        "zone_step_energy_max_gj": zone_step_energy_max,
        "zone_step_energy_norm_gj": energy_norm,
        "raw_resid_target_abs": 0.01, "finished_at": runlog.now_iso(),
    }
    clip = clip_rate_summary(clip_counts, n_obs)
    clip["finished_at"] = runlog.now_iso()

    runlog.write_json(os.path.join(a.out, "determinism.json"), determinism)
    runlog.write_json(os.path.join(a.out, "raw_resid.json"), raw_resid)
    runlog.write_json(os.path.join(a.out, "clip_rates.json"), clip)
    print(f"C-1 E={E_rep} ({float.hex(E_rep)}) match={determinism['E_hex_match']}")
    print(f"C-1 DR={DR_rep} ({float.hex(DR_rep)}) match={determinism['DR_hex_match']}")
    print(f"C-2 raw_resid_rel={raw_resid['raw_resid_rel']}（V2-S C-4: {V2S_RAW_RESID_REL}）")
    print(f"C-3 +25 max={clip['obs25_zone_step_energy']['max']}, +29 max={clip['obs29_setpoint_readback']['max']}, "
          f"CO2 max={clip['co2_40_43']['max']}, load max={clip['system_load_47_52_57_62']['max']}, "
          f"progress={clip['progress_10']}")


if __name__ == "__main__":
    main()
