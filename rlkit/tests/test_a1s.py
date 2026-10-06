"""指示書 A1-S 手順 5: 集計の道具（rlkit/diag/a1s_tables.py）を、模擬のデータで検算（MUST-4）する。エミュレータは使わない。

模擬のデータは、本物と同じ列の名前の CSV（週 10,081 行）と、決まった値の電力・不満足率・行動から作る。電力・不満足率の値が決まっているので、
O-1 の合計と E、O-2・O-3 の合計と O-1、O-4 の合成と DR の検算が成り立つ。成績の E を書き換えると検算が通らなくなることも確かめる。
"""
import csv
import datetime
import json
import os

import numpy as np

from rlkit import constants, exogenous, reward as reward_mod
from rlkit.diag import a1s_tables as T
from rlkit.v8 import action as act8

START = datetime.datetime(1999, 7, 21, 0, 0)
ZONES = list(constants.ZONE_LABELS)
N_STEP = 336                                # 30 分刻みの 336 ステップ（模擬）
PER_MIN_KW_VRF = 10.0
PER_MIN_KW_HEX = 1.0
CO2_PPM = 500.0
RATES = {"thermal": 0.10, "draft": 0.02, "vtmp": 0.005}


def _stamp(i):
    d = START + datetime.timedelta(minutes=i)
    return [d.strftime("%Y/%m/%d"), d.strftime("%H:%M:%S")]


def _write(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_mock_raw(out_dir, label="T1", e_scale=1.0):
    raw = os.path.join(out_dir, "raw", label)
    os.makedirs(raw, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "raw", "actions"), exist_ok=True)
    k = constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0 / 60.0
    n = T.N_MIN + 1
    per_min_gj = (4 * PER_MIN_KW_VRF + 2 * PER_MIN_KW_HEX) * k
    co2z = reward_mod.dr_co2(CO2_PPM)
    comp = 1.0 - (1 - RATES["thermal"]) * (1 - RATES["draft"]) * (1 - RATES["vtmp"]) * (1 - co2z)
    exo = exogenous.load()
    W = exo.total_occupant_minutes()
    _write(os.path.join(raw, "general.csv"),
           ["date", "time", "Outdoor drybulb temperature[C]", "Outdoor humidity ratio[g/kg]", "Global horizontal radiation [W/m2]",
            "Energy consumption (Total) [GJ]", "Energy consumption (Current) [GJ/h]", "Averaged dissatisfaction[-]", "Dissatisfaction rate (All)[-]",
            "Dissatisfaction rate (Thermal comfort)[-]", "Dissatisfaction rate (Draft)[-]", "Dissatisfaction rate (Vertical temp. dif)[-]",
            "Dissatisfaction rate (CO2 level)[-]"],
           [[*_stamp(i), 25.0, 12.0, 0.0, i * per_min_gj, per_min_gj * 60, 0.0, comp, RATES["thermal"], RATES["draft"], RATES["vtmp"], co2z]
            for i in range(n)])
    _write(os.path.join(raw, "vent.csv"),
           ["date", "time", "CO2 level (South)[ppm]", "CO2 level (North)[ppm]", "Fan electricity (South)[kW]", "Fan electricity (North)[kW]"],
           [[*_stamp(i), CO2_PPM, CO2_PPM, PER_MIN_KW_HEX, PER_MIN_KW_HEX] for i in range(n)])
    head = ["date", "time"]
    for s in range(1, 5):
        head += [f"VRF{s} electricity [kW]", f"VRF{s} heat load[kW]"]
    head += ["VRF1-1 Mode"]
    _write(os.path.join(raw, "vrf.csv"), head, [[*_stamp(i), *sum(([PER_MIN_KW_VRF, -5.0] for _ in range(4)), []), "Cooling"] for i in range(n)])
    zhead = ["date", "time"] + [f"{z} drybulb temperature [CDB]" for z in ZONES] + [f"{z}_Up drybulb temperature [CDB]" for z in ZONES] \
        + [f"{z} relative humidity [%]" for z in ZONES] + [f"{z}_Up relative humidity [%]" for z in ZONES]
    _write(os.path.join(raw, "zone.csv"), zhead, [[*_stamp(i), *([26.0] * 18), *([27.0] * 18), *([50.0] * 18), *([45.0] * 18)] for i in range(n)])
    dhead = ["date", "time"] + [f"{z} dissatisfaction rate ({c})" for c in ("Thermal", "Draft", "Temperature distribution") for z in ZONES]
    _write(os.path.join(raw, "dissatisfaction.csv"), dhead,
           [[*_stamp(i), *([RATES["thermal"]] * 18), *([RATES["draft"]] * 18), *([RATES["vtmp"]] * 18)] for i in range(n)])
    _write(os.path.join(raw, "occupant.csv"), ["date", "time", "A B Clo value [clo]", "A B Thermal sensation [-]"], [[*_stamp(i), 0.6, 0.1] for i in range(3)])
    # 成績: E は電力の積分（e_scale で書き換えて検算が破れることを確かめる）、DR は建物の合成の重み付き平均
    E = float(T.N_MIN * per_min_gj) * e_scale
    DR = float((exo.total_occupants[:T.N_MIN] * comp).sum() / W)
    open(os.path.join(raw, "result.txt"), "w", encoding="utf-8").write(
        f"Energy consumption[GJ]:{E!r}\nAverage dissatisfied rate[-]:{DR!r}\nVersion:1.1.1\nrseed_o:19800614\n")
    open(os.path.join(raw, "result.szk"), "wb").write(b"mock")
    acts = np.zeros((N_STEP, act8.ACTION_DIMS), dtype=np.int8)
    acts[:, act8.dim_slice("zone_duty")] = act8.DUTY_FULL
    np.savez_compressed(os.path.join(out_dir, "raw", "actions", f"{label}_best_harvest.npz"), actions=acts)
    # 再生中の記録（30 分刻み）
    def snap(t_end_min):
        idx = min(t_end_min, T.N_MIN) - 1
        zones = {lab: {"occupant_number": float(exo.zone_occupants[idx][kk]), "dissatisfied_thermal": RATES["thermal"],
                       "dissatisfied_draft": RATES["draft"], "dissatisfied_vtmp": RATES["vtmp"], "drybulb_c": 26.0}
                 for kk, lab in enumerate(ZONES)}
        return {"zones": zones, "building": {"instantaneous_dissatisfaction_rate": comp, "total_energy_consumption_mj": 0.0},
                "co2_ppm": {"South": CO2_PPM, "North": CO2_PPM}}
    steps = []
    for t in range(N_STEP):
        a, b = 30 * t, 30 * (t + 1)
        calls = [("read_outdoor_unit_power", [5.0] * 4), ("read_indoor_unit_power", [0.5] * 18), ("read_step_snapshot", snap(b)),
                 ("read_setpoint_readback", [22.0] * 18)]
        steps.append({"t": t, "dt_start": (START + datetime.timedelta(minutes=a)).isoformat(),
                      "dt_end": (START + datetime.timedelta(minutes=b)).isoformat(), "seconds": 1800.0, "delta_E_gj": 0.0, "calls": calls})
    json.dump({"reset_calls": [("read_step_snapshot", snap(1))], "steps": steps}, open(os.path.join(raw, "captures.json"), "w", encoding="utf-8"))
    json.dump({"label": label, "E_hex": float.hex(E), "DR_hex": float.hex(DR), "E": E, "DR": DR, "ECI": 0.7, "P_E": 0.7, "P_DR": 0.7,
               "hex_match": True, "expected_E_hex": float.hex(E), "expected_DR_hex": float.hex(DR),
               "final_info": {"setpoint_override_rate": 0.0}}, open(os.path.join(raw, "replay.json"), "w", encoding="utf-8"))
    return E, DR


def test_mock_checks_pass(tmp_path):
    out = str(tmp_path / "A1S")
    E, DR = make_mock_raw(out)
    summary, checks = T.build_all(out, labels=("T1",))
    assert all(checks["_pass"]["T1"].values()), checks["_pass"]
    en = summary["T1"]["energy"]
    assert abs(en["total_csv"] - E) < 1e-6 and abs(en["other"]) < 1e-6
    assert abs(en["outdoor_est"] + en["indoor_est"] - en["vrf_csv"]) < 1e-12                       # 室外機 + 室内機 = VRF の合計（構成上）
    assert abs(en["hex_csv"] - 2 * PER_MIN_KW_HEX * T.N_MIN * constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0 / 60.0) < 1e-9
    d = summary["T1"]["dissatisfaction"]
    assert abs(d["building"]["composite"] - DR) < 1e-9 and abs(d["zone"]["composite"] - DR) < 1e-9
    assert abs(d["building"]["thermal"] - RATES["thermal"]) < 1e-9                                  # 在室者・分で重み付けした平均が、一定の率に等しい
    # O-2・O-3 の合計が O-1 に等しい
    rows = list(csv.reader(open(os.path.join(out, "tables", "O2_energy_by_weekday_T1.csv"), encoding="utf-8")))[1:]
    assert abs(sum(float(r[17]) for r in rows) - en["total_csv"]) < 1e-9
    # 行動の集計: 全ゾーン割合 1（運転）、設定温度 22 ℃、読み戻しは指令どおり（書き換え 0）
    a = summary["T1"]["actions"]
    assert sum(a["duty_counts"]["occupied"]) + sum(a["duty_counts"]["unoccupied"]) == N_STEP * 18
    assert a["override"]["overwritten"] == 0 and a["override"]["operated"] == N_STEP * 18
    assert a["remote_prohibit"]["total"] == 0 and a["bypass"]["total"] == 0
    assert os.path.exists(os.path.join(out, "results", "T1", "result.szk")) and checks["T1.O14"]["matches_P1_hex"]
    for name in ("O1_energy_week_T1.csv", "O3_energy_window_weekday_T1.csv", "O4_dissatisfaction_week_T1.csv", "O5_dissatisfaction_by_zone_T1.csv",
                 "O7_zone_state_T1.csv", "O7_co2_by_tenant_T1.csv", "O8_duty_counts_T1.csv", "O12_setpoint_overwrite_T1.csv",
                 "actions_readable_T1.csv", "O13_comparison.csv", "inventory_outputs.csv"):
        assert os.path.exists(os.path.join(out, "tables", name)), name


def test_mock_checks_fail_when_energy_is_inconsistent(tmp_path):
    out = str(tmp_path / "A1S")
    make_mock_raw(out, e_scale=1.5)                          # 成績の E を 1.5 倍にする（電力の積分と合わない）
    _summary, checks = T.build_all(out, labels=("T1",))
    assert checks["_pass"]["T1"]["O1_total_vs_E"] is False
    assert checks["_pass"]["T1"]["O2_vs_O1"] is True          # 曜日別・時刻別の合計は O-1 と一致のまま（E の書き換えの影響を受けない）
