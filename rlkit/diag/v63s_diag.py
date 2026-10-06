"""指示書 V6.3-S の R8-1・R8-2（第 3-1 節・第 6 章）。学習は行わない。

  R8-2  組 6 で基準の軌跡を再生して P-6 と 16 進で照合し、組の分単位の出力を保存する。そのうえで R-8 の n を
        R8-A（ゾーン別）と R8-B（建物）の作り方で作り、Σn と成績の DR の相対差を測って分岐を機械的に決める

使い方（リポジトリルートから）:
  python -m rlkit.diag.v63s_diag --step r8
"""
import argparse
import csv
import datetime
import json
import os
import shutil
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from rlkit import constants, exogenous, localdata, reward as reward_mod, runlog
from rlkit.diag import v62s_diag
from rlkit.parallel import launch as launch_mod
from rlkit.parallel import slots

OUT_DIR = os.path.join("runs", "diag", "V63S")
R8_ACCEPT_REL = 1e-9
MINUTE_FILES = ("dissatisfaction.csv", "general.csv", "vent.csv", "occupant.csv", "zone.csv", "vrf.csv", "result.txt")


def _read_csv(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    head = rows[0]
    dts = [datetime.datetime.strptime(r[0] + " " + r[1], "%Y/%m/%d %H:%M:%S") for r in rows[1:]]
    return head, dts, rows[1:]


def _decimals(rows, col):
    """列の値の小数点以下の桁数（最大）。"""
    d = 0
    for r in rows:
        v = r[col]
        if "." in v:
            d = max(d, len(v.split(".")[1]))
    return d


def r8_candidates(data_dir, step_starts, official_DR):
    """R8-A・R8-B の n を作り、Σn と成績の DR の相対差を返す。"""
    exo = exogenous.load()
    W = exo.total_occupant_minutes()
    hd, dts, rows = _read_csv(os.path.join(data_dir, "dissatisfaction.csv"))
    hg, dtg, rowsg = _read_csv(os.path.join(data_dir, "general.csv"))
    hv, dtv, rowsv = _read_csv(os.path.join(data_dir, "vent.csv"))
    assert dts == dtg == dtv
    n_min = len(dts)
    occ = np.array([exo.zone_occupants_at(d) for d in dts])                       # (M, 18) 外生表
    comp = {}
    for key, name in (("thermal", "Thermal"), ("draft", "Draft"), ("vtmp", "Temperature distribution")):
        cols = [hd.index(f"{lab} dissatisfaction rate ({name})") for lab in constants.ZONE_LABELS]
        comp[key] = np.array([[float(r[c]) for c in cols] for r in rows])
    co2_cols = {"South": hv.index("CO2 level (South)[ppm]"), "North": hv.index("CO2 level (North)[ppm]")}
    co2 = np.array([[reward_mod.dr_co2(float(r[co2_cols[z[2]]])) for z in constants.ZONE_DEFS] for r in rowsv])
    zone_rate = 1.0 - (1 - comp["thermal"]) * (1 - comp["draft"]) * (1 - comp["vtmp"]) * (1 - co2)
    D_zone = occ * zone_rate                                                      # (M, 18) 不満足者数
    i_all = hg.index("Dissatisfaction rate (All)[-]")
    i_avg = hg.index("Averaged dissatisfaction[-]")
    rate_all = np.array([float(r[i_all]) for r in rowsg])
    avg = np.array([float(r[i_avg]) for r in rowsg])
    total_occ = np.array([exo.total_occupants_at(d) for d in dts])
    # 分をステップへ割り当てる（ステップ i は (start_i, start_i+1] の分。0 分の行は最初のステップに含めない）
    starts = np.array([s.timestamp() for s in step_starts])
    sidx = np.searchsorted(starts, np.array([d.timestamp() for d in dts]), side="left") - 1
    T = len(step_starts)
    n_A = np.zeros((T, constants.N_ZONES))
    n_B = np.zeros(T)
    for m in range(n_min):
        if 0 <= sidx[m] < T:
            n_A[sidx[m]] += D_zone[m] / W
            n_B[sidx[m]] += rate_all[m] * total_occ[m] / W
    out = {
        "W_exogenous": W, "minutes": n_min,
        "column_decimals": {"dissatisfaction.csv": _decimals(rows[:2000], 2), "general.csv(All)": _decimals(rowsg[:2000], i_all),
                            "general.csv(Averaged)": _decimals(rowsg[:2000], i_avg), "vent.csv(CO2)": _decimals(rowsv[:2000], co2_cols["South"])},
        "R8A_sum_n": float(n_A.sum()), "R8A_rel_diff": float(abs(n_A.sum() - official_DR) / official_DR),
        "R8A_negative": bool((n_A < 0).any()),
        "R8B_sum_n": float(n_B.sum()), "R8B_rel_diff": float(abs(n_B.sum() - official_DR) / official_DR),
        "general_averaged_last": float(avg[-1]), "general_averaged_last_rel_diff": float(abs(avg[-1] - official_DR) / official_DR),
        "official_DR": official_DR,
    }
    return out, n_A, n_B


def run_r8():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = launch_mod.preflight(OUT_DIR, [6])
    os.environ[launch_mod.LAUNCH_LOG_ENV] = report["log_dir"]
    slot = slots.make_slot(6)
    slots.prepare_replica(slot)
    v62s_diag.CONFIG = os.path.join("rlkit", "config_summer_v61.json")
    t0 = time.time()
    rec = v62s_diag.replay_record(6, v62s_diag.baseline_actions())
    dst = localdata.diag_csv_dir(os.path.join(OUT_DIR, "R8_baseline_output"))      # OneDrive の外
    os.makedirs(dst, exist_ok=True)
    for name in MINUTE_FILES:
        shutil.copy2(os.path.join(slot.data_dir, name), os.path.join(dst, name))
    np.savez_compressed(os.path.join(OUT_DIR, "R8_baseline_steps.npz"), e=rec["e"], n=rec["n"], dE=rec["dE"], dt=rec["dt"],
                        E=rec["E"], DR=rec["DR"])
    match = rec["E_hex"] == v62s_diag.BASELINE_E_HEX and rec["DR_hex"] == v62s_diag.BASELINE_DR_HEX
    steps = [datetime.datetime.fromisoformat(s) for s in rec["dt"]]
    res, n_A, n_B = r8_candidates(dst, steps, rec["DR"])
    res.update({"replay_E_hex": rec["E_hex"], "replay_DR_hex": rec["DR_hex"], "hex_match": match,
                "endpoint_sum_n": float(rec["n"].sum()),
                "endpoint_rel_diff": float(abs(rec["n"].sum() - rec["DR"]) / rec["DR"]), "elapsed_s": time.time() - t0})
    if res["R8A_rel_diff"] <= R8_ACCEPT_REL and not res["R8A_negative"]:
        res["branch"] = "R8-A"
    elif res["R8B_rel_diff"] <= R8_ACCEPT_REL:
        res["branch"] = "R8-B"
    else:
        res["branch"] = "R8-C"
    np.savez_compressed(os.path.join(OUT_DIR, "R8_candidates.npz"), n_A=n_A, n_B=n_B)
    runlog.write_json(os.path.join(OUT_DIR, "R8_decision.json"), res)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", required=True, choices=["r8"])
    p.parse_args(argv)
    runlog.check_thread_env()
    run_r8()


if __name__ == "__main__":
    main()
