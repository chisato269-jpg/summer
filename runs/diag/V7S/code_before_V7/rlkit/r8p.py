"""rlkit.r8p — R-8'：建物の累積平均の差分による不満足の項（指示書 V6.4-S 第 2 章）。

建物のステップ別の値
  n_t = (A_t·O_t − A_{t−1}·O_{t−1}) / W       （A_0·O_0 = 0、W = O_T）
    A_t: ステップ t の終わりに BACnet で読んだ Averaged dissatisfaction rate
    O_t: 外生表の建物の在室者数を、外生表の時刻が「A_t の読み取り時刻 + δ」以下の分について足した値
ゾーンへの配分
  n_k,t = n_t·s_k,t、s_k,t = Ã_k,t / Σ_j Ã_j,t、Ã_k,t = Σ_{m∈t} N_k,m·D̃_k,m
    N_k,m: 外生表のゾーンの在室者数、D̃_k,m: dissatisfaction.csv の 3 成分と vent.csv の CO2 から作る成分の合成
  Σ_j Ã_j,t = 0 のステップは在室者数の比、在室者のいないステップは 18 ゾーンに等分
R8-B のステップ別の値 n_t^B = Σ_{m∈t} All_m·N_m / W（general.csv の Dissatisfaction rate (All) × 建物の在室者数）。
分の行はステップ [開始, 終わり) に割り当てる。出力の時刻の列は「その分の始まり」を指す。V6.3-S で置いた
(開始, 終わり] は 1 分ずれていた（BACnet の A_t の差分との週の L1 距離が 3.5e-3 → 3.3e-5 になることで確認。V64S-a）。
"""
import csv
import datetime
import os

import numpy as np

from rlkit import constants, exogenous, reward as reward_mod

OFFSETS_MIN = (-1, 0, 1)
A_WEEK_REL_TOL = 1e-9
A_L1_MAX = 1e-3
A_MIN_N = -1e-7
B_WEEK_REL_TOL = 1e-6
MONOTONIC_DROP = 1e-6
S11_TOL = {"R8'-A": 1e-9, "R8'-B": 1e-5}


def _read_csv(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    head = rows[0]
    dts = [datetime.datetime.strptime(r[0] + " " + r[1], "%Y/%m/%d %H:%M:%S") for r in rows[1:]]
    return head, dts, rows[1:]


def load_minute_data(data_dir):
    """組の分単位の出力から、分の時刻・ゾーン別の合成の不満足率 D̃_k,m・建物の All の率を読む。"""
    hd, dts, rows = _read_csv(os.path.join(data_dir, "dissatisfaction.csv"))
    hg, dtg, rowsg = _read_csv(os.path.join(data_dir, "general.csv"))
    hv, dtv, rowsv = _read_csv(os.path.join(data_dir, "vent.csv"))
    if not (dts == dtg == dtv):
        raise ValueError("分単位の出力の時刻がファイル間で一致しない")
    comp = []
    for name in ("Thermal", "Draft", "Temperature distribution"):
        cols = [hd.index(f"{lab} dissatisfaction rate ({name})") for lab in constants.ZONE_LABELS]
        comp.append(np.array([[float(r[c]) for c in cols] for r in rows]))
    co2_col = {"South": hv.index("CO2 level (South)[ppm]"), "North": hv.index("CO2 level (North)[ppm]")}
    co2 = np.array([[reward_mod.dr_co2(float(r[co2_col[z[2]]])) for z in constants.ZONE_DEFS] for r in rowsv])
    rate = 1.0 - (1 - comp[0]) * (1 - comp[1]) * (1 - comp[2]) * (1 - co2)
    i_all = hg.index("Dissatisfaction rate (All)[-]")
    rate_all = np.array([float(r[i_all]) for r in rowsg])
    return {"dt": dts, "zone_rate": rate, "rate_all": rate_all}


def minute_step_index(minute_dts, step_starts, final_end):
    """分の行 → ステップの索引。ステップ t は [開始_t, 開始_{t+1})、最後は [開始_T, final_end)。範囲外は −1。"""
    starts = np.array([s.timestamp() for s in step_starts])
    ts = np.array([d.timestamp() for d in minute_dts])
    idx = np.searchsorted(starts, ts, side="right") - 1
    idx[ts >= final_end.timestamp()] = -1
    return idx


def cumulative_occupants(read_times, delta_min, exo=None):
    """O_t: 外生表の建物の在室者数を、外生表の時刻 ≤ 読み取り時刻 + δ の分について足した値。"""
    exo = exo or exogenous.load()
    cum = np.concatenate([[0.0], np.cumsum(exo.total_occupants)])
    times = np.array([d.timestamp() for d in exo.datetimes])
    out = []
    for t in read_times:
        lim = (t + datetime.timedelta(minutes=delta_min)).timestamp()
        out.append(cum[int(np.searchsorted(times, lim, side="right"))])
    return np.array(out, dtype=np.float64)


def building_n(A, read_times, delta_min, exo=None):
    O = cumulative_occupants(read_times, delta_min, exo)
    W = O[-1]
    prod = np.asarray(A, dtype=np.float64) * O
    return np.diff(np.concatenate([[0.0], prod])) / W, prod, W


def building_n_B(minute, step_starts, final_end, exo=None):
    exo = exo or exogenous.load()
    W = exo.total_occupant_minutes()
    occ = np.array([exo.total_occupants_at(d) for d in minute["dt"]])
    sidx = minute_step_index(minute["dt"], step_starts, final_end)
    n = np.zeros(len(step_starts))
    for m, i in enumerate(sidx):
        if 0 <= i < len(step_starts):
            n[i] += minute["rate_all"][m] * occ[m] / W
    return n


def zone_shares(minute, step_starts, final_end, exo=None):
    """s_k,t（18 ゾーンの和が 1）と、在室者のいないステップの旗。"""
    exo = exo or exogenous.load()
    N = np.array([exo.zone_occupants_at(d) for d in minute["dt"]])
    sidx = minute_step_index(minute["dt"], step_starts, final_end)
    T = len(step_starts)
    Ad = np.zeros((T, constants.N_ZONES))
    Nt = np.zeros((T, constants.N_ZONES))
    for m, i in enumerate(sidx):
        if 0 <= i < T:
            Ad[i] += N[m] * minute["zone_rate"][m]
            Nt[i] += N[m]
    shares = np.full((T, constants.N_ZONES), 1.0 / constants.N_ZONES)
    empty = Nt.sum(axis=1) <= 0
    for i in range(T):
        if Ad[i].sum() > 0:
            shares[i] = Ad[i] / Ad[i].sum()
        elif Nt[i].sum() > 0:
            shares[i] = Nt[i] / Nt[i].sum()
    return shares, empty


def evaluate(A, read_times, step_starts, minute, exo=None):
    """R8'-A・B・C の判定に要る量を δ ごとに計算し、分岐を決める（第 2-2 節）。"""
    exo = exo or exogenous.load()
    A = np.asarray(A, dtype=np.float64)
    A_T = A[-1]
    nB = building_n_B(minute, step_starts, read_times[-1], exo)
    out = {"A_T": A_T, "n_B_week_rel": float(abs(nB.sum() - A_T) / A_T), "by_delta": {}}
    best = None
    for d in OFFSETS_MIN:
        n, prod, W = building_n(A, read_times, d, exo)
        drops = np.diff(prod)
        rec = {"W": W, "week_rel": float(abs(n.sum() - A_T) / A_T), "L1": float(np.abs(n - nB).sum() / A_T),
               "min_n": float(n.min()), "max_drop": float(max(0.0, -drops.min())) if drops.size else 0.0,
               "n_steps_drop_gt_tol": int((drops < -MONOTONIC_DROP).sum())}
        out["by_delta"][str(d)] = rec
        if best is None or rec["L1"] < out["by_delta"][str(best)]["L1"]:
            best = d
    out["delta"] = best
    b = out["by_delta"][str(best)]
    out["P4_monotonic"] = b["n_steps_drop_gt_tol"] == 0          # 採った δ で A_t·O_t が 1e-6 を超えて減らない
    condA = {"week_rel": b["week_rel"] <= A_WEEK_REL_TOL, "L1": b["L1"] <= A_L1_MAX, "min_n": b["min_n"] >= A_MIN_N}
    out["conditions_A"] = condA
    if out["P4_monotonic"] and all(condA.values()):
        out["branch"] = "R8'-A"
    elif out["n_B_week_rel"] <= B_WEEK_REL_TOL:
        out["branch"] = "R8'-B"
    else:
        out["branch"] = "R8'-C"
    return out


def zone_n(branch, delta, A, read_times, step_starts, minute, exo=None):
    """採った分岐で n_k,t (T, 18) と建物の n_t を作る。"""
    exo = exo or exogenous.load()
    if branch == "R8'-A":
        n, _, _ = building_n(A, read_times, delta, exo)
    elif branch == "R8'-B":
        n = building_n_B(minute, step_starts, read_times[-1], exo)
    else:
        raise ValueError(branch)
    shares, empty = zone_shares(minute, step_starts, read_times[-1], exo)
    return n[:, None] * shares, n, empty
