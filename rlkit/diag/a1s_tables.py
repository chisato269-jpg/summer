"""指示書 A1-S 手順 7（C-2・C-4・C-5）: 再生で複写したエミュレータの出力と行動から、O-1〜O-14 の表を作る。エミュレータは使わない。

入力: raw/<ラベル>/ の data の 8 ファイル（general・vent・vrf・zone・dissatisfaction・occupant の各 csv、result.txt、result.szk）、
      replay.json、captures.json（再生中に環境が BACnet で読んだ値）、raw/actions/ の行動列、外生表（verify/exogenous/summer.npz）。
出力: tables/ の CSV と checks.json（検算）、summary.json（報告書用の主な値）、results/T1/（運用結果ファイル）。

時刻の割り付け（A1S-b）: 「分 m」（m = 0..10079、開始時刻 = 1999-07-21 00:00 + m 分）は、全ての分単位の出力で CSV の行 i = m + 1 に対応させる。
  確かめた根拠: 電力 [kW] の行 i は累積エネルギーの行 i−1 から i の差分と一致する（最大差 1.2e-5 GJ）。建物の不満足率・累積エネルギーの行は、
  環境が BACnet で読んだ値（時刻 t_i の読み取り）と行 i−1 が一致する（BACnet の読み取りは 1 分遅れる）。外生表の在室者数は、行と同じ時刻の
  もの（外生表の索引 i−1）を重みにすると、建物の不満足率の重み付き平均が成績の DR と 1e-6 以内で一致する。行 0（00:00）は初期状態で、重みを持たない。
CSV の 1 行目は列の名前（単位つき）。時刻はエピソードの開始からの分と、曜日・時刻の両方を付ける（C-4）。曜日は 1999-07-21（水）から。
"""
import argparse
import csv
import datetime
import json
import os
import shutil
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rlkit import constants, exogenous
from rlkit import reward as reward_mod
from rlkit.v8 import action as act8

OUT_DEFAULT = r"C:\Users\chisa\rl_local\diag\A1S"
LABELS = ("T1", "T2", "T3")
WEEKDAYS = ["水", "木", "金", "土", "日", "月", "火"]
WINDOWS = [(0, 7), (7, 10), (10, 13), (13, 16), (16, 19), (19, 24)]
WINDOW_NAMES = [f"{a}-{b}h" for a, b in WINDOWS]
START = datetime.datetime(1999, 7, 21, 0, 0)
N_MIN = 10080
KWH_TO_GJ = constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0
ZONES = list(constants.ZONE_LABELS)
N_Z = constants.N_ZONES
ZONE_SYS = [z[0] - 1 for z in constants.ZONE_DEFS]
ZONE_TEN = [0 if z[2] == "South" else 1 for z in constants.ZONE_DEFS]
TENANTS = ["South", "North"]
OVERRIDE_TOL = constants.SETPOINT_OVERRIDE_TOL_C
COMP_NAMES = ["thermal", "draft", "vtmp", "co2", "composite"]

OUTPUT_DOC = {
    "general.csv": ("(3a) general.csv", "外気・建物全体の電力量・不満足率", "O-1 の検算、O-4（建物の成分）、O-10（外気温）、O-3"),
    "occupant.csv": ("(3b) occupant.csv", "執務者別の着衣量 [clo] と温冷感 [-]（86 人 × 2 列）", "使わない（在室の判定は外生表と BACnet の在室者数）"),
    "vent.csv": ("(3c) vent.csv", "テナント別の CO2 [ppm] と換気（全熱交換器）の電力 [kW]", "O-1〜O-3（全熱交換器）、O-4（CO2 成分）、O-7（CO2）"),
    "vrf.csv": ("(3d) vrf.csv", "系統別の VRF の電力 [kW]・熱負荷 [kW]、室内機別の運転状態・温湿度・風量・設定温度", "O-1〜O-3（系統別の電力）、設定温度の確認"),
    "zone.csv": ("(3e) zone.csv", "ゾーン別の室温 [℃]・絶対湿度 [g/kg]・相対湿度 [%]（下層と上層 _Up）、屋根裏", "O-7、O-10"),
    "dissatisfaction.csv": ("記載なし（取扱説明書の出力の説明に無い）", "ゾーン別の不満足率（温冷感・気流・上下温度差の 3 成分）", "O-4・O-5・O-6（ゾーン別。A1S-e の確認つき）"),
    "result.txt": ("(3f) result.txt", "エネルギー消費量 [GJ]・平均不満足者率・計算の設定", "O-14、検算の基準（E・DR）"),
    "result.szk": ("(3g) result.szk", "結果（暗号化。取扱説明書による）", "O-14（保存のみ。中身は読まない）"),
}


# =====================================================================
# 読み込み
# =====================================================================

def read_csv(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def minute_arrays(n=N_MIN):
    m = np.arange(n)
    day = m // 1440
    hour = (m % 1440) // 60
    window = np.zeros(n, dtype=int)
    for w, (a, b) in enumerate(WINDOWS):
        window[(hour >= a) & (hour < b)] = w
    return {"m": m, "day": day, "hour": hour, "window": window}


def parse_result_txt(path):
    out = {}
    for line in open(path, encoding="utf-8").read().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


class Run:
    """1 本の再生の入力を読む。"""

    def __init__(self, raw_dir, actions_path, label, kind, exo=None):
        self.label, self.kind, self.raw = label, kind, raw_dir
        self.exo = exo or exogenous.load()
        self.rep = json.load(open(os.path.join(raw_dir, "replay.json"), encoding="utf-8"))
        self.result = parse_result_txt(os.path.join(raw_dir, "result.txt"))
        self.E_result = float(self.result["Energy consumption[GJ]"])
        self.DR_result = float(self.result["Average dissatisfied rate[-]"])
        self.actions = np.load(actions_path)["actions"].astype(np.int64)
        self.T = self.actions.shape[0]
        hg, g = read_csv(os.path.join(raw_dir, "general.csv"))
        hv, v = read_csv(os.path.join(raw_dir, "vent.csv"))
        hr, r = read_csv(os.path.join(raw_dir, "vrf.csv"))
        hz, z = read_csv(os.path.join(raw_dir, "zone.csv"))
        hd, d = read_csv(os.path.join(raw_dir, "dissatisfaction.csv"))
        self.hg, self.hv, self.hr, self.hz, self.hd = hg, hv, hr, hz, hd
        self.dts = [datetime.datetime.strptime(x[0] + " " + x[1], "%Y/%m/%d %H:%M:%S") for x in g]
        if len(self.dts) != N_MIN + 1:
            raise ValueError(f"{label}: 分単位の出力の行数が {len(self.dts)}（期待 {N_MIN + 1}）")
        self.G = np.array([[float(c) for c in x[2:]] for x in g])
        self.V = np.array([[float(c) for c in x[2:]] for x in v])
        self.Z = np.array([[float(c) for c in x[2:]] for x in z])
        self.D = np.array([[float(c) for c in x[2:]] for x in d])
        self.R_rows = r
        self.R_head = hr
        self.cap = json.load(open(os.path.join(raw_dir, "captures.json"), encoding="utf-8"))
        self.ma = minute_arrays()
        # 外生表の索引 i−1 は行 i と同じ時刻
        self.N_zone = self.exo.zone_occupants[:N_MIN]                       # (10080, 18) 分 m ↔ 行 m+1
        self.N_tot = self.exo.total_occupants[:N_MIN]
        self.W = self.exo.total_occupant_minutes()

    # ---- 列の取り出し（行 m+1 を分 m とする） ----
    def col_g(self, name):
        return self.G[1:N_MIN + 1, self.hg.index(name) - 2]

    def col_v(self, name):
        return self.V[1:N_MIN + 1, self.hv.index(name) - 2]

    def col_z(self, name):
        return self.Z[1:N_MIN + 1, self.hz.index(name) - 2]

    def col_vrf(self, name):
        """vrf.csv の数値の列（行 m+1 を分 m とする）。モードなど文字の列は対象外。"""
        i = self.R_head.index(name)
        return np.array([float(x[i]) for x in self.R_rows[1:N_MIN + 1]])

    def vrf_power_kw(self):
        return np.stack([self.col_vrf(f"VRF{s} electricity [kW]") for s in range(1, 5)], axis=1)        # (10080, 4)

    def hex_power_kw(self):
        return np.stack([self.col_v(f"Fan electricity ({t})[kW]") for t in TENANTS], axis=1)             # (10080, 2)

    def zone_rate_components(self):
        """分 m・ゾーン k の不満足率の成分 (10080, 18) を、thermal・draft・vtmp・co2・composite の順に返す。"""
        out = []
        for name in ("Thermal", "Draft", "Temperature distribution"):
            cols = [self.hd.index(f"{lab} dissatisfaction rate ({name})") - 2 for lab in ZONES]
            out.append(self.D[1:N_MIN + 1][:, cols])
        co2 = np.stack([self.col_v(f"CO2 level ({t})[ppm]") for t in TENANTS], axis=1)                  # (10080, 2)
        co2z = np.array([[reward_mod.dr_co2(float(co2[m, ZONE_TEN[k]])) for k in range(N_Z)] for m in range(N_MIN)])
        out.append(co2z)
        comp = 1.0 - (1 - out[0]) * (1 - out[1]) * (1 - out[2]) * (1 - out[3])
        out.append(comp)
        return out

    # ---- 再生中の記録 ----
    def snapshots(self):
        """決定時点 t の BACnet のスナップショット [リセット直後, 各ステップの終端]。"""
        snaps = [v for n, v in self.cap["reset_calls"] if n == "read_step_snapshot"]
        for s in self.cap["steps"]:
            snaps.append([v for n, v in s["calls"] if n == "read_step_snapshot"][-1])
        return snaps

    def step_calls(self, name):
        return [[v for n, v in s["calls"] if n == name] for s in self.cap["steps"]]

    def step_minutes(self):
        """各ステップの [開始分, 終了分)。"""
        out = []
        for s in self.cap["steps"]:
            a = int((datetime.datetime.fromisoformat(s["dt_start"]) - START).total_seconds() // 60)
            b = int((datetime.datetime.fromisoformat(s["dt_end"]) - START).total_seconds() // 60)
            out.append((a, b))
        return out


# =====================================================================
# 集計の部品
# =====================================================================

def hhmm(m):
    return f"{(m % 1440) // 60:02d}:{m % 60:02d}"


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def energy_categories(run):
    """O-1〜O-3 の分単位の電力量 [GJ]（行 m+1 の電力 × 1 分）。{名前: (10080,)}、推測の印の名前の集合を返す。
    室外機・室内機の分離は、各ステップの標本（室外機の電力・室内機の電力の平均）の室外機の割合で、VRF の系統別の電力（CSV）を割る（推測）。"""
    k = KWH_TO_GJ / 60.0
    vrf = run.vrf_power_kw() * k
    hexe = run.hex_power_kw() * k
    out, estimated = {}, set()
    for s in range(4):
        out[f"vrf_s{s + 1}"] = vrf[:, s]
    for t, name in enumerate(TENANTS):
        out[f"hex_{name}"] = hexe[:, t]
    # 標本からの分離（推測）
    outs = run.step_calls("read_outdoor_unit_power")
    ins = run.step_calls("read_indoor_unit_power")
    ratio = np.full((N_MIN, 4), np.nan)
    sample_out = np.zeros((N_MIN, 4))
    sample_in = np.zeros((N_MIN, 4))
    for (a, b), o, i in zip(run.step_minutes(), outs, ins):
        o = np.mean(np.asarray(o, dtype=float), axis=0)                                    # (4,) 標本の平均 [kW]
        iz = np.mean(np.asarray(i, dtype=float), axis=0)                                   # (18,)
        isys = np.array([sum(iz[kk] for kk in range(N_Z) if ZONE_SYS[kk] == s) for s in range(4)])
        tot = o + isys
        r = np.where(tot > 0, o / np.where(tot > 0, tot, 1.0), np.nan)
        ratio[a:b] = r
        sample_out[a:b] = o
        sample_in[a:b] = isys
    for s in range(4):
        r = np.nan_to_num(ratio[:, s], nan=0.5)
        out[f"outdoor_s{s + 1}_est"] = vrf[:, s] * r
        out[f"indoor_s{s + 1}_est"] = vrf[:, s] * (1.0 - r)
        estimated.update({f"outdoor_s{s + 1}_est", f"indoor_s{s + 1}_est"})
    return out, estimated, {"sample_out_kw": sample_out, "sample_in_kw": sample_in}


def group_sum(values, index, n):
    out = np.zeros((n,) + values.shape[1:])
    np.add.at(out, index, values)
    return out


def table_O1_O3(run, tables_dir, checks):
    cat, est, samples = energy_categories(run)
    L = run.label
    ma = run.ma
    names = [f"vrf_s{s}" for s in range(1, 5)] + [f"outdoor_s{s}_est" for s in range(1, 5)] + [f"indoor_s{s}_est" for s in range(1, 5)] \
        + [f"hex_{t}" for t in TENANTS]
    week = {n: float(cat[n].sum()) for n in names}
    vrf_sum = sum(week[f"vrf_s{s}"] for s in range(1, 5))
    hex_sum = sum(week[f"hex_{t}"] for t in TENANTS)
    out_sum = sum(week[f"outdoor_s{s}_est"] for s in range(1, 5))
    in_sum = sum(week[f"indoor_s{s}_est"] for s in range(1, 5))
    total_csv = vrf_sum + hex_sum
    other = run.E_result - total_csv
    summary = {"outdoor_est": out_sum, "indoor_est": in_sum, "vrf_csv": vrf_sum, "hex_csv": hex_sum, "other": other,
               "total_csv": total_csv, "E_result": run.E_result, "E_minus_total_csv": run.E_result - total_csv}
    checks[f"{L}.O1_total_vs_E"] = {"total_csv_GJ": total_csv, "E_result_GJ": run.E_result, "diff_GJ": run.E_result - total_csv,
                                    "rel": (run.E_result - total_csv) / run.E_result,
                                    "outdoor+indoor_vs_vrf": out_sum + in_sum - vrf_sum,
                                    "general_csv_total_GJ": float(run.G[-1, run.hg.index("Energy consumption (Total) [GJ]") - 2])}
    rows = []
    for s in range(1, 5):
        rows.append([L, f"outdoor_unit_system{s}", 2, week[f"outdoor_s{s}_est"], "推測（標本の室外機の割合で VRF の電力を分離）"])
        rows.append([L, f"indoor_units_system{s}", 2, week[f"indoor_s{s}_est"], "推測（同上）"])
        rows.append([L, f"vrf_total_system{s} (outdoor+indoor)", 1, week[f"vrf_s{s}"], "vrf.csv の系統の電力の積分"])
    for t in TENANTS:
        rows.append([L, f"total_heat_exchanger_{t}", 1, week[f"hex_{t}"], "vent.csv の換気の電力の積分"])
    rows.append([L, "other (E_result - vrf - hex)", 3, other, "成績の E との残り"])
    rows.append([L, "total (vrf + hex, CSV)", 1, total_csv, ""])
    rows.append([L, "E (result.txt)", 0, run.E_result, "成績"])
    write_csv(os.path.join(tables_dir, f"O1_energy_week_{L}.csv"),
              ["label", "category", "source_priority", "value_GJ", "note"], rows)

    # O-2 曜日別
    day = ma["day"]
    cols = names
    mat = np.stack([group_sum(cat[n][:, None], day, 7)[:, 0] for n in cols], axis=1)               # (7, ncat)
    tot_by_day = mat[:, :4].sum(axis=1) + mat[:, -2:].sum(axis=1)
    rows = []
    for d in range(7):
        rows.append([L, d, WEEKDAYS[d], *[float(x) for x in mat[d]], float(tot_by_day[d]),
                     float(tot_by_day[d] / total_csv), float(mat[d, 4:8].sum()), float(mat[d, 8:12].sum()), float(mat[d, 12:14].sum())])
    write_csv(os.path.join(tables_dir, f"O2_energy_by_weekday_{L}.csv"),
              ["label", "day_index_from_wed", "weekday", *[f"{n}_GJ" for n in cols], "total_vrf_plus_hex_GJ", "fraction_of_week",
               "outdoor_est_sum_GJ", "indoor_est_sum_GJ", "hex_sum_GJ"], rows)
    checks[f"{L}.O2_sum_vs_O1"] = {"sum_days": float(tot_by_day.sum()), "O1_total": total_csv, "diff": float(tot_by_day.sum() - total_csv)}
    # O-3 時刻別（24 時間）と、時間帯 6 × 曜日 7
    hour = ma["hour"]
    mat_h = np.stack([group_sum(cat[n][:, None], hour, 24)[:, 0] for n in cols], axis=1)
    tot_h = mat_h[:, :4].sum(axis=1) + mat_h[:, -2:].sum(axis=1)
    rows = [[L, h, *[float(x) for x in mat_h[h]], float(tot_h[h]), float(mat_h[h, 4:8].sum()), float(mat_h[h, 8:12].sum()),
             float(mat_h[h, 12:14].sum())] for h in range(24)]
    write_csv(os.path.join(tables_dir, f"O3_energy_by_hour_{L}.csv"),
              ["label", "hour_of_day", *[f"{n}_GJ" for n in cols], "total_vrf_plus_hex_GJ", "outdoor_est_sum_GJ", "indoor_est_sum_GJ",
               "hex_sum_GJ"], rows)
    checks[f"{L}.O3_sum_vs_O1"] = {"sum_hours": float(tot_h.sum()), "O1_total": total_csv, "diff": float(tot_h.sum() - total_csv)}
    wd = ma["window"] * 7 + ma["day"]
    tot_min = sum(cat[f"vrf_s{s}"] for s in range(1, 5)) + sum(cat[f"hex_{t}"] for t in TENANTS)
    out_min = sum(cat[f"outdoor_s{s}_est"] for s in range(1, 5))
    in_min = sum(cat[f"indoor_s{s}_est"] for s in range(1, 5))
    hex_min = sum(cat[f"hex_{t}"] for t in TENANTS)
    A = np.stack([tot_min, out_min, in_min, hex_min], axis=1)
    G = group_sum(A, wd, 42)
    rows = []
    for w in range(6):
        for d in range(7):
            g = G[w * 7 + d]
            rows.append([L, WINDOW_NAMES[w], d, WEEKDAYS[d], float(g[0]), float(g[1]), float(g[2]), float(g[3])])
    write_csv(os.path.join(tables_dir, f"O3_energy_window_weekday_{L}.csv"),
              ["label", "time_window", "day_index_from_wed", "weekday", "total_vrf_plus_hex_GJ", "outdoor_est_GJ", "indoor_est_GJ", "hex_GJ"], rows)
    checks[f"{L}.O3_window_sum_vs_O1"] = {"sum": float(G[:, 0].sum()), "O1_total": total_csv, "diff": float(G[:, 0].sum() - total_csv)}
    # 標本から分離した推測の妥当性: 標本の合計（室外機＋室内機）と VRF の系統別の電力（CSV）の比
    k = KWH_TO_GJ / 60.0
    sample_total = (samples["sample_out_kw"] + samples["sample_in_kw"]).sum(axis=0) * k
    vrf_total = np.array([week[f"vrf_s{s}"] for s in range(1, 5)])
    checks[f"{L}.O1_sample_vs_vrf_csv"] = {"sample_outdoor_plus_indoor_GJ_by_system": sample_total.tolist(), "vrf_csv_GJ_by_system": vrf_total.tolist(),
                                          "rel_diff_by_system": ((sample_total - vrf_total) / vrf_total).tolist()}
    return summary, cat


def table_O4_O6(run, tables_dir, checks):
    L = run.label
    ma = run.ma
    comps = run.zone_rate_components()                         # 4 成分 + 合成、各 (10080, 18)
    N = run.N_zone
    W = run.W
    # 建物（general.csv）
    names = {"thermal": "Dissatisfaction rate (Thermal comfort)[-]", "draft": "Dissatisfaction rate (Draft)[-]",
             "vtmp": "Dissatisfaction rate (Vertical temp. dif)[-]", "co2": "Dissatisfaction rate (CO2 level)[-]",
             "composite": "Dissatisfaction rate (All)[-]"}
    bld = {c: float((run.N_tot * run.col_g(names[c])).sum() / W) for c in COMP_NAMES}
    zon = {c: float((N * comps[i]).sum() / W) for i, c in enumerate(COMP_NAMES)}
    rows = [[L, c, bld[c], zon[c], zon[c] - bld[c]] for c in COMP_NAMES]
    rows.append([L, "DR (result.txt)", run.DR_result, run.DR_result, 0.0])
    write_csv(os.path.join(tables_dir, f"O4_dissatisfaction_week_{L}.csv"),
              ["label", "component", "building_general_csv_weighted_rate", "zone_level_weighted_rate", "zone_minus_building"], rows)
    checks[f"{L}.O4_composite_vs_DR"] = {"building_All": bld["composite"], "zone_composite": zon["composite"], "DR_result": run.DR_result,
                                         "building_minus_DR": bld["composite"] - run.DR_result, "zone_minus_DR": zon["composite"] - run.DR_result,
                                         "W_person_minutes": W}
    # O-5 ゾーン別
    rows = []
    for k, lab in enumerate(ZONES):
        occ = float(N[:, k].sum())
        vals = [float((N[:, k] * comps[i][:, k]).sum()) for i in range(5)]
        rows.append([L, lab, ZONE_TEN[k] and "North" or "South", occ, *vals, *[(v / occ if occ else 0.0) for v in vals]])
    write_csv(os.path.join(tables_dir, f"O5_dissatisfaction_by_zone_{L}.csv"),
              ["label", "zone", "tenant", "occupant_minutes", *[f"dissatisfied_person_minutes_{c}" for c in COMP_NAMES],
               *[f"mean_rate_{c}" for c in COMP_NAMES]], rows)
    checks[f"{L}.O5_sum_vs_O4"] = {"sum_composite_person_min_over_W": float(sum(r[8] for r in rows) / W), "O4_zone_composite": zon["composite"]}
    # O-6 時刻別・曜日別
    def by(index, n, label_fn, fname, key):
        tot_occ = group_sum(N.sum(axis=1)[:, None], index, n)[:, 0]
        arr = np.stack([group_sum((N * comps[i]).sum(axis=1)[:, None], index, n)[:, 0] for i in range(5)], axis=1)
        rows = []
        for j in range(n):
            rows.append([L, *label_fn(j), float(tot_occ[j]), *[float(x) for x in arr[j]],
                         *[float(x / tot_occ[j]) if tot_occ[j] else 0.0 for x in arr[j]]])
        write_csv(os.path.join(tables_dir, fname), ["label", *key, "occupant_minutes",
                                                    *[f"dissatisfied_person_minutes_{c}" for c in COMP_NAMES],
                                                    *[f"mean_rate_{c}" for c in COMP_NAMES]], rows)
        return arr.sum(axis=0), tot_occ.sum()
    s_h, o_h = by(ma["hour"], 24, lambda j: [j], f"O6_dissatisfaction_by_hour_{L}.csv", ["hour_of_day"])
    s_d, o_d = by(ma["day"], 7, lambda j: [j, WEEKDAYS[j]], f"O6_dissatisfaction_by_weekday_{L}.csv", ["day_index_from_wed", "weekday"])
    checks[f"{L}.O6_sums"] = {"hour_sum_composite_over_W": float(s_h[4] / W), "day_sum_composite_over_W": float(s_d[4] / W),
                              "occupant_minutes_hour": float(o_h), "occupant_minutes_day": float(o_d), "W": W}
    return {"building": bld, "zone": zon}


def table_O7(run, tables_dir, checks):
    L = run.label
    ma = run.ma
    N = run.N_zone
    rows = []
    stats = {}
    for kind, suffix in (("drybulb_c", "{z} drybulb temperature [CDB]"), ("drybulb_up_c", "{z}_Up drybulb temperature [CDB]"),
                         ("rh_pct", "{z} relative humidity [%]"), ("rh_up_pct", "{z}_Up relative humidity [%]")):
        for k, z in enumerate(ZONES):
            x = run.col_z(suffix.format(z=z))
            occ = N[:, k] > 0
            for w in range(-1, 6):
                sel = occ if w < 0 else (occ & (ma["window"] == w))
                if sel.sum() == 0:
                    rows.append([L, kind, z, "all" if w < 0 else WINDOW_NAMES[w], 0, "", "", ""])
                    continue
                rows.append([L, kind, z, "all" if w < 0 else WINDOW_NAMES[w], int(sel.sum()), float(x[sel].mean()), float(x[sel].max()),
                             float(x[sel].min())])
                if w < 0:
                    stats[f"{kind}.{z}"] = (float(x[sel].mean()), float(x[sel].max()), float(x[sel].min()))
    write_csv(os.path.join(tables_dir, f"O7_zone_state_{L}.csv"),
              ["label", "quantity", "zone", "time_window", "occupied_minutes", "mean", "max", "min"], rows)
    # CO2（テナント別）
    rows = []
    co2s = {}
    for t, name in enumerate(TENANTS):
        x = run.col_v(f"CO2 level ({name})[ppm]")
        occ = np.array([N[:, [k for k in range(N_Z) if ZONE_TEN[k] == t]].sum(axis=1) > 0]).reshape(-1)
        for scope, sel in (("occupied", occ), ("all_minutes", np.ones(N_MIN, dtype=bool))):
            row = [L, name, scope, int(sel.sum()), float(x[sel].mean()) if sel.any() else "", float(x[sel].max()) if sel.any() else "",
                   int((x[sel] > 900).sum()), int((x[sel] > 1000).sum())]
            rows.append(row)
            co2s[f"{name}.{scope}"] = row[3:]
    write_csv(os.path.join(tables_dir, f"O7_co2_by_tenant_{L}.csv"),
              ["label", "tenant", "scope", "minutes", "mean_ppm", "max_ppm", "minutes_over_900ppm", "minutes_over_1000ppm"], rows)
    return {"co2": co2s}


# =====================================================================
# 行動
# =====================================================================

def decode_v8(a):
    """a (T, 80) → 辞書（配列）。"""
    T = a.shape[0]
    duty = a[:, act8.dim_slice("zone_duty")]
    sp_idx = a[:, act8.dim_slice("zone_setpoint")]
    return {"duty_idx": duty, "setpoint_c": np.array(act8.SETPOINT_GRID)[sp_idx],
            "fan": np.array(constants.ZONE_FAN_SPEED)[a[:, act8.dim_slice("zone_fan")]], "remote": a[:, act8.dim_slice("zone_remote")],
            "evap_c": np.array(act8.EVAP_GRID)[a[:, act8.dim_slice("system_evap")]], "vent": a[:, act8.dim_slice("tenant_vent")],
            "bypass": a[:, act8.dim_slice("tenant_bypass")], "on": duty > 0}


def decode_v7(a):
    from rlkit import action as act7
    L = act7.LAYOUT
    state = a[:, L["zone_state"][0]:L["zone_state"][0] + 18]
    cool = state == constants.ZONE_STATE.index("cool")
    grid = np.array(constants.ZONE_SETPOINT_C[1])
    return {"duty_idx": np.where(cool, 3, 0), "setpoint_c": grid[a[:, L["zone_setpoint"][0]:L["zone_setpoint"][0] + 18]],
            "fan": np.array(constants.ZONE_FAN_SPEED)[a[:, L["zone_fan"][0]:L["zone_fan"][0] + 18]], "remote": None,
            "evap_c": np.array(constants.SYSTEM_EVAPORATING_C)[a[:, L["system_evap"][0]:L["system_evap"][0] + 4]],
            "vent": a[:, L["tenant_vent"][0]:L["tenant_vent"][0] + 2], "bypass": None, "on": cool}


def table_actions(run, tables_dir, checks):
    L = run.label
    dec = decode_v8(run.actions) if run.kind == "v8" else decode_v7(run.actions)
    steps = run.cap["steps"]
    snaps = run.snapshots()
    occ_dec = np.array([[snaps[t]["zones"][lab]["occupant_number"] for lab in ZONES] for t in range(run.T)])        # 決定時点の在室者数
    mins = run.step_minutes()
    start_m = np.array([a for a, _ in mins])
    window = run.ma["window"][start_m]
    day = start_m // 1440
    seconds = np.array([s["seconds"] for s in steps])
    # 行動の一覧（人が読める形）
    head = ["step", "minute_from_start", "weekday", "hhmm", "seconds"]
    for lab in ZONES:
        head += [f"{lab}_duty_fraction", f"{lab}_setpoint_C", f"{lab}_fan", f"{lab}_remote_permit", f"{lab}_occupants_at_decision"]
    head += [f"system{s}_evap_C" for s in range(1, 5)] + [f"{t}_vent_level" for t in TENANTS] + [f"{t}_bypass_on" for t in TENANTS]
    frac = [0.0, 1 / 3, 2 / 3, 1.0]
    rows = []
    for t in range(run.T):
        row = [t, int(start_m[t]), WEEKDAYS[int(day[t]) % 7], hhmm(int(start_m[t])), seconds[t]]
        for k in range(N_Z):
            row += [frac[int(dec["duty_idx"][t, k])] if run.kind == "v8" else (1.0 if dec["on"][t, k] else 0.0),
                    float(dec["setpoint_c"][t, k]) if dec["on"][t, k] else "", dec["fan"][t, k],
                    ("" if dec["remote"] is None else ("permit" if dec["remote"][t, k] == 0 else "prohibit")), occ_dec[t, k]]
        row += [int(x) for x in dec["evap_c"][t]] + [constants.VENT_LEVELS[int(x)] for x in dec["vent"][t]]
        row += ([""] * 2 if dec["bypass"] is None else [int(x) for x in dec["bypass"][t]])
        rows.append(row)
    write_csv(os.path.join(tables_dir, f"actions_readable_{L}.csv"), head, rows)
    out = {}
    occupied = occ_dec >= 1
    # O-8 運転の割合
    rows = []
    counts = {}
    for k, lab in enumerate(ZONES):
        for w in range(6):
            for d in range(7):
                for occ_flag in (True, False):
                    sel = (window == w) & (day == d) & (occupied[:, k] == occ_flag)
                    c = [int(((dec["duty_idx"][:, k] == i) & sel).sum()) for i in range(4)]
                    rows.append([L, lab, WINDOW_NAMES[w], d, WEEKDAYS[d], "occupied" if occ_flag else "unoccupied", *c])
    write_csv(os.path.join(tables_dir, f"O8_duty_counts_{L}.csv"),
              ["label", "zone", "time_window", "day_index_from_wed", "weekday", "occupancy_at_decision",
               "steps_duty_0", "steps_duty_1of3", "steps_duty_2of3", "steps_duty_1"], rows)
    for occ_flag in (True, False):
        sel = occupied == occ_flag
        counts["occupied" if occ_flag else "unoccupied"] = [int(((dec["duty_idx"] == i) & sel).sum()) for i in range(4)]
    out["duty_counts"] = counts
    # O-9 設定温度と風量
    rows = []
    grid = sorted(set(np.unique(dec["setpoint_c"][dec["on"]]).tolist()))
    sp_counts_all = {g: int(((dec["setpoint_c"] == g) & dec["on"]).sum()) for g in grid}
    fan_counts_all = {f: int(((dec["fan"] == f) & dec["on"]).sum()) for f in constants.ZONE_FAN_SPEED}
    for kind, vals in (("setpoint_C", grid), ("fan", constants.ZONE_FAN_SPEED)):
        arr = dec["setpoint_c"] if kind == "setpoint_C" else dec["fan"]
        for k, lab in enumerate(ZONES):
            for w in range(-1, 6):
                for v in vals:
                    sel = dec["on"][:, k] & (arr[:, k] == v)
                    if w >= 0:
                        sel = sel & (window == w)
                    rows.append([L, kind, lab, "all" if w < 0 else WINDOW_NAMES[w], v, int(sel.sum())])
    write_csv(os.path.join(tables_dir, f"O9_setpoint_fan_distribution_{L}.csv"),
              ["label", "quantity", "zone", "time_window", "value", "steps_operated"], rows)
    out["setpoint_counts"] = sp_counts_all
    out["fan_counts"] = fan_counts_all
    n_on = int(dec["on"].sum())
    out["setpoint_mean_c"] = float(dec["setpoint_c"][dec["on"]].mean()) if n_on else None
    # O-10 リモコンの禁止とバイパス
    outdoor = run.col_g("Outdoor drybulb temperature[C]")
    rows = []
    n_prohibit_occ = n_prohibit_unocc = 0
    if dec["remote"] is not None:
        for t in range(run.T):
            for k in range(N_Z):
                if dec["remote"][t, k] == act8.REMOTE_PROHIBIT:
                    rows.append([L, t, int(start_m[t]), WEEKDAYS[int(day[t]) % 7], hhmm(int(start_m[t])), ZONES[k], occ_dec[t, k],
                                 int(dec["duty_idx"][t, k])])
                    if occ_dec[t, k] >= 1:
                        n_prohibit_occ += 1
                    else:
                        n_prohibit_unocc += 1
    write_csv(os.path.join(tables_dir, f"O10_remote_prohibit_events_{L}.csv"),
              ["label", "step", "minute_from_start", "weekday", "hhmm", "zone", "occupants_at_decision", "duty_index"], rows)
    out["remote_prohibit"] = {"total": n_prohibit_occ + n_prohibit_unocc, "occupied": n_prohibit_occ, "unoccupied": n_prohibit_unocc,
                              "available": dec["remote"] is not None}
    rows = []
    n_bypass = n_lt = 0
    if dec["bypass"] is not None:
        for t in range(run.T):
            m = int(start_m[t])
            for ti, name in enumerate(TENANTS):
                if dec["bypass"][t, ti] == act8.BYPASS_ON:
                    zs = [k for k in range(N_Z) if ZONE_TEN[k] == ti]
                    indoor = float(np.mean([run.col_z(f"{ZONES[k]} drybulb temperature [CDB]")[m] for k in zs]))
                    out_t = float(outdoor[m])
                    rows.append([L, t, m, WEEKDAYS[int(day[t]) % 7], hhmm(m), name, out_t, indoor, out_t < indoor])
                    n_bypass += 1
                    n_lt += int(out_t < indoor)
    write_csv(os.path.join(tables_dir, f"O10_bypass_events_{L}.csv"),
              ["label", "step", "minute_from_start", "weekday", "hhmm", "tenant", "outdoor_temp_C", "indoor_mean_temp_C", "outdoor_lower_than_indoor"],
              rows)
    out["bypass"] = {"total": n_bypass, "outdoor_lower": n_lt, "available": dec["bypass"] is not None}
    # O-11 蒸発温度と換気
    rows = []
    for s in range(4):
        for w in range(6):
            for v in sorted(set(np.unique(dec["evap_c"][:, s]).tolist())):
                rows.append([L, "evap_C", f"system{s + 1}", WINDOW_NAMES[w], v, int(((dec["evap_c"][:, s] == v) & (window == w)).sum()),
                             float(seconds[(dec["evap_c"][:, s] == v) & (window == w)].sum() / 3600.0)])
    for ti, name in enumerate(TENANTS):
        for w in range(6):
            for v in range(4):
                sel = (dec["vent"][:, ti] == v) & (window == w)
                rows.append([L, "vent_level", name, WINDOW_NAMES[w], constants.VENT_LEVELS[v], int(sel.sum()), float(seconds[sel].sum() / 3600.0)])
    write_csv(os.path.join(tables_dir, f"O11_evap_vent_distribution_{L}.csv"),
              ["label", "quantity", "unit", "time_window", "value", "steps", "hours"], rows)
    out["evap_counts"] = {int(v): int((dec["evap_c"] == v).sum()) for v in np.unique(dec["evap_c"])}
    out["vent_counts"] = {constants.VENT_LEVELS[v]: int((dec["vent"] == v).sum()) for v in range(4)}
    # O-12 設定温度の書き換え
    rb = run.step_calls("read_setpoint_readback")
    readback = np.array([np.asarray(x[-1], dtype=float) for x in rb])                                     # (T, 18) ステップ終端の読み戻し
    over = (np.abs(readback - dec["setpoint_c"]) > OVERRIDE_TOL) & dec["on"]
    rows = []
    prohib = (dec["remote"] == act8.REMOTE_PROHIBIT) if dec["remote"] is not None else np.zeros_like(dec["on"])
    for k, lab in enumerate(ZONES):
        for w in range(-1, 6):
            for grp, gsel in (("all", np.ones(run.T, dtype=bool)), ("remote_prohibited", prohib[:, k]), ("not_prohibited", ~prohib[:, k])):
                sel = dec["on"][:, k] & gsel
                if w >= 0:
                    sel = sel & (window == w)
                n = int(sel.sum())
                c = int((over[:, k] & sel).sum())
                rows.append([L, lab, "all" if w < 0 else WINDOW_NAMES[w], grp, n, c, (c / n) if n else ""])
    write_csv(os.path.join(tables_dir, f"O12_setpoint_overwrite_{L}.csv"),
              ["label", "zone", "time_window", "group", "operated_steps", "overwritten_steps", "overwritten_fraction"], rows)
    n_on_tot = int(dec["on"].sum())
    out["override"] = {"operated": n_on_tot, "overwritten": int(over.sum()), "fraction": float(over.sum() / n_on_tot) if n_on_tot else None,
                       "prohibited_operated": int((dec["on"] & prohib).sum()), "prohibited_overwritten": int((over & prohib).sum()),
                       "not_prohibited_operated": int((dec["on"] & ~prohib).sum()), "not_prohibited_overwritten": int((over & ~prohib).sum())}
    checks[f"{L}.O12_vs_env_rate"] = {"computed_fraction": out["override"]["fraction"],
                                      "env_setpoint_override_rate": run.rep["final_info"].get("setpoint_override_rate")}
    return out


def semantic_checks(run):
    """MUST-3・A1S-e: 使った列の意味を、再生中に環境が BACnet で読んだ値との一致で確かめる（既存のコード・取扱説明書・実ファイルの 3 つ目の照合）。
    ステップ終端（分 m_end、時刻 t_end）の読み取りと比べる CSV の行は、列の種類ごとに、最もよく一致する行を実測で確かめて決めた:
    不満足率（dissatisfaction.csv・general.csv の率）と累積エネルギーは行 m_end − 1（BACnet のこれらの値は 1 分遅れて読める）、
    室温・CO2・VRF の電力は行 m_end（遅れなし）。行 m_end + s の s = −3..+1 の平均絶対誤差の比較は報告書に載せる（A1S-b の根拠）。"""
    snaps = run.snapshots()[1:]                                    # 各ステップの終端
    mins = run.step_minutes()
    rows_end = [b - 1 for _a, b in mins]                           # 不満足率・累積エネルギーの行（0 始まり）
    rows_now = [b for _a, b in mins]                               # 室温・CO2・VRF の電力の行
    out = {}
    def stat(name, errs):
        errs = np.asarray(errs, dtype=float)
        out[name] = {"n": int(errs.size), "max_abs": float(np.abs(errs).max()) if errs.size else None,
                     "mean_abs": float(np.abs(errs).mean()) if errs.size else None}
    # 不満足率（ゾーン別 3 成分）
    for key, col in (("dissatisfied_thermal", "Thermal"), ("dissatisfied_draft", "Draft"), ("dissatisfied_vtmp", "Temperature distribution")):
        cols = [run.hd.index(f"{lab} dissatisfaction rate ({col})") - 2 for lab in ZONES]
        errs = []
        for sn, r in zip(snaps, rows_end):
            if 0 <= r < run.D.shape[0]:
                errs.extend([sn["zones"][lab][key] - run.D[r, c] for lab, c in zip(ZONES, cols)])
        stat(f"dissatisfaction.csv:{col}_vs_BACnet", errs)
    # 建物の不満足率（瞬時）、累積エネルギー
    ci = run.hg.index("Dissatisfaction rate (All)[-]") - 2
    stat("general.csv:All_vs_BACnet_instantaneous", [sn["building"]["instantaneous_dissatisfaction_rate"] - run.G[r, ci] for sn, r in zip(snaps, rows_end)])
    ei = run.hg.index("Energy consumption (Total) [GJ]") - 2
    stat("general.csv:Total_GJ_vs_BACnet", [sn["building"]["total_energy_consumption_mj"] / 1000.0 - run.G[r, ei] for sn, r in zip(snaps, rows_end)])
    # CO2
    for t, name in enumerate(TENANTS):
        vi = run.hv.index(f"CO2 level ({name})[ppm]") - 2
        stat(f"vent.csv:CO2_{name}_vs_BACnet", [sn["co2_ppm"][name] - run.V[r, vi] for sn, r in zip(snaps, rows_now)])
    # ゾーンの室温
    errs = []
    for sn, r in zip(snaps, rows_now):
        for lab in ZONES:
            zi = run.hz.index(f"{lab} drybulb temperature [CDB]") - 2
            errs.append(sn["zones"][lab]["drybulb_c"] - run.Z[r, zi])
    stat("zone.csv:drybulb_vs_BACnet", errs)
    # VRF の系統の電力 = 室外機 + 室内機（終端の標本）
    outs = run.step_calls("read_outdoor_unit_power")
    ins = run.step_calls("read_indoor_unit_power")
    for s_i in range(4):
        vi = run.R_head.index(f"VRF{s_i + 1} electricity [kW]")
        errs = []
        for o, i, r in zip(outs, ins, rows_now):
            o_end = np.asarray(o[-1], dtype=float)
            i_end = np.asarray(i[-1], dtype=float)
            samp = o_end[s_i] + sum(i_end[k] for k in range(N_Z) if ZONE_SYS[k] == s_i)
            errs.append(samp - float(run.R_rows[r][vi]))
        stat(f"vrf.csv:VRF{s_i + 1}_electricity_vs_outdoor+indoor_BACnet", errs)
    # 在室者数: 外生表と BACnet
    errs = []
    for sn, (a, b) in zip(snaps, mins):
        idx = min(b, N_MIN) - 1 if b >= 1 else 0
        errs.extend([sn["zones"][lab]["occupant_number"] - run.exo.zone_occupants[idx][k] for k, lab in enumerate(ZONES)])
    stat("exogenous_occupants_vs_BACnet", errs)
    return out


# =====================================================================
# 運用結果ファイル・棚卸し・比較
# =====================================================================

def table_O14(run, out_dir, checks):
    dst = os.path.join(out_dir, "results", run.label)
    os.makedirs(dst, exist_ok=True)
    info = {}
    for name in ("result.txt", "result.szk"):
        src = os.path.join(run.raw, name)
        shutil.copy2(src, os.path.join(dst, name))
        import hashlib
        h = hashlib.sha256(open(os.path.join(dst, name), "rb").read()).hexdigest()
        info[name] = {"size": os.path.getsize(os.path.join(dst, name)), "sha256": h}
    res = run.result
    E, DR = float(res["Energy consumption[GJ]"]), float(res["Average dissatisfied rate[-]"])
    info["E_gj"], info["DR"] = E, DR
    info["E_hex"], info["DR_hex"] = float.hex(E), float.hex(DR)
    info["matches_P1_hex"] = bool(info["E_hex"] == run.rep["expected_E_hex"] and info["DR_hex"] == run.rep["expected_DR_hex"])
    info["settings_in_result_txt"] = {k: v for k, v in res.items() if k not in ("Energy consumption[GJ]", "Average dissatisfied rate[-]")}
    checks[f"{run.label}.O14"] = info
    return info


def inventory(raw_dir, out_dir, run):
    """4-1: 出力の棚卸し。ファイルごとの列の名前の型・単位・刻み・取扱説明書との対応・使い道。"""
    import re
    rows = []
    for name in ("general.csv", "occupant.csv", "vent.csv", "vrf.csv", "zone.csv", "dissatisfaction.csv"):
        head, body = read_csv(os.path.join(raw_dir, name))
        groups = {}
        for c in head:
            unit = re.findall(r"\[([^\]]*)\]\s*$", c)
            key = re.sub(r"\d+", "#", c)
            groups.setdefault(key, [0, c, unit[0] if unit else "（名前に単位なし）"])
            groups[key][0] += 1
        man = OUTPUT_DOC[name]
        for key, (n, ex, unit) in groups.items():
            rows.append([name, key, ex, n, unit, "60 秒（1 分）", len(body), man[0], man[1], man[2]])
    for name in ("result.txt", "result.szk"):
        man = OUTPUT_DOC[name]
        rows.append([name, "(ファイル)", "", "", "", "週に 1 つ", "", man[0], man[1], man[2]])
    write_csv(os.path.join(out_dir, "tables", "inventory_outputs.csv"),
              ["file", "column_pattern (数字を # に置換)", "example_column", "n_columns", "unit", "time_step", "n_rows", "manual_correspondence",
               "meaning", "used_for"], rows)
    return len(rows)


def build_all(out_dir=OUT_DEFAULT, labels=LABELS):
    exo = exogenous.load()
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    checks, summary = {}, {}
    kinds = {"T1": "v8", "T2": "v8", "T3": "v7"}
    runs = {}
    for L in labels:
        rd = os.path.join(out_dir, "raw", L)
        if not os.path.exists(os.path.join(rd, "replay.json")):
            continue
        runs[L] = Run(rd, os.path.join(out_dir, "raw", "actions", f"{L}_best_harvest.npz"), L, kinds[L], exo)
    if "T1" in runs:
        inventory(runs["T1"].raw, out_dir, runs["T1"])
    for L, run in runs.items():
        s = {"replay": {k: run.rep[k] for k in ("E_hex", "DR_hex", "E", "DR", "ECI", "P_E", "P_DR", "hex_match", "expected_E_hex", "expected_DR_hex")}}
        s["energy"], _cat = table_O1_O3(run, tables_dir, checks)
        s["dissatisfaction"] = table_O4_O6(run, tables_dir, checks)
        s["state"] = table_O7(run, tables_dir, checks)
        checks[f"{L}.semantic_checks"] = semantic_checks(run)
        s["actions"] = table_actions(run, tables_dir, checks)
        if L == "T1":
            s["O14"] = table_O14(run, out_dir, checks)
        summary[L] = s
    checks["_pass"] = _evaluate_checks(checks)
    with open(os.path.join(tables_dir, "checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(tables_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    write_O13(summary, tables_dir)
    return summary, checks


def _evaluate_checks(checks):
    """MUST-4 の検算の合否。許容: 電力量の合計と E の差は 1e-3 GJ（成績の丸め）、曜日・時刻の合計の差は 1e-9、
    建物の合成と DR の差は 1e-4、ゾーンの合成と DR の差は 5e-3。"""
    out = {}
    for L in LABELS:
        if f"{L}.O1_total_vs_E" not in checks:
            continue
        c = checks
        out[L] = {"O1_total_vs_E": abs(c[f"{L}.O1_total_vs_E"]["diff_GJ"]) < 1e-3,
                  "O1_outdoor_indoor_vs_vrf": abs(c[f"{L}.O1_total_vs_E"]["outdoor+indoor_vs_vrf"]) < 1e-9,
                  "O2_vs_O1": abs(c[f"{L}.O2_sum_vs_O1"]["diff"]) < 1e-9, "O3_vs_O1": abs(c[f"{L}.O3_sum_vs_O1"]["diff"]) < 1e-9,
                  "O3_window_vs_O1": abs(c[f"{L}.O3_window_sum_vs_O1"]["diff"]) < 1e-9,
                  "O4_building_vs_DR": abs(c[f"{L}.O4_composite_vs_DR"]["building_minus_DR"]) < 1e-4,
                  "O4_zone_vs_DR": abs(c[f"{L}.O4_composite_vs_DR"]["zone_minus_DR"]) < 5e-3,
                  "O5_vs_O4": abs(c[f"{L}.O5_sum_vs_O4"]["sum_composite_person_min_over_W"] - c[f"{L}.O5_sum_vs_O4"]["O4_zone_composite"]) < 1e-9,
                  "O6_vs_O4": abs(c[f"{L}.O6_sums"]["hour_sum_composite_over_W"] - c[f"{L}.O5_sum_vs_O4"]["O4_zone_composite"]) < 1e-9
                  and abs(c[f"{L}.O6_sums"]["day_sum_composite_over_W"] - c[f"{L}.O5_sum_vs_O4"]["O4_zone_composite"]) < 1e-9}
    return out


def write_O13(summary, tables_dir):
    """O-13: 3 本の比較（T-2 → T-1 の差、T-3 → T-2 の差）。"""
    keys = [("E_GJ", lambda s: s["replay"]["E"]), ("DR", lambda s: s["replay"]["DR"]), ("ECI", lambda s: s["replay"]["ECI"]),
            ("P_E", lambda s: s["replay"]["P_E"]), ("P_DR", lambda s: s["replay"]["P_DR"]),
            ("outdoor_unit_est_GJ", lambda s: s["energy"]["outdoor_est"]), ("indoor_units_est_GJ", lambda s: s["energy"]["indoor_est"]),
            ("vrf_total_csv_GJ", lambda s: s["energy"]["vrf_csv"]), ("heat_exchanger_GJ", lambda s: s["energy"]["hex_csv"]),
            ("DR_thermal_building", lambda s: s["dissatisfaction"]["building"]["thermal"]),
            ("DR_draft_building", lambda s: s["dissatisfaction"]["building"]["draft"]),
            ("DR_vtmp_building", lambda s: s["dissatisfaction"]["building"]["vtmp"]),
            ("DR_co2_building", lambda s: s["dissatisfaction"]["building"]["co2"]),
            ("DR_composite_building", lambda s: s["dissatisfaction"]["building"]["composite"]),
            ("duty_occupied_steps_zone_total", lambda s: sum(s["actions"]["duty_counts"]["occupied"])),
            ("duty_full_fraction_occupied", lambda s: s["actions"]["duty_counts"]["occupied"][3] / max(sum(s["actions"]["duty_counts"]["occupied"]), 1)),
            ("duty_off_fraction_occupied", lambda s: s["actions"]["duty_counts"]["occupied"][0] / max(sum(s["actions"]["duty_counts"]["occupied"]), 1)),
            ("duty_partial_fraction_occupied", lambda s: (s["actions"]["duty_counts"]["occupied"][1] + s["actions"]["duty_counts"]["occupied"][2])
             / max(sum(s["actions"]["duty_counts"]["occupied"]), 1)),
            ("setpoint_mean_C_operated", lambda s: s["actions"]["setpoint_mean_c"]),
            ("fan_High_fraction_operated", lambda s: s["actions"]["fan_counts"].get("High", 0) / max(sum(s["actions"]["fan_counts"].values()), 1)),
            ("remote_prohibit_zone_steps", lambda s: s["actions"]["remote_prohibit"]["total"] if s["actions"]["remote_prohibit"]["available"] else None),
            ("bypass_tenant_steps", lambda s: s["actions"]["bypass"]["total"] if s["actions"]["bypass"]["available"] else None),
            ("setpoint_overwrite_fraction", lambda s: s["actions"]["override"]["fraction"]),
            ("setpoint_overwrite_fraction_not_prohibited", lambda s: (s["actions"]["override"]["not_prohibited_overwritten"]
                                                                     / max(s["actions"]["override"]["not_prohibited_operated"], 1)))]
    rows = []
    for name, fn in keys:
        vals = {}
        for L in ("T1", "T2", "T3"):
            try:
                vals[L] = fn(summary[L]) if L in summary else None
            except Exception:
                vals[L] = None
        def diff(a, b):
            return (vals[a] - vals[b]) if (vals.get(a) is not None and vals.get(b) is not None) else ""
        rows.append([name, *[("" if vals[L] is None else vals[L]) for L in ("T1", "T2", "T3")], diff("T1", "T2"), diff("T2", "T3")])
    write_csv(os.path.join(tables_dir, "O13_comparison.csv"),
              ["quantity", "T1_trial03_best", "T2_trial03_start", "T3_trial08_best", "T1_minus_T2", "T2_minus_T3"], rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    a = ap.parse_args(argv)
    summary, checks = build_all(a.out)
    print(json.dumps(checks["_pass"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
