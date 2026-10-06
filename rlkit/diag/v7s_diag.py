"""指示書 V7-S の準備（学習を伴わない）。

  prev       3-1: 走行 v2s_trial07 の最良（更新 281・組 5）の行動列を組 6 で再生し、E・DR の 16 進一致を確かめる。
                  同じ再生の途中で、運転中の室内機の風向の状態（BACnet 1112）を読み戻す（3-2）
  co2        3-3: 既存の分単位の出力（V6-S の K-11 と基準の再生）から、物質収支の式の係数を推定して検証する
  influence  3-4: V6.4-S の反事実の記録から、影響範囲の水準の ΔC を計算して判定する

使い方（リポジトリルートから）: python -m rlkit.diag.v7s_diag --step prev|co2|influence
"""
import argparse
import datetime
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from rlkit import action, comm, config as config_mod, constants, driver, exogenous, localdata, runlog
from rlkit.parallel import launch as launch_mod
from rlkit.parallel import slots

OUT_DIR = os.path.join("runs", "diag", "V7S")
PREV_BEST = os.path.join("runs", "v2s_trial07", "harvest", "best_harvest.npz")
PREV_E_HEX = "0x1.3667fdf3b645ap+2"
PREV_DR_HEX = "0x1.801f5a0000000p-3"
CONFIG_V64 = os.path.join("rlkit", "config_summer_v64.json")
DIRECTION_READ_EVERY = 25          # 3-2: 何ステップごとに風向を読み戻すか

# 3-3 の既存の記録（V6-S の K-11 と基準の再生）。推定 3 本・検証 1 本は事前登録で決めた
V6S_DIR = os.path.join("runs", "diag", "V6S")
CO2_RUNS = {"k11_a": "Low 固定", "k11_d": "High 固定", "z1_slot6": "基準の軌跡（Middle と停止）", "k11_b": "CO2 による切替"}
CO2_FIT_RUNS = ("k11_a", "k11_d", "z1_slot6")
CO2_VALID_RUN = "k11_b"
CO2_ACCEPT_MAE = 10.0
CO2_ACCEPT_MAX = 30.0
# 論文 Fig. 2・Fig. 3・表 3・表 4
CEILING_HEIGHT_M = 2.7
TENANT_FLOOR_M2 = 26.0 * 3 + 32.5 * 6
TENANT_VOLUME_M3 = TENANT_FLOOR_M2 * CEILING_HEIGHT_M
TENANT_RATED_AIRFLOW_M3H = 150.0 * 3 + 250.0 * 6

# 3-4: 論文 Fig. 2 の平面図（上から下、左から右）。同じテナントの中で壁を共有するゾーンを隣接とする
FLOOR_GRID = {"North": [["N3", "N8", "N9"], ["N2", "N6", "N7"], ["N1", "N4", "N5"]],
              "South": [["S9", "S8", "S3"], ["S7", "S6", "S2"], ["S5", "S4", "S1"]]}
INFLUENCE_MIN_ABS_DC = 0.0005
INFLUENCE_PASS_SIGN = 0.90
INFLUENCE_PASS_MEDIAN_REL = 0.30


# =====================================================================
# 3-1・3-2
# =====================================================================

def run_prev():
    os.makedirs(OUT_DIR, exist_ok=True)
    from rlkit.env import SummerEnv
    report = launch_mod.preflight(OUT_DIR, [6])
    os.environ[launch_mod.LAUNCH_LOG_ENV] = report["log_dir"]
    slot = slots.make_slot(6)
    slots.prepare_replica(slot)
    cfg = config_mod.load(CONFIG_V64)
    env = SummerEnv(include_indoor_unit_power=cfg["include_indoor_unit_power"],
                    zone_step_energy_norm_gj=cfg["zone_step_energy_norm_gj"], slot=slot, m9_enabled=cfg["m9_enabled"])
    acts = np.load(PREV_BEST)["actions"].astype(np.int64)
    st0 = action.LAYOUT["zone_state"][0]
    reads = []
    t0 = time.time()
    try:
        env.reset()
        for t in range(acts.shape[0]):
            _o, _r, done, _tr, info = env.step(acts[t])
            if done:
                break
            if t % DIRECTION_READ_EVERY == 0:
                on = [bool(acts[t, st0 + k]) for k in range(constants.N_ZONES)]
                vals = driver.run_coro(env._driver, comm.read_direction_status(env._driver.comms))
                reads.append({"step": t, "dt": info["sim_datetime"].isoformat(), "zone_on": on, "direction": vals})
        written = {f"{ou}-{iu}": getattr(env._dw, "_last", {}).get(("zone_dir_fixed", ou, iu))
                   for ou, iu, *_ in constants.ZONE_DEFS}
    finally:
        env.close()
    E, DR = info["E_rep"], info["DR_rep"]
    on_vals = [v for r in reads for v, on in zip(r["direction"], r["zone_on"]) if on]
    off_vals = [v for r in reads for v, on in zip(r["direction"], r["zone_on"]) if not on]
    sp0 = action.LAYOUT["zone_setpoint"][0]
    grid = constants.ZONE_SETPOINT_C[1]
    on_mask = acts[:, st0:st0 + constants.N_ZONES] == 1
    sp_used = acts[:, sp0:sp0 + constants.N_ZONES][on_mask]
    out = {
        "time": runlog.now_iso(), "elapsed_s": time.time() - t0,
        "3-1": {"E": E, "DR": DR, "E_hex": float.hex(E), "DR_hex": float.hex(DR),
                "expected_E_hex": PREV_E_HEX, "expected_DR_hex": PREV_DR_HEX,
                "match": float.hex(E) == PREV_E_HEX and float.hex(DR) == PREV_DR_HEX,
                "ECI": info["ECI"], "r8p_sum_rel": info["r8p_sum_rel"], "raw_resid_rel": info["raw_resid_rel"]},
        "3-2": {
            "code_direction_constant": constants.ZONE_DIRECTION_FIXED,
            "szklib_value_of_constant": 1 if constants.ZONE_DIRECTION_FIXED == "Horizontal" else None,
            "diff_writer_last_written": written,
            "readback_every_steps": DIRECTION_READ_EVERY, "n_reads": len(reads),
            "readback_values_when_on": {str(v): on_vals.count(v) for v in sorted(set(on_vals))},
            "readback_values_when_off": {str(v): off_vals.count(v) for v in sorted(set(off_vals))},
            "setpoint_grid_C": grid,
            "setpoint_used_counts_when_cooling": {str(grid[i]): int((sp_used == i).sum()) for i in range(len(grid))},
            "reads": reads,
        },
    }
    runlog.write_json(os.path.join(OUT_DIR, "prep_prev.json"), out)
    print(json.dumps({k: v for k, v in out["3-1"].items()}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out["3-2"].items() if k != "reads"}, ensure_ascii=False))
    return out


# =====================================================================
# 3-3 CO2 の物質収支
# =====================================================================

TENANTS = ("South", "North")


def _load_run(label):
    """ステップの記録（開始時刻・長さ・水準・BACnet の CO2）と、分単位の CO2（vent.csv）。"""
    rec = json.load(open(os.path.join(V6S_DIR, "raw", label + ".json"), encoding="utf-8"))
    steps = rec["steps"]
    rows = [l.rstrip("\n").split(",") for l in open(os.path.join(localdata.diag_csv_dir(os.path.join(V6S_DIR, "slot_output", label)), "vent.csv"), encoding="utf-8")][1:]
    minute = {datetime.datetime.strptime(r[0] + " " + r[1], "%Y/%m/%d %H:%M:%S"): (float(r[2]), float(r[3])) for r in rows}
    return steps, minute, rec


_BASE_ACTS = None


def _levels(step, label):
    """そのステップで適用した換気の水準（K-11 は規則の値、基準の軌跡は V5 の行動: 運転 → Middle、停止 → 停止）。"""
    global _BASE_ACTS
    vl = step.get("vent_levels")
    if vl:
        return {t: int(vl[t]) for t in TENANTS}
    if label.startswith("z1_"):
        if _BASE_ACTS is None:
            v5 = np.load(os.path.join("runs", "v2s_trial04", "harvest", "best_harvest.npz"))["actions"].astype(np.int64)
            _BASE_ACTS = action.v5_to_v6_actions(v5)
        v0 = action.LAYOUT["tenant_vent"][0]
        return {t: int(_BASE_ACTS[step["step"], v0 + j]) for j, t in enumerate(TENANTS)}
    raise ValueError(f"{label}: 換気の水準の記録が無い")


def build_pairs(label, exo):
    """ステップ t の終わりの CO2 と、ステップ t+1 の水準・長さ・平均在室者数・終わりの CO2 の組。

    CO2 は分単位の出力から取る（ステップの終わりの時刻の行）。BACnet の値も並べて返す（検証と照合に使う）。
    """
    steps, minute, rec = _load_run(label)
    out = {t: [] for t in TENANTS}
    for i in range(len(steps) - 1):
        s0, s1 = steps[i], steps[i + 1]
        start1 = datetime.datetime.fromisoformat(s1["dt"])
        end1 = start1 + datetime.timedelta(seconds=s1["seconds"])
        c_start_min = minute.get(start1)
        c_end_min = minute.get(end1)
        if c_start_min is None or c_end_min is None:
            continue
        lv = _levels(s1, label)
        mins = [start1 + datetime.timedelta(minutes=m) for m in range(int(s1["seconds"] // 60))]
        for j, tenant in enumerate(TENANTS):
            zs = [k for k, z in enumerate(constants.ZONE_DEFS) if z[2] == tenant]
            nbar = float(np.mean([exo.zone_occupants_at(m)[zs].sum() for m in mins]))
            out[tenant].append({"label": label, "step": i + 1, "dt": s1["dt"], "seconds": s1["seconds"], "level": lv[tenant],
                                "N": nbar, "c0_min": c_start_min[j], "c1_min": c_end_min[j],
                                "c0_bacnet": float(s0["co2_ppm"][tenant]), "c1_bacnet": float(s1["co2_ppm"][tenant])})
    return out, rec


def predict(p, c0, N, level, seconds):
    """C(t+Δ) = C_out + (C − C_out)·e^{−aΔ} + (b·N/a)(1 − e^{−aΔ})。a = Q_ℓ/V [1/h]、b = g/V [ppm/h/人]。"""
    a = np.asarray(p["a"])[np.asarray(level)]
    h = np.asarray(seconds, dtype=np.float64) / 3600.0
    e = np.exp(-a * h)
    return p["c_out"] + (np.asarray(c0) - p["c_out"]) * e + p["b"] * np.asarray(N) / a * (1.0 - e)


def fit_tenant(pairs):
    from scipy.optimize import least_squares
    c0 = np.array([q["c0_min"] for q in pairs])
    c1 = np.array([q["c1_min"] for q in pairs])
    N = np.array([q["N"] for q in pairs])
    lv = np.array([q["level"] for q in pairs])
    sec = np.array([q["seconds"] for q in pairs], dtype=np.float64)

    def unpack(x):
        return {"c_out": x[0], "a": np.exp(x[1:5]), "b": np.exp(x[5])}

    def resid(x):
        return predict(unpack(x), c0, N, lv, sec) - c1
    x0 = np.array([400.0, np.log(0.1), np.log(1.0), np.log(1.8), np.log(2.6), np.log(25.0)])
    r = least_squares(resid, x0, method="lm", max_nfev=20000)
    p = unpack(r.x)
    res = resid(r.x)
    return p, {"n_pairs": int(len(c1)), "rmse": float(np.sqrt(np.mean(res ** 2))), "mae": float(np.mean(np.abs(res))),
               "max_abs": float(np.max(np.abs(res))), "levels_count": {str(l): int((lv == l).sum()) for l in range(4)},
               "success": bool(r.success)}


def evaluate(p, pairs, source):
    """在室時間帯（N̄ > 0）の 1 ステップ先の予測誤差。source: "bacnet" または "min"。"""
    sel = [q for q in pairs if q["N"] > 0]
    c0 = np.array([q["c0_" + source] for q in sel])
    c1 = np.array([q["c1_" + source] for q in sel])
    pred = predict(p, c0, [q["N"] for q in sel], [q["level"] for q in sel], [q["seconds"] for q in sel])
    err = pred - c1
    worst = int(np.argmax(np.abs(err)))
    return {"n_steps": len(sel), "mae": float(np.mean(np.abs(err))), "max_abs": float(np.max(np.abs(err))),
            "bias": float(np.mean(err)), "worst": dict(sel[worst], pred=float(pred[worst])),
            "levels_count": {str(l): sum(q["level"] == l for q in sel) for l in range(4)}}


def run_co2():
    exo = exogenous.load()
    allp = {}
    meta = {}
    for label in CO2_RUNS:
        pr, rec = build_pairs(label, exo)
        allp[label] = pr
        meta[label] = {"E_hex": rec["final"]["E_hex"], "DR_hex": rec["final"]["DR_hex"], "n_steps": len(rec["steps"])}
    out = {"time": runlog.now_iso(), "runs": {k: dict(meta[k], desc=v) for k, v in CO2_RUNS.items()},
           "fit_runs": list(CO2_FIT_RUNS), "valid_run": CO2_VALID_RUN, "V_m3": TENANT_VOLUME_M3,
           "V_source": "論文 Fig. 2（テナントの床面積 273 m²）と Fig. 3（床から天井まで 2,700 mm）",
           "rated_airflow_m3h": TENANT_RATED_AIRFLOW_M3H, "tenants": {}}
    # 分単位と BACnet の CO2 の照合（ステップの終わりの時刻の行が BACnet の値と同じか）
    diffs = [q["c1_min"] - q["c1_bacnet"] for lab in CO2_RUNS for t in TENANTS for q in allp[lab][t]]
    out["minute_vs_bacnet_step_end"] = {"n": len(diffs), "max_abs": float(np.max(np.abs(diffs))),
                                        "mean_abs": float(np.mean(np.abs(diffs)))}
    accept = True
    for tenant in TENANTS:
        fit_pairs = [q for lab in CO2_FIT_RUNS for q in allp[lab][tenant]]
        p, stat = fit_tenant(fit_pairs)
        Q = [float(a * TENANT_VOLUME_M3) for a in p["a"]]
        rec = {"c_out_ppm": float(p["c_out"]), "a_per_h": [float(a) for a in p["a"]], "b_ppm_per_h_person": float(p["b"]),
               "Q_m3h_stop_low_middle_high": Q, "g_ppm_m3_per_h_person": float(p["b"] * TENANT_VOLUME_M3),
               "g_L_per_h_person": float(p["b"] * TENANT_VOLUME_M3 / 1000.0),
               "fit": stat,
               "fit_eval_occupied_bacnet": {lab: evaluate(p, allp[lab][tenant], "bacnet") for lab in CO2_FIT_RUNS},
               "valid_eval_occupied_bacnet": evaluate(p, allp[CO2_VALID_RUN][tenant], "bacnet"),
               "valid_eval_occupied_minute": evaluate(p, allp[CO2_VALID_RUN][tenant], "min"),
               "monotonic_stop_ge_low_ge_middle_ge_high": bool(p["a"][0] <= p["a"][1] <= p["a"][2] <= p["a"][3])}
        v = rec["valid_eval_occupied_bacnet"]
        rec["accept"] = v["mae"] <= CO2_ACCEPT_MAE and v["max_abs"] <= CO2_ACCEPT_MAX
        accept &= rec["accept"]
        out["tenants"][tenant] = rec
    out["accept"] = bool(accept)
    out["criteria"] = {"mae_ppm": CO2_ACCEPT_MAE, "max_ppm": CO2_ACCEPT_MAX}
    runlog.write_json(os.path.join(OUT_DIR, "co2_fit.json"), out)
    print(json.dumps({t: {k: v for k, v in r.items() if k not in ("fit_eval_occupied_bacnet",)} for t, r in out["tenants"].items()},
                     ensure_ascii=False, indent=1, default=str))
    print("minute vs bacnet:", out["minute_vs_bacnet_step_end"], "accept:", out["accept"])
    return out


# =====================================================================
# 3-4 影響範囲の水準
# =====================================================================

def influence_sets():
    """ゾーン k の影響範囲 = 自ゾーン ∪ 隣接ゾーン（論文 Fig. 2）∪ 同じ系統のゾーン。索引は constants.ZONE_DEFS の順。"""
    labels = constants.ZONE_LABELS
    idx = {l: k for k, l in enumerate(labels)}
    adj = {l: set() for l in labels}
    for grid in FLOOR_GRID.values():
        for r in range(3):
            for c in range(3):
                for dr, dc in ((0, 1), (1, 0)):
                    rr, cc = r + dr, c + dc
                    if rr < 3 and cc < 3:
                        a, b = grid[r][c], grid[rr][cc]
                        adj[a].add(b)
                        adj[b].add(a)
    sys_of = {z[4]: z[0] for z in constants.ZONE_DEFS}
    sets = []
    for l in labels:
        s = {l} | adj[l] | {m for m in labels if sys_of[m] == sys_of[l]}
        sets.append(sorted(idx[m] for m in s))
    return sets, {l: sorted(adj[l]) for l in labels}


def run_influence():
    rows = json.load(open(os.path.join("runs", "diag", "V64S", "K14", "K14p_rows.json"), encoding="utf-8"))
    sets, adj = influence_sets()
    fams = {"S": ("S", "S2"), "P": ("P",)}
    out = {"time": runlog.now_iso(), "adjacency_source": "論文 Fig. 2 の平面図（エミュレータの建物の入力データは実行形式に組み込まれていて読めない）",
           "adjacency": adj, "influence_sets": {constants.ZONE_LABELS[k]: [constants.ZONE_LABELS[j] for j in s] for k, s in enumerate(sets)},
           "residual": "反事実の ΔC_res の全額を足す", "families": {}}
    for fam, members in fams.items():
        fr = [r for r in rows if r["family"] in members]
        pop = [r for r in fr if abs(r["dC_true"]) >= INFLUENCE_MIN_ABS_DC]
        vals = []
        for r in pop:
            s = sets[r["zone"]]
            vals.append(sum(r["dC_o"][j] for j in s) + r["dC_res"])
        sign = float(np.mean([np.sign(v) == np.sign(r["dC_true"]) for v, r in zip(vals, pop)]))
        rel = float(np.median([abs(v - r["dC_true"]) / abs(r["dC_true"]) for v, r in zip(vals, pop)]))
        ok = sign >= INFLUENCE_PASS_SIGN and rel <= INFLUENCE_PASS_MEDIAN_REL
        prev = {lv: (float(np.mean([np.sign(r["levels"][lv]) == np.sign(r["dC_true"]) for r in pop])),
                     float(np.median([abs(r["levels"][lv] - r["dC_true"]) / abs(r["dC_true"]) for r in pop])))
                for lv in ("local", "system", "building")}
        out["families"][fam] = {"members": list(members), "n": len(fr), "n_excluded_small": len(fr) - len(pop),
                                "influence": {"sign_agreement": sign, "median_rel_err": rel, "passed": ok},
                                "reference_levels_sign_rel": prev,
                                "chosen": "influence" if ok else "building",
                                "mean_influence_set_size": float(np.mean([len(sets[r["zone"]]) for r in pop]))}
    runlog.write_json(os.path.join(OUT_DIR, "influence_decision.json"), out)
    print(json.dumps(out["families"], ensure_ascii=False, indent=1))
    return out


# =====================================================================
# 手順 6 書き込み層の確認
# =====================================================================

BASE_E_HEX = "0x1.a228e147ae148p+2"
BASE_DR_HEX = "0x1.863c140000000p-3"


def _replay(actions, setpoint_grid=None, slot_index=6):
    from rlkit.env import SummerEnv
    slot = slots.make_slot(slot_index)
    slots.prepare_replica(slot)
    cfg = config_mod.load(os.path.join("rlkit", "config_summer_v7.json"))
    env = SummerEnv(include_indoor_unit_power=cfg["include_indoor_unit_power"],
                    zone_step_energy_norm_gj=cfg["zone_step_energy_norm_gj"], slot=slot, m9_enabled=cfg["m9_enabled"],
                    setpoint_grid=setpoint_grid)
    t0 = time.time()
    try:
        env.reset()
        for t in range(actions.shape[0]):
            _o, _r, done, _tr, info = env.step(actions[t])
            if done:
                break
    finally:
        env.close()
    return {"E": info["E_rep"], "DR": info["DR_rep"], "E_hex": float.hex(info["E_rep"]), "DR_hex": float.hex(info["DR_rep"]),
            "ECI": info["ECI"], "P_E": info["P_E"], "P_DR": info["P_DR"], "r8p_sum_rel": info["r8p_sum_rel"],
            "raw_resid_rel": info["raw_resid_rel"], "co2_pred_obs_clip_rate": info["co2_pred_obs_clip_rate"],
            "elapsed_s": time.time() - t0}


def run_wcheck():
    """V7-S 手順 6: (1) 旧設定（設定温度の格子を V6.4-S の 20〜28 ℃に上書き。風向は C-1 で変えていないので同じ）で
    P-7 の基準の軌跡を再生し 16 進一致を確かめる。(2) 新設定で P-1 の行動列（旧格子の索引）を物理値で写して再生する。
    20・21 ℃は新しい格子に無いため 22 ℃に写す（比較の目安で合否は問わない）。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    report = launch_mod.preflight(OUT_DIR, [6])
    os.environ[launch_mod.LAUNCH_LOG_ENV] = report["log_dir"]
    old_grid = constants.ZONE_SETPOINT_C_V64
    new_grid = constants.ZONE_SETPOINT_C[1]
    sp0 = action.LAYOUT["zone_setpoint"][0]
    st0 = action.LAYOUT["zone_state"][0]
    n = constants.N_ZONES
    v5 = np.load(os.path.join("runs", "v2s_trial04", "harvest", "best_harvest.npz"))["actions"].astype(np.int64)
    base_new = action.v5_to_v6_actions(v5)                     # 新しい格子の索引
    base_old = base_new.copy()
    base_old[:, sp0:sp0 + n] = np.array([old_grid.index(new_grid[i]) for i in range(len(new_grid))])[base_new[:, sp0:sp0 + n]]
    old = _replay(base_old, setpoint_grid=old_grid)
    old["match"] = old["E_hex"] == BASE_E_HEX and old["DR_hex"] == BASE_DR_HEX
    print("旧設定の基準の再生:", json.dumps(old, ensure_ascii=False))
    out = {"time": runlog.now_iso(), "old_settings_baseline": old}
    if not old["match"]:
        runlog.write_json(os.path.join(OUT_DIR, "wcheck.json"), out)
        print("旧設定の基準の再生が 16 進で一致しない（停止して報告）")
        return out
    prev = np.load(PREV_BEST)["actions"].astype(np.int64)
    phys = np.asarray(old_grid)[prev[:, sp0:sp0 + n]]
    on = prev[:, st0:st0 + n] == constants.ZONE_STATE.index("cool")
    mapped = prev.copy()
    mapped[:, sp0:sp0 + n] = np.array([[new_grid.index(max(p, new_grid[0])) for p in row] for row in phys])
    new = _replay(mapped)
    new["clipped_to_22_cooling_zone_steps"] = int(((phys < new_grid[0]) & on).sum())
    new["clipped_to_22_all_zone_steps"] = int((phys < new_grid[0]).sum())
    new["prev_E_hex"], new["prev_DR_hex"], new["prev_ECI"] = PREV_E_HEX, PREV_DR_HEX, 0.6066094784948464
    out["new_settings_prev_best"] = new
    runlog.write_json(os.path.join(OUT_DIR, "wcheck.json"), out)
    print("新設定の P-1 の再生:", json.dumps(new, ensure_ascii=False))
    return out


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", required=True, choices=["prev", "co2", "influence", "wcheck"])
    a = p.parse_args(argv)
    runlog.check_thread_env()
    {"prev": run_prev, "co2": run_co2, "influence": run_influence, "wcheck": run_wcheck}[a.step]()


if __name__ == "__main__":
    main()
