"""指示書 V6.4-S の段階 Z'（R8'-1・R8'-2、K-14'、K-15）。学習は行わない。

  r8p   組 6 で基準の軌跡を再生し（16 進照合）、毎ステップの A_t と分単位の出力から R-8' の分岐を決める
  k14   V6.2-S の 76 本（生データに A_t と分単位の出力が無いため再生し直す）と家族 S2 の 36 本を 7 組で再生する
  k14analyze  R-8' の n で ΔC を作り、恒等式の検査（第 3 章）と判定表（V6.2-S 第 4-2 節、S は S+S2）で水準を決める
  k15   trial06 の SIL の保存庫 8 本の行動列を 7 組で再生し直し、R-8' の n の報酬で λ_hour を決める

使い方（リポジトリルートから）: python -m rlkit.diag.v64s_diag --step r8p|k14|k14analyze|k15
"""
import argparse
import datetime
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from rlkit import action, config as config_mod, constants, r8p, runlog
from rlkit.diag import v62s_diag
from rlkit.parallel import launch as launch_mod
from rlkit.parallel import slots

OUT_DIR = os.path.join("runs", "diag", "V64S")
K14_DIR = os.path.join(OUT_DIR, "K14")
K15_DIR = os.path.join(OUT_DIR, "K15")
CONFIG = os.path.join("rlkit", "config_summer_v61.json")
ALL_SLOTS = [0, 1, 2, 3, 4, 5, 6]
RHO_EVAL = 72.3
TOL_IDENTITY = {"R8'-A": 1e-9, "R8'-B": 1e-5}
S2_HOURS = (7, 19)


def replay_full(slot_index, actions):
    """1 本を再生し、ステップごとの e・終端 1 点の n・ΔE・開始時刻・読み取り時刻・A_t と、分単位の出力の配列を返す。"""
    from rlkit.env import SummerEnv
    cfg = config_mod.load(CONFIG)
    slot = slots.make_slot(slot_index)
    env = SummerEnv(include_indoor_unit_power=cfg["include_indoor_unit_power"],
                    zone_step_energy_norm_gj=cfg["zone_step_energy_norm_gj"], slot=slot, m9_enabled=cfg["m9_enabled"])
    T = actions.shape[0]
    e = np.zeros((T, constants.N_ZONES))
    n_end = np.zeros((T, constants.N_ZONES))
    dE = np.zeros(T)
    A = np.zeros(T)
    starts, reads = [], []
    try:
        env.reset()
        for t in range(T):
            starts.append(env._pending_dt)
            _o, _r, done, _tr, info = env.step(actions[t])
            reads.append(info["sim_datetime"])
            e[t], n_end[t], dE[t] = info["zone_energy_gj"], info["zone_dissatisfaction_increment"], info["delta_E_gj"]
            A[t] = info["averaged_dissatisfaction_rate"]
            if done:
                break
    finally:
        env.close()
    minute = r8p.load_minute_data(slot.data_dir)
    return {"e": e, "n_end": n_end, "dE": dE, "A": A, "starts": starts, "reads": reads, "minute": minute,
            "E": info["E_rep"], "DR": info["DR_rep"], "E_hex": float.hex(info["E_rep"]),
            "DR_hex": float.hex(info["DR_rep"]), "raw_resid_rel": info["raw_resid_rel"]}


def save_full(path, rec, **extra):
    np.savez_compressed(path, e=rec["e"], n_end=rec["n_end"], dE=rec["dE"], A=rec["A"],
                        starts=np.array([s.isoformat() for s in rec["starts"]]),
                        reads=np.array([s.isoformat() for s in rec["reads"]]),
                        minute_dt=np.array([d.isoformat() for d in rec["minute"]["dt"]]),
                        zone_rate=rec["minute"]["zone_rate"], rate_all=rec["minute"]["rate_all"],
                        E=rec["E"], DR=rec["DR"], **extra)


def load_full(path):
    d = np.load(path, allow_pickle=True)
    fd = datetime.datetime.fromisoformat
    return {"e": d["e"], "n_end": d["n_end"], "dE": d["dE"], "A": d["A"],
            "starts": [fd(s) for s in d["starts"]], "reads": [fd(s) for s in d["reads"]],
            "minute": {"dt": [fd(s) for s in d["minute_dt"]], "zone_rate": d["zone_rate"], "rate_all": d["rate_all"]},
            "E": float(d["E"]), "DR": float(d["DR"]), "actions": d["actions"] if "actions" in d else None}


# =====================================================================
# R8'-1・R8'-2
# =====================================================================

def run_r8p():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = launch_mod.preflight(OUT_DIR, [6])
    os.environ[launch_mod.LAUNCH_LOG_ENV] = report["log_dir"]
    slots.prepare_replica(slots.make_slot(6))
    t0 = time.time()
    rec = replay_full(6, v62s_diag.baseline_actions())
    save_full(os.path.join(OUT_DIR, "R8p_baseline.npz"), rec)
    match = rec["E_hex"] == v62s_diag.BASELINE_E_HEX and rec["DR_hex"] == v62s_diag.BASELINE_DR_HEX
    # V6.3-S で保存した分単位の出力と同じか
    prev = r8p.load_minute_data(os.path.join("runs", "diag", "V63S", "R8_baseline_output"))
    same_as_v63 = (prev["dt"] == rec["minute"]["dt"] and np.array_equal(prev["zone_rate"], rec["minute"]["zone_rate"])
                   and np.array_equal(prev["rate_all"], rec["minute"]["rate_all"]))
    ev = r8p.evaluate(rec["A"], rec["reads"], rec["starts"], rec["minute"])
    out = {"hex_match": match, "E_hex": rec["E_hex"], "DR_hex": rec["DR_hex"], "A_T_equals_DR": rec["A"][-1] == rec["DR"],
           "minute_output_same_as_V63S": bool(same_as_v63), "elapsed_s": time.time() - t0, **ev}
    if ev["branch"] != "R8'-C":
        n_kt, n_t, empty = r8p.zone_n(ev["branch"], ev["delta"], rec["A"], rec["reads"], rec["starts"], rec["minute"])
        out["zone_sum_equals_building_max_abs"] = float(np.abs(n_kt.sum(axis=1) - n_t).max())
        out["empty_steps"] = int(empty.sum())
        out["empty_steps_max_abs_n"] = float(np.abs(n_t[empty]).max()) if empty.any() else 0.0
        out["n_nonneg_min"] = float(n_kt.min())
        # ゾーンへの配分の比と終端 1 点の比の比較（在室者のいるステップ、系統別の中央値）
        end_share = rec["n_end"] / np.maximum(rec["n_end"].sum(axis=1, keepdims=True), 1e-300)
        shares = n_kt / np.where(n_t[:, None] != 0, n_t[:, None], 1.0)
        occupied = rec["n_end"].sum(axis=1) > 0
        rel = np.abs(shares - end_share)[occupied]
        out["share_vs_endpoint_abs_diff_by_system_median"] = {
            f"system{s + 1}": float(np.median(rel[:, np.array(action.SYSTEM_OF_ZONE) == s])) for s in range(4)}
        big = end_share[occupied] > 0.08
        out["share_vs_endpoint_rel_diff_large_zones_median"] = float(np.median(
            (np.abs(shares[occupied] - end_share[occupied]) / np.maximum(end_share[occupied], 1e-12))[big]))
        np.savez_compressed(os.path.join(OUT_DIR, "R8p_baseline_n.npz"), n_kt=n_kt, n_t=n_t, empty=empty)
    runlog.write_json(os.path.join(OUT_DIR, "R8p_decision.json"), out)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return out



# =====================================================================
# K-14'（V6.3-S 第 3-2 節、許容差は V6.4-S 第 3 章）
# =====================================================================

def s2_jobs(base, dts):
    """家族 S2: 2 日目の 07:00 と 19:00 に始まるステップから 1 時間、18 ゾーンそれぞれの発停を反転する。"""
    grid = constants.ZONE_SETPOINT_C[1]
    st0, sp0, fan0 = (action.LAYOUT[k][0] for k in ("zone_state", "zone_setpoint", "zone_fan"))
    cool, off = constants.ZONE_STATE.index("cool"), constants.ZONE_STATE.index("off")
    jobs = []
    for hh in S2_HOURS:
        i = v62s_diag.flip_start(dts, hh)
        n_steps = 1
        while (dts[i + n_steps] - dts[i]).total_seconds() < 3600:
            n_steps += 1
        for k in range(constants.N_ZONES):
            a = base.copy()
            to_off = int(base[i, st0 + k]) == cool
            for t in range(i, i + n_steps):
                if int(base[t, st0 + k]) == cool:
                    a[t, st0 + k] = off
                else:
                    a[t, st0 + k] = cool
                    a[t, sp0 + k] = grid.index(v62s_diag.K14_S_ON_SETPOINT_C)
                    a[t, fan0 + k] = constants.ZONE_FAN_SPEED.index(v62s_diag.K14_S_ON_FAN)
            jobs.append({"label": f"S2_{hh:02d}_{constants.ZONE_LABELS[k]}", "family": "S2", "zone": k, "start": i,
                         "n_flip_steps": n_steps, "to_off": to_off, "actions": a})
    return jobs


def _worker(slot_index, launch_dir, out_dir, jobs_q, out_q):
    os.environ[launch_mod.LAUNCH_LOG_ENV] = launch_dir
    try:
        slots.prepare_replica(slots.make_slot(slot_index))
    except Exception:
        out_q.put(("error", f"slot{slot_index}", traceback.format_exc()))
        return
    while True:
        job = jobs_q.get()
        if job is None:
            return
        try:
            t0 = time.time()
            rec = replay_full(slot_index, job["actions"])
            save_full(os.path.join(out_dir, "raw", job["label"] + ".npz"), rec, actions=job["actions"])
            meta = {k: v for k, v in job.items() if k != "actions"}
            meta.update({"E": rec["E"], "DR": rec["DR"], "E_hex": rec["E_hex"], "DR_hex": rec["DR_hex"],
                         "slot": slot_index, "elapsed_s": time.time() - t0})
            out_q.put(("ok", job["label"], meta))
        except Exception:
            out_q.put(("error", job["label"], traceback.format_exc()))


def run_pool(jobs, out_dir):
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    done = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(out_dir, "raw"))}
    todo = [j for j in jobs if j["label"] not in done]
    print(f"{out_dir}: {len(jobs)} 本（未実行 {len(todo)} 本）", flush=True)
    if not todo:
        return
    report = launch_mod.preflight(out_dir, ALL_SLOTS)
    ctx = mp.get_context("spawn")
    jobs_q, out_q = ctx.Queue(), ctx.Queue()
    for j in todo:
        jobs_q.put(j)
    for _ in ALL_SLOTS:
        jobs_q.put(None)
    procs = [ctx.Process(target=_worker, args=(k, report["log_dir"], out_dir, jobs_q, out_q)) for k in ALL_SLOTS]
    for p in procs:
        p.start()
    errors = {}
    for _ in todo:
        status, label, payload = out_q.get()
        if status == "ok":
            runlog.append_jsonl(os.path.join(out_dir, "jobs_meta.jsonl"), payload)
            print(f"  {label}: E {payload['E']:.6f} DR {payload['DR']:.6f} ({payload['elapsed_s']:.0f} 秒)", flush=True)
        else:
            errors[label] = payload
            print(f"  {label}: 失敗", flush=True)
            if label.startswith("slot"):
                break
    for p in procs:
        p.join(timeout=120)
    if errors:
        runlog.write_json(os.path.join(out_dir, "errors.json"), errors)
        raise RuntimeError(f"再生に失敗: {sorted(errors)}")


def k14_all_jobs():
    base = v62s_diag.baseline_actions()
    rb = load_full(os.path.join(OUT_DIR, "R8p_baseline.npz"))
    return v62s_diag.k14_jobs(base, rb["starts"]) + s2_jobs(base, rb["starts"])


def run_k14():
    run_pool(k14_all_jobs(), K14_DIR)


def _n_for(rec, decision):
    n_kt, n_t, empty = r8p.zone_n(decision["branch"], decision["delta"], rec["A"], rec["reads"], rec["starts"], rec["minute"])
    return n_kt


def analyze_k14():
    decision = json.load(open(os.path.join(OUT_DIR, "R8p_decision.json"), encoding="utf-8"))
    tol = TOL_IDENTITY[decision["branch"]]
    base = load_full(os.path.join(OUT_DIR, "R8p_baseline.npz"))
    nb = _n_for(base, decision)
    from rlkit import exogenous
    exo = exogenous.load()
    metas = [json.loads(l) for l in open(os.path.join(K14_DIR, "jobs_meta.jsonl"), encoding="utf-8")]
    rows = []
    for m in metas:
        r = load_full(os.path.join(K14_DIR, "raw", m["label"] + ".npz"))
        n = _n_for(r, decision)
        i = int(m["start"])
        T = base["dE"].shape[0]
        de, dn, ddE = r["e"] - base["e"], n - nb, r["dE"] - base["dE"]
        pre_same = bool(np.all(de[:i] == 0) and np.all(ddE[:i] == 0) and np.all(r["A"][:i] == base["A"][:i]))
        dC = de[i:].sum(axis=0) + RHO_EVAL * dn[i:].sum(axis=0)
        dC_res = float((ddE[i:] - de[i:].sum(axis=1)).sum())
        dC_true = float(dC.sum() + dC_res)
        dE_true, dDR_true = r["E"] - base["E"], r["DR"] - base["DR"]
        official = dE_true + RHO_EVAL * dDR_true
        fam_for_levels = "S" if m["family"] == "S2" else m["family"]
        row = {"label": m["label"], "family": m["family"], "start": i, "zone": m.get("zone"), "tenant": m.get("tenant"),
               "to_off": m.get("to_off"), "dC_o": dC.tolist(), "dC_res": dC_res, "dC_true": dC_true,
               "dE_true": dE_true, "dDR_true": dDR_true, "dC_official": official,
               "identity_abs_err": abs(dC_true - official), "pre_flip_identical": pre_same,
               "levels": v62s_diag._levels_for(dict(m, family=fam_for_levels), dC, dC_res)}
        if m["family"] == "S2":
            row["occupants_at_flip"] = float(exo.zone_occupants_at(r["starts"][i])[m["zone"]])
        if m["family"] in ("S", "S2") and m.get("to_off"):
            k = m["zone"]
            aux = {}
            h1 = i + (m.get("n_flip_steps") or v62s_diag.K14_FLIP_STEPS)
            h4 = next(j for j in range(i, T + 1) if j == T or (r["starts"][j] - r["starts"][i]).total_seconds() >= 4 * 3600)
            for name, hi in (("1h", h1), ("4h", h4), ("end", T)):
                w_dE = ddE[i:hi].sum()
                w_dn = dn[i:hi].sum()
                aux[name] = {"de_k_over_dE": float(de[i:hi, k].sum() / w_dE) if w_dE != 0 else None,
                             "dn_k_over_dn": float(dn[i:hi, k].sum() / w_dn) if w_dn != 0 else None}
            row["aux"] = aux
        rows.append(row)
    out = {"r8p_branch": decision["branch"], "delta": decision["delta"], "identity_tol_GJeq": tol, "rho_eval": RHO_EVAL,
           "families": {}}
    fams = {"S": (("S", "S2"), ("local", "system", "building")), "P": (("P",), ("local", "system", "building")),
            "V": (("V",), ("tenant", "building"))}
    for fam, (members, levels) in fams.items():
        fr = [r for r in rows if r["family"] in members]
        pop = [r for r in fr if abs(r["dC_true"]) >= v62s_diag.K14_MIN_ABS_DC]
        need = v62s_diag.K14_PASS_SIGN_V if fam == "V" else v62s_diag.K14_PASS_SIGN
        stats, chosen = {}, None
        for lv in levels:
            if not pop:
                break
            sign = float(np.mean([np.sign(r["levels"][lv]) == np.sign(r["dC_true"]) for r in pop]))
            rel = float(np.median([abs(r["levels"][lv] - r["dC_true"]) / abs(r["dC_true"]) for r in pop]))
            ok = sign >= need and rel <= v62s_diag.K14_PASS_MEDIAN_REL
            stats[lv] = {"sign_agreement": sign, "median_rel_err": rel, "passed": ok}
            if ok and chosen is None:
                chosen = lv
        out["families"][fam] = {"members": list(members), "n": len(fr), "n_excluded_small": len(fr) - len(pop),
                                "levels": stats, "chosen": chosen}
    s2 = [r for r in rows if r["family"] == "S2" and abs(r["dC_true"]) >= v62s_diag.K14_MIN_ABS_DC]
    out["S2_by_occupancy"] = {}
    for name, sel in (("unoccupied", [r for r in s2 if r["occupants_at_flip"] <= 0]),
                      ("occupied", [r for r in s2 if r["occupants_at_flip"] > 0])):
        out["S2_by_occupancy"][name] = {
            "n": len(sel),
            "local_sign_agreement": float(np.mean([np.sign(r["levels"]["local"]) == np.sign(r["dC_true"]) for r in sel])) if sel else None,
            "local_median_rel_err": float(np.median([abs(r["levels"]["local"] - r["dC_true"]) / abs(r["dC_true"]) for r in sel])) if sel else None,
            "median_dC_true": float(np.median([r["dC_true"] for r in sel])) if sel else None,
            "median_dC_local": float(np.median([r["levels"]["local"] for r in sel])) if sel else None}
    out["identity_violations"] = [r["label"] for r in rows if r["identity_abs_err"] > tol]
    out["identity_abs_err_max"] = max(r["identity_abs_err"] for r in rows)
    out["pre_flip_all_identical"] = all(r["pre_flip_identical"] for r in rows)
    out["n_rows"] = len(rows)
    out["current_levels_all_pass"] = (out["families"]["S"]["chosen"] == "local" and out["families"]["P"]["chosen"] == "local"
                                      and out["families"]["V"]["chosen"] == "tenant")
    runlog.write_json(os.path.join(K14_DIR, "K14p_rows.json"), rows)
    runlog.write_json(os.path.join(OUT_DIR, "K14_decision.json"), out)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


# =====================================================================
# K-15（V6.3-S 第 3-3 節、n は R-8'）
# =====================================================================

def run_k15_replay():
    import torch as th
    buf = th.load(os.path.join(v62s_diag.TRIAL06, "sil_buffer.pt"), weights_only=False)
    jobs = [{"label": f"K15_u{e['update']}", "update": e["update"], "actions": np.asarray(e["actions"], dtype=np.int64)}
            for e in buf]
    run_pool(jobs, K15_DIR)


def run_k15():
    import torch as th
    from rlkit import advantage as adv_mod, observation
    from rlkit.policy import ValueNetwork
    from rlkit.reward import R7Reference, RewardCalculator
    decision = json.load(open(os.path.join(OUT_DIR, "R8p_decision.json"), encoding="utf-8"))
    level = json.load(open(os.path.join(OUT_DIR, "K14_decision.json"), encoding="utf-8"))["families"]["S"]["chosen"]
    buf = th.load(os.path.join(v62s_diag.TRIAL06, "sil_buffer.pt"), weights_only=False)
    state = th.load(os.path.join(v62s_diag.TRIAL06, "checkpoint_latest.pt"), weights_only=False)
    ref = R7Reference.from_dict(state["reference"])
    vnet = ValueNetwork()
    vnet.load_state_dict(state["model"]["weights"]["value_net"])
    vnet.set_reward_context(*ref.value_context())
    vnet.eval()
    checks, data = [], []
    for entry in buf:
        r = load_full(os.path.join(K15_DIR, "raw", f"K15_u{entry['update']}.npz"))
        n = _n_for(r, decision)
        checks.append({"update": entry["update"], "E_match": bool(r["E"] == entry["E"]), "DR_match": bool(r["DR"] == entry["DR"]),
                       "e_match": bool(np.array_equal(r["e"], entry["zone_energy"])),
                       "sum_n_rel": float(abs(n.sum() - r["DR"]) / r["DR"])})
        data.append((entry, r, n))
    results = {}
    for lam in v62s_diag.K15_LAMBDAS:
        advs, offs, occs, valid = [], [], [], []
        for entry, r, n in data:
            T = entry["actions"].shape[0]
            calc = RewardCalculator(ref, T, entry["W"])
            e, dE = r["e"], r["dE"]
            r19 = np.concatenate([calc.zone_rewards(e, n), (-calc.scale * (dE - e.sum(axis=1)))[:, None]], axis=1)
            with th.no_grad():
                V = vnet(th.as_tensor(entry["obs"]).float()).numpy()
            A19, _ = adv_mod.segmented_gae(r19, V, entry["step_seconds"], [(0, T)], lambda_hour=lam)
            A = v62s_diag.level_zone_advantages(A19, entry["actions"], level)
            masks = entry["masks"]
            n_opt = np.stack([masks[:, s0:s0 + w].sum(axis=1) for s0, w in action.dim_logit_offsets(1)[:constants.N_ZONES]], axis=1)
            occ = np.stack([entry["obs"][:, observation.zone_base(k) + 12] for k in range(constants.N_ZONES)], axis=1) >= 0.1 - 1e-6
            advs.append(A)
            offs.append(entry["actions"][:, :constants.N_ZONES] == constants.ZONE_STATE.index("off"))
            occs.append(occ)
            valid.append(n_opt >= 2)
        A, off, occ, ok = (np.concatenate(x) for x in (advs, offs, occs, valid))
        mu, sd = A[ok].mean(), A[ok].std()
        An = (A - mu) / (sd + constants.ADVANTAGE_NORM_EPS)
        sel = ok & occ
        results[str(lam)] = {"D": float(An[sel & off].mean() - An[sel & ~off].mean()),
                             "n_off_occupied": int((sel & off).sum()), "n_on_occupied": int((sel & ~off).sum()),
                             "norm_mean": float(mu), "norm_std": float(sd)}
    s095, s099, s10 = (np.sign(results[str(l)]["D"]) for l in v62s_diag.K15_LAMBDAS)
    lam = 0.95 if s095 == s10 else (0.99 if s099 == s10 else 1.0)
    out = {"level_used": level, "r8p_branch": decision["branch"], "delta": decision["delta"],
           "reference": {"rho": ref.rho, "E_ref": ref.E_ref, "DR_ref": ref.DR_ref}, "replay_checks": checks,
           "results": results, "lambda_hour": lam}
    runlog.write_json(os.path.join(OUT_DIR, "K15_decision.json"), out)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out

def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", required=True, choices=["r8p", "k14", "k14analyze", "k15replay", "k15"])
    a = p.parse_args(argv)
    runlog.check_thread_env()
    {"r8p": run_r8p, "k14": run_k14, "k14analyze": analyze_k14, "k15replay": run_k15_replay, "k15": run_k15}[a.step]()


if __name__ == "__main__":
    main()
