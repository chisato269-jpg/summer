"""指示書 V6.2-S の段階 Z'（Z'-1、K-14 反事実の再生、K-15 λ の比較）。学習は行わない。

  Z'-1  組 6 で基準の軌跡をそのまま再生し、P-5 と 16 進で照合する。ステップごとの e_k,t・n_k,t・ΔE_t を保存する
  K-14  基準の軌跡から、指定のステップから 4 ステップだけ発停・設定温度・換気を変えた反事実を 7 組で並列に再生し、
        出力ごとの ΔC を求めて、家族 S・P・V の信用割当の水準を判定表で決める（第 4-1 節・第 4-2 節）
  K-15  runs/v2s_trial06 の SIL の保存庫 8 本と価値ネットワークで、λ_hour = 0.95・0.99・1.0 の GAE を計算し、
        在室ゾーン・ステップの D_λ = mean(Â | 停止) − mean(Â | 運転) の符号で λ を決める（第 4-3 節）

使い方（リポジトリルートから）:
  python -m rlkit.diag.v62s_diag --step z1
  python -m rlkit.diag.v62s_diag --step k14
  python -m rlkit.diag.v62s_diag --step k14analyze
  python -m rlkit.diag.v62s_diag --step k15
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

from rlkit import action, config as config_mod, constants, runlog
from rlkit.parallel import launch as launch_mod
from rlkit.parallel import slots

OUT_DIR = os.path.join("runs", "diag", "V62S")
K14_DIR = os.path.join(OUT_DIR, "K14")
BASELINE = os.path.join("runs", "v2s_trial04", "harvest")
BASELINE_E_HEX = "0x1.a228e147ae148p+2"
BASELINE_DR_HEX = "0x1.863c140000000p-3"
CONFIG = os.path.join("rlkit", "config_summer_v61.json")
ALL_SLOTS = [0, 1, 2, 3, 4, 5, 6]

# 第 13 章
K14_FLIP_STEPS = 4
K14_TIMES_S = (10, 13, 16)
K14_TIME_PV = 13
K14_S_ON_SETPOINT_C = 26.0
K14_S_ON_FAN = "Middle"
K14_P_DELTA_C = 2.0
K14_P_CAP_C = 28.0
RHO_EVAL = 72.3
K14_MIN_ABS_DC = 0.0005
K14_PASS_SIGN = 0.90
K14_PASS_SIGN_V = 1.00
K14_PASS_MEDIAN_REL = 0.30
K14_IDENTITY_TOL = 0.01
K15_LAMBDAS = (0.95, 0.99, 1.0)
TRIAL06 = os.path.join("runs", "v2s_trial06")

SYSTEM_OF_ZONE = np.array(action.SYSTEM_OF_ZONE)
TENANT_OF_ZONE = np.array(action.TENANT_OF_ZONE)


def baseline_actions():
    v5 = np.load(os.path.join(BASELINE, "best_harvest.npz"))["actions"].astype(np.int64)
    return action.v5_to_v6_actions(v5)


def step_datetimes():
    """再生の記録（Z'-1）から各ステップの開始時刻。"""
    d = np.load(os.path.join(OUT_DIR, "Z1_steps.npz"), allow_pickle=True)
    return [datetime.datetime.fromisoformat(s) for s in d["dt"]]


def flip_start(dts, hour):
    """エピソードの 2 日目（平日でなければ最初の平日）の、hour:00 かその直後に始まるステップ。"""
    day0 = dts[0].date()
    days = sorted({d.date() for d in dts})
    target = day0 + datetime.timedelta(days=1)
    if target.weekday() >= 5:
        target = next(d for d in days if d.weekday() < 5)
    return next(i for i, t in enumerate(dts) if t.date() == target and t.hour * 60 + t.minute >= hour * 60)


# =====================================================================
# 反事実の組み立て（第 4-1 節）
# =====================================================================

def k14_jobs(base, dts):
    grid = constants.ZONE_SETPOINT_C[1]
    st0, sp0, fan0 = (action.LAYOUT[k][0] for k in ("zone_state", "zone_setpoint", "zone_fan"))
    v0 = action.LAYOUT["tenant_vent"][0]
    cool = constants.ZONE_STATE.index("cool")
    off = constants.ZONE_STATE.index("off")
    jobs = []
    for hh in K14_TIMES_S:
        i = flip_start(dts, hh)
        for k in range(constants.N_ZONES):
            a = base.copy()
            to_off = int(base[i, st0 + k]) == cool
            for t in range(i, i + K14_FLIP_STEPS):
                if int(base[t, st0 + k]) == cool:
                    a[t, st0 + k] = off
                else:
                    a[t, st0 + k] = cool
                    a[t, sp0 + k] = grid.index(K14_S_ON_SETPOINT_C)
                    a[t, fan0 + k] = constants.ZONE_FAN_SPEED.index(K14_S_ON_FAN)
            jobs.append({"label": f"S_{hh:02d}_{constants.ZONE_LABELS[k]}", "family": "S", "zone": k, "start": i,
                         "to_off": to_off, "actions": a})
    i = flip_start(dts, K14_TIME_PV)
    for k in range(constants.N_ZONES):
        if int(base[i, st0 + k]) != cool:
            continue
        a = base.copy()
        for t in range(i, i + K14_FLIP_STEPS):
            cur = grid[int(base[t, sp0 + k])]
            a[t, sp0 + k] = grid.index(min(cur + K14_P_DELTA_C, K14_P_CAP_C))
        jobs.append({"label": f"P_13_{constants.ZONE_LABELS[k]}", "family": "P", "zone": k, "start": i, "actions": a})
    for j, tenant in enumerate(action.VENT_TENANT_ORDER):
        base_level = int(base[i, v0 + j])
        levels = ("Middle", "High") if base_level == constants.VENT_OFF else ("High", "Low")
        for name in levels:
            a = base.copy()
            a[i:i + K14_FLIP_STEPS, v0 + j] = constants.VENT_LEVELS.index(name)
            jobs.append({"label": f"V_13_{tenant}_{name}", "family": "V", "tenant": j, "start": i, "actions": a})
    return jobs


# =====================================================================
# 再生（子プロセス）
# =====================================================================

def replay_record(slot_index, actions):
    from rlkit.env import SummerEnv
    cfg = config_mod.load(CONFIG)
    slot = slots.make_slot(slot_index)
    env = SummerEnv(include_indoor_unit_power=cfg["include_indoor_unit_power"],
                    zone_step_energy_norm_gj=cfg["zone_step_energy_norm_gj"], slot=slot, m9_enabled=cfg["m9_enabled"])
    T = actions.shape[0]
    e = np.zeros((T, constants.N_ZONES))
    n = np.zeros((T, constants.N_ZONES))
    dE = np.zeros(T)
    dts = []
    try:
        env.reset()
        for t in range(T):
            dts.append(env._pending_dt.isoformat())
            _o, _r, done, _tr, info = env.step(actions[t])
            e[t] = info["zone_energy_gj"]
            n[t] = info["zone_dissatisfaction_increment"]
            dE[t] = info["delta_E_gj"]
            if done:
                break
    finally:
        env.close()
    return {"e": e, "n": n, "dE": dE, "dt": np.array(dts), "E": info["E_rep"], "DR": info["DR_rep"],
            "E_hex": float.hex(info["E_rep"]), "DR_hex": float.hex(info["DR_rep"]), "raw_resid_rel": info["raw_resid_rel"]}


def _worker(slot_index, launch_dir, jobs_q, out_q):
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
            rec = replay_record(slot_index, job["actions"])
            path = os.path.join(K14_DIR, "raw", job["label"] + ".npz")
            np.savez_compressed(path, e=rec["e"], n=rec["n"], dE=rec["dE"], dt=rec["dt"], actions=job["actions"],
                                E=rec["E"], DR=rec["DR"])
            meta = {k: v for k, v in job.items() if k != "actions"}
            meta.update({"E": rec["E"], "DR": rec["DR"], "E_hex": rec["E_hex"], "DR_hex": rec["DR_hex"],
                         "raw_resid_rel": rec["raw_resid_rel"], "slot": slot_index, "elapsed_s": time.time() - t0})
            out_q.put(("ok", job["label"], meta))
        except Exception:
            out_q.put(("error", job["label"], traceback.format_exc()))


def run_z1():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = launch_mod.preflight(OUT_DIR, [6])
    os.environ[launch_mod.LAUNCH_LOG_ENV] = report["log_dir"]
    slots.prepare_replica(slots.make_slot(6))
    t0 = time.time()
    rec = replay_record(6, baseline_actions())
    np.savez_compressed(os.path.join(OUT_DIR, "Z1_steps.npz"), e=rec["e"], n=rec["n"], dE=rec["dE"], dt=rec["dt"],
                        E=rec["E"], DR=rec["DR"])
    out = {"E": rec["E"], "DR": rec["DR"], "E_hex": rec["E_hex"], "DR_hex": rec["DR_hex"],
           "raw_resid_rel": rec["raw_resid_rel"], "elapsed_s": time.time() - t0,
           "match": rec["E_hex"] == BASELINE_E_HEX and rec["DR_hex"] == BASELINE_DR_HEX}
    runlog.write_json(os.path.join(OUT_DIR, "Z1_baseline.json"), out)
    print("Z'-1:", out, flush=True)
    return out


def run_k14():
    base = baseline_actions()
    jobs = k14_jobs(base, step_datetimes())
    os.makedirs(os.path.join(K14_DIR, "raw"), exist_ok=True)
    done = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(K14_DIR, "raw"))}
    meta_path = os.path.join(K14_DIR, "jobs_meta.jsonl")
    todo = [j for j in jobs if j["label"] not in done]
    print(f"K-14: 反事実 {len(jobs)} 本（未実行 {len(todo)} 本）", flush=True)
    report = launch_mod.preflight(K14_DIR, ALL_SLOTS)
    ctx = mp.get_context("spawn")
    jobs_q, out_q = ctx.Queue(), ctx.Queue()
    for j in todo:
        jobs_q.put(j)
    for _ in ALL_SLOTS:
        jobs_q.put(None)
    procs = [ctx.Process(target=_worker, args=(k, report["log_dir"], jobs_q, out_q)) for k in ALL_SLOTS]
    for p in procs:
        p.start()
    errors = {}
    for _ in todo:
        status, label, payload = out_q.get()
        if status == "ok":
            runlog.append_jsonl(meta_path, payload)
            print(f"  {label}: E {payload['E']:.6f} DR {payload['DR']:.6f} ({payload['elapsed_s']:.0f} 秒)", flush=True)
        else:
            errors[label] = payload
            print(f"  {label}: 失敗", flush=True)
            if label.startswith("slot"):
                break
    for p in procs:
        p.join(timeout=120)
    if errors:
        runlog.write_json(os.path.join(K14_DIR, "errors.json"), errors)
        raise RuntimeError(f"K-14 の再生に失敗: {sorted(errors)}")


# =====================================================================
# K-14 の解析
# =====================================================================

def _levels_for(meta, dC, dC_res):
    """家族ごとの水準の ΔC。dC: (18,) 出力ごとの ΔC_o、dC_res: 残差。"""
    total = float(dC.sum() + dC_res)
    if meta["family"] in ("S", "P"):
        k = meta["zone"]
        return {"local": float(dC[k]), "system": float(dC[SYSTEM_OF_ZONE == SYSTEM_OF_ZONE[k]].sum() + dC_res),
                "building": total}
    t = meta["tenant"]
    return {"tenant": float(dC[TENANT_OF_ZONE == t].sum() + dC_res), "building": total}


def _window_sums(d, lo, hi):
    return {k: d[k][lo:hi].sum(axis=0) for k in ("e", "n", "dE")}


def analyze_k14():
    z = np.load(os.path.join(OUT_DIR, "Z1_steps.npz"), allow_pickle=True)
    base = {"e": z["e"], "n": z["n"], "dE": z["dE"]}
    E0, DR0 = float(z["E"]), float(z["DR"])
    metas = [json.loads(l) for l in open(os.path.join(K14_DIR, "jobs_meta.jsonl"), encoding="utf-8")]
    rows = []
    for m in metas:
        r = np.load(os.path.join(K14_DIR, "raw", m["label"] + ".npz"))
        i = int(m["start"])
        T = base["dE"].shape[0]
        de, dn, ddE = r["e"] - base["e"], r["n"] - base["n"], r["dE"] - base["dE"]
        pre_same = bool(np.all(de[:i] == 0) and np.all(dn[:i] == 0) and np.all(ddE[:i] == 0))
        dC = de[i:].sum(axis=0) + RHO_EVAL * dn[i:].sum(axis=0)
        dC_res = float((ddE[i:] - de[i:].sum(axis=1)).sum())
        dC_true = float(dC.sum() + dC_res)
        dE_true, dDR_true = float(r["E"]) - E0, float(r["DR"]) - DR0
        official = dE_true + RHO_EVAL * dDR_true
        id_err = abs(dC_true - official) / abs(official) if official != 0 else (0.0 if dC_true == 0 else float("inf"))
        row = {"label": m["label"], "family": m["family"], "start": i, "zone": m.get("zone"), "tenant": m.get("tenant"),
               "to_off": m.get("to_off"), "dC_o": dC.tolist(), "dC_res": dC_res, "dC_true": dC_true,
               "dE_true": dE_true, "dDR_true": dDR_true, "dC_official": official, "identity_rel_err": id_err,
               "pre_flip_identical": pre_same, "levels": _levels_for(m, dC, dC_res),
               "sum_dE_steps": float(ddE[i:].sum()), "sum_dn_steps": float(dn[i:].sum())}
        if m["family"] == "S" and m.get("to_off"):
            k = m["zone"]
            aux = {}
            for name, hi in (("1h", i + K14_FLIP_STEPS), ("4h", i + 4 * K14_FLIP_STEPS), ("end", T)):
                w_e, w_n, w_dE = de[i:hi, k].sum(), dn[i:hi, k].sum(), ddE[i:hi].sum()
                w_dn_all = dn[i:hi].sum()
                aux[name] = {"de_k": float(w_e), "dE_window": float(w_dE),
                             "de_k_over_dE": float(w_e / w_dE) if w_dE != 0 else None,
                             "dn_k": float(w_n), "dn_all_window": float(w_dn_all),
                             "dn_k_over_dn_all": float(w_n / w_dn_all) if w_dn_all != 0 else None}
            aux["end"]["de_k_over_dE_true"] = float(de[i:, k].sum() / dE_true) if dE_true != 0 else None
            aux["end"]["dn_k_over_dDR_true"] = float(dn[i:, k].sum() / dDR_true) if dDR_true != 0 else None
            row["aux"] = aux
        rows.append(row)
    decision = {"rho_eval": RHO_EVAL, "min_abs_dC": K14_MIN_ABS_DC, "families": {}}
    candidates = {"S": ("local", "system", "building"), "P": ("local", "system", "building"), "V": ("tenant", "building")}
    for fam, levels in candidates.items():
        fr = [r for r in rows if r["family"] == fam]
        pop = [r for r in fr if abs(r["dC_true"]) >= K14_MIN_ABS_DC]
        need_sign = K14_PASS_SIGN_V if fam == "V" else K14_PASS_SIGN
        stats = {}
        chosen = None
        for lv in levels:
            if not pop:
                break
            sign = float(np.mean([np.sign(r["levels"][lv]) == np.sign(r["dC_true"]) for r in pop]))
            rel = float(np.median([abs(r["levels"][lv] - r["dC_true"]) / abs(r["dC_true"]) for r in pop]))
            passed = sign >= need_sign and rel <= K14_PASS_MEDIAN_REL
            stats[lv] = {"sign_agreement": sign, "median_rel_err": rel, "passed": passed}
            if passed and chosen is None:
                chosen = lv
        decision["families"][fam] = {"n": len(fr), "n_excluded_small": len(fr) - len(pop), "levels": stats,
                                     "chosen": chosen, "sign_threshold": need_sign}
    bad_id = [r["label"] for r in rows if abs(r["dC_official"]) > 0 and r["identity_rel_err"] > K14_IDENTITY_TOL]
    decision["identity_violations"] = bad_id
    decision["identity_rel_err_max"] = max(r["identity_rel_err"] for r in rows)
    decision["pre_flip_all_identical"] = all(r["pre_flip_identical"] for r in rows)
    decision["current_levels_all_pass"] = (decision["families"]["S"]["chosen"] == "local"
                                           and decision["families"]["P"]["chosen"] == "local"
                                           and decision["families"]["V"]["chosen"] == "tenant")
    runlog.write_json(os.path.join(K14_DIR, "K14_rows.json"), rows)
    runlog.write_json(os.path.join(OUT_DIR, "K14_decision.json"), decision)
    print(json.dumps({k: v for k, v in decision.items()}, ensure_ascii=False, indent=1))
    return decision, rows


# =====================================================================
# K-15
# =====================================================================

def level_zone_advantages(A19, actions, level):
    """zone_state の次元に与える優位度（第 3 章）。A19: (T,19)、actions: (T,62)。"""
    A_zone, A_sys = A19[:, :constants.N_ZONES], A19[:, constants.N_ZONES]
    if level == "local":
        return A_zone.copy()
    if level == "building":
        return np.repeat((A_zone.sum(axis=1) + A_sys)[:, None], constants.N_ZONES, axis=1)
    on = action.zone_on_flags_batch(actions)
    n_all = on.sum(axis=1)
    out = np.zeros_like(A_zone)
    for s in range(4):
        members = SYSTEM_OF_ZONE == s
        share = np.where(n_all > 0, on[:, members].sum(axis=1) / np.maximum(n_all, 1), 0.0)
        out[:, members] = (A_zone[:, members].sum(axis=1) + A_sys * share)[:, None]
    return out


def run_k15(level):
    import torch as th
    from rlkit import advantage as adv_mod, observation
    from rlkit.policy import ValueNetwork
    from rlkit.reward import R7Reference, RewardCalculator
    buf = th.load(os.path.join(TRIAL06, "sil_buffer.pt"), weights_only=False)
    state = th.load(os.path.join(TRIAL06, "checkpoint_latest.pt"), weights_only=False)
    ref = R7Reference.from_dict(state["reference"])
    vnet = ValueNetwork()
    vnet.load_state_dict(state["model"]["weights"]["value_net"])
    vnet.set_reward_context(*ref.value_context())
    vnet.eval()
    results = {}
    for lam in K15_LAMBDAS:
        advs, offs, occs, valid = [], [], [], []
        for entry in buf:
            T = entry["actions"].shape[0]
            calc = RewardCalculator(ref, T, entry["W"])
            e, n, dE = entry["zone_energy"], entry["zone_n"], entry["delta_E"]
            r19 = np.concatenate([calc.zone_rewards(e, n), (-calc.scale * (dE - e.sum(axis=1)))[:, None]], axis=1)
            with th.no_grad():
                V = vnet(th.as_tensor(entry["obs"]).float()).numpy()
            A19, _ = adv_mod.segmented_gae(r19, V, entry["step_seconds"], [(0, T)], lambda_hour=lam)
            A = level_zone_advantages(A19, entry["actions"], level)
            masks = entry["masks"]
            n_opt = np.stack([masks[:, s:s + w].sum(axis=1) for s, w in action.dim_logit_offsets(1)[:constants.N_ZONES]], axis=1)
            occ = np.stack([entry["obs"][:, observation.zone_base(k) + 12] for k in range(constants.N_ZONES)], axis=1) >= 0.1 - 1e-6
            advs.append(A)
            offs.append(entry["actions"][:, :constants.N_ZONES] == constants.ZONE_STATE.index("off"))
            occs.append(occ)
            valid.append(n_opt >= 2)
        A, off, occ, ok = (np.concatenate(x) for x in (advs, offs, occs, valid))
        mu, sd = A[ok].mean(), A[ok].std()
        An = (A - mu) / (sd + constants.ADVANTAGE_NORM_EPS)
        sel = ok & occ
        D = float(An[sel & off].mean() - An[sel & ~off].mean())
        results[str(lam)] = {"D": D, "n_off_occupied": int((sel & off).sum()), "n_on_occupied": int((sel & ~off).sum()),
                             "norm_mean": float(mu), "norm_std": float(sd)}
    s095, s099, s10 = (np.sign(results[str(l)]["D"]) for l in K15_LAMBDAS)
    if s095 == s10:
        lam = 0.95
    elif s099 == s10:
        lam = 0.99
    else:
        lam = 1.0
    out = {"level_used": level, "reference": {"rho": ref.rho, "E_ref": ref.E_ref, "DR_ref": ref.DR_ref},
           "n_trajectories": len(buf), "buffer_updates": [e["update"] for e in buf], "results": results,
           "lambda_hour": lam}
    runlog.write_json(os.path.join(OUT_DIR, "K15_decision.json"), out)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", required=True, choices=["z1", "k14", "k14analyze", "k15"])
    a = p.parse_args(argv)
    runlog.check_thread_env()
    if a.step == "z1":
        run_z1()
    elif a.step == "k14":
        run_k14()
        analyze_k14()
    elif a.step == "k14analyze":
        analyze_k14()
    else:
        d = json.load(open(os.path.join(OUT_DIR, "K14_decision.json"), encoding="utf-8"))
        run_k15(d["families"]["S"]["chosen"])


if __name__ == "__main__":
    main()
