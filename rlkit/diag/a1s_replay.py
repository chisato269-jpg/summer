"""指示書 A1-S 手順 6（C-2・C-3）: 3 本の軌跡の再生と、エミュレータの出力の複写。学習はしない。

記録した行動列だけを、環境（V8: rlkit/v8/env.py、T-3 は trial08 の rlkit/env.py）で 1 エピソード再生する。方策は使わない。
再生中、環境が BACnet で読んだ値（ステップの終端の建物・ゾーンの物理量、室外機・室内機の電力の標本、設定温度の読み戻しなど）を、
rlkit.comm の読み取り関数を包んで記録する（環境の挙動は変えない）。再生の後、エミュレータが書いた data/ の 8 ファイルを
raw/<ラベル>/ へ複写して SHA-256 を記録する。3 本は別々の組（既定 12・11・10）で同時に再生する。

  python -m rlkit.diag.a1s_replay [--labels T1 T2 T3] [--out C:\\Users\\chisa\\rl_local\\diag\\A1S]
"""
import argparse
import copy
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

OUT_DEFAULT = r"C:\Users\chisa\rl_local\diag\A1S"
OUTPUT_FILES = ("general.csv", "occupant.csv", "vent.csv", "vrf.csv", "zone.csv", "dissatisfaction.csv", "result.txt", "result.szk")
ENV_KWARGS = {"stage": 1, "include_indoor_unit_power": True, "zone_step_energy_norm_gj": 0.02, "m9_enabled": True, "comm_rpm": True}
WRAPPED = ("read_step_snapshot", "read_outdoor_unit_power", "read_indoor_unit_power", "read_setpoint_readback",
           "read_onoff_status", "read_remote_permission_status", "read_tenant_thermal_sensation", "read_direction_status")
JOBS = {
    "T1": {"kind": "v8", "slot": 12, "actions": "T1_best_harvest.npz", "E_hex": "0x1.1e0f7ae147ae1p+2", "DR_hex": "0x1.58efe40000000p-3",
           "name": "trial03 の最良（更新 600 の兄弟、組 6）"},
    "T2": {"kind": "v8", "slot": 11, "actions": "T2_best_harvest.npz", "E_hex": "0x1.1ac98b4395810p+2", "DR_hex": "0x1.6bec4c0000000p-3",
           "name": "trial03 の始点（trial02 の更新 5 の兄弟）"},
    "T3": {"kind": "v7", "slot": 10, "actions": "T3_best_harvest.npz", "E_hex": "0x1.1fae9374bc6a8p+2", "DR_hex": "0x1.6b6c220000000p-3",
           "name": "trial08 の収穫の最良（更新 1268、組 4）"},
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_seeds(setting_ini):
    import re
    seeds = {}
    with open(setting_ini, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*(rseed_o|rseed_w|rseed_oprm|use_rso|use_rsw|period|timestep|outputSpan|weather)\s*=\s*(-?\d+)", line)
            if m:
                seeds[m.group(1)] = int(m.group(2))
    return seeds


def install_capture(events):
    """rlkit.comm の読み取り関数を包んで、値を events に記録する。戻り値は元に戻す関数。"""
    from rlkit import comm
    originals = {}
    for name in WRAPPED:
        fn = getattr(comm, name, None)
        if fn is None:
            continue
        originals[name] = fn

        def make(name, fn):
            async def wrapper(comms, *a, **kw):
                v = await fn(comms, *a, **kw)
                events.append((name, copy.deepcopy(v)))
                return v
            return wrapper
        setattr(comm, name, make(name, fn))

    def restore():
        for name, fn in originals.items():
            setattr(comm, name, fn)
    return restore


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, datetime.datetime):
        return o.isoformat()
    return o


def replay_job(job, out_dir):
    """子プロセスで 1 本を再生する。job: {label, kind, slot, actions_path, E_hex, DR_hex}。"""
    from rlkit import constants, driver
    from rlkit.parallel import launch as launch_mod
    from rlkit.parallel import slots
    label = job["label"]
    raw = os.path.join(out_dir, "raw", label)
    os.makedirs(raw, exist_ok=True)
    actions = np.load(job["actions_path"])["actions"].astype(np.int64)
    expected_dims = 80 if job["kind"] == "v8" else 60
    if actions.shape != (constants.ZONE_STEPS_PER_WEEK, expected_dims):
        raise RuntimeError(f"{label}: 行動列の形が {actions.shape}（期待 ({constants.ZONE_STEPS_PER_WEEK}, {expected_dims})）")
    slot = slots.make_slot(job["slot"])
    slots.prepare_replica(slot)
    seeds_before = read_seeds(os.path.join(slot.data_dir, "..", "setting.ini")) if hasattr(slot, "data_dir") else {}
    if job["kind"] == "v8":
        from rlkit.v8.env import SummerEnvV8
        env = SummerEnvV8(slot=slot, **ENV_KWARGS)
    else:
        from rlkit.env import SummerEnv
        env = SummerEnv(slot=slot, **ENV_KWARGS)
    events = []
    restore = install_capture(events)
    steps = []
    final = None
    t0 = time.time()
    try:
        env.reset()
        init_events = list(events)
        n_prev = len(events)
        for t in range(actions.shape[0]):
            dt_start = env._pending_dt
            _obs, _r, done, _trunc, info = env.step(actions[t])
            steps.append({"t": t, "dt_start": dt_start.isoformat(), "dt_end": info["sim_datetime"].isoformat(),
                          "seconds": float(info["step_seconds"]), "delta_E_gj": float(info["delta_E_gj"]),
                          "calls": [(n, v) for n, v in events[n_prev:]]})
            n_prev = len(events)
            if done:
                final = info
                break
    finally:
        restore()
        env.close()
    if final is None:
        raise RuntimeError(f"{label}: 再生が終端しなかった")
    elapsed = time.time() - t0
    # エミュレータの出力の複写（環境を閉じた後）
    outputs = {}
    src = slot.data_dir
    for name in OUTPUT_FILES:
        p = os.path.join(src, name)
        if os.path.exists(p):
            dst = os.path.join(raw, name)
            shutil.copy2(p, dst)
            outputs[name] = {"size": os.path.getsize(dst), "sha256": sha256_file(dst), "source": p,
                             "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat()}
    rec = {"label": label, "name": job["name"], "kind": job["kind"], "slot": job["slot"], "elapsed_s": elapsed,
           "replayed_at": datetime.datetime.now().isoformat(), "actions_path": job["actions_path"],
           "actions_sha256": sha256_file(job["actions_path"]),
           "E": float(final["E_rep"]), "DR": float(final["DR_rep"]), "E_hex": float.hex(float(final["E_rep"])),
           "DR_hex": float.hex(float(final["DR_rep"])), "ECI": float(final["ECI"]), "P_E": float(final["P_E"]),
           "P_DR": float(final["P_DR"]), "expected_E_hex": job["E_hex"], "expected_DR_hex": job["DR_hex"],
           "seeds_in_setting_ini": read_seeds(os.path.join(os.path.dirname(src), "setting.ini")),
           "outputs": outputs, "n_steps": len(steps),
           "final_info": {k: to_jsonable(final[k]) for k in ("E_rep", "DR_rep", "ECI", "setpoint_override_rate", "sum_delta_E_gj",
                                                              "raw_resid_sum_gj", "raw_resid_rel", "r8p_sum_rel") if k in final}}
    rec["hex_match"] = bool(rec["E_hex"] == rec["expected_E_hex"] and rec["DR_hex"] == rec["expected_DR_hex"])
    with open(os.path.join(raw, "replay.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(rec), f, ensure_ascii=False, indent=1)
    with open(os.path.join(raw, "captures.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable({"reset_calls": [(n, v) for n, v in init_events], "steps": steps}), f, ensure_ascii=False)
    return rec


def _child(job, out_dir, launch_dir, q):
    try:
        from rlkit.parallel import launch as launch_mod
        os.environ[launch_mod.LAUNCH_LOG_ENV] = launch_dir
        rec = replay_job(job, out_dir)
        q.put(("ok", job["label"], {k: rec[k] for k in ("label", "E_hex", "DR_hex", "ECI", "hex_match", "elapsed_s")}))
    except Exception:
        q.put(("error", job["label"], traceback.format_exc()))


def run_parallel(labels, out_dir, actions_dir):
    from rlkit import runlog
    from rlkit.parallel import launch as launch_mod
    runlog.check_thread_env()
    jobs = []
    for lab in labels:
        j = dict(JOBS[lab], label=lab, actions_path=os.path.join(actions_dir, JOBS[lab]["actions"]))
        jobs.append(j)
    slot_ids = [j["slot"] for j in jobs]
    report = launch_mod.preflight(os.path.join(out_dir, "raw"), slot_ids, limit=13, min_free_gb=3.0)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for j in jobs:
        p = ctx.Process(target=_child, args=(j, out_dir, report["log_dir"], q))
        p.start()
        procs.append(p)
    results, errors = {}, {}
    for _ in jobs:
        status, label, payload = q.get()
        (results if status == "ok" else errors)[label] = payload
        print(f"[a1s_replay] {label}: {status} {payload if status == 'ok' else ''}", flush=True)
    for p in procs:
        p.join(timeout=120)
    return results, errors


def main(argv=None):
    os.chdir(_REPO_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["T1", "T2", "T3"])
    ap.add_argument("--out", default=OUT_DEFAULT)
    a = ap.parse_args(argv)
    results, errors = run_parallel(a.labels, a.out, os.path.join(a.out, "raw", "actions"))
    summary = {"results": results, "errors": {k: v[-3000:] for k, v in errors.items()}}
    with open(os.path.join(a.out, "raw", "replay_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: results[k] for k in results}, ensure_ascii=False))
    if errors:
        print("エラー:", list(errors), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
