"""並列数拡張の診断（並列数拡張マニュアル第 4〜6 節を夏季機に移したもの。学習は行わない）。

  probe   D-4: 一括読み取り（ReadPropertyMultiple）をエミュレータが受け付けるか。点数ごとの成否・時間・値の一致（組 1 つ）
  verify  D-5: 基準の軌跡を 1 エピソード再生し、毎ステップの読み取りで一括版と逐次版を両方読んで辞書が == か確かめる（組 1 つ）
  scale   第 6 節: 組数 N の組に基準の軌跡を同時に再生し、1 エピソードの時間・CPU・メモリ・全組の 16 進一致を測る

基準の軌跡は runs/v2s_trial04/harvest/best_harvest.npz（V5 の最良、P-7）を V7 の行動（60 次元、設定温度 22〜28 ℃）に写したもの。
全組に同じ行動列を与えるので、全組の E・DR が P-7 と 16 進で一致するかが判定になる。

使い方（リポジトリルートから。環境変数 OMP_NUM_THREADS=1 などを設定）:
  python -m rlkit.diag.v7s_par --step probe --slot 6
  python -m rlkit.diag.v7s_par --step verify --slot 6
  python -m rlkit.diag.v7s_par --step scale --n 13 [--rpm]
測定・学習・単体試験を同時に走らせない（組のロックを壊す。並列数拡張マニュアル B-1）。
"""
import argparse
import asyncio
import datetime
import json
import os
import sys
import threading
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

OUT_ROOT = os.path.join("runs", "diag", "V7S", "par")
BASELINE_NPZ = os.path.join("runs", "v2s_trial04", "harvest", "best_harvest.npz")
BASELINE_E_HEX = "0x1.a228e147ae148p+2"
BASELINE_DR_HEX = "0x1.863c140000000p-3"
T = 378


def baseline_actions():
    from rlkit import action
    return action.v5_to_v6_actions(np.load(BASELINE_NPZ)["actions"].astype(np.int64))


def env_kwargs(rpm):
    from rlkit import config as config_mod
    cfg = config_mod.load(os.path.join("rlkit", "config_summer_v7.json"))
    return {"stage": 1, "include_indoor_unit_power": cfg["include_indoor_unit_power"],
            "zone_step_energy_norm_gj": cfg["zone_step_energy_norm_gj"], "m9_enabled": cfg["m9_enabled"],
            "comm_rpm": bool(rpm)}


def _launch(out_dir, slot_indices):
    from rlkit.parallel import launch, slots
    specs = [slots.make_slot(k) for k in slot_indices]
    report = launch.preflight(out_dir, slot_indices, limit=max(len(slot_indices), launch.SLOT_LIMIT), min_free_gb=1.0)
    os.environ[launch.LAUNCH_LOG_ENV] = report["log_dir"]
    for s in specs:
        slots.prepare_replica(s)
    return specs


# =====================================================================
# D-4: 一括読み取りの探査
# =====================================================================

def run_probe(slot_index, label):
    from bacpypes3.pdu import Address
    from bacpypes3.primitivedata import ObjectIdentifier
    from rlkit import comm, driver
    out_dir = os.path.join(OUT_ROOT, "probe", f"{label}_slot{slot_index}")
    os.makedirs(out_dir, exist_ok=True)
    _launch(out_dir, [slot_index])
    from rlkit.parallel import slots
    drv = driver.EpisodeDriver(slot=slots.make_slot(slot_index))
    drv.start()
    result = {"label": label, "slot": slot_index}

    def call(coro):
        return asyncio.run_coroutine_threadsafe(coro, drv.loop).result()

    recorded = []
    PV = next(c for c in type(drv.comms.vrf).__mro__ if c.__name__ == "PresentValueReadWriter")
    orig = PV.read_present_value

    async def rec_read(self, addr, obj_id):
        ok, val = await orig(self, addr, obj_id)
        recorded.append((self.id, addr, obj_id, ok, val))
        return ok, val

    async def go():
        comms = drv.comms
        for _ in range(10):                      # 1999-07-21 00:00 から 10 時間進める（水曜 10 時、在室時間帯）
            await drv.advance(3600)
        PV.read_present_value = rec_read
        try:
            t = time.perf_counter()
            c0 = time.process_time()
            await comm.read_step_snapshot(comms)
            await comm.read_outdoor_unit_power(comms)
            await comm.read_indoor_unit_power(comms)
            await comm.read_setpoint_readback(comms)
            await comm.read_tenant_thermal_sensation(comms)
            await comm.read_remote_permission_status(comms)
            await comm.read_onoff_status(comms)
            single_wall, single_cpu = time.perf_counter() - t, time.process_time() - c0
        finally:
            PV.read_present_value = orig
        result["single_reads"] = {"n": len(recorded), "wall_s": single_wall, "cpu_s": single_cpu,
                                  "n_failed": sum(1 for r in recorded if not r[3])}
        by_dev = {}
        for dev_id, addr, obj_id, ok, val in recorded:
            by_dev.setdefault((dev_id, addr), []).append((obj_id, val))
        result["groups"] = {f"id{d}@{a}": len(v) for (d, a), v in by_dev.items()}
        apps = {getattr(comms, n).id: getattr(comms, n) for n in ("vrf", "vent", "env", "occ")}
        result["rpm"] = []
        for (dev_id, addr), items in by_dev.items():
            comm_obj = apps[dev_id]
            for chunk in (1, 5, 20, 40, 50, 100):
                part = items[:chunk]
                if len(part) < chunk and chunk > 1:
                    continue
                plist = []
                for obj_id, _v in part:
                    plist += [ObjectIdentifier(obj_id), ["present-value"]]
                t = time.perf_counter()
                c0 = time.process_time()
                rec = {"device_id": dev_id, "addr": addr, "chunk": chunk}
                try:
                    resp = await asyncio.wait_for(comm_obj.bacdevice.read_property_multiple(Address(addr), plist), timeout=5.0)
                    rec["wall_s"] = time.perf_counter() - t
                    rec["cpu_s"] = time.process_time() - c0
                    vals = [r[3] for r in resp]
                    rec["n_returned"] = len(vals)
                    same = 0
                    for (obj_id, v1), v2 in zip(part, vals):
                        try:
                            same += int(float(v1) == float(v2))
                        except (TypeError, ValueError):
                            same += int(str(v1) == str(v2))
                    rec["n_same_as_single"] = same
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    rec["wall_s"] = time.perf_counter() - t
                result["rpm"].append(rec)
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                if "error" in rec:
                    break

    try:
        call(go())
    finally:
        drv.close()
    ok_chunks = [r["chunk"] for r in result["rpm"] if "error" not in r and r.get("n_same_as_single") == r["chunk"]]
    result["max_ok_chunk"] = max(ok_chunks) if ok_chunks else None
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: result[k] for k in ("single_reads", "groups", "max_ok_chunk")}, ensure_ascii=False), flush=True)


# =====================================================================
# D-5: 一括版と逐次版の全点一致
# =====================================================================

def run_verify(slot_index, label):
    from rlkit import comm, comm_rpm
    from rlkit.env import SummerEnv
    from rlkit.parallel import slots
    out_dir = os.path.join(OUT_ROOT, "verify", f"{label}_slot{slot_index}")
    os.makedirs(out_dir, exist_ok=True)
    _launch(out_dir, [slot_index])
    comm_rpm.install()
    stats = {"calls": 0, "n_mismatch": 0, "mismatch_examples": []}
    for name in comm_rpm.PATCHED + (comm_rpm.THERMAL,):
        rpm_fn, seq_fn = getattr(comm, name), getattr(comm, name + "_sequential")

        def make(name=name, rpm_fn=rpm_fn, seq_fn=seq_fn):
            async def both(comms, *a, **k):
                r = await rpm_fn(comms, *a, **k)
                s = await seq_fn(comms, *a, **k)
                stats["calls"] += 1
                if r != s:
                    stats["n_mismatch"] += 1
                    if len(stats["mismatch_examples"]) < 5:
                        stats["mismatch_examples"].append({"fn": name, "rpm": repr(r)[:300], "seq": repr(s)[:300]})
                return r
            return both
        setattr(comm, name, make())
    kw = env_kwargs(rpm=False)
    kw.pop("comm_rpm")
    env = SummerEnv(slot=slots.make_slot(slot_index), **kw)
    acts = baseline_actions()
    t0 = time.time()
    try:
        env.reset()
        for t in range(T):
            _o, _r, done, _tr, info = env.step(acts[t])
            if done:
                break
    finally:
        env.close()
    res = {"label": label, "slot": slot_index, "elapsed_s": time.time() - t0, "E_hex": float(info["E_rep"]).hex(),
           "DR_hex": float(info["DR_rep"]).hex(), "calls": stats["calls"], "n_mismatch": stats["n_mismatch"],
           "mismatch_examples": stats["mismatch_examples"], "rpm_stats": dict(comm_rpm.COMM_RPM_STATS),
           "hex_match": float(info["E_rep"]).hex() == BASELINE_E_HEX and float(info["DR_rep"]).hex() == BASELINE_DR_HEX}
    res["passed"] = res["hex_match"] and res["n_mismatch"] == 0 and res["rpm_stats"]["fallback_chunks"] == 0
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return 0 if res["passed"] else 2


# =====================================================================
# 第 6 節: 組数 N の同時再生
# =====================================================================

class Sampler(threading.Thread):
    """1 秒ごとに、システム全体の CPU（論理コアごと）・利用可能メモリ、組のプロセスの常駐メモリを記録する。"""

    def __init__(self, parent_pid):
        super().__init__(daemon=True)
        self.parent_pid = parent_pid
        self.samples = []
        self._halt = threading.Event()

    def run(self):
        import psutil
        me = psutil.Process(self.parent_pid)
        psutil.cpu_percent(percpu=True)
        while not self._halt.wait(1.0):
            try:
                kids = me.children(recursive=True)
                rss_py = sum(p.memory_info().rss for p in kids if "python" in p.name().lower())
                emus = [p for p in psutil.process_iter(["name"]) if p.info["name"] == "Shizuku2.exe"]
                rss_emu = sum(p.memory_info().rss for p in emus)
                per = psutil.cpu_percent(percpu=True)
                self.samples.append({"t": time.time(), "cpu_total": float(np.mean(per)), "cpu_max_core": float(np.max(per)),
                                     "avail_gb": psutil.virtual_memory().available / 2 ** 30, "rss_py_gb": rss_py / 2 ** 30,
                                     "rss_emu_gb": rss_emu / 2 ** 30, "n_emu": len(emus), "n_py": len(kids)})
            except Exception:
                pass

    def stop(self):
        self._halt.set()


def run_scale(n, label, rpm):
    from rlkit.parallel import slots
    from rlkit.parallel.batched import SlotVecEnv
    if slots.N_SLOTS < n:
        raise SystemExit(f"環境変数 RLKIT_N_SLOTS を {n} 以上にして起動する（今は {slots.N_SLOTS}）")
    slot_indices = list(range(n))
    out_dir = os.path.abspath(os.path.join(OUT_ROOT, "scale", label))
    os.makedirs(out_dir, exist_ok=True)
    _launch(out_dir, slot_indices)
    actions = baseline_actions()
    sampler = Sampler(os.getpid())
    sampler.start()
    vec = SlotVecEnv(slot_indices, env_kwargs(rpm), T, factory="real")
    step_wait, finals = [], [None] * n
    t_run0 = None
    try:
        t_boot0 = time.time()
        vec.reset()
        boot_s = time.time() - t_boot0
        t_run0 = time.time()
        for t in range(T):
            ts = time.perf_counter()
            _o, _r, dones, infos = vec.step(np.repeat(actions[t][None, :], n, axis=0))
            step_wait.append(time.perf_counter() - ts)
            for i, d in enumerate(dones):
                if d:
                    finals[i] = infos[i]
    finally:
        t_run1 = time.time()
        vec.close()
        sampler.stop()
    hexes = [(float(f["E_rep"]).hex(), float(f["DR_rep"]).hex()) for f in finals]
    ok = [h == (BASELINE_E_HEX, BASELINE_DR_HEX) for h in hexes]
    mid = [s for s in sampler.samples if t_run0 + 30 <= s["t"] <= t_run1 - 10]
    sw = np.array(step_wait)
    res = {"n": n, "rpm": bool(rpm), "label": label, "time": datetime.datetime.now().isoformat(timespec="seconds"),
           "boot_s": round(boot_s, 1), "episode_s": round(t_run1 - t_run0, 1),
           "step_wait_s": {"mean": float(sw.mean()), "median": float(np.median(sw)), "p90": float(np.percentile(sw, 90)),
                           "p99": float(np.percentile(sw, 99)), "max": float(sw.max())},
           "all_hex_match": bool(all(ok)), "n_hex_match": int(sum(ok)),
           "cpu_total_mean_pct": float(np.mean([s["cpu_total"] for s in mid])) if mid else None,
           "cpu_busiest_core_mean_pct": float(np.mean([s["cpu_max_core"] for s in mid])) if mid else None,
           "avail_gb_start": float(sampler.samples[0]["avail_gb"]) if sampler.samples else None,
           "avail_gb_min": float(min(s["avail_gb"] for s in sampler.samples)) if sampler.samples else None,
           "rss_python_children_gb_max": float(max(s["rss_py_gb"] for s in sampler.samples)) if sampler.samples else None,
           "rss_emulators_gb_max": float(max(s["rss_emu_gb"] for s in sampler.samples)) if sampler.samples else None,
           "n_emulators_seen_max": int(max(s["n_emu"] for s in sampler.samples)) if sampler.samples else None,
           "episodes_per_hour": round(n * 3600.0 / (t_run1 - t_run0), 1)}
    if res["rss_python_children_gb_max"] is not None:
        res["rss_per_pair_gb"] = (res["rss_python_children_gb_max"] + res["rss_emulators_gb_max"]) / n
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "samples.json"), "w", encoding="utf-8") as f:
        json.dump(sampler.samples, f)
    np.save(os.path.join(out_dir, "step_wait.npy"), sw)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return 0 if res["all_hex_match"] else 2


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True, choices=["probe", "verify", "scale"])
    p.add_argument("--slot", type=int, default=6)
    p.add_argument("--n", type=int, default=7)
    p.add_argument("--label", default=None)
    p.add_argument("--rpm", action="store_true")
    a = p.parse_args(argv)
    from rlkit import runlog
    runlog.check_thread_env()
    if a.step == "probe":
        run_probe(a.slot, a.label or "rpm1")
    elif a.step == "verify":
        sys.exit(run_verify(a.slot, a.label or "verify1"))
    else:
        sys.exit(run_scale(a.n, a.label or f"{'rpm' if a.rpm else 'seq'}_n{a.n:02d}", a.rpm))


if __name__ == "__main__":
    main()
