"""段階 0 の診断 C-2（K-3、指示書 V2-S 第 2-3 節）: 7 組同時の決定論性の確認。学習は行わない。

学習用 6 組（0〜5）と評価用 1 組（6）で規則ベース最良軌跡を同時に再生し、各組の E・DR・eci を 16 進で
単独再生の値と比べる。評価用の組が一致すれば `eval_concurrent = true`、一致しなければ `false` とする。

単独再生の基準値は、C-1 の summary.json（`--reference`）があればそれを、無ければ指示書 P-3 の値を使う。

使い方（リポジトリルートから）: python -m rlkit.diag.c2_concurrent7 [--out runs/diag/V2S/C2]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 指示書 V2-S P-3（V1 夏季レポートの規則ベース最良軌跡の再生値）
REFERENCE_E_GJ = 4.587348191951224
REFERENCE_DR = 0.18481666361514731


def replay_worker(slot_index, out_dir, p2_offset_min):
    """子プロセス: 組 slot_index で規則ベース最良軌跡を 1 エピソード再生し、結果を JSON へ書く。"""
    os.chdir(_REPO_ROOT)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from rlkit import comm, driver, eci as eci_mod, timing
    from rlkit.parallel import slots

    result = {"slot": slot_index, "ok": False}
    t0 = time.time()
    drv = None
    try:
        slot = slots.make_slot(slot_index)
        drv = driver.EpisodeDriver(slot=slot)
        drv.start()
        schedule = driver.make_summer_best_schedule(p2_offset_min)
        run = lambda coro: driver.run_coro(drv, coro, timeout=300)
        dt = run(drv.true_sim_datetime())
        n_steps = 0
        while True:
            run(schedule(drv, dt, timing.is_system_boundary(dt)))
            dt = run(drv.advance(timing.zone_step_seconds(dt)))
            n_steps += 1
            if drv.final_dtime is not None and dt >= drv.final_dtime:
                run(drv.park())
                break
        E = run(comm.call_write(drv.comms.env.get_total_energy_consumption(), "total energy")) / 1000.0
        DR = run(comm.call_write(drv.comms.env.get_averaged_dissatisfaction_rate(), "avg DR"))
        smooth = eci_mod.eci_smooth(E, DR)
        result.update({"ok": True, "n_steps": n_steps, "E": E, "DR": DR, "eci_smooth": smooth,
                       "eci_official": eci_mod.eci(E, DR), "E_hex": float.hex(E), "DR_hex": float.hex(DR),
                       "eci_smooth_hex": float.hex(smooth)})
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if drv is not None:
            try:
                drv.close()
            except Exception:
                pass
        result["wall_s"] = time.time() - t0
        with open(os.path.join(out_dir, f"slot{slot_index}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def main(argv=None):
    os.chdir(_REPO_ROOT)
    from rlkit import config as config_mod
    from rlkit import driver, eci as eci_mod, runlog
    from rlkit.parallel import slots

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join("runs", "diag", "V2S", "C2"))
    p.add_argument("--reference", default=os.path.join("runs", "diag", "V2S", "C1", "summary.json"))
    p.add_argument("--timeout", type=float, default=1800.0)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    cfg = config_mod.load()
    slot_indices = list(cfg["train_slots"]) + [cfg["eval_slot"]]
    trajectory = driver.load_summer_best_trajectory()
    p2 = trajectory["p2_offset_min"]

    reference = {"E": REFERENCE_E_GJ, "DR": REFERENCE_DR, "source": "指示書 V2-S P-3"}
    if os.path.exists(a.reference):
        with open(a.reference, encoding="utf-8") as f:
            c1 = json.load(f)
        reference = {"E": c1["E_rep"], "DR": c1["DR_rep"], "source": a.reference}
    reference["eci_smooth"] = eci_mod.eci_smooth(reference["E"], reference["DR"])
    reference["E_hex"] = float.hex(reference["E"])
    reference["DR_hex"] = float.hex(reference["DR"])
    reference["eci_smooth_hex"] = float.hex(reference["eci_smooth"])

    replica_checks = [slots.prepare_replica(slots.make_slot(k)) for k in slot_indices]

    ctx = mp.get_context("spawn")
    t0 = time.time()
    procs = [ctx.Process(target=replay_worker, args=(k, os.path.abspath(a.out), p2)) for k in slot_indices]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=max(a.timeout - (time.time() - t0), 1.0))
    wall_s = time.time() - t0
    for proc in procs:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)

    results = {}
    for k in slot_indices:
        path = os.path.join(a.out, f"slot{k}.json")
        results[k] = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"slot": k, "ok": False,
                                                                                           "error": "結果ファイルが無い"}
    def matches(r):
        return bool(r.get("ok")) and r["E_hex"] == reference["E_hex"] and r["DR_hex"] == reference["DR_hex"] \
            and r["eci_smooth_hex"] == reference["eci_smooth_hex"]

    eval_slot = cfg["eval_slot"]
    eval_ok = matches(results[eval_slot])
    summary = {"slot_indices": slot_indices, "eval_slot": eval_slot, "reference": reference,
               "results": results, "matches": {k: matches(r) for k, r in results.items()},
               "all_completed": all(r.get("ok") for r in results.values()),
               "all_match": all(matches(r) for r in results.values()),
               "eval_slot_matches": eval_ok, "eval_concurrent": eval_ok,
               "concurrent_wall_s": wall_s, "replica_checks": replica_checks, "finished_at": runlog.now_iso()}
    runlog.write_json(os.path.join(a.out, "determinism7.json"), summary)
    print(f"eval_concurrent = {eval_ok}")
    print(f"all_completed = {summary['all_completed']} all_match = {summary['all_match']} wall_s = {wall_s:.1f}")
    for k, r in results.items():
        print(f"  slot{k}: ok={r.get('ok')} E={r.get('E')} DR={r.get('DR')} wall_s={r.get('wall_s')}")


if __name__ == "__main__":
    main()
