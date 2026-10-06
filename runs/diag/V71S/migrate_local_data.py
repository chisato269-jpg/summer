"""エミュレータの組の複製と、診断の分単位の出力（CSV の保全）を、OneDrive の外（rlkit.localdata）へ移す。

  python runs/diag/V71S/migrate_local_data.py --dry-run     # 計画だけを表示
  python runs/diag/V71S/migrate_local_data.py               # 実行

移すのは「移動」だけで、削除はしない（同じドライブ内の改名。同期の側からは削除に見える）。診断の出力は、移す前後で全ファイルの
SHA-256 を比べる。記録は runs/diag/local_data_manifest.json（OneDrive に残す）。原本 Shizuku2/ は動かさない。
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from rlkit import localdata, runlog  # noqa: E402
from rlkit.parallel import slots  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
CSV_DIRS = [os.path.join("runs", "diag", "V6S", "slot_output"), os.path.join("runs", "diag", "V6S", "K10", "source"),
            os.path.join("runs", "diag", "V61S", "slot_output"), os.path.join("runs", "diag", "V63S", "R8_baseline_output")]
MANIFEST = os.path.join("runs", "diag", "local_data_manifest.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree(path):
    out = {}
    for d, _x, fs in os.walk(path):
        for f in fs:
            fp = os.path.join(d, f)
            out[os.path.relpath(fp, path)] = (os.path.getsize(fp), sha(fp))
    return out


def move(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    runlog.retrying(lambda: shutil.move(src, dst), attempts=10, delay=3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import psutil
    running = [p.pid for p in psutil.process_iter(["name"]) if p.info["name"] == "Shizuku2.exe"]
    if running:
        raise SystemExit(f"エミュレータが動いている（PID {running}）。止めてから実行する")
    replicas = sorted(d for d in os.listdir(".") if d.startswith("Shizuku2_slot") and os.path.isdir(d))
    partial = [d for d in replicas if d.endswith(".partial")]
    if partial:
        raise SystemExit(f"途中で止まった複製の跡がある: {partial}")
    plan = {"replicas": [(d, os.path.join(localdata.EMULATOR_ROOT, d)) for d in replicas],
            "csv_dirs": [(d, localdata.diag_csv_dir(d)) for d in CSV_DIRS if os.path.isdir(d)]}
    for src, dst in plan["replicas"] + plan["csv_dirs"]:
        if os.path.exists(dst):
            raise SystemExit(f"移動先が既にある（上書きしない）: {dst}")
    mb = lambda p: sum(os.path.getsize(os.path.join(d, f)) for d, _x, fs in os.walk(p) for f in fs) / 2 ** 20
    print("移動先の根:", localdata.LOCAL_ROOT)
    for src, dst in plan["replicas"] + plan["csv_dirs"]:
        print(f"  {mb(src):7.1f} MB  {src}  ->  {dst}")
    if a.dry_run:
        return
    manifest = {"moved_at": runlog.now_iso(), "local_root": localdata.LOCAL_ROOT, "replicas": [], "csv_dirs": []}
    for src, dst in plan["csv_dirs"]:
        before = tree(src)
        move(src, dst)
        after = tree(dst)
        assert before == after, f"移動の前後で内容が違う: {src}"
        os.makedirs(src, exist_ok=True)
        with open(os.path.join(src, "LOCATION.txt"), "w", encoding="utf-8") as f:
            f.write(f"この診断の出力（エミュレータの分単位の CSV など）は、容量が大きいため OneDrive の外に移した。\n場所: {dst}\n"
                    f"全ファイルの SHA-256 は runs/diag/local_data_manifest.json にある（移動の前後で一致を確認済み）。\n")
        manifest["csv_dirs"].append({"from": src.replace("\\", "/"), "to": dst, "n_files": len(after),
                                     "mb": round(sum(v[0] for v in after.values()) / 2 ** 20, 1),
                                     "files": {k: {"size": v[0], "sha256": v[1]} for k, v in sorted(after.items())}})
        print("移した（内容一致）:", src)
    for src, dst in plan["replicas"]:
        size = mb(src)
        move(src, dst)
        manifest["replicas"].append({"from": src, "to": dst, "mb": round(size, 1)})
        print("移した:", src)
    runlog.write_json(MANIFEST, manifest)
    # 移した複製を原本と照合する。error.log だけが違う場合は、退避してから原本の内容に戻す（これまでの手順と同じ）
    keep = os.path.join("runs", "diag", "V71S", "replica_error_logs_migrate")
    orig_log = os.path.join(slots.ORIGINAL_DIR, "error.log")
    results = []
    for src, dst in plan["replicas"]:
        k = int(src.split("slot")[1])
        slot = slots.make_slot(k)
        try:
            slots.verify_replica(slot)
            results.append((k, "ok"))
        except slots.SlotError as exc:
            lg = os.path.join(dst, "error.log")
            if "error.log" in str(exc) and os.path.exists(lg) and os.path.exists(orig_log):
                os.makedirs(keep, exist_ok=True)
                shutil.copy2(lg, os.path.join(keep, f"slot{k:02d}_error.log"))
                shutil.copy2(orig_log, lg)
                try:
                    slots.verify_replica(slot)
                    results.append((k, "error.log を退避して原本に戻し、照合 ok"))
                    continue
                except slots.SlotError as exc2:
                    exc = exc2
            results.append((k, f"照合に失敗: {str(exc)[:200]}"))
    for k, r in results:
        print(f"  組 {k}: {r}")
    manifest["replica_verify"] = {str(k): r for k, r in results}
    runlog.write_json(MANIFEST, manifest)


if __name__ == "__main__":
    main()
