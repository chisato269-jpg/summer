"""指示書 V9-S 第 6 章（凍結の証跡 F-1・F-3）: コード・設定・エミュレータの複製の SHA-256 の一覧を作り、あとで照合する。

対象（V9S-c）:
  コード      起動に使うコードの複製（このファイルの親の親）の rlkit/ の下の全ファイル（__pycache__・*.pyc を除く。設定 config_summer_v9.json を含む）、
              szklib/ の *.py、verify/exogenous/summer.npz
  原本        Shizuku2/（原本）と、13 組の複製（~/rl_local/emulators/Shizuku2_slot{0..12}）のうち、実行で書き換わる出力を除いたもの
              （data/、error.log、slot_manifest.json、*.log を除く）
一覧そのものの SHA-256（正規化した JSON の SHA-256）も記録する。

  python -m rlkit.diag.v9s_freeze --make <manifest.json>            # F-1（起動の直前）
  python -m rlkit.diag.v9s_freeze --check <manifest.json> [<out>]   # F-3（終了の後）。一致しなければ終了コード 1
"""
import argparse
import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rlkit import localdata

EXCLUDE_DIRS = {"__pycache__", ".pytest_cache"}
REPLICA_EXCLUDE_TOP = {"data", "error.log", "slot_manifest.json"}
REPLICA_SLOTS = list(range(13))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _walk(root, skip_top=(), skip_ext=(".pyc",), only_ext=None):
    for d, dirs, files in os.walk(root):
        rel_dir = os.path.relpath(d, root)
        dirs[:] = sorted(x for x in dirs if x not in EXCLUDE_DIRS and not (rel_dir == "." and x in skip_top))
        for f in sorted(files):
            if rel_dir == "." and f in skip_top:
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in skip_ext or (only_ext is not None and ext not in only_ext):
                continue
            yield os.path.join(d, f)


def collect(repo_root=REPO_ROOT, emulator_root=None, slots=REPLICA_SLOTS):
    emulator_root = emulator_root or localdata.EMULATOR_ROOT
    out = {}
    for p in _walk(os.path.join(repo_root, "rlkit")):
        out["code/" + os.path.relpath(p, repo_root).replace("\\", "/")] = sha256_file(p)
    for p in _walk(os.path.join(repo_root, "szklib"), only_ext={".py"}):
        out["code/" + os.path.relpath(p, repo_root).replace("\\", "/")] = sha256_file(p)
    exo = os.path.join(repo_root, "verify", "exogenous", "summer.npz")
    out["code/verify/exogenous/summer.npz"] = sha256_file(exo)
    orig = os.path.join(repo_root, "Shizuku2")
    for p in _walk(orig, skip_top=REPLICA_EXCLUDE_TOP, skip_ext=(".pyc", ".log")):
        out["original/" + os.path.relpath(p, orig).replace("\\", "/")] = sha256_file(p)
    for k in slots:
        r = os.path.join(emulator_root, f"Shizuku2_slot{k}")
        for p in _walk(r, skip_top=REPLICA_EXCLUDE_TOP, skip_ext=(".pyc", ".log")):
            out[f"slot{k}/" + os.path.relpath(p, r).replace("\\", "/")] = sha256_file(p)
    return out


def canonical_sha256(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=True).encode("ascii")).hexdigest()


def make(path):
    files = collect()
    rec = {"created_at": __import__("datetime").datetime.now().isoformat(), "repo_root": REPO_ROOT,
           "emulator_root": localdata.EMULATOR_ROOT, "n_files": len(files), "files_sha256": canonical_sha256(files), "files": files}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=0)
    print(json.dumps({"manifest": path, "n_files": rec["n_files"], "files_sha256": rec["files_sha256"],
                      "manifest_file_sha256": sha256_file(path)}, ensure_ascii=False))
    return rec


def check(path, out=None):
    old = json.load(open(path, encoding="utf-8"))
    new = collect(old["repo_root"], old["emulator_root"])
    missing = sorted(set(old["files"]) - set(new))
    added = sorted(set(new) - set(old["files"]))
    changed = sorted(k for k in set(old["files"]) & set(new) if old["files"][k] != new[k])
    res = {"manifest": path, "n_files_before": len(old["files"]), "n_files_after": len(new), "files_sha256_before": old["files_sha256"],
           "files_sha256_after": canonical_sha256(new), "missing": missing, "added": added, "changed": changed,
           "identical": not (missing or added or changed)}
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else v[:20]) for k, v in res.items()}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make")
    ap.add_argument("--check")
    ap.add_argument("out", nargs="?")
    a = ap.parse_args()
    if a.make:
        make(a.make)
    elif a.check:
        sys.exit(0 if check(a.check, a.out)["identical"] else 1)
    else:
        ap.error("--make か --check が要る")
