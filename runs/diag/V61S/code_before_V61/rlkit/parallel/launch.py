"""rlkit.parallel.launch — 起動時の安全装置（指示書 V4-S 第 2-7 節。照会 W-7 への回答）。

20 側 `rlkit/parallel/launch.py`（指示書 PL2 の K-b・K-f・H-11）が持つ 3 つの検査だけを、21(s) の
学習の起動経路へ移した。型 B（独立並列）の起動機構そのものは移していない（21(s) は型 A だけを使う）。

  組数の上限   : 同時に使う組の数が SLOT_LIMIT を超えないこと（PL2-8 の N_pass 検査に相当）
  空きメモリ   : 起動前の空きメモリが MIN_FREE_MEM_GB 以上であること（H-11 の下限に相当）
  孤児の掃除   : 前回の走行が残したロックと、複製の中の実行ファイルで動いているプロセスを片づける（K-f）

掃除は「その組を持っている学習側プロセスが生きていない」ことを確かめてから行う。生きていれば、
その組は使用中として起動を拒否する（走行中の別セッションを殺さないため）。
プロセスの判定と終了は PID と生成時刻だけで行い、名前は使わない（PL2-f）。

組ごとの起動結果（開始・BACnet 応答・終了）は、環境変数 LAUNCH_LOG_ENV が指すディレクトリの
slot{k}.jsonl へ追記する。子プロセス（組ごとのワーカー）は spawn で環境変数を受け継ぐ。
組ごとにファイルを分けるのは、1 つのファイルへ 7 プロセスが同時に追記すると行が混ざるためである
（試走 runs/v2s_trial03 で 741 行中 3 行が混線した。組ごとなら書き手は同時に 1 つだけになる）。
"""
import datetime
import json
import os

import psutil

from rlkit.parallel import procs, slots

# 夏季機では 7 組同時（学習 6 ＋ 評価 1）の全桁一致を実測で確認済み（V2-S 診断 C-2）。
# 指示書 V4-S 第 2-7 節「夏季機では 7 組同時が動作しているため値は変えない」に従いこの値を上限とする。
SLOT_LIMIT = 7
MIN_FREE_MEM_GB = 2.0
LAUNCH_LOG_ENV = "RLKIT_LAUNCH_LOG"
LAUNCH_DIR_NAME = "launch"
LAUNCH_LOG_NAME_FMT = "slot{}.jsonl"
PREFLIGHT_NAME = "preflight.json"


class LaunchError(RuntimeError):
    """起動前の検査で不合格だった。"""


def _now():
    return datetime.datetime.now().isoformat(timespec="microseconds")


def check_slot_limit(slot_indices, limit=SLOT_LIMIT):
    """同時に使う組数の上限（PL2-8 相当）。重複も誤りとする。"""
    idx = list(slot_indices)
    if len(set(idx)) != len(idx):
        raise LaunchError(f"組番号が重複している: {idx}")
    if len(idx) > limit:
        raise LaunchError(f"同時に使う組数 {len(idx)} が上限 {limit} を超える（指示書 V4-S 第 2-7 節）")
    return {"n_slots": len(idx), "limit": limit, "slots": idx}


def free_memory_gb():
    return psutil.virtual_memory().available / 2 ** 30


def check_free_memory(min_free_gb=MIN_FREE_MEM_GB):
    """起動前の空きメモリの下限（H-11 相当）。"""
    free = free_memory_gb()
    rec = {"free_gb": free, "min_free_gb": min_free_gb}
    if free < min_free_gb:
        raise LaunchError(f"空きメモリ {free:.2f} GB が下限 {min_free_gb} GB を下回る（指示書 V4-S 第 2-7 節）")
    return rec


def cleanup_stale_slots(slot_indices, api=None):
    """組ごとに、死んだ所有者のロックを外し、複製の中で動いている残留プロセスを終了させる（K-f 相当）。"""
    api = api or procs.DEFAULT_API
    reports = []
    for k in slot_indices:
        slot = slots.make_slot(k)
        rep = {"slot": k, "lock": None, "lock_released": False, "killed": [], "residual_after": []}
        lock = slots.SlotLock(slot, api)
        rec = lock.read()
        if rec is not None:
            rep["lock"] = {"learner_pid": rec.get("learner_pid"), "emulator_pid": rec.get("emulator_pid")}
            if api.is_alive(rec.get("learner_pid"), rec.get("learner_create_time")):
                raise LaunchError(f"組{k}は使用中（学習側 PID {rec.get('learner_pid')} が生きている）。起動を拒否する")
            rep["lock_released"] = lock.release_if_owner_dead(rec.get("learner_pid"))
        if os.path.isdir(slot.replica_dir):
            for p in api.processes_with_exe_inside(slot.replica_dir):
                rep["killed"].append({"pid": p["pid"], "exe": p["exe"],
                                      "gone": api.kill(p["pid"], p.get("create_time"))})
            rep["residual_after"] = api.processes_with_exe_inside(slot.replica_dir)
        if rep["residual_after"]:
            raise LaunchError(f"組{k}の複製の中に残留プロセスがある: {rep['residual_after']}")
        reports.append(rep)
    return reports


def preflight(run_dir, slot_indices, api=None, min_free_gb=MIN_FREE_MEM_GB, limit=SLOT_LIMIT):
    """3 つの検査をこの順で行い、結果を <run_dir>/launch/preflight.json に残して報告を返す。

    どれかが不合格なら LaunchError を投げる（学習は始めない）。
    戻り値には組ごとの起動記録を置くディレクトリを入れる。呼び出し側はそれを LAUNCH_LOG_ENV に設定する。
    """
    launch_dir = os.path.join(run_dir, LAUNCH_DIR_NAME)
    os.makedirs(launch_dir, exist_ok=True)
    report = {"time": _now(), "pid": os.getpid(), "slots": list(slot_indices), "log_dir": launch_dir}
    try:
        report["slot_limit"] = check_slot_limit(slot_indices, limit=limit)
        report["memory"] = check_free_memory(min_free_gb=min_free_gb)
        report["cleanup"] = cleanup_stale_slots(slot_indices, api=api)
        report["passed"] = True
    except LaunchError as exc:
        report["passed"] = False
        report["error"] = str(exc)
        with open(os.path.join(launch_dir, PREFLIGHT_NAME), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        raise
    with open(os.path.join(launch_dir, PREFLIGHT_NAME), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def log_slot_event(slot_index, event, **fields):
    """組ごとの起動結果を slot{k}.jsonl へ 1 行追記する。環境変数が無ければ何もしない。"""
    log_dir = os.environ.get(LAUNCH_LOG_ENV)
    if not log_dir:
        return False
    path = os.path.join(log_dir, LAUNCH_LOG_NAME_FMT.format(int(slot_index)))
    rec = {"time": _now(), "pid": os.getpid(), "slot": slot_index, "event": event}
    rec.update(fields)
    line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        return False
    return True
