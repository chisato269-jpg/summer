"""走行状況を、テキストで 3 行に短く表示する（スマホの Claude アプリのコードタブなど、文字だけの画面向け）。

  python -m rlkit.diag.status_text

走行ディレクトリは読むだけ。稼働の判定は updates.jsonl の最終更新から 20 分以内かどうか（20〜60 分は「止まっている可能性」、それ以上は「停止」）。
"""
import json
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from rlkit import localdata

_RUNS = localdata.OLD_RUNS_ROOT


def _latest_run():
    """更新の記録（updates.jsonl）が最も新しい走行。OneDrive の外（~/rl_local/runs）と以前の走行（runs/）の両方から探す。
    計画更新数は走行の最初の meta_start の設定から読む。"""
    return localdata.latest_run()


_B = _latest_run()
RUN_DIR = _B[1] if _B else os.path.join(_RUNS, "v2s_trial08")
PLANNED = _B[2] if _B else 1500
RUN_NAME = _B[3] if _B else "v2s_trial08"


def ago(sec):
    m = int(sec // 60)
    return f"{m} 分前" if m < 120 else f"{m / 60:.1f} 時間前"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    path = os.path.join(RUN_DIR, "updates.jsonl")
    lines = open(path, encoding="utf-8").read().splitlines()
    u = json.loads(lines[-1])
    age = time.time() - os.path.getmtime(path)
    n = u["update"]
    stop_file = os.path.exists(os.path.join(RUN_DIR, "STOP"))
    if n >= PLANNED:
        state = "完了"
    elif stop_file:
        state = "■ 停止指示あり"
    elif age <= 1200:
        state = "● 稼働中"
    elif age <= 3600:
        state = "▲ 止まっている可能性"
    else:
        state = "■ 停止"
    reason = ""
    try:                                    # 走行が自分で止まった（status.json があり、更新番号が同じ）なら、最終更新が新しくても停止と表示する
        _st = json.load(open(os.path.join(RUN_DIR, "status.json"), encoding="utf-8"))
        if _st.get("update_index") == n and state == "● 稼働中":
            state = "■ 停止"
    except Exception:
        pass
    if state.startswith("■"):
        try:
            st = json.load(open(os.path.join(RUN_DIR, "status.json"), encoding="utf-8"))
            if st.get("update_index") == n and st.get("stop_reason"):
                reason = "（例外で停止）" if st.get("status") == "error" else "（" + str(st["stop_reason"])[:30] + "）"
        except Exception:
            pass
    recent = [json.loads(l) for l in lines[-20:]]
    sec = sorted(r["update_seconds"] for r in recent)[len(recent) // 2]
    c13 = [r["c13"] for r in recent if r.get("c13")]
    best = u["harvest_best"]
    alive = sum(1 for p in _emulators())
    print(f"{RUN_NAME} {state}{reason}  更新 {n}/{PLANNED}（{100 * n / PLANNED:.0f}%）  最終更新 {ago(age)}")
    eta = f"  残り約 {(PLANNED - n) * sec / 86400:.1f} 日" if state == "● 稼働中" else ""
    print(f"最良 ECI {best['ECI']:.4f}（更新 {best['update']}）  直近の収穫 {u['harvest_update_ECI']:.4f}  {sec:.0f} 秒/更新{eta}")
    mem = f"  空き最小 {min(c['min_free_memory_gb'] for c in c13):.1f} GB  戻し {sum(c['rpm_fallback_chunks'] for c in c13)} 回" if c13 else ""
    print(f"エミュレータ {alive} 台{mem}")


def _emulators():
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Shizuku2.exe", "/NH"], capture_output=True, text=True).stdout
        return [l for l in out.splitlines() if "Shizuku2.exe" in l]
    except Exception:
        return []


if __name__ == "__main__":
    main()
