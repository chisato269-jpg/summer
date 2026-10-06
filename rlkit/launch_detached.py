"""rlkit.launch_detached — 指示書 V7.1-S C-15 (1): 学習を、起動した端末や Claude の終了に巻き込まれない形で起動する。

Windows で、起動したシェルの木（taskkill /T や端末の終了で終わる範囲）とジョブから切り離した別のプロセスとして
コマンドを起動し、すぐに戻る。使い方:

    python -m rlkit.launch_detached --log runs/v2s_trial08/stdout_resume.log -- \\
        python -m rlkit.train_summer --config rlkit/config_summer_v7p.json --resume

切り離しの方法: DETACHED_PROCESS（コンソールを持たない）・CREATE_NEW_PROCESS_GROUP（Ctrl+C を受けない）・
CREATE_BREAKAWAY_FROM_JOB（親のジョブから外れる）。ジョブが外れることを許していない環境では最後の 1 つを外して
やり直し、その事実を標準出力に書く（JSON の "breakaway": false）。起動したプロセスの PID を --pidfile にも書く。

できないこと: サインアウトや PC の再起動は防げない（利用者のプロセスが OS によって終了される。2026-10-01 の終了の原因）。
"""
import argparse
import json
import os
import subprocess
import sys
import time

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000

THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "KMP_BLOCKTIME": "0",
              "PYTHONIOENCODING": "utf-8"}


def launch(cmd, log_path, cwd=None, pidfile=None, env_extra=None):
    """cmd を切り離して起動し、{"pid", "breakaway", "log", "started_at"} を返す。"""
    env = dict(os.environ)
    env.update(THREAD_ENV)
    env.update(env_extra or {})
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log = open(log_path, "ab")
    base = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    last_error = None
    for breakaway in (True, False):
        flags = base | (CREATE_BREAKAWAY_FROM_JOB if breakaway else 0)
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=flags, close_fds=True)
            break
        except OSError as exc:                       # ジョブが切り離しを許していない場合（アクセス拒否）
            last_error = repr(exc)
            proc = None
    if proc is None:
        raise RuntimeError(f"起動できなかった: {last_error}")
    info = {"pid": proc.pid, "breakaway": breakaway, "breakaway_error": last_error if not breakaway else None,
            "log": os.path.abspath(log_path), "cmd": list(cmd), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if pidfile:
        with open(pidfile, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description="コマンドを、起動した端末から切り離して起動する（C-15）")
    p.add_argument("--log", required=True, help="標準出力・標準エラーの追記先")
    p.add_argument("--pidfile", default=None, help="起動したプロセスの情報を書く JSON")
    p.add_argument("--cwd", default=None)
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- の後に起動するコマンド")
    a = p.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        raise SystemExit("起動するコマンドが無い（-- の後に書く）")
    print(json.dumps(launch(cmd, a.log, a.cwd, a.pidfile), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
