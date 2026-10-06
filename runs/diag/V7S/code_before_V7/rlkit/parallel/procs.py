"""rlkit.parallel.procs — PID による終了、子孫プロセスの追跡、組の中に限った残留の検出
（指示書PL2 第6-1節）。

判定はすべて PID・生成時刻・実行ファイルのパスで行い、プロセスの名前は一切使わない
（不変条件PL2-f）。PID の再利用で無関係なプロセスを終了させないよう、記録した生成時刻と
一致しないプロセスは「別のプロセス」とみなして触れない。
"""
import os

import psutil

# 生成時刻の比較に使う許容幅[秒]。psutil は同じプロセスに対して同じ値を返すが、
# JSON を経由した往復で丸めが入っても一致とみなせるように小さな幅を持たせる。
CREATE_TIME_TOLERANCE_S = 0.01


def norm_path(path):
    if not path:
        return None
    return os.path.normcase(os.path.abspath(path))


def path_is_inside(path, root):
    """path が root ディレクトリの中（root 自身を含む）にあるか。"""
    p, r = norm_path(path), norm_path(root)
    if p is None or r is None:
        return False
    r = r.rstrip("\\/")
    return p == r or p.startswith(r + os.sep)


def same_path(a, b):
    pa, pb = norm_path(a), norm_path(b)
    return pa is not None and pa == pb


def _create_time_matches(actual, expected):
    return expected is None or (actual is not None and abs(actual - expected) <= CREATE_TIME_TOLERANCE_S)


class ProcApi:
    """psutil を薄く包んだプロセス操作。単体試験では同じ形の偽物に差し替える。"""

    def create_time(self, pid):
        try:
            return psutil.Process(pid).create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            return None

    def is_alive(self, pid, create_time=None):
        if pid is None:
            return False
        try:
            p = psutil.Process(pid)
            if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
                return False
            return _create_time_matches(p.create_time(), create_time)
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied:
            return create_time is None

    def exe(self, pid):
        try:
            return psutil.Process(pid).exe()
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            return None

    def kill(self, pid, create_time=None, timeout=10.0):
        """PID で強制終了し、消滅を待つ。生成時刻が一致しなければ（元のプロセスはすでに
        消えている）何もしない。戻り値: 対象がもう存在しなければ True。"""
        try:
            p = psutil.Process(pid)
            if not _create_time_matches(p.create_time(), create_time):
                return True
            p.kill()
            p.wait(timeout=timeout)
            return True
        except psutil.NoSuchProcess:
            return True
        except psutil.TimeoutExpired:
            return False

    def children(self, pid):
        """pid の子孫プロセス（再帰）を {pid, create_time, exe} の一覧で返す。"""
        try:
            kids = psutil.Process(pid).children(recursive=True)
        except psutil.NoSuchProcess:
            return []
        out = []
        for c in kids:
            try:
                out.append({"pid": c.pid, "create_time": c.create_time(), "exe": c.exe()})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return out

    def processes_with_exe_inside(self, root):
        """実行ファイルが root の中にあるプロセスの一覧（組の中に限った残留の検出）。"""
        out = []
        for p in psutil.process_iter(["pid", "exe", "create_time"]):
            exe = p.info.get("exe")
            if exe and path_is_inside(exe, root):
                out.append({"pid": p.info["pid"], "create_time": p.info.get("create_time"), "exe": exe})
        return out


DEFAULT_API = ProcApi()
