"""rlkit.runlog — 走行記録の書き出しと、数値演算スレッドの環境変数の確認（現行 parallel/common.py の小道具）。"""
import datetime
import json
import os
import time

import numpy as np

from rlkit import constants


def now_iso():
    return datetime.datetime.now().isoformat()


def _default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_default) + "\n")


def retrying(fn, attempts=10, delay=3.0, errors=(PermissionError,)):
    """V7.1-S: ファイルの置き換え・複写が拒否されたとき（OneDrive の同期がファイルを掴んでいる間など）、
    delay 秒おきに attempts 回まで試す。戻り値は (fn の値, やり直した回数)。最後も失敗したら、その例外を投げる。"""
    for k in range(attempts):
        try:
            return fn(), k
        except errors:
            if k == attempts - 1:
                raise
            time.sleep(delay)


def replace_with_retries(src, dst, attempts=10, delay=3.0):
    """os.replace を retrying で行う。戻り値はやり直した回数。"""
    return retrying(lambda: os.replace(src, dst), attempts, delay)[1]


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)
    replace_with_retries(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_thread_env():
    """OMP/MKL/OPENBLAS_NUM_THREADS=1、KMP_BLOCKTIME=0（v2 第 5-8 節）が設定されていなければ停止する。"""
    bad = {k: os.environ.get(k) for k, v in constants.REQUIRED_THREAD_ENV.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f"数値演算スレッドの環境変数が未設定または値が違う: {bad}")
