"""起動時の安全装置の試験（指示書 V4-S 第 2-7 節・B-5）。実エミュレータも実組も使わない。"""
import json

import pytest

from rlkit.parallel import launch


def test_slot_limit_check():
    """組数の上限（PL2-8 相当）。夏季機の上限は 7 組で、重複も誤りとする。"""
    assert launch.SLOT_LIMIT == 7
    assert launch.check_slot_limit([0, 1, 2, 3, 4, 5, 6])["n_slots"] == 7
    with pytest.raises(launch.LaunchError):
        launch.check_slot_limit(list(range(8)))
    with pytest.raises(launch.LaunchError):
        launch.check_slot_limit([0, 0, 1])


def test_free_memory_check():
    """空きメモリの下限（H-11 相当）。"""
    rec = launch.check_free_memory(min_free_gb=0.0)
    assert rec["free_gb"] > 0.0 and rec["min_free_gb"] == 0.0
    with pytest.raises(launch.LaunchError):
        launch.check_free_memory(min_free_gb=10 ** 6)


def test_preflight_records_failure(tmp_path):
    """不合格のときも記録を残してから止める（組の掃除には進まない）。"""
    with pytest.raises(launch.LaunchError):
        launch.preflight(str(tmp_path), [0, 1], min_free_gb=10 ** 6)
    rec = json.load(open(tmp_path / "launch" / "preflight.json", encoding="utf-8"))
    assert rec["passed"] is False and "空きメモリ" in rec["error"]
    assert rec["slot_limit"]["n_slots"] == 2 and "cleanup" not in rec
    assert rec["log_dir"].endswith("launch")


def test_slot_event_log(tmp_path, monkeypatch):
    """組ごとの起動結果を組ごとの JSONL に残す（同時追記で行が混ざらないようにするため）。"""
    monkeypatch.setenv(launch.LAUNCH_LOG_ENV, str(tmp_path / "launch"))
    assert launch.log_slot_event(3, "emulator_started", pid=123)
    assert launch.log_slot_event(3, "bacnet_ready", pid=123)
    assert launch.log_slot_event(5, "emulator_started", pid=456)
    rows3 = [json.loads(l) for l in open(tmp_path / "launch" / "slot3.jsonl", encoding="utf-8")]
    rows5 = [json.loads(l) for l in open(tmp_path / "launch" / "slot5.jsonl", encoding="utf-8")]
    assert [r["event"] for r in rows3] == ["emulator_started", "bacnet_ready"]
    assert rows3[0]["slot"] == 3 and rows3[0]["pid"] == 123
    assert len(rows5) == 1 and rows5[0]["slot"] == 5          # 組ごとに別のファイル
    monkeypatch.delenv(launch.LAUNCH_LOG_ENV)
    assert launch.log_slot_event(3, "emulator_exited") is False
