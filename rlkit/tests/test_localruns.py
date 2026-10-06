"""新しい走行の記録を OneDrive の外（RUNS_ROOT）に置く仕組みの試験（通信量の対策）。エミュレータは使わない（模擬の環境）。"""
import os

from rlkit import localdata
from rlkit.v8 import trainer3 as tr3

from test_v82 import _start_from_first_update, _trainer_cfg


def _repo_tree(root):
    out = set()
    for d, _dirs, files in os.walk(root):
        if os.sep + "__pycache__" in d or ".pytest_cache" in d:
            continue
        for f in files:
            out.add(os.path.join(d, f))
    return out


def test_run_dir_outside_the_project_writes_nothing_inside_and_status_finds_it(tmp_path, monkeypatch):
    e_hex, d_hex = _start_from_first_update(tmp_path)
    local = tmp_path / "local"
    monkeypatch.setattr(localdata, "RUNS_ROOT", str(local / "runs"))
    monkeypatch.setattr(localdata, "OLD_RUNS_ROOT", str(tmp_path / "no_old_runs"))
    cfg = dict(_trainer_cfg(tmp_path / "x", planned=3), start_E_hex=e_hex, start_DR_hex=d_hex)
    cfg["run_dir"] = localdata.run_dir_for("v8s_trialXX")                       # 絶対パス（OneDrive の外）
    repo_runs = os.path.join(localdata._REPO_ROOT, "runs")
    before = _repo_tree(repo_runs)
    t = tr3.V83Trainer(cfg, cfg["run_dir"], factory="mock_v8")
    t.run()
    assert t.update_index == 3
    after = _repo_tree(repo_runs)
    assert after == before                                                        # プロジェクトの runs/ に、記録は 1 つも出来ない
    assert os.path.exists(os.path.join(cfg["run_dir"], "updates.jsonl"))
    b = localdata.latest_run()
    assert b is not None and os.path.normpath(b[1]) == os.path.normpath(cfg["run_dir"])
    assert b[2] == 3 and b[3] == "v8s_trialXX"                                  # 計画更新数は最初の meta_start の設定から読む


def test_work_dir_is_created_under_the_local_work_root(tmp_path, monkeypatch):
    monkeypatch.setattr(localdata, "WORK_ROOT", str(tmp_path / "work"))
    p = localdata.work_dir("V84S", "calib")
    assert os.path.isdir(p) and p == os.path.join(str(tmp_path / "work"), "V84S", "calib")


def test_latest_run_picks_the_newest_across_both_roots(tmp_path, monkeypatch):
    new_root, old_root = tmp_path / "new", tmp_path / "old"
    for root, name, t in ((new_root, "a", 1000), (old_root, "b", 2000), (old_root, "v2s_trial08", 500)):
        d = root / name
        d.mkdir(parents=True)
        f = d / "updates.jsonl"
        f.write_text("{}\n")
        os.utime(f, (t, t))
    monkeypatch.setattr(localdata, "RUNS_ROOT", str(new_root))
    monkeypatch.setattr(localdata, "OLD_RUNS_ROOT", str(old_root))
    assert localdata.latest_run()[3] == "b"
    os.utime(new_root / "a" / "updates.jsonl", (3000, 3000))
    assert localdata.latest_run()[3] == "a"
    os.utime(new_root / "a" / "updates.jsonl", (100, 100))
    os.utime(old_root / "b" / "updates.jsonl", (100, 100))
    assert localdata.latest_run()[3:] == ("v2s_trial08",) and localdata.latest_run()[2] == 1500
