"""指示書 V7.1-S の単体試験（UT-42〜UT-44）。エミュレータは使わない（模擬の環境）。

実機を使う 8 本（test_v6_live.py・test_v64_live.py・test_v7_live.py）は RLKIT_LIVE=1 で別に行う。
"""
import json
import os

import numpy as np
import pytest

from rlkit import config, constants
from rlkit import train_summer as ts


def M(r=0.0, a=0.0, e=0.0, d=0.0):
    return {"energy_identity_rel": e, "diss_identity_rel": d, "r": r, "a_gj": a}


# ---- UT-42: C-11 の判定 ----

@pytest.mark.parametrize("m, stops", [
    (M(r=0.0683, a=0.0728), False),       # 実際の最大に近い値。a が 0.15 GJ 以下なので止めない
    (M(r=0.06, a=0.20), True),            # 両方を超えた
    (M(r=0.03, a=0.20), False),           # r が 5% 以下
    (M(r=-0.06, a=-0.30), True),          # 負の側も絶対値で判定する
    (M(r=0.04, a=0.18), False),
    (M(e=1e-8), True),                    # エネルギーの側の恒等式が 1e-9 を超えた
    (M(e=1e-9), False),                   # 1e-9 ちょうどは超えていない
    (M(d=2e-9), True),                    # 不満足の側の恒等式
    (M(r=0.05, a=0.15), False),           # 基準ちょうどは超えていない
])
def test_ut42_c11_rule_examples(m, stops):
    assert bool(ts.c11_stop_reasons([("決定論的評価", m)])) == stops


def test_ut42_one_of_twelve_training_episodes_stops():
    train = [dict(slot=k, **M(r=0.01, a=0.03)) for k in range(12)]
    rec = {"eval_c11": M(r=0.01, a=0.03), "train_c11": train}
    assert ts.c11_stop_reasons(ts.c11_entries_from_record(rec)) == []
    train[7] = dict(slot=7, **M(r=0.06, a=0.20))                       # 1 本だけが条件を満たす
    reasons = ts.c11_stop_reasons(ts.c11_entries_from_record(rec))
    assert len(reasons) == 1 and "組 7" in reasons[0]
    rec["eval_c11"] = M(e=1e-8)                                        # 決定論的評価だけが当たる場合も止まる
    assert any("決定論的評価" in r for r in ts.c11_stop_reasons(ts.c11_entries_from_record(rec)))


def test_ut42_metrics_from_info():
    info = {"sum_delta_E_gj": 4.0, "sum_zone_energy_gj": 3.6, "sum_header_energy_gj": 0.4 + 2e-8,
            "r8p_sum_rel": 3e-10, "raw_resid_rel": 0.0683, "raw_resid_sum_gj": 0.2732}
    m = ts.c11_metrics(info)
    assert m["energy_identity_rel"] == pytest.approx(5e-9, rel=1e-3) and m["diss_identity_rel"] == 3e-10
    assert (m["r"], m["a_gj"]) == (0.0683, 0.2732)
    assert ts.c11_stop_reasons([("x", m)])                              # 恒等式 5e-9 > 1e-9


def test_ut42_constants_unchanged():
    assert (constants.C11_ENERGY_IDENTITY_TOL, constants.C11_DISS_IDENTITY_TOL,
            constants.C11_GAP_R, constants.C11_GAP_A_GJ) == (1e-9, 1e-9, 0.05, 0.15)


def _cfg(tmp, **kw):
    c = config.load()
    c.update({"include_indoor_unit_power": False, "zone_step_energy_norm_gj": 0.01, "planned_updates": 3,
              "run_dir": str(tmp)})
    c.update(kw)
    return c


def _count_replays(monkeypatch):
    calls = []
    orig = ts.Trainer.verify_harvest_replay

    def counting(self):
        calls.append(1)
        return orig(self)
    monkeypatch.setattr(ts.Trainer, "verify_harvest_replay", counting)
    return calls


def test_records_c13_and_c11_in_updates(tmp_path):
    tr = ts.Trainer(_cfg(tmp_path, planned_updates=1), str(tmp_path), factory="mock")
    tr.run()
    u = json.loads(open(tmp_path / "updates.jsonl", encoding="utf-8").readline())
    c = u["c13"]
    for k in ("update_wall_s", "rpm_fallback_chunks", "parent_wait_median_s", "parent_wait_p99_s", "min_free_memory_gb",
              "eval_raw_resid_rel", "eval_raw_resid_a_gj", "train_raw_resid_rel", "train_raw_resid_a_gj"):
        assert k in c, k
    assert c["parent_wait_steps"] == tr.T * tr.eps_per_slot and c["parent_wait_p99_s"] >= c["parent_wait_median_s"] >= 0
    assert c["min_free_memory_gb"] > 0 and c["rpm_fallback_chunks"] == 0
    assert len(c["train_raw_resid_rel"]) == 12 and len(c["train_raw_resid_a_gj"]) == 12 and len(u["train_c11"]) == 12
    assert u["update_seconds"] == c["update_wall_s"] and "eval_c11" in u
    ep = json.loads(open(tmp_path / "episodes.jsonl", encoding="utf-8").readline())
    assert "c11" in ep and "raw_resid_sum_gj" in ep


# ---- UT-43: C-12（停止の理由を問わず F-1 を 1 回行う） ----

def test_ut43_replay_on_planned_end(tmp_path, monkeypatch):
    calls = _count_replays(monkeypatch)
    tr = ts.Trainer(_cfg(tmp_path, planned_updates=2), str(tmp_path), factory="mock")
    tr.run()
    assert tr.stop_reason.startswith("計画更新数") and len(calls) == 1
    assert tr.harvest_replay["F-1_passed"] is True


def test_ut43_replay_on_c11_stop(tmp_path, monkeypatch):
    calls = _count_replays(monkeypatch)
    monkeypatch.setattr(ts, "c11_stop_reasons", lambda entries: ["C-11: 試験"] if os.path.exists(tmp_path / "updates.jsonl") else [])
    tr = ts.Trainer(_cfg(tmp_path, planned_updates=10), str(tmp_path), factory="mock")
    tr.run()
    assert "C-11" in tr.stop_reason and tr.model.update_index == 1 and len(calls) == 1


def test_ut43_replay_on_collapse_watch_stop(tmp_path, monkeypatch):
    calls = _count_replays(monkeypatch)
    monkeypatch.setattr(constants, "COLLAPSE_WATCH_START", 2)
    monkeypatch.setattr(constants, "COLLAPSE_WATCH_EVERY", 2)
    monkeypatch.setattr(ts, "collapse_check", lambda h, u: {"update": u, "triggered": True, "median_recent": 0.1,
                                                            "best_previous": 0.5, "threshold": 0.425,
                                                            "block_medians_before": []})
    tr = ts.Trainer(_cfg(tmp_path, planned_updates=10), str(tmp_path), factory="mock")
    tr.run()
    assert tr.stop_reason.startswith("崩れの見張り") and len(calls) == 1


def test_ut43_replay_on_divergence_stop(tmp_path, monkeypatch):
    calls = _count_replays(monkeypatch)
    from rlkit.learner import SummerPPO
    orig = SummerPPO.train

    def nan_train(self, *a, **k):
        out = orig(self, *a, **k)
        out["nonfinite"] = True
        return out
    monkeypatch.setattr(SummerPPO, "train", nan_train)
    tr = ts.Trainer(_cfg(tmp_path, planned_updates=10), str(tmp_path), factory="mock")
    tr.run()
    assert tr.stop_reason.startswith("S-6") and len(calls) == 1


def test_ut43_replay_failure_retries_once_then_records(tmp_path, monkeypatch):
    attempts = []

    def failing(self):
        attempts.append(1)
        raise RuntimeError("エミュレータの不具合（模擬）")
    monkeypatch.setattr(ts.Trainer, "verify_harvest_replay", failing)
    tr = ts.Trainer(_cfg(tmp_path, planned_updates=1), str(tmp_path), factory="mock")
    tr.run()
    assert len(attempts) == 2                                           # 失敗後に 1 回だけやり直す
    hr = tr.harvest_replay
    assert hr["failed"] is True and hr["F-1_passed"] is None and len(hr["attempts"]) == 2
    status = json.load(open(tmp_path / "status.json", encoding="utf-8"))
    assert status["status"] == "stopped" and status["harvest_replay"]["failed"] is True
    ev = [json.loads(l) for l in open(tmp_path / "events.jsonl", encoding="utf-8")]
    assert sum(e.get("event") == "harvest_replay_failed" for e in ev) == 2


def test_ut43_replay_retry_succeeds(tmp_path, monkeypatch):
    orig = ts.Trainer.verify_harvest_replay
    state = {"n": 0}

    def flaky(self):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("一時的な不具合（模擬）")
        return orig(self)
    monkeypatch.setattr(ts.Trainer, "verify_harvest_replay", flaky)
    tr = ts.Trainer(_cfg(tmp_path, planned_updates=1), str(tmp_path), factory="mock")
    tr.run()
    assert state["n"] == 2 and tr.harvest_replay["F-1_passed"] is True


# ---- UT-44: C-15 の起動の保護 ----

def test_ut44_start_refused_when_records_exist_and_resume_works(tmp_path, monkeypatch):
    ts.Trainer(_cfg(tmp_path, planned_updates=2), str(tmp_path), factory="mock").run()
    upd = (tmp_path / "updates.jsonl").read_bytes()
    stat = (tmp_path / "status.json").read_bytes()
    names = sorted(os.listdir(tmp_path))
    with pytest.raises(SystemExit):                                     # --resume なしの起動は拒否し、何も書かない
        ts.Trainer(_cfg(tmp_path, planned_updates=3), str(tmp_path), factory="mock").run(resume=False)
    assert (tmp_path / "updates.jsonl").read_bytes() == upd and (tmp_path / "status.json").read_bytes() == stat
    assert sorted(os.listdir(tmp_path)) == names

    from rlkit.learner import SummerPPO

    def must_not_load(self, *a, **k):
        raise AssertionError("--resume で warm_start_from を読んではならない")
    monkeypatch.setattr(SummerPPO, "load_warm_start", must_not_load)
    cfg = _cfg(tmp_path, planned_updates=3, warm_start_from="runs/does_not_exist/checkpoint_latest.pt")
    tr = ts.Trainer(cfg, str(tmp_path), factory="mock")
    tr.run(resume=True)
    ups = [json.loads(l) for l in open(tmp_path / "updates.jsonl", encoding="utf-8")]
    assert [u["update"] for u in ups] == [1, 2, 3]                     # 更新 3（= 続きの最初）から始まり、既存の行は残る
    assert (tmp_path / "updates.jsonl").read_bytes().startswith(upd)
    ev = [json.loads(l) for l in open(tmp_path / "events.jsonl", encoding="utf-8")]
    ctx = [e for e in ev if e.get("event") == "resume_context"]
    assert len(ctx) == 1 and ctx[0]["instruction_version"] == "V7.1-S" and ctx[0]["resumed_from_update"] == 2
    assert not any(e.get("event") == "warm_start" for e in ev)


def test_ut44_fresh_dir_starts_without_resume(tmp_path):
    tr = ts.Trainer(_cfg(tmp_path, planned_updates=1), str(tmp_path), factory="mock")
    tr.run(resume=False)
    assert tr.model.update_index == 1


# ---- 第 4-1 節: 反事実の行動列の作り方（エミュレータなし。再生は rlkit/diag/v71s_cf.py） ----

def _cf_base():
    from rlkit import action as action_mod
    rng = np.random.default_rng(3)
    base = np.zeros((20, constants.ACTION_DIMS), dtype=np.int64)
    sz, sp, fan = (action_mod.LAYOUT[k][0] for k in ("zone_state", "zone_setpoint", "zone_fan"))
    base[:, sz:sz + 18] = rng.integers(0, 3, size=(20, 18))            # 停止・冷房・送風
    base[:, sp:sp + 18] = rng.integers(0, 7, size=(20, 18))
    base[:, fan:fan + 18] = rng.integers(0, 3, size=(20, 18))
    base[:, 54:58] = rng.integers(0, 8, size=(20, 4))
    base[:, 58:60] = rng.integers(0, 4, size=(20, 2))
    return base


def test_cf_actions_change_only_cooling_setpoint_and_fan():
    from rlkit.diag import v71s_cf as cf_mod
    base = _cf_base()
    cool = base[:, :18] == cf_mod.COOL
    sp, fan = base[:, 18:36], base[:, 36:54]
    o0, n0 = cf_mod.make_cf_actions(base, 0)
    assert np.array_equal(o0, base) and n0["setpoint"] == 0 and n0["fan"] == 0
    for k, new_c in ((1, 23.0), (2, 24.0), (3, 26.0)):                  # 22 ℃（索引 0）の冷房だけが動く
        out, n = cf_mod.make_cf_actions(base, k)
        sel = cool & (sp == 0)
        assert n["setpoint"] == int(sel.sum()) and n["fan"] == 0
        assert (out[:, 18:36][sel] == cf_mod.SETPOINT_GRID.index(new_c)).all()
        assert np.array_equal(out[:, 18:36][~sel], sp[~sel]) and np.array_equal(out[:, 36:54], fan)
    out, n = cf_mod.make_cf_actions(base, 4)                            # 冷房の全ゾーン・ステップが 26 ℃
    assert (out[:, 18:36][cool] == 4).all() and np.array_equal(out[:, 18:36][~cool], sp[~cool])
    out, n = cf_mod.make_cf_actions(base, 5)                            # High → Middle
    sel = cool & (fan == 2)
    assert n["fan"] == int(sel.sum()) and (out[:, 36:54][sel] == 1).all() and np.array_equal(out[:, 18:36], sp)
    out, n = cf_mod.make_cf_actions(base, 6)                            # High → Low
    assert (out[:, 36:54][sel] == 0).all()
    out7, n7 = cf_mod.make_cf_actions(base, 7)                          # CF-7 = CF-2 + CF-5
    o2, _ = cf_mod.make_cf_actions(base, 2)
    o5, _ = cf_mod.make_cf_actions(base, 5)
    assert np.array_equal(out7[:, 18:36], o2[:, 18:36]) and np.array_equal(out7[:, 36:54], o5[:, 36:54])
    out8, _ = cf_mod.make_cf_actions(base, 8)                           # CF-8 = CF-3 + CF-6
    o3, _ = cf_mod.make_cf_actions(base, 3)
    o6, _ = cf_mod.make_cf_actions(base, 6)
    assert np.array_equal(out8[:, 18:36], o3[:, 18:36]) and np.array_equal(out8[:, 36:54], o6[:, 36:54])
    for k in range(9):                                                  # 状態・蒸発温度・換気は不変
        o, _ = cf_mod.make_cf_actions(base, k)
        assert np.array_equal(o[:, :18], base[:, :18]) and np.array_equal(o[:, 54:], base[:, 54:])


# ---- 保存点の置き換えの拒否に対するやり直し ----

def test_replace_retries_then_succeeds(tmp_path, monkeypatch):
    from rlkit import runlog
    src, dst = tmp_path / "a.tmp", tmp_path / "a"
    src.write_text("x")
    real, calls = os.replace, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(5, "アクセスが拒否されました")
        return real(a, b)
    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(runlog.time, "sleep", lambda s: None)
    assert runlog.replace_with_retries(str(src), str(dst)) == 3 and dst.read_text() == "x"


def test_replace_retries_gives_up_after_attempts(tmp_path, monkeypatch):
    from rlkit import runlog
    src = tmp_path / "a.tmp"
    src.write_text("x")
    calls = {"n": 0}

    def always(a, b):
        calls["n"] += 1
        raise PermissionError(5, "拒否")
    monkeypatch.setattr(os, "replace", always)
    monkeypatch.setattr(runlog.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        runlog.replace_with_retries(str(src), str(tmp_path / "a"), attempts=4)
    assert calls["n"] == 4
    with pytest.raises(FileNotFoundError):                              # 拒否以外の失敗はやり直さない
        monkeypatch.setattr(os, "replace", os.rename)
        runlog.replace_with_retries(str(tmp_path / "none"), str(tmp_path / "b"))


def test_trainer_survives_checkpoint_replace_denials(tmp_path, monkeypatch):
    from rlkit import runlog
    real, state = os.replace, {"fail": 2}

    def deny_sil(a, b):
        if str(b).endswith("sil_buffer.pt") and state["fail"] > 0:
            state["fail"] -= 1
            raise PermissionError(5, "拒否")
        return real(a, b)
    monkeypatch.setattr(os, "replace", deny_sil)
    monkeypatch.setattr(runlog.time, "sleep", lambda s: None)
    tr = ts.Trainer(_cfg(tmp_path, planned_updates=1), str(tmp_path), factory="mock")
    tr.run()
    assert tr.stop_reason.startswith("計画更新数") and tr.model.update_index == 1
    ev = [json.loads(l) for l in open(tmp_path / "events.jsonl", encoding="utf-8")]
    assert any(e.get("event") == "checkpoint_replace_retried" and e["retries"] >= 1 for e in ev)


# ---- モデルだけを別の場所（OneDrive の外）に置く（model_dir） ----

def test_model_dir_default_is_run_dir(tmp_path):
    tr = ts.Trainer(_cfg(tmp_path), str(tmp_path), factory="mock")
    assert tr.model_dir == str(tmp_path) and tr._model_path("a.pt") == os.path.join(str(tmp_path), "a.pt")
    assert config.model_dir_of({"run_dir": str(tmp_path)}) == str(tmp_path)


def test_model_dir_separates_models_from_records_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "CHECKPOINT_EVERY", 1)
    run, models = tmp_path / "run", tmp_path / "models"
    cfg = _cfg(run, planned_updates=2, model_dir=str(models))
    tr = ts.Trainer(cfg, str(run), factory="mock")
    tr.run()
    # モデルは model_dir にだけ出来る
    for name in ("checkpoint_latest.pt", "checkpoint_u1.pt", "checkpoint_u2.pt", "sil_buffer.pt"):
        assert (models / name).exists() and not (run / name).exists(), name
    assert (models / "best").is_dir() and any(p.suffix == ".pt" for p in (models / "best").iterdir())
    assert not any(p.suffix == ".pt" for p in (run / "best").iterdir())          # best の .pt は run_dir に出来ない
    # 記録・行動列は run_dir のまま
    for name in ("updates.jsonl", "episodes.jsonl", "events.jsonl", "status.json"):
        assert (run / name).exists() and not (models / name).exists(), name
    assert (run / "harvest" / "best_harvest.npz").exists() and any((run / "best").glob("*_actions.npz"))
    assert not list(run.glob("*.pt")) and not list(run.rglob("*.pt"))
    meta = [json.load(open(p, encoding="utf-8")) for p in run.glob("meta_start_*.json")][0]
    assert os.path.normcase(meta["model_dir"]) == os.path.normcase(str(models))
    # --resume は model_dir から続ける
    cfg3 = _cfg(run, planned_updates=3, model_dir=str(models))
    tr3 = ts.Trainer(cfg3, str(run), factory="mock")
    tr3.run(resume=True)
    assert tr3.model.update_index == 3 and (models / "checkpoint_u3.pt").exists()
    assert [json.loads(l)["update"] for l in open(run / "updates.jsonl", encoding="utf-8")] == [1, 2, 3]


def test_model_dir_missing_checkpoint_stops_without_falling_back_to_run_dir(tmp_path):
    run, models = tmp_path / "run", tmp_path / "models"
    ts.Trainer(_cfg(run, planned_updates=1), str(run), factory="mock").run()      # model_dir なし: run_dir に保存点が出来る
    assert (run / "checkpoint_latest.pt").exists()
    cfg = _cfg(run, planned_updates=2, model_dir=str(models))                      # 別の空の model_dir を指定して再開
    tr = ts.Trainer(cfg, str(run), factory="mock")
    tr.run(resume=True)
    assert tr.stop_reason.startswith("例外") and "checkpoint_latest.pt" in tr.stop_reason
    assert [json.loads(l)["update"] for l in open(run / "updates.jsonl", encoding="utf-8")] == [1]   # 進んでいない


def test_resolve_model_dir_new_runs_go_outside_old_runs_stay(tmp_path):
    run = tmp_path / "runs" / "v2s_trial09"
    run.mkdir(parents=True)
    # 設定に無い新しい走行: OneDrive の外（MODEL_ROOT の下、走行名のフォルダ）
    got = config.resolve_model_dir({"run_dir": str(run)}, str(run), resume=False)
    assert got == os.path.join(config.MODEL_ROOT, "v2s_trial09")
    # 設定で明示（null は従来どおり run_dir）
    assert config.resolve_model_dir({"model_dir": None}, str(run), resume=False) == str(run)
    assert config.resolve_model_dir({"model_dir": str(tmp_path / "m")}, str(run), resume=False) == str(tmp_path / "m")
    # 以前の走行（meta_start に model_dir が無い）の再開は run_dir のまま
    (run / "meta_start_1.json").write_text(json.dumps({"started_at": "x"}), encoding="utf-8")
    assert config.resolve_model_dir({}, str(run), resume=True) == str(run)
    # 新しい走行の再開は、最初の meta_start に記録された model_dir
    (run / "meta_start_1.json").write_text(json.dumps({"model_dir": str(tmp_path / "outside")}), encoding="utf-8")
    assert config.resolve_model_dir({}, str(run), resume=True) == str(tmp_path / "outside")
    # 既定値には model_dir を入れていない（設定に書かれたかを区別するため）
    assert "model_dir" not in config.load()


def test_existing_trial08_config_keeps_models_in_run_dir():
    cfg = config.load(os.path.join("rlkit", "config_summer_v7p.json"))
    run = config.repo_path(cfg["run_dir"])
    assert "model_dir" not in cfg
    assert config.resolve_model_dir(cfg, run, resume=True) == run      # 今保存されているモデルはそのまま


# ---- エミュレータの複製と診断の CSV を OneDrive の外へ（rlkit.localdata） ----

def test_replicas_default_outside_project_custom_original_stays_sibling(tmp_path, monkeypatch):
    from rlkit import localdata
    from rlkit.parallel import slots
    monkeypatch.setattr(localdata, "EMULATOR_ROOT", str(tmp_path / "local_emulators"))
    d = slots.replica_dir_for(3)                                           # 既定の原本: 外に置く
    assert d == os.path.join(str(tmp_path / "local_emulators"), "Shizuku2_slot3")
    assert slots.make_slot(3).replica_dir == d
    orig = tmp_path / "somewhere" / "Shizuku2"                             # 別の原本（試験など）は従来どおり隣
    assert slots.replica_dir_for(3, str(orig)) == os.path.join(str(orig.parent), "Shizuku2_slot3")
    assert os.path.basename(slots.replica_dir_for(12)) == "Shizuku2_slot12"
    assert os.path.commonpath([os.path.abspath(slots.ORIGINAL_DIR), os.path.abspath(d)]) != os.path.abspath(slots.ORIGINAL_DIR)


def test_diag_csv_dir_maps_runs_diag_only(tmp_path, monkeypatch):
    from rlkit import localdata
    monkeypatch.setattr(localdata, "DIAG_CSV_ROOT", str(tmp_path / "csv"))
    inside = os.path.join(localdata.DIAG_DIR, "V6S", "slot_output", "k11_a")
    assert localdata.diag_csv_dir(inside) == os.path.join(str(tmp_path / "csv"), "V6S", "slot_output", "k11_a")
    assert localdata.diag_csv_dir(os.path.join("runs", "diag", "V63S", "R8_baseline_output")) == \
        os.path.join(str(tmp_path / "csv"), "V63S", "R8_baseline_output")
    other = str(tmp_path / "elsewhere" / "x")
    assert localdata.diag_csv_dir(other) == other                          # runs/diag の外はそのまま
    assert localdata.diag_csv_dir(os.path.join("runs", "v2s_trial08", "x")) == os.path.join("runs", "v2s_trial08", "x")


def test_local_root_is_outside_the_repo():
    from rlkit import localdata
    root = os.path.abspath(localdata._REPO_ROOT)
    for p in (localdata.EMULATOR_ROOT, localdata.DIAG_CSV_ROOT):
        assert not os.path.abspath(p).lower().startswith(root.lower() + os.sep)
