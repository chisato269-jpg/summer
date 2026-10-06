"""学習器と学習の入口の試験（模擬環境。v2 第 5-5 節、指示書 V2-S 段階 B）。"""
import json
import math
import os

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv

from rlkit import action, config, constants, learner
from rlkit.mockenv import MockSummerEnv
from rlkit.reward import ReferencePoint
from rlkit.train_summer import (M5ReleaseState, Trainer, gate_g1_satisfied, gate_g2_relative_difference,
                                gate_g3_release_ok, is_degenerate_actions, median_of)

REF = ReferencePoint(4.587348191951224, 0.18481666361514731).to_dict()


def _model(stage=1, n_envs=2, planned=10, entropy_coef_init=0.01):
    vec = DummyVecEnv([lambda i=i: MockSummerEnv(stage=stage, seed=i) for i in range(n_envs)])
    vec.env_method("set_reference", REF)
    return learner.SummerPPO(vec, stage, entropy_coef_init=entropy_coef_init, planned_updates=planned,
                             seed=constants.RANDOM_SEED, n_steps_per_env=756)


def _trial_config(tmp_path, **overrides):
    """模擬環境の走行設定（entropy_coef_init は本物の設定と同じ 0.01）。"""
    cfg = config.load()
    cfg.update({"include_indoor_unit_power": False, "zone_step_energy_norm_gj": 0.01, "entropy_coef_init": 0.01,
                "planned_updates": 10, "stage2_start_fraction": 1.0, "run_dir": str(tmp_path)})
    cfg.update(overrides)
    return cfg


def test_schedules_and_rules():
    assert learner.floor_fraction_at(0, 100) == pytest.approx(0.30)
    assert learner.floor_fraction_at(30, 100) == pytest.approx(0.175)
    assert learner.floor_fraction_at(60, 100) == pytest.approx(0.05)
    assert learner.floor_fraction_at(99, 100) == pytest.approx(0.05)
    # target_kl 0.05（V4-S 第 2-3 節）、学習率の範囲 [1e-5, 5e-4]（V5-S S-7）
    assert learner.adapt_learning_rate(1e-4, 0.11) == pytest.approx(1e-4 / 1.5)   # 2 × target 超で縮める
    assert learner.adapt_learning_rate(1e-4, 0.02) == pytest.approx(1.5e-4)       # 0.5 × target 未満で伸ばす
    assert learner.adapt_learning_rate(1e-4, 0.05) == pytest.approx(1e-4)         # target 付近は変えない
    assert learner.adapt_learning_rate(5e-4, 0.001) == pytest.approx(5e-4)        # 上限で止まる
    assert learner.adapt_learning_rate(1e-5, 0.5) == pytest.approx(1e-5)          # 下限で止まる
    assert constants.POLICY_LR_MIN == 1e-5
    # S-6: 群ごとの床。換気だけ 0.10 で一定、他の群は既定の減衰に従う
    assert learner.floor_fraction_at(0, 100, group="tenant_vent") == pytest.approx(0.10)
    assert learner.floor_fraction_at(60, 100, group="tenant_vent") == pytest.approx(0.10)
    assert learner.floor_fraction_at(0, 100, group="zone_state") == pytest.approx(0.30)
    assert learner.floor_fraction_at(60, 100, group="zone_state") == pytest.approx(0.05)
    gl = action.group_layout(1)
    c = learner.dual_ascent_update({g: 0.01 for g in gl}, {g: 0.0 for g in gl}, gl, 0.3, 0.001, 1.0)
    assert c["zone_state"] == pytest.approx(0.01 * math.exp(0.2 * 0.3))
    with pytest.raises(ValueError):
        learner.SummerPPO(DummyVecEnv([lambda: MockSummerEnv()]), 1, None, 10, 0, 756)


def test_gate_functions():
    """B-3: G-1 ダッシュは (a) 連続回数と (c) 通算更新だけで決まる。G-2・G-3 は指示書 V2-S のまま。"""
    assert gate_g1_satisfied(3, 200)
    assert not gate_g1_satisfied(2, 200)                # (a) 連続回数が足りない
    assert not gate_g1_satisfied(3, 199)                # (c) 通算更新が足りない
    assert gate_g1_satisfied(5, 300)                    # (b) 非占有冷房率は条件に含まない（D-21）
    assert gate_g2_relative_difference(0.5, 0.5) == 0.0
    assert gate_g2_relative_difference(0.5, 0.5001) == pytest.approx(2e-4)
    assert gate_g2_relative_difference(0.0, 0.0) == 0.0
    assert math.isinf(gate_g2_relative_difference(0.0, 0.1))
    assert gate_g3_release_ok(0.5, 0.49) and not gate_g3_release_ok(0.5, 0.489)


def test_harvest_helpers():
    """B-1: 収穫の系列で判定するための小道具（中央値と、決定論的評価の退化の検出）。"""
    assert median_of([0.1, 0.3, 0.2]) == pytest.approx(0.2)
    assert median_of([]) is None
    flat = np.zeros((378, 60), dtype=np.int64)
    assert is_degenerate_actions(flat)                      # 全ステップ・全ゾーンが停止
    flat2 = flat.copy(); flat2[100, 3] = 1
    assert not is_degenerate_actions(flat2)
    allcool = np.zeros((378, 60), dtype=np.int64); allcool[:, :18] = 1
    assert is_degenerate_actions(allcool)                   # 全ゾーンが冷房でも「同一」なら退化


def test_m5_release_is_judged_once_per_attempt():
    """V5S-a の回帰: 解除の予約は判定が済むまで外れる（毎更新の再解除を防ぐ）。"""
    st = M5ReleaseState()
    st.fail(10)                                   # 1 回目の失敗 → 50 更新後に再試行
    assert st.retry_at == 10 + constants.G3_RETRY_UPDATES and st.due(st.retry_at)
    st.pending_since, st.before_median, st.retry_at = 60, 0.3, None    # 再試行で解除した状態
    assert not st.due(61) and not st.due(100)     # 判定が済むまで再解除しない
    st.fail(63)                                   # 判定で失敗 → 次の予約が入る
    assert st.pending_since is None and st.due(63 + constants.G3_RETRY_UPDATES)


def test_m5_release_state():
    st = M5ReleaseState()
    assert not st.due(10)
    st.fail(10)
    assert st.retries == 1 and st.retry_at == 10 + constants.G3_RETRY_UPDATES and not st.exhausted
    assert not st.due(10 + constants.G3_RETRY_UPDATES - 1) and st.due(10 + constants.G3_RETRY_UPDATES)
    st.fail(70)
    st.fail(130)
    assert st.exhausted
    st.pending_since, st.before_median = 5, 0.25
    st2 = M5ReleaseState.from_dict(st.to_dict())
    assert (st2.retries, st2.retry_at, st2.released) == (st.retries, st.retry_at, st.released)
    assert (st2.pending_since, st2.before_median) == (5, 0.25)
    st2.succeed()
    assert st2.released and st2.retry_at is None and not st2.due(1000)


def test_policy_separation_and_optimizers():
    model = _model(n_envs=1)
    pol = model.policy
    ids_pi = {id(p) for p in pol.policy_network.parameters()}
    ids_v = {id(p) for p in pol.value_network.parameters()}
    assert not ids_pi & ids_v
    assert {id(p) for g in pol.optimizer.param_groups for p in g["params"]} == ids_pi
    assert {id(p) for g in pol.value_optimizer.param_groups for p in g["params"]} == ids_v
    assert pol.optimizer.param_groups[0]["lr"] == constants.POLICY_LR_INIT
    assert pol.value_optimizer.param_groups[0]["lr"] == constants.VALUE_LR


def test_collect_and_train_one_update():
    model = _model()
    terminal = model.collect_one()
    assert len(terminal) == 4  # 2 組 × 2 周
    ro = model._rollout
    assert ro["observations"].shape == (1512, 638) and ro["advantages19"].shape == (1512, 19)
    assert ro["segments"] == [(0, 378), (378, 756), (756, 1134), (1134, 1512)]
    assert np.allclose(ro["rewards19"].sum(axis=1),
                       learner.adv_mod.flatten_env_major(model.rollout_buffer.rewards).reshape(-1))
    v_before = [p.detach().clone() for p in model.policy.value_network.parameters()]
    rec = model.train()
    assert 1 <= rec["epochs_applied"] <= 4 and rec["update"] == 1
    assert len(rec["approx_kl_per_epoch"]) >= rec["epochs_applied"]
    assert constants.POLICY_LR_MIN <= rec["policy_lr_after"] <= constants.POLICY_LR_MAX
    assert math.isfinite(rec["value_loss_last"]) and rec["floor_fraction"] == pytest.approx(0.30)
    assert any(not th.equal(a, b) for a, b in zip(v_before, model.policy.value_network.parameters()))
    # E-2: 更新 1 で勾配ノルム比を測る（記録のみ。判定には使わない）
    assert rec["grad_norm_policy"] > 0 and rec["grad_norm_entropy"] > 0
    assert rec["grad_norm_ratio"] == pytest.approx(rec["grad_norm_entropy"] / rec["grad_norm_policy"])


def test_grad_norm_ratio_measured_only_at_listed_updates():
    """B-1: 勾配ノルム比は E-2 の更新（1・10・50・100・200）でだけ測る。合否の判定には使わない。"""
    assert constants.GRAD_NORM_RATIO_MEASURE_AT == (1, 10, 50, 100, 200)
    model = _model(planned=200)
    model.collect_one()
    rec1 = model.train()
    assert rec1["update"] == 1 and rec1["grad_norm_ratio"] > 0
    model.collect_one()
    rec2 = model.train()
    assert rec2["update"] == 2 and rec2["grad_norm_policy"] is None and rec2["grad_norm_ratio"] is None
    model.update_index = 9  # 次の学習が更新 10 になるようにする
    model.collect_one()
    rec10 = model.train()
    assert rec10["update"] == 10 and rec10["grad_norm_ratio"] > 0


def test_kl_early_stop_discards_epoch():
    model = _model()
    model.collect_one()
    learner.set_optimizer_lr(model.policy.optimizer, 0.5)
    rec = model.train()
    assert rec["kl_early_stop"] is True
    assert rec["epochs_applied"] == len(rec["approx_kl_per_epoch"]) - 1
    assert rec["approx_kl_per_epoch"][-1] > constants.KL_STOP_FACTOR * constants.TARGET_KL


def test_stage2_transfer_and_deterministic_eval():
    m1 = _model(stage=1, n_envs=1)
    m2 = _model(stage=2, n_envs=1)
    m1.update_index, m1.policy_lr = 7, 3e-4
    learner.transfer_to_stage2(m1, m2)
    assert m2.update_index == 7 and m2.policy_lr == 3e-4
    obs = th.rand(4, 638)
    with th.no_grad():
        l1, l2 = m1.policy.policy_network(obs), m2.policy.policy_network(obs)
    # 第 1 段階の設定温度は索引 36 から（運転状態 2 × 18）、第 2 段階は 54 から（3 × 18）
    assert th.allclose(l2[:, 54:54 + 234].reshape(4, 18, 13)[:, :, 0::2], l1[:, 36:36 + 126].reshape(4, 18, 7), atol=1e-6)
    vec = DummyVecEnv([lambda: MockSummerEnv(stage=2, seed=5)])
    info, acts = learner.run_deterministic_episode(m2.policy, vec)
    assert acts.shape == (378, 60) and "E_rep" in info
    m2.load_weights(m1.weights())  # 第 1 段階の重みを第 2 段階へ読み直せる（退行時の再読み込み）


def test_trainer_orchestration_with_mock(tmp_path):
    cfg = _trial_config(tmp_path, planned_updates=3)
    trainer = Trainer(cfg, str(tmp_path), factory="mock")
    trainer.run()
    status = json.load(open(tmp_path / "status.json", encoding="utf-8"))
    assert status["status"] == "stopped", status
    evals = [json.loads(l) for l in open(tmp_path / "evals.jsonl", encoding="utf-8")]
    updates = [json.loads(l) for l in open(tmp_path / "updates.jsonl", encoding="utf-8")]
    assert evals[0]["update"] == 0 and evals[0]["tag"] == "initial"
    # 同じ重みの 2 回の決定論的評価（R-1 と更新 1 の評価）は一致する
    assert evals[1]["eci_smooth"] == evals[0]["eci_smooth"]
    assert len(updates) == 3 and all(u["train_episodes"] == 12 for u in updates)
    # 指示書 V4-S M-4 の記録項目（V3-S M-5 から退行回数を除き、基準点・内側率・収穫を加えたもの）
    u0 = updates[0]
    for key in ("approx_kl_after", "epochs_applied", "policy_lr_after", "c_g", "H_g", "floor_fraction",
                "value_loss_last", "reference", "reference_update", "eval_raw_resid_rel",
                "eval_cooling_fraction_zone_unoccupied", "eval_obs25_clip_rate_mean", "grad_norm_ratio",
                "m5_enabled", "eval_obs29_clip_rate_mean", "eval_obs29_clip_rate_max", "eval_co2_clip_rate_max",
                "eval_load_clip_rate_max", "eval_progress_clip_rate", "train_E_mean", "train_DR_mean",
                "inside_rate_recent20", "harvest_update_best", "harvest_best",
                # 指示書 V5-S M-5 の追加分
                "harvest_best_official", "harvest_official_streak", "approx_kl_by_group",
                "floor_fraction_by_group", "eval_m8_zone_steps", "eval_m9_tenant_steps",
                "eval_degenerate_actions", "m8_enabled", "m9_enabled"):
        assert key in u0, key
    assert set(u0["approx_kl_by_group"]) == set(constants.ENTROPY_GROUPS)        # S-7
    assert u0["floor_fraction_by_group"]["tenant_vent"] == pytest.approx(0.10)   # S-6
    assert u0["eval_m8_zone_steps"] > 0                                          # M-8 が効いている
    assert {"rho", "rho_clipped"} <= set(u0["reference"])
    assert "regression_count" not in u0                              # Q-1 で廃止
    assert os.path.exists(tmp_path / "checkpoint_latest.pt")
    assert status["stage"] == 1 and status["m5_enabled"] is True


def test_trainer_stage_switch_and_release(tmp_path):
    """B-1: 手動切替で G-2 が走り、G-3' が解除 → 3 更新後の中央値で判定される。"""
    cfg = _trial_config(tmp_path, planned_updates=5, manual_stage2_update=1)
    trainer = Trainer(cfg, str(tmp_path), factory="mock", max_updates=5)
    trainer.run()
    gates = [json.loads(l) for l in open(tmp_path / "gates.jsonl", encoding="utf-8")]
    kinds = [g["gate"] for g in gates]
    assert kinds[:2] == ["G-1''", "G-2"]
    g2 = gates[1]
    assert g2["passed"] and g2["relative_difference"] <= constants.G2_ECI_REL_TOLERANCE
    assert trainer.model.stage == 2
    released = [g for g in gates if g["gate"] == "G-3'" and g["phase"] == "released"]
    judged = [g for g in gates if g["gate"] == "G-3'" and g["phase"] == "judged"]
    assert len(released) == 1 and released[0]["judge_at"] == released[0]["update"] + 3
    assert len(judged) == 1 and judged[0]["update"] >= released[0]["judge_at"]
    # 解除の直後は M-5・M-8 の両方が外れる（G-3' は両方をまとめて扱う）
    if judged[0]["passed"]:
        assert trainer.m5_state.released and trainer.m5_enabled is False and trainer.m8_enabled is False
    else:
        assert trainer.m5_enabled is True and trainer.m8_enabled is True
        assert trainer.m5_state.retry_at == judged[0]["update"] + constants.G3_RETRY_UPDATES
    evals = [json.loads(l) for l in open(tmp_path / "evals.jsonl", encoding="utf-8")]
    assert {"g2_pre", "g2_post"} <= {e["tag"] for e in evals}
    updates = [json.loads(l) for l in open(tmp_path / "updates.jsonl", encoding="utf-8")]
    # G-1'' は収穫の公式 ECI の連続で数える
    assert all("harvest_official_streak" in u for u in updates)


def test_trainer_records_grad_norm_ratio_without_stopping(tmp_path):
    """B-1: 勾配ノルム比は記録するだけで停止しない（E-1 で検査を廃止した）。"""
    cfg = _trial_config(tmp_path, planned_updates=2, entropy_coef_init=0.01)
    trainer = Trainer(cfg, str(tmp_path), factory="mock")
    reason = trainer.run()
    assert "勾配ノルム比" not in reason and "計画更新数 2" in reason
    updates = [json.loads(l) for l in open(tmp_path / "updates.jsonl", encoding="utf-8")]
    assert len(updates) == 2
    assert updates[0]["grad_norm_policy"] > 0 and updates[0]["grad_norm_entropy"] > 0
    assert updates[0]["grad_norm_ratio"] > 0          # 値の大小は問わない
    assert updates[1]["grad_norm_ratio"] is None      # 更新 2 では測らない


def test_reference_follows_training_mean_and_no_regression(tmp_path):
    """B-1・B-2: 基準点は訓練 12 エピソードの平均の指数移動平均で動き、退行処理は起きない。"""
    cfg = _trial_config(tmp_path, planned_updates=4)
    trainer = Trainer(cfg, str(tmp_path), factory="mock")
    trainer.run()
    updates = [json.loads(l) for l in open(tmp_path / "updates.jsonl", encoding="utf-8")]
    assert len(updates) == 4
    c = constants.REFERENCE_EMA_COEF
    prev = None
    for u in updates:
        ru = u["reference_update"]
        assert ru["E_bar"] == pytest.approx(u["train_E_mean"])       # 入力は訓練 12 本の平均
        assert ru["DR_bar"] == pytest.approx(u["train_DR_mean"])
        assert ru["E_bar"] != pytest.approx(u["eval_E"])             # 決定論的評価の点は使わない
        if u["update"] == 1:
            assert ru["first"] is True
            assert ru["E_star"] == pytest.approx(ru["E_bar"]) and ru["DR_star"] == pytest.approx(ru["DR_bar"])
        else:
            assert ru["first"] is False
            assert ru["E_star"] == pytest.approx((1 - c) * prev["E_star"] + c * ru["E_bar"])
            assert ru["DR_star"] == pytest.approx((1 - c) * prev["DR_star"] + c * ru["DR_bar"])
            assert prev["rho"] / 2.0 - 1e-12 <= ru["rho"] <= prev["rho"] * 2.0 + 1e-12   # 常時 2 倍制限
        prev = ru
        assert "regression_reload" not in u and "train_skipped" not in u   # Q-1: 退行処理は無い
        assert "train_s" in u                                              # 毎更新で学習する
    assert not hasattr(constants, "REGRESSION_CONSECUTIVE")
    assert os.path.exists(tmp_path / "best" / "best_smooth.pt")            # Q-2: 最良重みの保存は続く


def test_harvest_best_action_sequence(tmp_path):
    """B-4: 12 本から公式 ECI 最大の行動列が保存され、走行全体の最良が更新される。"""
    cfg = _trial_config(tmp_path, planned_updates=3)
    trainer = Trainer(cfg, str(tmp_path), factory="mock")
    trainer.run()
    updates = [json.loads(l) for l in open(tmp_path / "updates.jsonl", encoding="utf-8")]
    eps = [json.loads(l) for l in open(tmp_path / "episodes.jsonl", encoding="utf-8")]
    for u in updates:
        best = u["harvest_update_best"]
        same = [e for e in eps if e["update"] == u["update"]]
        assert len(same) == 12
        top = max(same, key=lambda e: (e["eci_official"], e["eci_smooth"]))
        assert best["eci_official"] == pytest.approx(top["eci_official"])
        assert best["eci_smooth"] == pytest.approx(top["eci_smooth"])
        assert best["E"] == pytest.approx(top["E"]) and best["slot"] == top["slot"]
    run_best = [(u["harvest_best"]["eci_official"], u["harvest_best"]["eci_smooth"]) for u in updates]
    assert run_best == sorted(run_best)          # 走行全体の最良は下がらない
    assert run_best[-1] == max((u["harvest_update_best"]["eci_official"],
                                u["harvest_update_best"]["eci_smooth"]) for u in updates)
    meta = json.load(open(tmp_path / "harvest" / "best_harvest.json", encoding="utf-8"))
    actions = np.load(tmp_path / "harvest" / "best_harvest.npz")["actions"]
    assert actions.shape == (constants.ZONE_STEPS_PER_WEEK, constants.ACTION_DIMS)
    assert meta["E_hex"] == float.hex(meta["E"]) and meta["stage"] == 1
    assert trainer.best_harvest["update"] == meta["update"]


def test_trainer_stage1_ceiling_stops(tmp_path):
    """U-5: 上限に達しても第 1 段階のままなら停止する。"""
    cfg = _trial_config(tmp_path, planned_updates=4, stage2_start_fraction=0.25)
    trainer = Trainer(cfg, str(tmp_path), factory="mock")
    reason = trainer.run()
    assert "U-5" in reason
    assert trainer.model.update_index == 1 and trainer.model.stage == 1


def test_trainer_resume_keeps_update_index_and_new_planned(tmp_path):
    """B-3: 更新番号の通算と、再開時の planned_updates 変更の反映。"""
    cfg = _trial_config(tmp_path, planned_updates=10)
    Trainer(cfg, str(tmp_path), factory="mock", max_updates=1).run()
    first = json.load(open(tmp_path / "status.json", encoding="utf-8"))
    assert first["update_index"] == 1
    saved = th.load(tmp_path / "checkpoint_latest.pt", weights_only=False)
    cfg2 = _trial_config(tmp_path, planned_updates=20)
    # --updates（max_updates）は通算の意味（指示書 V4-S W-3）。20 更新の次は 21 ではなく通算 2 を指定する
    trainer2 = Trainer(cfg2, str(tmp_path), factory="mock", max_updates=2)
    trainer2.run(resume=True)
    second = json.load(open(tmp_path / "status.json", encoding="utf-8"))
    assert second["update_index"] == 2                  # 更新番号は再開をまたいで通算する
    assert trainer2.model.planned_updates == 20         # planned_updates の変更が以後に反映される
    # 基準点は保存した値から引き継がれる（初期基準点は同じ）
    assert trainer2.reference.history[0]["E_star"] == saved["reference"]["history"][0]["E_star"]
