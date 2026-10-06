"""学習器と学習の入口の試験（模擬環境。v2 第 5-5 節、指示書 V6-S 段階 A の UT-4・UT-6〜UT-9・UT-12〜UT-14）。

V6-S で差し替え・廃止した試験（報告書に理由を記す）:
  差し替え: 模擬の学習器の作り方（entropy_coef_init の廃止）、採取の形（642 次元・62 次元）、学習の記録項目
            （床・双対上昇の代わりに c_g・GC-1 の除外数）、学習率の上限（5e-4 → 1e-3）、収穫の順位（同点は J）、
            基準点の傾き（R-5）、再開時の引き継ぎ（SIL の保存庫・c_sil を追加）、決定論的評価（第 1 段階のみ）
  廃止: 床の減衰と双対上昇の試験（L-1 で廃止）、段階の切替と G-3' の解除の試験・第 1 段階の上限の試験（A-7 で廃止。
        門の判定関数と解除の状態の単体の試験は残す）、第 2 段階への移行と決定論的評価の試験（第 1 段階の決定論的評価に置換）
V6.1-S で差し替え・廃止した試験:
  差し替え: SIL の保存庫の順位（公式 ECI・J → ECI）、判定関数（H-4' → H-4''、S-8' を除く）、学習の入口の記録項目
            （基準点の項目を R-7 の比と評価点へ）、基準点の追従（R-2'' の指数移動平均 → R-7）、収穫の順位（ECI）、再開の確認
  廃止: S-8' で止まることの試験（S-8' の廃止。呼ばれないことは test_v61.py の UT-21 で確かめる）
"""
import json
import math
import os

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv

from rlkit import action, config, constants, learner, observation, policy
from rlkit import train_summer as ts
from rlkit.mockenv import MOCK_W, MockSummerEnv
from rlkit.reward import ReferencePoint, RewardCalculator, reward_context
from rlkit.train_summer import (M5ReleaseState, Trainer, gate_g1_satisfied, gate_g2_relative_difference,
                                gate_g3_release_ok, harvest_key, is_degenerate_actions, median_of)

REF = ReferencePoint(4.587348191951224, 0.18481666361514731).to_dict()


def _model(n_envs=2, planned=10):
    vec = DummyVecEnv([lambda i=i: MockSummerEnv(seed=i) for i in range(n_envs)])
    vec.env_method("set_reference", REF)
    return learner.SummerPPO(vec, 1, planned_updates=planned, seed=constants.RANDOM_SEED, n_steps_per_env=756)


def _trial_config(tmp_path, **overrides):
    cfg = config.load()
    cfg.update({"include_indoor_unit_power": False, "zone_step_energy_norm_gj": 0.01,
                "planned_updates": 10, "run_dir": str(tmp_path)})
    cfg.update(overrides)
    return cfg


def _read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


# =====================================================================
# UT-6・UT-7: エントロピー係数と学習率
# =====================================================================

def test_ut6_entropy_coefficient_schedule():
    """UT-6: 更新 1・900・1,200・1,500 で c0/lnK_g、同、半分、0（計画 1,500 更新）。"""
    gl = action.group_layout(1)
    c1 = learner.entropy_coefficients(1, 1500, gl)
    c900 = learner.entropy_coefficients(900, 1500, gl)
    c1200 = learner.entropy_coefficients(1200, 1500, gl)
    c1500 = learner.entropy_coefficients(1500, 1500, gl)
    K = {"zone_state": 2, "zone_setpoint": 9, "zone_fan": 3, "system_evap": 8, "tenant_vent": 4, "tenant_remote": 2}
    assert set(c1) == set(K)
    for g, k in K.items():
        assert c1[g] == pytest.approx(0.002 / math.log(k))
        assert c900[g] == pytest.approx(c1[g])
        assert c1200[g] == pytest.approx(0.5 * c1[g])
        assert c1500[g] == 0.0
    assert learner.entropy_c0_at(901, 1500) == pytest.approx(0.002 * 599 / 600)   # c0 × (1500 − u)/600
    assert not hasattr(learner, "dual_ascent_update") and not hasattr(learner, "floor_fraction_at")   # 床は廃止


def test_ut7_learning_rate_range():
    """UT-7: 方策学習率は [1e-5, 1e-3]、初期 1e-4。調整の規則は変えない。"""
    assert constants.POLICY_LR_INIT == 1e-4 and constants.POLICY_LR_MIN == 1e-5 and constants.POLICY_LR_MAX == 1e-3
    assert learner.adapt_learning_rate(1e-4, 0.11) == pytest.approx(1e-4 / 1.5)   # 2 × target 超で縮める
    assert learner.adapt_learning_rate(1e-4, 0.02) == pytest.approx(1.5e-4)       # 0.5 × target 未満で伸ばす
    assert learner.adapt_learning_rate(1e-4, 0.05) == pytest.approx(1e-4)
    lr = 1e-4
    for _ in range(20):
        lr = learner.adapt_learning_rate(lr, 0.0)
    assert lr == pytest.approx(1e-3)                                               # 上限（旧 5e-4 を超えて伸びる）
    for _ in range(40):
        lr = learner.adapt_learning_rate(lr, 1.0)
    assert lr == pytest.approx(1e-5)
    m = _model(n_envs=1)
    assert m.policy.optimizer.param_groups[0]["lr"] == constants.POLICY_LR_INIT


# =====================================================================
# UT-4: GC-1
# =====================================================================

def _gc1_sample():
    """停止ゾーン・運転ゾーンの無い系統・1 択の要素を含む合成の標本（T=3）。"""
    acts = np.zeros((3, 62), dtype=np.int64)
    acts[0, [0, 9]] = 1          # t=0: S1（系統1）と N1（系統3）が運転
    acts[2, :18] = 1             # t=2: 全ゾーン運転
    masks = np.ones((3, 296), dtype=bool)
    # t=1 のリモコン（南）を 1 択にする（M-3 の退化に相当）
    r0 = sum(action.nvec(1)[:60])
    masks[1, r0:r0 + 2] = [False, True]
    n_opt = learner.compute_n_options(th.as_tensor(masks), 1)
    include, gc1, one = learner.include_mask(acts, n_opt)
    return acts, masks, include, gc1, one


def test_ut4_gc1_include_mask():
    acts, masks, include, gc1, one = _gc1_sample()
    assert include[0, 18] and include[0, 36] and include[0, 27]          # 運転ゾーンの設定温度・風量
    assert not include[0, 19] and not include[0, 37]                     # 停止ゾーン S2
    assert include[0, 54] and not include[0, 55] and include[0, 56] and not include[0, 57]   # 系統
    assert not include[1, 18:58].any()                                   # 全停止のステップ
    assert include[1, :18].all() and include[1, 58:60].all()             # 運転状態・換気は残る
    assert not include[1, 60] and include[1, 61]                         # 1 択の要素は除く
    assert include[2].all()
    excl = learner.exclusion_counts(gc1, one, action.group_layout(1))
    assert excl["gc1"]["zone_setpoint"] == 16 + 18 and excl["gc1"]["system_evap"] == 2 + 4
    assert excl["one_option"]["tenant_remote"] == 1


def test_ut4_gc1_excluded_elements_do_not_contribute():
    """UT-4: 除外した要素は方策の損失・エントロピー・approx_kl・正規化の母集団・SIL の損失に寄与しない。"""
    acts, masks, include, gc1, one = _gc1_sample()
    th.manual_seed(0)
    logp = th.randn(3, 62, requires_grad=True)
    old = th.randn(3, 62)
    adv = th.randn(3, 62)
    # 方策の損失（比のクリップ項）: 除外した要素への勾配は 0
    loss, _ = learner.clipped_policy_loss(logp, old, adv, include)
    loss.backward()
    assert th.all(logp.grad[~include] == 0) and th.any(logp.grad[include] != 0)
    # 除外した要素の値を変えても損失は変わらない
    logp2 = logp.detach().clone()
    logp2[~include] += 5.0
    adv2 = adv.clone()
    adv2[~include] *= -3.0
    assert learner.clipped_policy_loss(logp2, old, adv2, include)[0].item() == pytest.approx(loss.item())
    # approx_kl
    kl = learner.approx_kl_from(logp.detach(), old, include).item()
    assert learner.approx_kl_from(logp2, old, include).item() == pytest.approx(kl)
    # エントロピー（群ごとの平均）
    ent = th.rand(3, 62)
    ent2 = ent.clone()
    ent2[~include] = 100.0
    gl = action.group_layout(1)
    h1, h2 = learner.compute_group_entropy(ent, include, gl), learner.compute_group_entropy(ent2, include, gl)
    for g in gl:
        assert (h1[g] is None and h2[g] is None) or float(h1[g]) == pytest.approx(float(h2[g]))
    # 優位度の正規化の母集団
    m = np.random.default_rng(0).normal(size=(3, 62))
    m2 = m.copy()
    m2[~include.numpy()] = 1e6
    n1, mu1, sd1 = learner.normalize_over(m, include.numpy())
    n2, mu2, sd2 = learner.normalize_over(m2, include.numpy())
    assert (mu1, sd1) == pytest.approx((mu2, sd2)) and np.allclose(n1[include.numpy()], n2[include.numpy()])
    # SIL の方策損失
    lp = th.randn(3, 62, requires_grad=True)
    sil = learner.sil_policy_loss(lp, th.rand(3, 62), 2.0, include, 0.1)
    sil.backward()
    assert th.all(lp.grad[~include] == 0) and th.any(lp.grad[include] != 0)


def test_ut4_train_uses_gc1(monkeypatch):
    """train() が GC-1 の除外数を記録し、損失・KL の計算に include を渡していること。"""
    model = _model()
    model.collect_one()
    seen = []
    orig = learner.clipped_policy_loss

    def spy(log_prob, old, adv, include):
        seen.append(include)
        return orig(log_prob, old, adv, include)
    monkeypatch.setattr(learner, "clipped_policy_loss", spy)
    ro = model._rollout
    n_opt = learner.compute_n_options(th.as_tensor(ro["action_masks"]), 1)
    expected, _, _ = learner.include_mask(ro["actions"], n_opt)
    rec = model.train()
    assert seen and rec["n_included_elements"] == int(expected.sum())
    assert rec["gc1_excluded_by_group"]["zone_setpoint"] > 0
    assert rec["one_option_excluded_by_group"]["system_evap"] > 0      # M-3 の退化
    assert rec["gc1_excluded_by_group"]["zone_state"] == 0


# =====================================================================
# UT-8: 価値ネットワークの入力
# =====================================================================

def test_ut8_value_context_input():
    v = policy.ValueNetwork()
    assert v.context[0].in_features == 322
    obs = th.rand(4, 642)
    v.set_reward_context(0.2, 0.3)
    a = v(obs)
    v.set_reward_context(0.9, 0.1)
    b = v(obs)
    assert not th.allclose(a, b)                                      # (ρ_n, D_n) が価値に効く
    # 式: ρ_n = clip(log10(ρ)/4, 0, 1)、D_n = clip(log10(E* + ρ·DR*)/3, 0, 1)
    rho_n, D_n = reward_context(40.0, 6.5, 0.19)
    assert rho_n == pytest.approx(math.log10(40.0) / 4) and D_n == pytest.approx(math.log10(6.5 + 40 * 0.19) / 3)
    assert reward_context(1e5, 6.5, 0.19)[0] == 1.0 and reward_context(0.5, 0.5, 0.0)[1] == 0.0   # クリップ
    ref = ReferencePoint(6.53, 0.190)
    assert ref.value_context() == pytest.approx(reward_context(ref.rho, ref.E_star, ref.DR_star))
    # 学習器の方策から設定でき、採取時の価値（GAE）にも同じ値が使われる
    model = _model(n_envs=1)
    model.policy.set_reward_context(0.4, 0.6)
    assert model.policy.value_network.reward_context.tolist() == pytest.approx([0.4, 0.6])


# =====================================================================
# UT-9: SIL
# =====================================================================

def _entry(update, E, DR, official, rng, T=378):
    return {"update": update, "E": E, "DR": DR, "eci_official": official, "eci_smooth": ts.eci_mod.eci_smooth(E, DR),
            "ECI": ts.eci_mod.eci_smooth(E, DR), "J": None, "W": MOCK_W,
            "actions": rng.integers(0, 2, size=(T, 62)), "obs": rng.random((T, 642)).astype(np.float32),
            "masks": np.ones((T, 296), dtype=bool), "step_seconds": np.full(T, 900.0),
            "delta_E": rng.uniform(0, 0.02, T), "zone_energy": rng.uniform(0, 0.001, (T, 18)),
            "zone_n": rng.uniform(0, 1e-5, (T, 18))}


def test_ut9_sil_buffer_ranking(tmp_path):
    tr = Trainer(_trial_config(tmp_path), str(tmp_path), factory="mock")
    rng = np.random.default_rng(0)
    vals = [(0.0, 7.5, 0.20), (0.3, 6.0, 0.20), (0.0, 7.3, 0.20), (0.41, 6.0, 0.19), (0.2, 6.1, 0.2),
            (0.1, 6.5, 0.2), (0.0, 9.0, 0.2), (0.35, 6.2, 0.2), (0.05, 6.8, 0.2), (0.0, 7.21, 0.2)]
    for u, (o, E, DR) in enumerate(vals, start=1):
        tr.add_to_sil_buffer(_entry(u, E, DR, o, rng, T=4))
    assert len(tr.sil_buffer) == constants.SIL_BUFFER_SIZE == 8
    eci_list = [e["ECI"] for e in tr.sil_buffer]
    assert eci_list == sorted(eci_list, reverse=True)                     # V6.1-S: ECI の大きい順
    all_eci = sorted((ts.eci_mod.eci_smooth(E, DR) for _, E, DR in vals), reverse=True)
    assert eci_list == pytest.approx(all_eci[:8])


def test_ut9_sil_returns_match_step_rewards():
    ref = ReferencePoint(5.0, 0.20)
    rng = np.random.default_rng(1)
    e = _entry(1, 5.0, 0.2, 0.3, rng)
    R = learner.sil_returns(e, ref, 378)
    calc = RewardCalculator(ref, 378, MOCK_W)
    step = np.array([calc.step_reward(e["delta_E"][t], e["zone_n"][t].sum()) for t in range(378)])
    assert R.shape == (378, 19)
    assert R[0].sum() == pytest.approx(step.sum(), rel=1e-10)          # 収益の和 = ステップ報酬の和
    assert R[100].sum() == pytest.approx(step[100:].sum(), rel=1e-10)  # γ = 1、t 以降の和
    # 基準点が変わると計算し直した収益も変わる（基準点ごとに計算し直す）
    R2 = learner.sil_returns(e, ReferencePoint(6.0, 0.21), 378)
    assert not np.allclose(R, R2)


def test_ut9_sil_loss_zero_when_returns_below_values_and_sigma_scaling():
    include = th.ones(5, 62, dtype=th.bool)
    lp = th.randn(5, 62, requires_grad=True)
    assert learner.sil_policy_loss(lp, -th.rand(5, 62), 1.0, include, 0.1).item() == 0.0   # 全要素で R ≤ V
    R, V = th.rand(5, 19), th.rand(5, 19) + 1.0
    assert learner.sil_value_loss(R, V).item() == 0.0
    adv = th.rand(5, 62)
    l1 = learner.sil_policy_loss(lp, adv, 1.0, include, 0.1).item()
    l2 = learner.sil_policy_loss(lp, adv, 2.0, include, 0.1).item()
    assert l2 == pytest.approx(l1 / 2.0)                                # σ_A で割る（平均は引かない）
    manual = -0.1 * (lp * adv).sum().item() / include.sum().item()
    assert l1 == pytest.approx(manual, rel=1e-5)


def _sil_ready_model():
    model = _model()
    model.collect_one()
    rng = np.random.default_rng(3)
    buf = [_entry(k, 5.0, 0.2, 0.3, rng) for k in range(2)]
    for e in buf:   # 実際の標本に近いマスク（1 択を含む）にする
        e["masks"] = model._rollout["action_masks"][:378].copy()
        e["actions"] = model._rollout["actions"][:378].astype(np.int64)
    return model, buf


def test_ut9_sil_kl_adjustment_and_revert(monkeypatch):
    model, buf = _sil_ready_model()
    ref = ReferencePoint.from_dict(REF)
    obs = th.as_tensor(model._rollout["observations"]).float()
    acts = th.as_tensor(model._rollout["actions"]).long()
    masks = th.as_tensor(model._rollout["action_masks"]).bool()
    include = th.ones(acts.shape[0], 62, dtype=th.bool)
    kl_values = []
    monkeypatch.setattr(model, "_approx_kl", lambda *a, **k: kl_values.pop(0))

    def run():
        return model._sil(buf, "apply", ref, 1.0, obs, acts, masks, include)
    # 0.075 超で半分
    kl_values.append(0.08)
    rec = run()
    assert rec["c_sil_used"] == 0.1 and model.c_sil == pytest.approx(0.05) and not rec["reverted"]
    # 0.15 超で SIL 直前の重みに戻して半分
    before = {k: v.clone() for k, v in model.policy.policy_network.state_dict().items()}
    kl_values.append(0.2)
    rec = run()
    assert rec["reverted"] and model.c_sil == pytest.approx(0.025)
    after = model.policy.policy_network.state_dict()
    assert all(th.equal(before[k], after[k]) for k in before)
    # 下限 0.0125
    for _ in range(3):
        kl_values.append(0.1)
        run()
    assert model.c_sil == pytest.approx(constants.SIL_C_MIN)
    # 0.025 未満が 10 更新続いたら 2 倍（上限 0.1）
    for i in range(10):
        kl_values.append(0.01)
        rec = run()
    assert model.c_sil == pytest.approx(2 * constants.SIL_C_MIN) and rec["low_kl_streak"] == 0
    model.c_sil = 0.08
    model.sil_low_kl_streak = 9
    kl_values.append(0.01)
    run()
    assert model.c_sil == pytest.approx(constants.SIL_C_MAX)
    # 中間の値は連続を切る
    model.sil_low_kl_streak = 5
    kl_values.append(0.05)
    run()
    assert model.sil_low_kl_streak == 0


def test_ut9_sil_modes_in_train():
    model, buf = _sil_ready_model()
    ref = ReferencePoint.from_dict(REF)
    before = {k: v.clone() for k, v in model.policy.policy_network.state_dict().items()}
    rec = model.train(sil_buffer=buf, sil_mode="stats", reference=ref)
    assert rec["sil"]["mode"] == "stats" and 0.0 <= rec["sil"]["positive_advantage_fraction"] <= 1.0
    assert "approx_kl_sil" not in rec["sil"]                        # 更新 1〜300: 損失は適用しない
    model.collect_one()
    rec2 = model.train(sil_buffer=buf, sil_mode="apply", reference=ref)
    assert rec2["sil"]["mode"] == "apply" and "approx_kl_sil" in rec2["sil"] and rec2["sil"]["n_samples"] == 756
    assert rec2["sil"]["sigma_A"] == pytest.approx(rec2["advantage_norm_std"])


# =====================================================================
# UT-12・UT-13・UT-14
# =====================================================================

def test_ut12_observation_dimension_for_k9_result():
    assert constants.K9_RESULT == "B"
    assert constants.OBSERVATION_DIM == 84 + 31 * 18 == 642
    assert constants.GLOBAL_BLOCK_DIM == 84 and constants.ZONE_BLOCK_DIM == 31
    assert MockSummerEnv().observation_space.shape == (642,)
    from rlkit.env import SummerEnv
    assert SummerEnv(include_indoor_unit_power=False, zone_step_energy_norm_gj=0.02).observation_space.shape == (642,)
    assert observation.IDX_VENT_LEVEL == 80 and observation.IDX_REMOTE == 82


def test_ut13_stage_and_gate_processing_not_called(tmp_path, monkeypatch):
    """UT-13: 段階と門（G-1''・G-2・G-3'）、第 2 段階への変換が設定で呼ばれない。"""
    def boom(*a, **k):
        raise AssertionError("段階・門の処理が呼ばれた")
    for name in ("gate_g1_satisfied", "gate_g2_relative_difference", "gate_g3_release_ok"):
        monkeypatch.setattr(ts, name, boom)
    monkeypatch.setattr(learner, "transfer_to_stage2", boom)
    monkeypatch.setattr(learner, "transfer_policy_stage1_to_stage2", boom)
    monkeypatch.setattr(action, "convert_action_stage1_to_stage2", boom)
    monkeypatch.setattr(policy, "expand_zone_head_to_stage2", boom)
    for name in ("maybe_switch_stage", "convert_to_stage2", "attempt_m5_release", "check_stage1_ceiling"):
        assert not hasattr(Trainer, name)
    tr = Trainer(_trial_config(tmp_path, planned_updates=2), str(tmp_path), factory="mock")
    reason = tr.run()
    assert "計画更新数 2" in reason and tr.model.stage == 1
    assert not os.path.exists(tmp_path / "gates.jsonl")


def _rows(n, smooth, cut=None, secs=None, C=None):
    rows = []
    for u in range(1, n + 1):
        row = {k: 0 for k in ts.REQUIRED_UPDATE_KEYS}
        e = C[u - 1] if C else 7.0 - 0.01 * u
        row.update({"update": u, "harvest_update_best": {"eci_smooth": smooth[u - 1]},
                    "kl_early_stop": bool(cut[u - 1]) if cut else False,
                    "update_seconds": secs[u - 1] if secs else 400.0, "eval_raw_resid_rel": 0.01,
                    "train_E_list": [e] * 12, "train_DR_list": [0.2] * 12, "reference_used": {"rho": 60.0}})
        rows.append(row)
    return rows


def test_ut14_judgment_functions():
    # H（更新 20）
    rows = _rows(20, [0.1 + 0.01 * i for i in range(20)])
    h = ts.h_checks(rows, ts.REQUIRED_UPDATE_KEYS)
    assert h["all_passed"] and "H-7" not in h
    bad = _rows(20, [0.3 - 0.01 * i for i in range(20)], cut=[1] * 20, secs=[530] * 20,
                C=[6.0 + 0.01 * u for u in range(1, 21)])
    hb = ts.h_checks(bad, ts.REQUIRED_UPDATE_KEYS)
    assert not hb["H-2"]["passed"] and not hb["H-3"]["passed"] and not hb["H-4''"]["passed"] and not hb["H-5"]["passed"]
    del bad[3]["c_g"]
    assert not ts.h_checks(bad, ts.REQUIRED_UPDATE_KEYS)["H-1"]["passed"]
    # I（更新 300）: 281〜300 の中央値が、それ以前の 20 更新ごとの中央値の最大の 0.8 倍以上
    s = [0.4] * 280 + [0.33] * 20
    i_ok = ts.interim_checks(_rows(300, s), ts.REQUIRED_UPDATE_KEYS)
    assert i_ok["all_passed"] and len(i_ok["I-2"]["block_medians_1_280"]) == 14
    s_bad = [0.4] * 280 + [0.31] * 20
    assert not ts.interim_checks(_rows(300, s_bad), ts.REQUIRED_UPDATE_KEYS)["I-2"]["passed"]
    cut = [0] * 280 + [1] * 16 + [0] * 4
    assert not ts.interim_checks(_rows(300, s, cut=cut), ts.REQUIRED_UPDATE_KEYS)["I-1"]["passed"]
    raw_bad = _rows(300, s)
    raw_bad[-1]["eval_raw_resid_rel"] = 0.031
    assert not ts.interim_checks(raw_bad, ts.REQUIRED_UPDATE_KEYS)["I-3"]["passed"]
    # SIL-6（更新 340）
    hist = [0.4] * 300 + [0.31] * 40
    assert ts.sil_guard_check(hist)["triggered"]
    assert not ts.sil_guard_check([0.4] * 300 + [0.33] * 40)["triggered"]


def test_ut14_trainer_wiring_interim_sil_and_guard(tmp_path, monkeypatch):
    """更新 300 の中間判定の合格で更新 301 から SIL が有効になり、更新 340 の見張りで c_sil = 0 になる（番号を縮めて試す）。"""
    monkeypatch.setattr(constants, "INTERIM_UPDATE", 2)
    monkeypatch.setattr(constants, "SIL_START_UPDATE", 3)
    monkeypatch.setattr(constants, "SIL_GUARD_UPDATE", 4)
    monkeypatch.setattr(ts, "interim_checks", lambda updates, keys: {"all_passed": True, "stub": True})
    monkeypatch.setattr(ts, "sil_guard_check", lambda hist: {"triggered": True, "stub": True})
    tr = Trainer(_trial_config(tmp_path, planned_updates=5), str(tmp_path), factory="mock")
    tr.run()
    ups = _read(tmp_path / "updates.jsonl")
    modes = [u["sil"]["mode"] for u in ups]
    assert modes == ["stats", "stats", "apply", "apply", "apply"]
    assert ups[2]["sil"]["c_sil_used"] == pytest.approx(constants.SIL_C_INIT)
    assert ups[4]["sil"]["c_sil"] == 0.0 and "approx_kl_sil" not in ups[4]["sil"]   # 見張りの後は c_sil = 0
    assert tr.sil_enabled and tr.sil_guard["triggered"]


def test_ut14_trainer_wiring_interim_fail_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "INTERIM_UPDATE", 2)
    monkeypatch.setattr(ts, "interim_checks", lambda updates, keys: {"all_passed": False, "stub": True})
    tr = Trainer(_trial_config(tmp_path, planned_updates=5), str(tmp_path), factory="mock")
    reason = tr.run()
    assert reason.startswith("S-5") and tr.model.update_index == 2 and not tr.sil_enabled


# =====================================================================
# 既存の試験（V6 の形に合わせたもの）
# =====================================================================

def test_gate_functions():
    """V5 までの門の判定関数（A-7 により V6 では呼ばないが、コードは残す）。"""
    assert gate_g1_satisfied(3, 200)
    assert not gate_g1_satisfied(2, 200)
    assert not gate_g1_satisfied(3, 199)
    assert gate_g2_relative_difference(0.5, 0.5) == 0.0
    assert gate_g2_relative_difference(0.5, 0.5001) == pytest.approx(2e-4)
    assert math.isinf(gate_g2_relative_difference(0.0, 0.1))
    assert gate_g3_release_ok(0.5, 0.49) and not gate_g3_release_ok(0.5, 0.489)


def test_harvest_helpers():
    assert median_of([0.1, 0.3, 0.2]) == pytest.approx(0.2)
    assert median_of([]) is None
    flat = np.zeros((378, 62), dtype=np.int64)
    assert is_degenerate_actions(flat)
    flat2 = flat.copy(); flat2[100, 3] = 1
    assert not is_degenerate_actions(flat2)


def test_m5_release_state():
    """V5S-a の回帰（解除の状態の管理。A-7 により V6 では使わない）。"""
    st = M5ReleaseState()
    st.fail(10)
    assert st.retry_at == 10 + constants.G3_RETRY_UPDATES and st.due(st.retry_at)
    st.pending_since, st.before_median, st.retry_at = 60, 0.3, None
    assert not st.due(61) and not st.due(100)
    st2 = M5ReleaseState.from_dict(st.to_dict())
    assert (st2.pending_since, st2.before_median) == (60, 0.3)


def test_policy_separation_and_optimizers():
    model = _model(n_envs=1)
    pol = model.policy
    ids_pi = {id(p) for p in pol.policy_network.parameters()}
    ids_v = {id(p) for p in pol.value_network.parameters()}
    assert not ids_pi & ids_v
    assert {id(p) for g in pol.optimizer.param_groups for p in g["params"]} == ids_pi
    assert {id(p) for g in pol.value_optimizer.param_groups for p in g["params"]} == ids_v
    assert pol.value_optimizer.param_groups[0]["lr"] == constants.VALUE_LR


def test_collect_and_train_one_update():
    model = _model()
    terminal = model.collect_one()
    assert len(terminal) == 4  # 2 組 × 2 周
    ro = model._rollout
    assert ro["observations"].shape == (1512, 642) and ro["advantages19"].shape == (1512, 19)
    assert ro["actions"].shape == (1512, 62) and ro["action_masks"].shape == (1512, 296)
    assert ro["zone_energy"].shape == (1512, 18) and ro["delta_E"].shape == (1512,)   # SIL-1 の保存量
    assert ro["segments"] == [(0, 378), (378, 756), (756, 1134), (1134, 1512)]
    assert np.allclose(ro["rewards19"].sum(axis=1),
                       learner.adv_mod.flatten_env_major(model.rollout_buffer.rewards).reshape(-1))
    v_before = [p.detach().clone() for p in model.policy.value_network.parameters()]
    rec = model.train()
    assert 1 <= rec["epochs_applied"] <= 4 and rec["update"] == 1
    assert constants.POLICY_LR_MIN <= rec["policy_lr_after"] <= constants.POLICY_LR_MAX
    assert math.isfinite(rec["value_loss_last"]) and rec["nonfinite"] is False
    assert rec["entropy_c0"] == pytest.approx(0.002) and set(rec["c_g"]) == set(constants.ENTROPY_GROUPS)
    assert rec["floor_fraction"] is None                               # L-1: 床は廃止
    assert any(not th.equal(a, b) for a, b in zip(v_before, model.policy.value_network.parameters()))
    assert rec["grad_norm_policy"] > 0 and rec["grad_norm_ratio"] is not None   # V3-S E-2 の記録（更新 1）


def test_grad_norm_ratio_measured_only_at_listed_updates():
    assert constants.GRAD_NORM_RATIO_MEASURE_AT == (1, 10, 50, 100, 200)
    model = _model(planned=200)
    model.collect_one()
    assert model.train()["grad_norm_ratio"] is not None
    model.collect_one()
    assert model.train()["grad_norm_ratio"] is None


def test_kl_early_stop_discards_epoch():
    model = _model()
    model.collect_one()
    learner.set_optimizer_lr(model.policy.optimizer, 0.5)
    rec = model.train()
    assert rec["kl_early_stop"] is True
    assert rec["epochs_applied"] == len(rec["approx_kl_per_epoch"]) - 1
    assert rec["approx_kl_per_epoch"][-1] > constants.KL_STOP_FACTOR * constants.TARGET_KL


def test_deterministic_eval_stage1():
    m = _model(n_envs=1)
    vec = DummyVecEnv([lambda: MockSummerEnv(seed=5)])
    info, acts = learner.run_deterministic_episode(m.policy, vec)
    assert acts.shape == (378, 62) and "E_rep" in info and "J" in info


def test_trainer_orchestration_with_mock(tmp_path):
    cfg = _trial_config(tmp_path, planned_updates=3)
    trainer = Trainer(cfg, str(tmp_path), factory="mock")
    trainer.run()
    status = json.load(open(tmp_path / "status.json", encoding="utf-8"))
    assert status["status"] == "stopped", status
    evals = _read(tmp_path / "evals.jsonl")
    updates = _read(tmp_path / "updates.jsonl")
    assert evals[0]["update"] == 0 and evals[0]["tag"] == "initial"
    assert evals[1]["eci_smooth"] == evals[0]["eci_smooth"]           # 同じ重みの決定論的評価は一致
    assert len(updates) == 3 and all(u["train_episodes"] == 12 for u in updates)
    for u in updates:                                                  # 第 3-8 節の項目がそろう（H-1 の対象）
        assert all(k in u for k in ts.REQUIRED_UPDATE_KEYS), [k for k in ts.REQUIRED_UPDATE_KEYS if k not in u]
    u0 = updates[0]
    # V5 から引き継いだ記録項目
    for key in ("approx_kl_after", "epochs_applied", "policy_lr_after", "c_g", "H_g", "value_loss_last", "reference",
                "reference_update", "eval_raw_resid_rel", "eval_obs25_clip_rate_mean", "grad_norm_ratio",
                "inside_rate_recent20", "harvest_best_official", "harvest_official_streak", "approx_kl_by_group",
                "official_positive_streak", "eval_inside_zero_floor", "m5_enabled", "m8_enabled", "floor_fraction"):
        assert key in u0, key
    assert set(u0["approx_kl_by_group"]) == set(constants.ENTROPY_GROUPS)
    for key in ("rho", "E_ref", "DR_ref", "ref_update", "ref_ECI", "rho_raw", "rho_n", "D_n",
                "alpha_s", "beta_s", "sigma_E", "sigma_D", "rho_r5"):            # R-5 の項目は None で残す
        assert key in u0["reference_used"], key
    assert {"rho_n", "D_n", "rho_raw", "rho_clip_30_200", "changed", "E_ref", "DR_ref"} <= set(u0["reference_update"])
    assert u0["eval_m9_tenant_steps"] > 0 and u0["eval_m5_zone_steps"] > 0        # M-9・M-5' が効いている
    assert set(u0["eval_vent_level_fraction"]) == {"South", "North"}
    assert u0["sil"]["mode"] == "stats" and len(u0["sil_buffer"]) == 1
    assert os.path.exists(tmp_path / "checkpoint_latest.pt") and os.path.exists(tmp_path / "sil_buffer.pt")
    assert status["stage"] == 1
    assert os.path.exists(tmp_path / "harvest" / "replay_check.json")              # F-1


def test_reference_follows_best_harvest(tmp_path):
    """R-7: 評価点は走行全体の収穫の最良（ECI 最大）で、その比が次の更新の採取に使われる（V6-S の R-2'' から差し替え）。"""
    cfg = _trial_config(tmp_path, planned_updates=4)
    Trainer(cfg, str(tmp_path), factory="mock").run()
    updates = _read(tmp_path / "updates.jsonl")
    prev = None
    for u in updates:
        ru = u["reference_update"]
        assert ru["E_ref"] == pytest.approx(u["harvest_best"]["E"]) and ru["DR_ref"] == pytest.approx(u["harvest_best"]["DR"])
        assert ru["ref_update"] == u["harvest_best"]["update"]
        if prev is not None:
            assert u["reference_used"]["rho"] == pytest.approx(prev["rho"])       # 前の更新の終わりの比で採取
            assert prev["rho"] / 2.0 - 1e-12 <= ru["rho"] <= prev["rho"] * 2.0 + 1e-12
            if not ru["changed"]:
                assert ru["rho"] == prev["rho"]                                   # 評価点が変わらなければ比も変えない
        assert constants.R7_RHO_MIN <= ru["rho"] <= constants.R7_RHO_MAX
        prev = ru


def test_harvest_best_action_sequence(tmp_path):
    """L-4: 12 本から公式 ECI 最大（同点は J 最大）の行動列が保存され、走行全体の最良が更新される。"""
    cfg = _trial_config(tmp_path, planned_updates=3)
    trainer = Trainer(cfg, str(tmp_path), factory="mock")
    trainer.run()
    updates = _read(tmp_path / "updates.jsonl")
    eps = _read(tmp_path / "episodes.jsonl")
    for u in updates:
        best = u["harvest_update_best"]
        same = [e for e in eps if e["update"] == u["update"]]
        assert len(same) == 12
        top = max(same, key=lambda e: harvest_key(e))
        assert best["E"] == pytest.approx(top["E"]) and best["slot"] == top["slot"]
        assert best["ECI"] == pytest.approx(top["ECI"])          # V6.1-S: ECI の大きい順
    run_best = [harvest_key(u["harvest_best"]) for u in updates]
    assert run_best == sorted(run_best)
    meta = json.load(open(tmp_path / "harvest" / "best_harvest.json", encoding="utf-8"))
    d = np.load(tmp_path / "harvest" / "best_harvest.npz")
    assert d["actions"].shape == (378, 62) and d["masks"].shape == (378, 296)
    assert meta["E_hex"] == float.hex(meta["E"])
    buf = th.load(tmp_path / "sil_buffer.pt", weights_only=False)
    assert [e["update"] for e in buf][0] == meta["update"]                       # 保存庫の先頭は走行全体の最良
    assert buf[0]["obs"].shape == (378, 642) and buf[0]["zone_n"].shape == (378, 18)


def test_trainer_resume_keeps_state(tmp_path):
    """MUST-9: 再開で更新番号・基準点・学習率・収穫の最良・SIL の保存庫・エントロピーの予定が引き継がれる。"""
    cfg = _trial_config(tmp_path, planned_updates=10)
    t1 = Trainer(cfg, str(tmp_path), factory="mock", max_updates=2)
    t1.run()
    snap1 = t1.resume_snapshot()
    t2 = Trainer(cfg, str(tmp_path), factory="mock", max_updates=3)
    t2.run(resume=True)
    events = [e for e in _read(tmp_path / "events.jsonl") if e.get("event") == "resumed"]
    assert len(events) == 1
    ev = events[0]
    assert ev["update_index"] == 2 and ev["policy_lr"] == pytest.approx(snap1["policy_lr"])
    assert ev["reference"]["E_ref"] == pytest.approx(snap1["reference"]["E_ref"])
    assert ev["rho"] == pytest.approx(snap1["rho"]) and ev["reference"]["ref_update"] == snap1["reference"]["ref_update"]
    assert ev["best_harvest"] == snap1["best_harvest"] and ev["sil_buffer"] == snap1["sil_buffer"]
    assert ev["c_sil"] == snap1["c_sil"] and ev["entropy_c0_next_update"] == pytest.approx(learner.entropy_c0_at(3, 10))
    updates = _read(tmp_path / "updates.jsonl")
    assert [u["update"] for u in updates] == [1, 2, 3]
    assert updates[2]["reference_used"]["E_ref"] == pytest.approx(updates[1]["reference_update"]["E_ref"])
    assert updates[2]["reference_used"]["rho"] == pytest.approx(updates[1]["reference_update"]["rho"])
    assert updates[2]["policy_lr_before"] == pytest.approx(updates[1]["policy_lr_after"])
