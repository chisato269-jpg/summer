"""指示書 V6.4-S の段階 A の単体試験（V6.2-S の UT-26〜UT-31、V6.4-S の UT-32'、V6.3-S の UT-33・UT-34）。

エミュレータを使う部分（UT-29 のリモコンの読み戻し、UT-31 の反事実の再生、UT-32' の基準の再生、UT-34 の 7 組同時）は
test_v64_live.py にある（RLKIT_LIVE=1 のときだけ実行）。
"""
import json
import os

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv

from rlkit import action, advantage, constants, learner, policy, r8p
from rlkit.mockenv import MockSummerEnv
from rlkit.reward import R7Reference, ReferencePoint

REF = ReferencePoint(4.587348191951224, 0.18481666361514731).to_dict()
SYS = np.array(action.SYSTEM_OF_ZONE)
TEN = np.array(action.TENANT_OF_ZONE)


def _adv_sample():
    rng = np.random.default_rng(7)
    T = 3
    zone_adv = rng.normal(size=(T, 18))
    sys_adv = rng.normal(size=T)
    acts = np.zeros((T, constants.ACTION_DIMS), dtype=np.int64)
    acts[0, [0, 1, 9]] = 1       # t=0: S1・S2（系統 1）と N1（系統 3）が運転
    acts[2, :18] = 1             # t=1: 全ゾーン停止、t=2: 全ゾーン運転
    return zone_adv, sys_adv, acts


# ---- UT-26 ----

def test_ut26_level_formulas():
    """UT-26: 局所・系統・テナント・建物の各水準が V6.2-S 第 3 章の式どおり（系統の水準で全運転ゾーン数 0 の場合を含む）。"""
    A, As, acts = _adv_sample()
    loc = advantage.level_zone_advantages(A, As, acts, "local")
    sysl = advantage.level_zone_advantages(A, As, acts, "system")
    ten = advantage.level_zone_advantages(A, As, acts, "tenant")
    bld = advantage.level_zone_advantages(A, As, acts, "building")
    on = action.zone_on_flags_batch(acts)
    for t in range(3):
        n_all = on[t].sum()
        for k in range(18):
            s, tt = SYS[k], TEN[k]
            assert loc[t, k] == A[t, k]
            share = on[t, SYS == s].sum() / n_all if n_all > 0 else 0.0     # 全運転ゾーン数 0 なら第 2 項は 0
            assert sysl[t, k] == pytest.approx(A[t, SYS == s].sum() + As[t] * share)
            assert ten[t, k] == pytest.approx(A[t, TEN == tt].sum())
            assert bld[t, k] == pytest.approx(A[t].sum() + As[t])
    assert sysl[1, 0] == pytest.approx(A[1, SYS == 0].sum())                  # t=1 は全停止
    tv = advantage.tenant_level_advantages(A, As, "tenant")
    assert tv[:, 0] == pytest.approx(A[:, TEN == 0].sum(axis=1)) and tv[:, 1] == pytest.approx(A[:, TEN == 1].sum(axis=1))
    assert advantage.tenant_level_advantages(A, As, "building")[:, 1] == pytest.approx(A.sum(axis=1) + As)
    with pytest.raises(ValueError):
        advantage.level_zone_advantages(A, As, acts, "impact")               # NOT-4: 判定表に無い水準
    # 既定の水準（K-14'）: S 建物、P 建物、V テナント。系統の次元は現行のまま
    assert constants.CREDIT_LEVELS == {"S": "building", "P": "building", "V": "tenant"}
    out = advantage.assign_per_dim_advantages(A, As, acts)
    L = action.LAYOUT
    for key in ("zone_state", "zone_setpoint", "zone_fan"):
        s0, n = L[key]
        assert np.allclose(out[:, s0:s0 + n], bld)
    v0 = L["tenant_vent"][0]
    assert np.allclose(out[:, v0:v0 + 2], tv)
    e0 = L["system_evap"][0]
    a = np.asarray(constants.OUTDOOR_UNIT_A_KW)
    assert out[0, e0] == pytest.approx(A[0, SYS == 0].sum() + As[0] * 2 / 3)
    assert out[1, e0] == pytest.approx(A[1, SYS == 0].sum() + As[1] * a[0] / a.sum())   # 系統の次元の補完は現行のまま
    assert out.shape == (3, 60)


# ---- UT-27 ----

def test_ut27_normalization_by_level_and_sil_sigma():
    """UT-27: 正規化が水準ごとに行われ、GC-1 と 1 択の除外が守られる。SIL の σ_A が水準ごとの値になる。"""
    rng = np.random.default_rng(3)
    T = 50
    m = rng.normal(size=(T, 60)) * np.r_[np.full(54, 5.0), np.full(4, 0.1), np.full(2, 30.0)] + 2.0
    include = rng.random((T, 60)) > 0.3
    levels = advantage.dim_levels()
    norm, stats, sigma = advantage.normalize_by_level(m, include, levels)
    assert set(stats) == {"building", "system_evap", "tenant"}               # S と P は同じ建物の水準で 1 つに正規化
    for lv, st in stats.items():
        cols = levels == lv
        vals = norm[:, cols][include[:, cols]]
        assert vals.mean() == pytest.approx(0.0, abs=1e-9) and vals.std() == pytest.approx(1.0, rel=1e-6)
        assert np.all(sigma[:, cols] == st["std"]) and st["n"] == int(include[:, cols].sum())
    # 除外した要素の値は、どの水準の平均・標準偏差にも影響しない
    m2 = m.copy()
    m2[~include] = 1e9
    _, stats2, _ = advantage.normalize_by_level(m2, include, levels)
    assert stats2 == stats
    # 学習器の include（GC-1 と 1 択）: 停止ゾーンの設定温度・風量、運転ゾーンの無い系統、1 択の要素を除く
    acts = np.zeros((2, 60), dtype=np.int64)
    acts[0, 0] = 1
    masks = np.ones((2, 292), dtype=bool)
    v0 = sum(action.nvec(1)[:58])
    masks[1, v0:v0 + 4] = [True, False, False, False]
    inc, _, _ = learner.include_mask(acts, learner.compute_n_options(th.as_tensor(masks), 1))
    inc = inc.numpy()
    assert inc[0, 18] and not inc[0, 19] and not inc[1, 58] and inc[1, 59] and inc[0, 54] and not inc[0, 55]
    # SIL の σ_A: 要素の水準の、PPO の正規化の標準偏差
    sig = learner.sil_sigma_matrix(stats, 4)
    assert sig.shape == (4, 60)
    for lv, st in stats.items():
        assert np.all(sig[:, levels == lv] == st["std"])
    # 要素の無い水準は 1（その水準の要素は SIL に寄与しない）
    sig0 = learner.sil_sigma_matrix({"building": {"std": 2.0, "n": 5}, "system_evap": {"std": 3.0, "n": 0},
                                     "tenant": {"std": 4.0, "n": 1}}, 1)
    assert np.all(sig0[0, levels == "system_evap"] == 1.0) and np.all(sig0[0, levels == "building"] == 2.0)
    # 損失は要素ごとの σ で割る
    lp = th.randn(1, 60, requires_grad=True)
    adv = th.rand(1, 60)
    one = th.ones(1, 60, dtype=th.bool)
    l_el = learner.sil_policy_loss(lp, adv, th.as_tensor(sig0).float(), one, 0.1).item()
    manual = -0.1 * (lp * adv / th.as_tensor(sig0).float()).sum().item() / 60
    assert l_el == pytest.approx(manual, rel=1e-5)


# ---- UT-28 ----

def test_ut28_ratio_follows_rho_raw_every_update():
    """UT-28（C-4）: 評価点が変わらない更新でも、比が ρ_raw へ 2 倍の制限の範囲で近づく（30 → 60 → 72.3）。"""
    E, DR = 6.63993017578125, 0.2595181465148926                  # trial06 の評価点（ρ_raw ≈ 72.3）
    raw = R7Reference.raw_ratio(E, DR)
    assert raw == pytest.approx(72.3, abs=0.01)
    r = R7Reference(rho=30.0, E_ref=E, DR_ref=DR, ref_update=5, ref_eci=0.4)
    best = {"update": 5, "E": E, "DR": DR, "ECI": 0.4}
    e1 = r.update_after_harvest(best, 6)
    assert not e1["changed"] and e1["rho"] == pytest.approx(60.0) and e1["rho_limited_by_ratio"]
    e2 = r.update_after_harvest(best, 7)
    assert not e2["changed"] and e2["rho"] == pytest.approx(raw) and not e2["rho_limited_by_ratio"]
    e3 = r.update_after_harvest(best, 8)
    assert e3["rho"] == pytest.approx(raw)                           # 届いた後は動かない
    # 評価点がまだ無い（更新 1 の前）なら ρ₀ のまま
    r0 = R7Reference()
    e0 = r0.update_after_harvest(None, 0)
    assert e0["rho"] == 60.0 and e0["rho_raw"] is None


# ---- UT-29（エミュレータを使わない部分） ----

def test_ut29_remote_removed_shapes_and_fixed_permit():
    """UT-29（C-5）: 行動 60 次元・292 ロジット、観測 640 次元、母数 145,980。リモコンは常に許可。"""
    assert constants.ACTION_DIMS == 60 and constants.ACTION_LOGITS[1] == 292 and constants.OBSERVATION_DIM == 640
    assert "tenant_remote" not in action.LAYOUT and "tenant_remote" not in constants.ENTROPY_GROUPS
    p, v = policy.PolicyNetwork(1), policy.ValueNetwork()
    assert policy.parameter_count(p) == 75610 and policy.parameter_count(v) == 70370
    assert policy.parameter_count(p) + policy.parameter_count(v) == 145980
    assert p.tenant_head.out_features == 4
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = np.array([rng.integers(0, n) for n in action.nvec(1)])
        assert action.decode_action(a, 1)["remote"] == {"South": True, "North": True}
    assert MockSummerEnv().action_space.nvec.tolist() == action.nvec(1)


# ---- UT-30 ----

def test_ut30_lambda_hour_per_step():
    """UT-30（C-3）: λ_hour = 0.99 が 900 秒・3,600 秒のステップで λ_hour^(Δt/3600) として使われる。"""
    assert constants.LAMBDA_HOUR == 0.99
    lam = advantage.lambda_per_step([900, 3600, 900])
    assert lam[0] == pytest.approx(0.99 ** 0.25) and lam[1] == pytest.approx(0.99) and lam[2] == lam[0]
    rng = np.random.default_rng(4)
    secs = np.array([3600, 3600, 900, 900, 900, 900, 3600])
    r, v = rng.normal(size=(7, 19)), rng.normal(size=(7, 19))
    adv, _ = advantage.segmented_gae(r, v, secs, [(0, 7)])
    ref, _ = advantage.gae_variable_lambda(r, v, 0.99 ** (secs / 3600.0))
    assert np.allclose(adv, ref)
    adv95, _ = advantage.segmented_gae(r, v, secs, [(0, 7)], lambda_hour=0.95)
    assert not np.allclose(adv, adv95)


# ---- UT-31（エミュレータを使わない部分） ----

def test_ut31_k14_tool_overrides_only_four_steps():
    """UT-31: K-14 の道具。指定したステップから 4 ステップだけ発停・設定温度・換気が上書きされ、その後は基準に戻る。
    恒等式の一致は、記録済みの K-14'（112 本）で確かめる（再生は test_v64_live.py）。"""
    from rlkit.diag import v62s_diag, v64s_diag
    base = v62s_diag.baseline_actions()
    assert base.shape == (378, 60)
    dts = v62s_diag.step_datetimes()
    jobs = v62s_diag.k14_jobs(base, dts) + v64s_diag.s2_jobs(base, dts)
    L = action.LAYOUT
    for j in jobs:
        i = j["start"]
        n = j.get("n_flip_steps") or v62s_diag.K14_FLIP_STEPS
        diff = np.argwhere(j["actions"] != base)
        assert diff.size > 0 and diff[:, 0].min() >= i and diff[:, 0].max() < i + n, j["label"]
        cols = set(diff[:, 1].tolist())
        if j["family"] in ("S", "S2"):
            k = j["zone"]
            assert cols <= {L["zone_state"][0] + k, L["zone_setpoint"][0] + k, L["zone_fan"][0] + k}
        elif j["family"] == "P":
            assert cols == {L["zone_setpoint"][0] + j["zone"]}
        else:
            assert cols == {L["tenant_vent"][0] + j["tenant"]}
    assert sum(j["family"] == "S" for j in jobs) == 54 and sum(j["family"] == "S2" for j in jobs) == 36
    d = json.load(open(os.path.join("runs", "diag", "V64S", "K14_decision.json"), encoding="utf-8"))
    assert d["n_rows"] == 112 and not d["identity_violations"] and d["identity_abs_err_max"] <= 1e-9
    rows = json.load(open(os.path.join("runs", "diag", "V64S", "K14", "K14p_rows.json"), encoding="utf-8"))
    for row in rows:
        denom = max(abs(row["dC_official"]), 1e-12)
        assert abs(row["dC_true"] - row["dC_official"]) <= 0.01 * denom or row["identity_abs_err"] <= 1e-9


# ---- UT-32' ----

def test_ut32p_r8p_on_baseline_record():
    """UT-32'（R-8'）: 基準の軌跡の 1 エピソード（R8'-2 の再生の記録）で (1)〜(4) を確かめる。"""
    from rlkit import exogenous
    from rlkit.diag import v64s_diag
    rec = v64s_diag.load_full(os.path.join("runs", "diag", "V64S", "R8p_baseline.npz"))
    exo = exogenous.load()
    n_kt, n_t, empty = r8p.zone_n(constants.R8P_BRANCH, constants.R8P_DELTA_MIN, rec["A"], rec["reads"],
                                  rec["starts"], rec["minute"], exo)
    A_T = rec["A"][-1]
    assert A_T == rec["DR"]
    assert abs(n_t.sum() - A_T) / A_T <= 1e-9                                         # (1)
    assert np.abs(n_kt.sum(axis=1) - n_t).max() <= 1e-15                              # (2)
    ev = r8p.evaluate(rec["A"], rec["reads"], rec["starts"], rec["minute"], exo)
    assert ev["branch"] == constants.R8P_BRANCH and ev["delta"] == constants.R8P_DELTA_MIN
    assert ev["by_delta"][str(ev["delta"])]["L1"] <= 1e-3                             # (3)
    assert empty.any() and np.abs(n_t[empty]).max() <= 1e-7                            # (4)
    assert n_t.min() >= -1e-7 and abs(n_kt.sum() - rec["DR"]) / rec["DR"] <= 1e-9
    assert constants.R8P_S11_TOL[constants.R8P_BRANCH] == 1e-9


# ---- UT-33 ----

def _model(n_envs=2):
    vec = DummyVecEnv([lambda i=i: MockSummerEnv(seed=i) for i in range(n_envs)])
    vec.env_method("set_reference", REF)
    return learner.SummerPPO(vec, 1, planned_updates=10, seed=constants.RANDOM_SEED, n_steps_per_env=756)


def test_ut33_r8p_replacement_precedes_returns(monkeypatch):
    """UT-33: R-8' の置き換えが、収益・優位度・価値の目標の計算より前に行われる。"""
    seen = {}
    orig = advantage.segmented_gae

    def spy(rewards, values, *a, **k):
        seen["rewards"] = np.array(rewards, copy=True)
        return orig(rewards, values, *a, **k)
    monkeypatch.setattr(learner.adv_mod, "segmented_gae", spy)
    model = _model()
    terminal = model.collect_one()
    ro = model._rollout
    T = 756
    exp_r = np.zeros((T, 2, 18))
    exp_n = np.zeros((T, 2, 18))
    for step, i, info in terminal:
        exp_r[step - 377:step + 1, i] = info["r8p_zone_rewards"]
        exp_n[step - 377:step + 1, i] = info["r8p_zone_n"]
        assert not np.allclose(info["r8p_zone_n"][:, 0], info["r8p_zone_n"][:, -1])   # 配分が終端 1 点（等分）と違う
    flat = learner.adv_mod.flatten_env_major
    assert np.array_equal(seen["rewards"][:, :18], flat(exp_r))                        # GAE に渡した報酬が R-8' の値
    assert np.array_equal(ro["rewards19"][:, :18], flat(exp_r)) and np.array_equal(ro["zone_n"], flat(exp_n))
    adv, ret = orig(ro["rewards19"], ro["values19"], ro["step_seconds"], ro["segments"])
    assert np.allclose(ro["advantages19"], adv) and np.allclose(ro["returns19"], ret)   # 価値の目標も R-8' の報酬から
    assert ro["r8p"]["episodes"] == 4 and ro["r8p"]["branch"] == constants.R8P_BRANCH
    # 置き換えられない行があれば例外（終端 1 点の n を残さない）
    zr, zn = np.zeros((10, 1, 18)), np.zeros((10, 1, 18))
    with pytest.raises(RuntimeError):
        learner.apply_r8p(zr, zn, [])


def test_ut33_r8p_in_sil_buffer_and_eval_records(tmp_path):
    """UT-33: SIL の保存庫と決定論的評価の記録にも R-8' の値が入る。"""
    from rlkit import config
    from rlkit import train_summer as ts
    cfg = config.load()
    cfg.update({"include_indoor_unit_power": False, "zone_step_energy_norm_gj": 0.01, "planned_updates": 2,
                "run_dir": str(tmp_path)})
    tr = ts.Trainer(cfg, str(tmp_path), factory="mock")
    tr.run()
    buf = th.load(tmp_path / "sil_buffer.pt", weights_only=False)
    share = np.arange(1, 19, dtype=float)
    share /= share.sum()
    zn = buf[0]["zone_n"]
    assert np.allclose(zn / zn.sum(axis=1, keepdims=True), share)                        # 模擬環境の R-8' の配分
    ups = [json.loads(l) for l in open(tmp_path / "updates.jsonl", encoding="utf-8")]
    evs = [json.loads(l) for l in open(tmp_path / "evals.jsonl", encoding="utf-8")]
    for u in ups:
        assert u["r8p_branch"] == "R8'-A" and u["train_r8p_sum_rel_max"] <= 1e-9
        assert "eval_raw_resid_dissatisfaction_rel" in u and "eval_raw_resid_energy_rel" in u
        assert u["credit_levels"] == constants.CREDIT_LEVELS and u["lambda_hour"] == 0.99
        assert set(u["advantage_norm_by_level"]) == {"building", "system_evap", "tenant"}
        assert u["occupied_zone_step_off_fraction"]["n_occupied_zone_steps"] > 0
        assert u["train_remote_permit_fraction_mean"] is None and u["eval_remote_permit_fraction"] is None
    for e in evs:
        assert "r8p_sum_rel" in e and "raw_resid_dissatisfaction_rel" in e


def test_s11_and_s7_dissatisfaction_stop(tmp_path, monkeypatch):
    """S-11（R8'-A で 1e-9）と、S-7 の不満足の側の判定。"""
    from rlkit import config
    from rlkit import train_summer as ts
    cfg = config.load()
    cfg.update({"include_indoor_unit_power": False, "zone_step_energy_norm_gj": 0.01, "planned_updates": 3,
                "run_dir": str(tmp_path)})
    tr = ts.Trainer(cfg, str(tmp_path), factory="mock")
    tr.stop_reason = None
    tr.judge_after_update(5, {"eval_raw_resid_rel": 0.0, "eval_raw_resid_dissatisfaction_rel": 0.0,
                              "train_r8p_sum_rel_max": 2e-9})
    assert tr.stop_reason.startswith("S-11")
    tr.stop_reason = None
    tr.judge_after_update(5, {"eval_raw_resid_rel": 0.0, "eval_raw_resid_dissatisfaction_rel": 0.06,
                              "train_r8p_sum_rel_max": 0.0})
    assert tr.stop_reason.startswith("S-7")
    tr.stop_reason = None
    tr.judge_after_update(5, {"eval_raw_resid_rel": 0.0, "eval_raw_resid_dissatisfaction_rel": 1e-12,
                              "train_r8p_sum_rel_max": 1e-12})
    assert tr.stop_reason is None


# ---- UT-34（エミュレータを使わない部分） ----

def test_ut34_each_slot_reads_its_own_minute_output():
    """UT-34: 組ごとの環境は自分の複製の data/ を読む（7 組で互いに異なる）。7 組同時の再生は test_v64_live.py。"""
    from rlkit.env import SummerEnv
    from rlkit.parallel import slots
    dirs = []
    for k in range(7):
        slot = slots.make_slot(k)
        env = SummerEnv(include_indoor_unit_power=True, zone_step_energy_norm_gj=0.02, slot=slot)
        d = env.minute_output_dir()
        assert d == slot.data_dir and os.path.normcase(slot.replica_dir) in os.path.normcase(d)
        dirs.append(os.path.normcase(os.path.abspath(d)))
    assert len(set(dirs)) == 7
