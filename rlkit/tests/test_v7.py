"""指示書 V7-S の単体試験（UT-36〜UT-41。UT-35 はエミュレータを使うため test_v7_live.py）。

引き継ぎの試験は、走行 v2s_trial07 の保存点を読むだけで使う（MUST-2）。
"""
import datetime
import json
import os

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv

from rlkit import action, advantage, co2pred, config, constants, exogenous, learner, observation, policy
from rlkit import train_summer as ts
from rlkit.mockenv import MockSummerEnv
from rlkit.reward import R7Reference, ReferencePoint

PREV_CKPT = os.path.join("runs", "v2s_trial07", "checkpoint_latest.pt")
REF = ReferencePoint(4.587348191951224, 0.18481666361514731).to_dict()


def _prev_model_state():
    return th.load(PREV_CKPT, weights_only=False)["model"]


def _old_and_new_nets():
    m = _prev_model_state()
    po = policy.PolicyNetwork(1, global_dim=policy.V64_GLOBAL_DIM, n_setpoint=len(policy.V64_SETPOINT_GRID))
    vo = policy.ValueNetwork(global_dim=policy.V64_GLOBAL_DIM)
    po.load_state_dict(m["weights"]["policy_net"])
    vo.load_state_dict(m["weights"]["value_net"])
    sp = policy.warm_start_specs()
    pn, vn = policy.PolicyNetwork(1), policy.ValueNetwork()
    pn.load_state_dict(policy.convert_state_dict(m["weights"]["policy_net"], sp["policy"]))
    vn.load_state_dict(policy.convert_state_dict(m["weights"]["value_net"], sp["value"]))
    return m, po, vo, pn, vn


# ---- UT-36 ----

def test_ut36_setpoint_grid_and_inherited_rows():
    """UT-36: 設定温度の格子が 22〜28 ℃の 7 択で、引き継いだ出力が旧出力の 22〜28 ℃の行と一致する。"""
    assert action.setpoint_grid(1) == [22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]
    assert constants.ACTION_DIMS == 60 and constants.ACTION_LOGITS[1] == 256 and sum(action.nvec(1)) == 256
    m, po, vo, pn, vn = _old_and_new_nets()
    W_old, b_old = po.zone_head.weight.detach(), po.zone_head.bias.detach()
    W_new, b_new = pn.zone_head.weight.detach(), pn.zone_head.bias.detach()
    assert W_old.shape == (14, 224) and W_new.shape == (12, 224)
    for j, g in enumerate(action.setpoint_grid(1)):
        i_old = 2 + constants.ZONE_SETPOINT_C_V64.index(g)
        assert th.equal(W_new[2 + j], W_old[i_old]) and b_new[2 + j] == b_old[i_old]
    assert th.equal(W_new[:2], W_old[:2]) and th.equal(W_new[9:], W_old[11:])    # 運転状態・風量はそのまま
    # 観測 +19 の正規化は新しい格子（索引 ÷ 6）
    assert observation._norm_index(action.setpoint_grid(1).index(26.0), 7) == pytest.approx(4 / 6)


# ---- UT-37 ----

def test_ut37_co2_prediction_order_and_formula():
    """UT-37: 推定した式で、水準の順（停止 ≥ Low ≥ Middle ≥ High）に予測値が並ぶ。

    順が成り立つのは室内の CO2 が外気（C_out）以上のとき。室内が外気より低いと換気で室内の濃度が上がるため、順は逆になる
    （推定の C_out は南 400.003 ppm で、室内 400 ppm・無人では 0.001 ppm の差で逆になる。V7S-d）。"""
    for tenant in ("South", "North"):
        c_out = constants.CO2_PRED_PARAMS[tenant]["c_out"]
        for c0 in (c_out, 450.0, 700.0, 950.0, 1500.0):
            for n in (0.0, 1.0, 10.0, 40.0):
                for sec in (900, 3600):
                    p = co2pred.predict_levels(tenant, c0, n, sec)
                    assert p[0] >= p[1] >= p[2] >= p[3], (tenant, c0, n, sec, p)
    # 式そのもの（南、C 800 ppm、N̄ 20 人、900 秒、Middle）
    pr = constants.CO2_PRED_PARAMS["South"]
    a, h = pr["a"][2], 0.25
    expect = pr["c_out"] + (800 - pr["c_out"]) * np.exp(-a * h) + pr["b"] * 20 / a * (1 - np.exp(-a * h))
    assert co2pred.predict_levels("South", 800.0, 20.0, 900)[2] == pytest.approx(expect, rel=1e-12)
    assert co2pred.obs_values([700.0, 900.0, 1000.0, 1200.0]).tolist() == pytest.approx([-1.0, 0.0, 0.5, 1.0])


def _snap(co2_s, co2_n):
    zones = {l: {"drybulb_c": 26.0, "relative_humidity_pct": 50.0, "occupant_number": 2, "thermal_sensation": 0.5,
                 "clothing_index": 0.6, "dissatisfied_thermal": 0.1, "dissatisfied_draft": 0.05,
                 "dissatisfied_vtmp": 0.02, "heatload_kw": -1.0} for l in constants.ZONE_LABELS}
    b = {"drybulb_c": 30.0, "relative_humidity_pct": 60.0, "global_horizontal_radiation": 300.0,
         "nocturnal_radiation": 20.0, "total_energy_consumption_mj": 0.0, "averaged_dissatisfaction_rate": 0.0}
    return {"building": b, "zones": zones, "co2_ppm": {"South": co2_s, "North": co2_n}}


def test_ut37_observation_82_89():
    """UT-37: 観測の 82〜89 が式どおりに計算される（南の停止・Low・Middle・High、北の同じ 4 つ）。"""
    exo = exogenous.load()
    ob = observation.ObservationBuilder(exo, zone_step_energy_norm_gj=0.02)
    dt = datetime.datetime(1999, 7, 22, 10, 0)            # 平日の在室時間帯（900 秒のステップ）
    dec = action.decode_action(action.initial_action(1), 1)
    obs, raw = ob.build(dt, 40, 1, _snap(850.0, 620.0), dec)
    assert constants.OBSERVATION_DIM == 648 and obs.shape == (648,) and observation.IDX_CO2_PRED == 82
    for ti, (tenant, c0) in enumerate((("South", 850.0), ("North", 620.0))):
        n_bar = co2pred.tenant_mean_occupants(exo, dt, 900, tenant)
        assert n_bar > 0
        exp_raw = (co2pred.predict_levels(tenant, c0, n_bar, 900) - 900.0) / 200.0
        assert raw[82 + 4 * ti:86 + 4 * ti] == pytest.approx(exp_raw, rel=1e-12)
        assert obs[82 + 4 * ti:86 + 4 * ti] == pytest.approx(np.clip(exp_raw, -1, 1), abs=1e-6)
    assert (observation.OBS_LOW[82:90] == -1.0).all() and (observation.OBS_HIGH[82:90] == 1.0).all()
    assert observation.zone_base(0) == 90


def test_ut37_new_inputs_zero_and_outputs_unchanged():
    """UT-37: 新しい入力の重みが 0 で、引き継いだ直後の方策（と価値）の出力が旧方策と一致する。"""
    m, po, vo, pn, vn = _old_and_new_nets()
    for net in (pn, vn):
        W = net.global_encoder[0].weight.detach()
        assert W.shape == (96, 90) and th.count_nonzero(W[:, 82:]) == 0
    g = th.Generator().manual_seed(1)
    old = th.rand(32, 640, generator=g)
    new = th.cat([old[:, :82], th.rand(32, 8, generator=g) * 2 - 1, old[:, 82:]], dim=1)
    lo, ln = po(old).detach(), pn(new).detach()
    B = old.shape[0]
    sp_old = lo[:, 36:36 + 162].reshape(B, 18, 9)[:, :, 2:].reshape(B, -1)
    assert th.equal(th.cat([lo[:, :36], sp_old, lo[:, 198:]], dim=1), ln)
    vo.set_reward_context(0.3, 0.4)
    vn.set_reward_context(0.3, 0.4)
    assert th.equal(vo(old).detach(), vn(new).detach())
    assert policy.parameter_count(pn) == 75928 and policy.parameter_count(vn) == 71138


def test_c8_warm_start_optimizer_and_learning_rate():
    """C-8: 最適化器の状態も同じ写し方で引き継ぎ、方策学習率を引き継ぐ。更新番号は 0 から数え直す。"""
    vec = DummyVecEnv([lambda: MockSummerEnv(seed=0)])
    vec.env_method("set_reference", REF)
    model = learner.SummerPPO(vec, 1, planned_updates=10, seed=constants.RANDOM_SEED, n_steps_per_env=756)
    rec = model.load_warm_start(PREV_CKPT)
    m = _prev_model_state()
    assert rec["source_update_index"] == 388 and model.policy_lr == pytest.approx(m["policy_lr"])
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(m["policy_lr"])
    assert model.update_index == 0 and model.c_sil == 0.0
    names = [n for n, _ in model.policy.policy_network.named_parameters()]
    st = model.policy.optimizer.state_dict()["state"]
    old_st = m["policy_optimizer"]["state"]
    i = names.index("zone_head.weight")
    assert st[i]["exp_avg"].shape == (12, 224)
    assert th.equal(st[i]["exp_avg"], old_st[i]["exp_avg"][policy.warm_start_row_keep()])
    j = names.index("global_encoder.0.weight")
    assert th.equal(st[j]["exp_avg_sq"][:, :82], old_st[j]["exp_avg_sq"]) and th.count_nonzero(st[j]["exp_avg_sq"][:, 82:]) == 0
    assert float(st[i]["step"]) == float(old_st[i]["step"])


# ---- UT-38 ----

def test_ut38_sil_disabled():
    """UT-38: 自己模倣学習の損失が 0 で、勾配に寄与しない。学習の入口は SIL を適用しない。"""
    assert constants.SIL_ENABLED is False
    lp = th.randn(4, 60, requires_grad=True)
    loss = learner.sil_policy_loss(lp, th.rand(4, 60), 1.0, th.ones(4, 60, dtype=th.bool), 0.0)
    loss.backward()
    assert loss.item() == 0.0 and th.count_nonzero(lp.grad) == 0
    vec = DummyVecEnv([lambda: MockSummerEnv(seed=0)])
    vec.env_method("set_reference", REF)
    model = learner.SummerPPO(vec, 1, planned_updates=10, seed=constants.RANDOM_SEED, n_steps_per_env=756)
    assert model.c_sil == 0.0
    model.collect_one()
    buf = [{"update": 1, "E": 5.0, "DR": 0.2, "eci_official": 0.3, "eci_smooth": 0.3, "ECI": 0.3, "J": None,
            "W": 159497.0, "actions": model._rollout["actions"][:378].astype(np.int64),
            "obs": model._rollout["observations"][:378].astype(np.float32),
            "masks": model._rollout["action_masks"][:378].copy(), "step_seconds": np.full(378, 900.0),
            "delta_E": np.full(378, 0.01), "zone_energy": np.full((378, 18), 0.0005), "zone_n": np.full((378, 18), 1e-6)}]
    before = {k: v.clone() for k, v in model.policy.policy_network.state_dict().items()}
    ref = ReferencePoint.from_dict(REF)
    stats = {lv: {"mean": 0.0, "std": 1.0, "n": 1} for lv in dict.fromkeys(advantage.dim_levels())}
    obs = th.as_tensor(model._rollout["observations"]).float()
    acts = th.as_tensor(model._rollout["actions"]).long()
    masks = th.as_tensor(model._rollout["action_masks"]).bool()
    rec = model._sil(buf, "apply", ref, stats, obs, acts, masks, th.ones(acts.shape[0], 60, dtype=th.bool))
    assert "approx_kl_sil" not in rec
    after = model.policy.policy_network.state_dict()
    assert all(th.equal(before[k], after[k]) for k in before)
    tr = ts.Trainer({"planned_updates": 1500}, ".", factory="mock")
    tr.sil_enabled = True
    assert tr.sil_mode(301) == "stats" and tr.sil_mode(1200) == "stats"


# ---- UT-39 ----

def test_ut39_lambda_per_step():
    """UT-39: λ の設定が 900 秒・3,600 秒のステップで λ^(Δt/3600) として使われる（3-5 で 0.99 のまま）。"""
    assert constants.LAMBDA_HOUR == 0.99
    lam = advantage.lambda_per_step([900, 3600])
    assert lam[0] == pytest.approx(0.99 ** 0.25) and lam[1] == pytest.approx(0.99)


# ---- UT-40 ----

def test_ut40_collapse_watch_rule():
    """UT-40: 崩れの見張りの規則（模擬の記録）。"""
    hist = [0.60] * 20 + [0.62] * 20 + [0.54] * 20
    r = ts.collapse_check(hist, 60)
    assert r["best_previous"] == pytest.approx(0.62) and r["median_recent"] == pytest.approx(0.54)
    assert r["threshold"] == pytest.approx(0.85 * 0.62) and not r["triggered"]         # 0.54 ≥ 0.527
    hist2 = [0.60] * 20 + [0.62] * 20 + [0.52] * 20
    assert ts.collapse_check(hist2, 60)["triggered"]
    assert not ts.collapse_check([0.6] * 40, 40)["triggered"]
    assert ts.collapse_check([0.6] * 20 + [0.3] * 20, 40)["triggered"]


def test_ut40_trainer_checkpoints_and_collapse_stop(tmp_path, monkeypatch):
    """UT-40: 保存点が規則どおりに残り、崩れの見張りに当たれば止まる（番号を縮めて試す）。"""
    monkeypatch.setattr(constants, "CHECKPOINT_EVERY", 2)
    monkeypatch.setattr(constants, "COLLAPSE_WATCH_START", 2)
    monkeypatch.setattr(constants, "COLLAPSE_WATCH_EVERY", 2)
    calls = []

    def fake(hist, u):
        calls.append(u)
        return {"update": u, "triggered": u >= 4, "median_recent": 0.1, "best_previous": 0.5, "threshold": 0.425,
                "block_medians_before": []}
    monkeypatch.setattr(ts, "collapse_check", fake)
    cfg = config.load()
    cfg.update({"include_indoor_unit_power": False, "zone_step_energy_norm_gj": 0.01, "planned_updates": 10,
                "run_dir": str(tmp_path)})
    tr = ts.Trainer(cfg, str(tmp_path), factory="mock")
    reason = tr.run()
    assert reason.startswith("崩れの見張り") and tr.model.update_index == 4 and calls == [2, 4]
    assert os.path.exists(tmp_path / "checkpoint_u2.pt") and os.path.exists(tmp_path / "checkpoint_u4.pt")
    assert not os.path.exists(tmp_path / "checkpoint_u3.pt")
    assert len([l for l in open(tmp_path / "collapse_watch.jsonl", encoding="utf-8")]) == 2


def test_u20_checks_v7():
    """第 9 章の更新 20 の判定。"""
    def rows(smooth, cut=0, secs=430.0):
        out = []
        for u in range(1, 21):
            r = {k: 0 for k in ts.REQUIRED_UPDATE_KEYS}
            r.update({"update": u, "harvest_update_best": {"eci_smooth": smooth[u - 1]}, "kl_early_stop": u > 20 - cut,
                      "update_seconds": secs, "reference_used": {"rho": 75.0}})
            out.append(r)
        return out
    ok = ts.u20_checks(rows([0.58] * 10 + [0.575] * 10), ts.REQUIRED_UPDATE_KEYS)
    assert ok["all_passed"]
    assert not ts.u20_checks(rows([0.58] * 10 + [0.565] * 10), ts.REQUIRED_UPDATE_KEYS)["harvest_median_kept"]["passed"]
    assert not ts.u20_checks(rows([0.45] * 20), ts.REQUIRED_UPDATE_KEYS)["harvest_max"]["passed"]
    assert not ts.u20_checks(rows([0.58] * 20, cut=8), ts.REQUIRED_UPDATE_KEYS)["kl_cutoff"]["passed"]
    assert ts.u20_checks(rows([0.58] * 20, cut=7), ts.REQUIRED_UPDATE_KEYS)["kl_cutoff"]["passed"]
    assert not ts.u20_checks(rows([0.58] * 20, secs=530.0), ts.REQUIRED_UPDATE_KEYS)["time"]["passed"]


def test_c7_reference_start_and_c8_trainer_warm_start(tmp_path):
    """C-7: 更新 1 の比 75.06、分母 4.850・0.1876。C-8: 学習の入口が保存点から引き継ぐ。"""
    r = R7Reference()
    assert (r.rho, r.E_ref, r.DR_ref, r.ref_update) == (75.06, 4.850, 0.1876, None)
    cfg = config.load(os.path.join("rlkit", "config_summer_v7.json"))
    assert cfg["run_dir"] == "runs/v2s_trial08" and cfg["warm_start_from"] == "runs/v2s_trial07/checkpoint_latest.pt"
    cfg.update({"include_indoor_unit_power": False, "planned_updates": 2, "run_dir": str(tmp_path)})
    tr = ts.Trainer(cfg, str(tmp_path), factory="mock")
    tr.run()
    ev = [json.loads(l) for l in open(tmp_path / "events.jsonl", encoding="utf-8")]
    ws = [e for e in ev if e.get("event") == "warm_start"]
    assert len(ws) == 1 and ws[0]["source_update_index"] == 388
    ups = [json.loads(l) for l in open(tmp_path / "updates.jsonl", encoding="utf-8")]
    assert ups[0]["reference_used"]["rho"] == 75.06 and ups[0]["reference_used"]["E_ref"] == 4.850
    assert ups[0]["policy_lr_before"] == pytest.approx(_prev_model_state()["policy_lr"])
    assert ups[0]["entropy_c0"] == pytest.approx(0.002) and ups[0]["sil"]["mode"] == "stats"


# ---- UT-41 ----

def test_ut41_influence_not_adopted():
    """UT-41: 影響範囲の割当は 3-4 で両家族とも不合格だったため採らない（割当は建物・テナントのまま）。"""
    d = json.load(open(os.path.join("runs", "diag", "V7S", "influence_decision.json"), encoding="utf-8"))
    assert d["families"]["S"]["chosen"] == "building" and d["families"]["P"]["chosen"] == "building"
    assert constants.CREDIT_LEVELS == {"S": "building", "P": "building", "V": "tenant"}
    with pytest.raises(ValueError):
        advantage.level_zone_advantages(np.zeros((1, 18)), np.zeros(1), np.zeros((1, 60), dtype=int), "influence")
