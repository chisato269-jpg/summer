"""指示書 V6.1-S の単体試験（UT-17〜UT-24。UT-25 は test_v6_live.py）。

V6.4-S で差し替え: UT-19 の「評価点が変わらなければ比も変えない」を C-4（毎更新 ρ_raw へ近づける）へ、UT-22 の観測の次元（642 → 640、C-5）。
"""
import datetime
import json
import math

import numpy as np
import pytest
import torch as th

from rlkit import action, config, constants, eci, exogenous, learner, observation, policy
from rlkit import train_summer as ts
from rlkit.mockenv import MOCK_W
from rlkit.reward import R7Reference, RewardCalculator, reward_context

# 第 3-1 節の参照値: (E, DR, ECI, ρ_raw, 制限後)
R7_TABLE = [
    (6.530, 0.1900, 0.4066, 41.02, 41.02),
    (5.762, 0.1917, 0.4860, 55.10, 55.10),
    (4.334, 0.1708, 0.7612, 62.36, 62.36),
    (6.883, 0.2816, 0.0632, 70.62, 70.62),
    (5.450, 0.2256, 0.3037, 93.37, 93.37),
    (1.644, 0.6210, 0.0001, 1575.69, 200.0),
]


def _trial_config(tmp_path, **overrides):
    cfg = config.load()
    cfg.update({"include_indoor_unit_power": False, "zone_step_energy_norm_gj": 0.01,
                "planned_updates": 10, "run_dir": str(tmp_path)})
    cfg.update(overrides)
    return cfg


def _read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


# ---- UT-17 ----

@pytest.mark.parametrize("E,DR,eci_ref,rho_raw,rho_clip", R7_TABLE)
def test_ut17_r7_ratio_reference_table(E, DR, eci_ref, rho_raw, rho_clip):
    """ρ_raw が有限差分による独立計算と ±0.5% で一致し、[30, 200] の制限後も表と一致する。"""
    h = 1e-6
    dE = -(eci.eci_smooth(E + h, DR) - eci.eci_smooth(E - h, DR)) / (2 * h)
    dD = -(eci.eci_smooth(E, DR + h) - eci.eci_smooth(E, DR - h)) / (2 * h)
    fd = dD / dE
    got = R7Reference.raw_ratio(E, DR)
    assert got == pytest.approx(fd, rel=constants.R7_TEST_TOLERANCE)
    assert got == pytest.approx(rho_raw, rel=constants.R7_TEST_TOLERANCE)
    assert eci.eci_smooth(E, DR) == pytest.approx(eci_ref, abs=6e-5)
    clipped, _ = R7Reference.clip_ratio(got, rho_prev=rho_clip)      # 2 倍の制限に掛からない前回値で確かめる
    assert clipped == pytest.approx(rho_clip, rel=constants.R7_TEST_TOLERANCE)


# ---- UT-18 ----

def test_ut18_ranking_by_eci_not_zero_point():
    """評価点と収穫の順位は ECI の大きい順。零点の外側でも ECI が高い軌跡は、内側で ECI が低い軌跡より上位。"""
    outside = {"E": 7.30, "DR": 0.190, "eci_official": 0.0}
    inside = {"E": 7.00, "DR": 0.2215, "eci_official": 0.2164}
    assert eci.eci(7.30, 0.190) == 0.0 and eci.eci(7.00, 0.2215) > 0      # 公式 ECI では内側が上
    assert ts.harvest_key(outside) > ts.harvest_key(inside)               # ECI では外側（0.315）が上
    assert ts.harvest_key(outside) == (pytest.approx(eci.eci_smooth(7.30, 0.190)),)


def test_ut18_trainer_buffer_and_best_use_eci(tmp_path):
    tr = ts.Trainer(_trial_config(tmp_path), str(tmp_path), factory="mock")
    rng = np.random.default_rng(0)
    pts = [(7.30, 0.190), (7.00, 0.2215), (6.0, 0.2219), (5.5, 0.25), (8.0, 0.2), (6.5, 0.19), (7.1, 0.23),
           (6.2, 0.20), (9.0, 0.3), (5.8, 0.195)]
    for u, (E, DR) in enumerate(pts, start=1):
        tr.add_to_sil_buffer({"update": u, "E": E, "DR": DR, "ECI": eci.eci_smooth(E, DR),
                              "eci_official": eci.eci(E, DR)})
    eci_list = [e["ECI"] for e in tr.sil_buffer]
    assert eci_list == sorted(eci_list, reverse=True) and len(eci_list) == 8
    assert {e["update"] for e in tr.sil_buffer} == set(range(1, 11)) - {9, 7}   # ECI の低い 2 本が落ちる
    assert [e["update"] for e in tr.sil_buffer][:4] == [10, 6, 8, 1]   # 零点の外側の更新 1（ECI 0.315）が内側の更新 3・2 より上


# ---- UT-19 ----

def test_ut19_ratio_limits_and_unchanged_reference():
    r = R7Reference()
    assert (r.rho, r.E_ref, r.DR_ref) == (60.0, 6.5, 0.20) and r.denominator() == pytest.approx(6.5 + 60 * 0.20)
    # 30〜200 の制限と 2 倍の制限
    e = r.update_after_harvest({"update": 1, "E": 1.644, "DR": 0.621, "ECI": 0.0001}, 1)
    assert e["rho_clip_30_200"] == 200.0 and e["rho"] == 120.0 and e["rho_limited_by_ratio"]
    # C-4（V6.2-S）: 評価点が変わらなくても、比は ρ_raw（制限後 200）へ 2 倍の制限の範囲で近づく（V6.1-S は 120 で据え置き）
    e2 = r.update_after_harvest({"update": 1, "E": 1.644, "DR": 0.621, "ECI": 0.0001}, 2)
    assert not e2["changed"] and r.rho == 200.0
    # 下限 30（ρ_raw 41 が 2 倍の制限で 60 に止まる）
    r2 = R7Reference(rho=120.0)
    e3 = r2.update_after_harvest({"update": 5, "E": 6.53, "DR": 0.19, "ECI": 0.4066}, 5)
    assert e3["rho_raw"] == pytest.approx(41.02, rel=1e-3) and r2.rho == pytest.approx(60.0)
    assert R7Reference.clip_ratio(10.0, 30.0)[0] == 30.0


def test_ut19_update1_uses_rho0_and_initial_denominator(tmp_path):
    tr = ts.Trainer(_trial_config(tmp_path, planned_updates=2), str(tmp_path), factory="mock")
    tr.run()
    ups = _read(tmp_path / "updates.jsonl")
    ru = ups[0]["reference_used"]
    assert (ru["rho"], ru["E_ref"], ru["DR_ref"], ru["ref_update"]) == (60.0, 6.5, 0.20, None)
    # 更新 1 の収穫の後に評価点が更新 1 の収穫の最良になり、その比が更新 2 の採取に使われる
    u1 = ups[0]["reference_update"]
    assert u1["changed"] and u1["ref_update"] == 1 and u1["E_ref"] == pytest.approx(ups[0]["harvest_best"]["E"])
    assert ups[1]["reference_used"]["rho"] == pytest.approx(u1["rho"])
    assert ups[1]["reference_used"]["rho_raw"] == pytest.approx(u1["rho_raw"])


# ---- UT-20 ----

def test_ut20_step_reward_formula_and_identity():
    r = R7Reference(rho=55.0, E_ref=5.76, DR_ref=0.19, ref_update=3, ref_eci=0.48)
    T, W = 378, MOCK_W
    calc = RewardCalculator(r, T, W)
    rng = np.random.default_rng(1)
    for _ in range(10):
        dE, e = rng.uniform(0, 0.05), rng.uniform(0, 0.002, 18)
        nz = rng.uniform(0, 1e-4, 18)
        n = nz.sum()
        expected = -T * (dE + 55.0 * n) / (5.76 + 55.0 * 0.19)
        assert calc.step_reward(dE, n) == pytest.approx(expected, rel=1e-12)
        total = calc.zone_rewards(e, nz).sum() + calc.system_reward(dE, e)
        assert total == pytest.approx(expected, rel=1e-10)            # r_t = Σ_k r_k,t + r_sys,t


# ---- UT-21 ----

def _rows(n, C_pairs, secs=450.0, rho=60.0):
    rows = []
    for u in range(1, n + 1):
        E, DR = C_pairs[u - 1]
        row = {k: 0 for k in ts.REQUIRED_UPDATE_KEYS}
        row.update({"update": u, "harvest_update_best": {"eci_smooth": 0.1 + 0.001 * u}, "kl_early_stop": False,
                    "update_seconds": secs, "train_E_list": [E] * 12, "train_DR_list": [DR] * 12,
                    "reference_used": {"rho": rho}, "eval_raw_resid_rel": 0.0})
        rows.append(row)
    return rows


def test_ut21_h4pp_h5_h9():
    good = _rows(20, [(7.0 - 0.02 * u, 0.25) for u in range(1, 21)])
    h = ts.h_checks(good, ts.REQUIRED_UPDATE_KEYS)
    assert h["H-4''"]["passed"] and h["all_passed"] and "H-7" not in h
    assert h["H-4''"]["C_u"][0] == pytest.approx(6.98 + 60 * 0.25)
    bad = _rows(20, [(6.0 + 0.02 * u, 0.25) for u in range(1, 21)])
    assert not ts.h_checks(bad, ts.REQUIRED_UPDATE_KEYS)["H-4''"]["passed"]
    # ρ₂₀ は更新 20 の採取で使った比
    rows = _rows(20, [(7.0, 0.30 - 0.001 * u) for u in range(1, 21)])
    rows[-1]["reference_used"]["rho"] = 150.0
    assert ts.h_checks(rows, ts.REQUIRED_UPDATE_KEYS)["H-4''"]["rho20"] == 150.0
    # H-5 の閾値 520 秒
    assert constants.H5_UPDATE_SECONDS_MAX == 520.0
    assert ts.h_checks(_rows(20, [(7 - 0.02 * u, 0.25) for u in range(1, 21)], secs=519.0),
                       ts.REQUIRED_UPDATE_KEYS)["H-5"]["passed"]
    assert not ts.h_checks(_rows(20, [(7 - 0.02 * u, 0.25) for u in range(1, 21)], secs=521.0),
                           ts.REQUIRED_UPDATE_KEYS)["H-5"]["passed"]
    # H-9: 比が 30 または 200 に 10 更新以上続けて張り付いたら報告
    stuck = _rows(20, [(7.0, 0.25)] * 20, rho=200.0)
    assert ts.h9_check(stuck)["report"] and ts.h9_check(stuck)["max_streak"] == 20
    mixed = _rows(12, [(7.0, 0.25)] * 12, rho=30.0)
    mixed[5]["reference_used"]["rho"] = 45.0
    assert not ts.h9_check(mixed)["report"]


def test_ut21_zero_point_items_not_called(tmp_path, monkeypatch):
    """S-8'・H-7・R-5 の J が呼ばれない（呼ばれたら例外）。S-8' の更新番号を縮めても呼ばれない。"""
    def boom(*a, **k):
        raise AssertionError("零点条件の項目が呼ばれた")
    monkeypatch.setattr(ts, "s8prime_check", boom)
    monkeypatch.setattr(eci, "objective_j", boom)
    monkeypatch.setattr(eci, "r5_slopes", boom)
    monkeypatch.setattr(constants, "S8PRIME_UPDATE", 2)
    monkeypatch.setattr(constants, "H_CHECK_UPDATE", 3)
    tr = ts.Trainer(_trial_config(tmp_path, planned_updates=3), str(tmp_path), factory="mock")
    reason = tr.run()
    assert "計画更新数 3" in reason or reason.startswith("S-3")
    j = json.load(open(tmp_path / "judgment_u3.json", encoding="utf-8"))
    assert "H-7" not in j and "H-4''" in j and "H-9" in j
    ups = _read(tmp_path / "updates.jsonl")
    assert all(u["train_inside_count"] is None for u in ups)
    assert all(u["reference_used"]["sigma_E"] is None and u["reference_used"]["rho_r5"] is None for u in ups)


# ---- UT-22 ----

def _snap(co2_s, co2_n):
    zones = {l: {"drybulb_c": 26.0, "relative_humidity_pct": 50.0, "occupant_number": 1, "thermal_sensation": 0.0,
                 "clothing_index": 0.6, "dissatisfied_thermal": 0.1, "dissatisfied_draft": 0.05,
                 "dissatisfied_vtmp": 0.02, "heatload_kw": -1.0} for l in constants.ZONE_LABELS}
    b = {"drybulb_c": 30.0, "relative_humidity_pct": 60.0, "global_horizontal_radiation": 300.0,
         "nocturnal_radiation": 20.0, "total_energy_consumption_mj": 0.0, "averaged_dissatisfaction_rate": 0.0}
    return {"building": b, "zones": zones, "co2_ppm": {"South": co2_s, "North": co2_n}}


def test_ut22_o6_co2_change():
    ob = observation.ObservationBuilder(exogenous.load(), zone_step_energy_norm_gj=0.02)
    dt = datetime.datetime(1999, 7, 21, 9, 0)
    dec = action.decode_action(action.initial_action(1), 1)
    o0, r0 = ob.build(dt, 0, 1, _snap(800.0, 900.0), dec)
    assert o0.shape == (640,) and r0[42] == 0.0 and r0[43] == 0.0          # 最初のステップは 0
    o1, r1 = ob.build(dt, 1, 1, _snap(1100.0, 700.0), dec)
    assert r1[42] == pytest.approx(300 / 500) and r1[43] == pytest.approx(-200 / 500)
    assert o1[42] == pytest.approx(0.6) and o1[43] == pytest.approx(-0.4)   # 負の値も残る（−1〜1）
    assert r1[40] == pytest.approx(1100 / 4000)                             # 現在値は 40・41 のまま
    o2, r2 = ob.build(dt, 2, 1, _snap(1900.0, 100.0), dec)
    assert r2[42] == pytest.approx(1.6) and o2[42] == 1.0 and o2[43] == -1.0   # −1〜1 の制限
    clipped = observation.clipped_entries(r2)
    assert clipped[42] and clipped[43] and not observation.clipped_entries(r1)[[42, 43]].any()
    ob.reset()
    assert ob.build(dt, 0, 1, _snap(500.0, 500.0), dec)[1][42] == 0.0      # エピソードの切れ目で前回値を消す
    assert observation.OBS_LOW[42] == -1.0 and observation.OBS_LOW[41] == 0.0 and constants.OBSERVATION_DIM == 640


# ---- UT-23 ----

def test_ut23_value_context_from_r7(tmp_path):
    r = R7Reference(rho=55.0, E_ref=5.76, DR_ref=0.19, ref_update=3)
    rho_n, D_n = r.value_context()
    assert rho_n == pytest.approx(math.log10(55.0) / 4) and D_n == pytest.approx(math.log10(5.76 + 55 * 0.19) / 3)
    assert (rho_n, D_n) == pytest.approx(reward_context(55.0, 5.76, 0.19))
    tr = ts.Trainer(_trial_config(tmp_path, planned_updates=2), str(tmp_path), factory="mock")
    tr.run()
    last = _read(tmp_path / "updates.jsonl")[-1]["reference_used"]
    ctx = tr.model.policy.value_network.reward_context.tolist()
    assert ctx == pytest.approx([last["rho_n"], last["D_n"]], rel=1e-6)    # その更新の採取で使った比から
    assert last["rho_n"] == pytest.approx(math.log10(last["rho"]) / 4)


# ---- UT-24 ----

def test_ut24_sil_returns_use_r7():
    r = R7Reference(rho=70.0, E_ref=6.0, DR_ref=0.22, ref_update=4)
    rng = np.random.default_rng(2)
    T = 378
    e = {"W": MOCK_W, "delta_E": rng.uniform(0, 0.02, T), "zone_energy": rng.uniform(0, 0.001, (T, 18)),
         "zone_n": rng.uniform(0, 1e-5, (T, 18))}
    R = learner.sil_returns(e, r, T)
    step = -T * (e["delta_E"] + 70.0 * e["zone_n"].sum(axis=1)) / (6.0 + 70.0 * 0.22)
    assert R[0].sum() == pytest.approx(step.sum(), rel=1e-10)
    assert R[200].sum() == pytest.approx(step[200:].sum(), rel=1e-10)
    # 比か評価点が変われば計算し直した収益も変わる
    assert not np.allclose(R, learner.sil_returns(e, R7Reference(rho=140.0, E_ref=6.0, DR_ref=0.22), T))
