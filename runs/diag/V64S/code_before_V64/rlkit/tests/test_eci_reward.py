"""ECI・基準点・報酬・比率法の試験（v2 第 2-2 節、第 4-2 節、第 5-1 節。指示書 V2-S 第 2-2 節・B-1）。"""
import math

import numpy as np
import pytest

from rlkit import constants, eci, reward


# 現行 current_rl_spec.md 第 1-1 節の夏季基準点（α・β は同書の値）
E0, DR0 = 4.587348191951224, 0.18481666361514731


def test_slopes_match_current_reference_point():
    alpha, beta = eci.slopes(E0, DR0)
    assert alpha == pytest.approx(0.08703358883799721, rel=1e-9)
    assert beta == pytest.approx(7.026245066901605, rel=1e-9)
    assert eci.eci_smooth(E0, DR0) == pytest.approx(0.6494687218297681, rel=1e-12)


@pytest.mark.parametrize("E,DR,PE,PDR,ECI,alpha,beta", [
    (9.18, 0.14694, 0.033, 0.961, 0.179, 0.098, 0.37),   # 夏 現行 RL 最良
    (4.334, 0.1708, 0.746, 0.776, 0.761, 0.089, 5.54),   # 夏 Youworks
])
def test_table_4_2_summer(E, DR, PE, PDR, ECI, alpha, beta):
    assert eci.percentile_energy(E) == pytest.approx(PE, abs=1.5e-3)
    assert eci.percentile_dissatisfaction(DR) == pytest.approx(PDR, abs=1.5e-3)
    assert eci.eci_smooth(E, DR) == pytest.approx(ECI, abs=1.5e-3)
    a, b = eci.slopes(E, DR)
    assert a == pytest.approx(alpha, abs=1.5e-3)
    assert b == pytest.approx(beta, rel=1.5e-2)


def test_zero_floor_and_smooth():
    assert eci.eci(7.3, 0.2) == 0.0
    assert eci.eci(7.1, 0.23) == 0.0
    assert eci.eci(7.1, 0.2) > 0.0
    assert eci.eci_smooth(7.3, 0.2) > 0.0


def test_slopes_by_finite_difference():
    E, DR, h = 6.0, 0.2, 1e-6
    a, b = eci.slopes(E, DR)
    assert a == pytest.approx(-(eci.eci_smooth(E + h, DR) - eci.eci_smooth(E - h, DR)) / (2 * h), rel=1e-5)
    assert b == pytest.approx(-(eci.eci_smooth(E, DR + h) - eci.eci_smooth(E, DR - h)) / (2 * h), rel=1e-5)


def test_slopes_defined_for_large_energy():
    a, b = eci.slopes(30.0, 0.15)  # 1 − Φ の桁落ちで 0 にならない
    assert a > 0 and b > 0


def _r5(E, DR):
    s = eci.r5_slopes(E, DR)
    return s["alpha"], s["beta"]


def test_reference_point_roundtrip():
    """基準点の保存と復元。傾きは指数移動平均で動いた先の点で取り直す（R-2''・R-3''）。
    V6-S で差し替え: 傾きは eci.slopes（eci_smooth）ではなく R-5 の J の勾配（eci.r5_slopes）。"""
    ref = reward.ReferencePoint(E0, DR0)
    rho_old = ref.rho
    entry = ref.update(9.18, 0.14694, update_index=3)
    E_ema, DR_ema = 0.5 * (E0 + 9.18), 0.5 * (DR0 + 0.14694)
    a_new, b_new = _r5(E_ema, DR_ema)
    assert ref.E_star == pytest.approx(E_ema) and ref.DR_star == pytest.approx(DR_ema)
    assert entry["rho_raw"] == pytest.approx(b_new / a_new)
    assert ref.alpha == pytest.approx(a_new)
    assert ref.beta == pytest.approx(ref.rho * a_new)
    assert rho_old / 2.0 - 1e-12 <= ref.rho <= rho_old * 2.0 + 1e-12
    ref2 = reward.ReferencePoint.from_dict(ref.to_dict())
    assert (ref2.E_star, ref2.DR_star, ref2.alpha, ref2.beta) == (ref.E_star, ref.DR_star, ref.alpha, ref.beta)
    assert len(ref2.history) == 2


def test_reference_point_r2_r3_double_prime():
    """B-1: R-2''・R-3''。更新 1 は訓練平均そのもの（制限なし）、以後は 0.5 の指数移動平均で常に 2 倍制限。
    V6-S で差し替え: 傾きは R-5（eci.r5_slopes）。"""
    ref = reward.ReferencePoint(E0, DR0)
    rho0 = ref.rho
    e1 = ref.update(5.0, 0.40, update_index=1, first=True)
    a1, b1 = _r5(5.0, 0.40)
    assert (ref.E_star, ref.DR_star) == (5.0, 0.40)          # 平均がそのまま基準点になる
    assert e1["rho_clipped"] is False and e1["first"] is True
    assert ref.alpha == pytest.approx(a1) and ref.beta == pytest.approx(b1)   # 更新 1 は制限なし
    assert e1["E_bar"] == 5.0 and e1["DR_bar"] == 0.40
    rho1 = ref.rho
    e2 = ref.update(7.0, 0.20, update_index=2)
    assert ref.E_star == pytest.approx(0.5 * 5.0 + 0.5 * 7.0)     # 指数移動平均（係数 0.5）
    assert ref.DR_star == pytest.approx(0.5 * 0.40 + 0.5 * 0.20)
    a2, _ = _r5(ref.E_star, ref.DR_star)
    assert ref.alpha == pytest.approx(a2)                          # α は新しい基準点の勾配
    assert ref.beta == pytest.approx(ref.rho * a2)
    assert rho1 / 2.0 - 1e-12 <= ref.rho <= 2.0 * rho1 + 1e-12     # 2 倍制限は内外を問わず常に効く
    assert e2["first"] is False and rho0 != ref.rho
    # 制限が実際に働く場合（傾きが 2 倍を超えて動く平均を与える）
    ref2 = reward.ReferencePoint(E0, DR0)
    ref2.update(1.2095, 0.7971, update_index=1, first=True)
    before = ref2.rho
    entry = ref2.update(7.0, 0.18, update_index=2)
    assert entry["rho_clipped"] is True
    assert ref2.rho == pytest.approx(before / constants.REFERENCE_RATIO_MAX_CHANGE)


def test_reward_scale_and_identity():
    ref = reward.ReferencePoint(E0, DR0)
    calc = reward.RewardCalculator(ref, T=378, W=159497.0)
    # 基準点の週間総量をそのまま消費すると報酬の総和は −T
    steps = 378
    total = sum(calc.step_reward(E0 / steps, DR0 / steps) for _ in range(steps))
    assert total == pytest.approx(-378.0, rel=1e-12)
    rng = np.random.default_rng(0)
    e = rng.uniform(0, 0.01, 18)
    n = rng.uniform(0, 0.001, 18)
    dE = e.sum() + 0.02
    r_t = calc.step_reward(dE, n.sum())
    assert r_t == pytest.approx(calc.zone_rewards(e, n).sum() + calc.system_reward(dE, e), abs=1e-12)


def _split_case(delta_E, include_indoor=False, step_seconds=3600):
    """系統1: 2 ゾーン運転（負荷 1:3）、系統2: 1 ゾーン運転（P_ou < a_s）、系統3: 運転無し、系統4: 2 ゾーン運転（負荷 0）。"""
    p_ou = [3.0, 0.5, 2.0, 1.0]
    p_hex = np.full(18, 0.01)
    load = np.zeros(18)
    on = np.zeros(18, dtype=bool)
    on[[0, 1]] = True
    load[[0, 1]] = [1.0, -3.0]
    on[5] = True
    load[5] = 2.0
    on[[14, 15]] = True
    p_iu = np.full(18, 0.05) if include_indoor else None
    return reward.zone_energy_split(delta_E, p_ou, p_iu, p_hex, load, on, step_seconds), p_ou


def test_share_rule_unchanged():
    out, p_ou = _split_case(1.0)
    share = out["outdoor_share_kw"]
    a1 = constants.OUTDOOR_UNIT_A_KW[0]
    assert share[0] == pytest.approx(a1 / 2 + (3.0 - a1) * 0.25)
    assert share[1] == pytest.approx(a1 / 2 + (3.0 - a1) * 0.75)
    assert share[5] == pytest.approx(0.5)                      # P_ou < a_s のときは全額が固定成分
    assert share[14] == pytest.approx(0.5) and share[15] == pytest.approx(0.5)  # 負荷 0 は等分
    assert out["standby_kw"].tolist() == [0.0, 0.0, 2.0, 0.0]  # 運転ゾーンの無い系統だけが待機電力
    assert out["q_header_kw"] == pytest.approx(2.0)
    assert share.sum() + out["standby_kw"].sum() == pytest.approx(sum(p_ou))


@pytest.mark.parametrize("include_indoor", [False, True])
def test_ratio_method_identity(include_indoor):
    """B-1: Σ_k e_k + e_hdr = ΔE（比率法の構成上の恒等式）。"""
    delta_E = 0.0123
    out, _ = _split_case(delta_E, include_indoor=include_indoor)
    assert out["e_gj"].sum() + out["e_header_gj"] == pytest.approx(delta_E, abs=1e-12)
    assert out["Q_kw"] == pytest.approx(out["q_kw"].sum() + out["q_header_kw"])
    # 比率は q_k / Q に一致する
    assert out["e_gj"][0] == pytest.approx(delta_E * out["q_kw"][0] / out["Q_kw"])
    if include_indoor:
        assert out["q_kw"][2] == pytest.approx(0.01 + 0.05)  # 停止ゾーンは全熱交換器と室内機のみ
    else:
        assert out["q_kw"][2] == pytest.approx(0.01)


def test_ratio_method_zero_power_and_raw_resid():
    """B-1: Q = 0 のとき e_k = 0、e_hdr = ΔE。raw_resid は瞬時値による見積もりとの差。"""
    delta_E = 0.004
    out = reward.zone_energy_split(delta_E, [0.0] * 4, None, np.zeros(18), np.zeros(18),
                                   np.zeros(18, dtype=bool), 3600)
    assert out["Q_kw"] == 0.0
    assert out["e_gj"].tolist() == [0.0] * 18
    assert out["e_header_gj"] == pytest.approx(delta_E)
    assert out["raw_resid_gj"] == pytest.approx(delta_E)
    out2, _ = _split_case(0.02, step_seconds=900)
    expected = 0.02 - out2["Q_kw"] * 0.25 * constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0
    assert out2["raw_resid_gj"] == pytest.approx(expected)


def test_dr_combination():
    assert reward.dr_co2(1050.0) == pytest.approx(0.5)
    assert reward.combine_dr_components(0.1, 0.2, 0.0, 0.0) == pytest.approx(1 - 0.9 * 0.8)
    assert reward.dissatisfaction_increment(4, 0.25, 900, 159497.0) == pytest.approx(4 * 0.25 * 15 / 159497.0)
