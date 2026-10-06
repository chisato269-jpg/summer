"""指示書 V6-S の単体試験のうち、報酬（R-5）と収穫の順位（L-4）に関するもの（UT-5・UT-10）。"""
import math

import numpy as np
import pytest

from rlkit import constants, eci
from rlkit.reward import ReferencePoint, RewardCalculator


# 第 3-3 節の参照値（2 倍制限を掛ける前の β/α）: (E*, DR*, eci_smooth だけの β/α, R-5 の β/α)
R5_TABLE = [
    (6.53, 0.190, 41.02, 39.40),
    (4.50, 0.180, 75.20, 75.21),
    (5.00, 0.215, 103.93, 281.48),
    (5.72, 0.241, 92.04, 1587.89),
]


@pytest.mark.parametrize("E,DR,rho_smooth,rho_r5", R5_TABLE)
def test_ut5_r5_ratio_reference_table(E, DR, rho_smooth, rho_r5):
    s = eci.r5_slopes(E, DR)
    assert s["rho_smooth"] == pytest.approx(rho_smooth, rel=constants.R5_TEST_TOLERANCE)
    assert s["rho_r5"] == pytest.approx(rho_r5, rel=constants.R5_TEST_TOLERANCE)


def _bisect(f, lo, hi, n=200):
    flo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (flo > 0):
            lo, flo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_ut5_zero_boundary_independent_recalculation():
    # 逆関数を使わず、P(x) = 0.2 を二分法で独立に解く
    E_z = _bisect(lambda x: eci.percentile_energy(x) - 0.2, 1.0, 20.0)
    DR_z = _bisect(lambda x: eci.percentile_dissatisfaction(x) - 0.2, 0.05, 0.6)
    assert round(E_z, 4) == constants.R5_E_Z_GJ_LISTED
    assert round(DR_z, 4) == constants.R5_DR_Z_LISTED
    assert eci.E_ZERO_GJ == pytest.approx(E_z, abs=1e-9)
    assert eci.DR_ZERO == pytest.approx(DR_z, abs=1e-9)


def test_ut5_r5_gradient_matches_finite_difference_of_j():
    for E, DR in [(6.53, 0.190), (5.0, 0.215), (7.3, 0.20)]:
        s = eci.r5_slopes(E, DR)
        h = 1e-6
        dJ_dE = (eci.objective_j(E + h, DR) - eci.objective_j(E - h, DR)) / (2 * h)
        dJ_dD = (eci.objective_j(E, DR + h) - eci.objective_j(E, DR - h)) / (2 * h)
        assert s["alpha"] == pytest.approx(-dJ_dE, rel=1e-5)
        assert s["beta"] == pytest.approx(-dJ_dD, rel=1e-5)


def test_ut5_barrier_negligible_inside():
    s = eci.r5_slopes(4.5, 0.18)
    assert s["sigma_E"] < 1e-10 and s["sigma_D"] < 1e-8
    assert s["rho_r5"] == pytest.approx(s["rho_smooth"], rel=1e-3)


def test_ut5_step_reward_invariant_to_common_scaling():
    ref = ReferencePoint(6.53, 0.190)
    scaled = ReferencePoint(6.53, 0.190, alpha=ref.alpha * 37.5, beta=ref.beta * 37.5)
    T, W = 378, 159497.0
    a, b = RewardCalculator(ref, T, W), RewardCalculator(scaled, T, W)
    rng = np.random.default_rng(0)
    for _ in range(20):
        dE, n = rng.uniform(0, 0.05), rng.uniform(0, 0.002)
        e = rng.uniform(0, 0.002, 18)
        nz = rng.uniform(0, 1e-4, 18)
        assert a.step_reward(dE, n) == pytest.approx(b.step_reward(dE, n), rel=1e-12)
        assert np.allclose(a.zone_rewards(e, nz), b.zone_rewards(e, nz), rtol=1e-12)
        assert a.system_reward(dE, e) == pytest.approx(b.system_reward(dE, e), rel=1e-12)


def test_ut5_reference_point_uses_r5_and_ratio_limit():
    ref = ReferencePoint(6.53, 0.190)
    s = eci.r5_slopes(6.53, 0.190)
    assert ref.alpha == pytest.approx(s["alpha"]) and ref.beta == pytest.approx(s["beta"])
    first = ref.update(6.53, 0.190, update_index=1, first=True)
    assert first["rho"] == pytest.approx(39.40, rel=5e-3)
    # (5.72, 0.241) へ向けて動かすと、R-5 の比は 1,500 を超えるが、2 倍制限で 1 更新あたり 2 倍まで
    entry = ref.update(5.72 * 2 - 6.53, 0.241 * 2 - 0.190, update_index=2)
    assert entry["E_star"] == pytest.approx(5.72) and entry["DR_star"] == pytest.approx(0.241)
    assert entry["rho_raw"] == pytest.approx(1587.89, rel=5e-3)
    assert entry["rho"] == pytest.approx(2.0 * first["rho"])
    assert entry["rho_clipped"] is True
    for key in ("alpha_s", "beta_s", "sigma_E", "sigma_D", "rho_n", "D_n"):
        assert key in entry


def test_ut10_harvest_tie_break_uses_j():
    from rlkit.train_summer import harvest_key
    # 公式 ECI が同点（零点の外側どうしで 0）のとき、J の大きい方（境界に近い方）が上位
    near = {"eci_official": 0.0, "E": 7.25, "DR": 0.20}
    far = {"eci_official": 0.0, "E": 8.50, "DR": 0.20}
    assert eci.eci(7.25, 0.20) == 0.0 and eci.eci(8.5, 0.20) == 0.0
    assert harvest_key(near) > harvest_key(far)
    # eci_smooth の順位と逆になる組でも J に従う
    a = {"eci_official": 0.0, "E": 6.0, "DR": 0.2235}   # DR が境界のわずかに外
    b = {"eci_official": 0.0, "E": 7.4, "DR": 0.2000}   # E が境界の外
    ja, jb = eci.objective_j(6.0, 0.2235), eci.objective_j(7.4, 0.2)
    assert (harvest_key(a) > harvest_key(b)) == (ja > jb)
    # 公式 ECI が違えば公式 ECI が優先
    inside = {"eci_official": 0.1, "E": 7.1, "DR": 0.221}
    assert harvest_key(inside) > harvest_key(near)
