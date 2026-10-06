"""rlkit.eci — 夏季の評価関数 ECI と、その勾配（v2 第 2-2 節・第 5-1-1 節）。

ECI = sqrt(P_E · P_DR)、P = 1 − Φ((BoxCox(x) − μ)/σ)。公式値は零点条件の外側で 0 とし、
eci_smooth は零点条件を適用しない同じ式とする（v2 第 5-1-1 節、第 5-1-3 節 R-4）。
1 − Φ(z) は 0.5·erfc(z/√2) で計算する。同じ関数だが、E が大きく z が 8 を超える領域で
1 − Φ(z) が桁落ちして 0 になり、傾き α・β が定義できなくなるのを避けるためである。
"""
import math

from rlkit import constants


def box_cox(x, lam):
    """Box-Cox変換。lam==0 のときは対数変換とする。"""
    if lam == 0.0:
        return math.log(x)
    return (x ** lam - 1.0) / lam


def _box_cox_grad(x, lam):
    """d box_cox(x, lam) / dx"""
    if lam == 0.0:
        return 1.0 / x
    return x ** (lam - 1.0)


def _norm_sf(z):
    """1 − Φ(z)"""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _norm_pdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _z_energy(E):
    p = constants.ECI_PARAMS
    return (box_cox(E, p["lamE"]) - p["muE"]) / p["sigE"]


def _z_dissatisfaction(DR):
    p = constants.ECI_PARAMS
    return (box_cox(DR, p["lamDR"]) - p["muDR"]) / p["sigDR"]


def percentile_energy(E):
    """P_E = 1 − Φ((BoxCox(E) − μ_E)/σ_E)"""
    return _norm_sf(_z_energy(E))


def percentile_dissatisfaction(DR):
    """P_DR = 1 − Φ((BoxCox(DR) − μ_DR)/σ_DR)"""
    return _norm_sf(_z_dissatisfaction(DR))


def is_zero_floor(E, DR):
    """零点条件に触れる（公式 ECI が 0 になる）とき真。正常領域は E <= 7.200 かつ DR <= 0.2220。"""
    return (E > constants.ZERO_FLOOR_ENERGY_GJ) or (DR > constants.ZERO_FLOOR_DR)


def eci(E, DR, apply_zero_floor=True):
    """apply_zero_floor=True（既定）が公式 ECI（報告・選抜用）、False が eci_smooth（学習用の傾きの計算用）。"""
    if apply_zero_floor and is_zero_floor(E, DR):
        return 0.0
    return math.sqrt(percentile_energy(E) * percentile_dissatisfaction(DR))


def eci_smooth(E, DR):
    return eci(E, DR, apply_zero_floor=False)


def slopes(E, DR):
    """基準点 (E, DR) における (α, β) = (−∂eci_smooth/∂E, −∂eci_smooth/∂DR)（v2 第 2-2 節）。

    零点条件の外側でも同じ式で定義する（R-4）。eci_smooth が浮動小数点で 0 になる点では
    傾きを定義できないため例外を送出する。
    """
    p = constants.ECI_PARAMS
    zE = _z_energy(E)
    zDR = _z_dissatisfaction(DR)
    PE = _norm_sf(zE)
    PDR = _norm_sf(zDR)
    value = math.sqrt(PE * PDR)
    if value == 0.0:
        raise ValueError(f"eci_smooth が 0 になる点 (E={E}, DR={DR}) では傾き α・β を定義できない")
    dPE_dE = -_norm_pdf(zE) / p["sigE"] * _box_cox_grad(E, p["lamE"])
    dPDR_dDR = -_norm_pdf(zDR) / p["sigDR"] * _box_cox_grad(DR, p["lamDR"])
    alpha = -(dPE_dE * PDR) / (2.0 * value)
    beta = -(dPDR_dDR * PE) / (2.0 * value)
    return alpha, beta


# =====================================================================
# R-5（指示書 V6-S 第 3-3 節、設計理論書 v3 第 4-1 節）: 零点境界の障壁を加えた目的関数 J
#   J = ln ECI_s(E, DR) − κ·[softplus((E − E_z)/τ_E) + softplus((DR − DR_z)/τ_D)]
#   α = −∂J/∂E = α_s/ECI_s + (κ/τ_E)·σ((E − E_z)/τ_E)
#   β = −∂J/∂DR = β_s/ECI_s + (κ/τ_D)·σ((DR − DR_z)/τ_D)
# =====================================================================

def _inverse_box_cox(y, lam):
    if lam == 0.0:
        return math.exp(y)
    return (lam * y + 1.0) ** (1.0 / lam)


def zero_boundary(percentile=None):
    """零点境界 (E_z, DR_z): Box-Cox 母数から P = 0.2 を解いた値（P-5）。

    P-5 と第 12 章の値（7.1998 GJ・0.2220）はこの解の 4 桁表記である。R-5 の参照値（第 3-3 節の表）は
    丸める前の解で計算されており、丸めた 0.2220 を使うと (5.00, 0.215) の比が 0.65% ずれるため、
    ここでは解そのものを使う（報告事項 V6S-a）。
    """
    from statistics import NormalDist
    q = constants.R5_ZERO_PERCENTILE if percentile is None else percentile
    z = NormalDist().inv_cdf(1.0 - q)
    p = constants.ECI_PARAMS
    return (_inverse_box_cox(p["muE"] + p["sigE"] * z, p["lamE"]),
            _inverse_box_cox(p["muDR"] + p["sigDR"] * z, p["lamDR"]))


E_ZERO_GJ, DR_ZERO = zero_boundary()


def _softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def log_eci_smooth(E, DR):
    """ln ECI_s = (ln P_E + ln P_DR)/2。P が浮動小数点で 0 になる点では −inf。"""
    PE = percentile_energy(E)
    PDR = percentile_dissatisfaction(DR)
    if PE <= 0.0 or PDR <= 0.0:
        return -math.inf
    return 0.5 * (math.log(PE) + math.log(PDR))


def objective_j(E, DR, kappa=None, tau_E=None, tau_D=None):
    """R-5 の J(E, DR)。L-4 の収穫の同点の順位と、SIL の保存庫の順位に使う。"""
    kappa = constants.R5_KAPPA if kappa is None else kappa
    tau_E = constants.R5_TAU_E_GJ if tau_E is None else tau_E
    tau_D = constants.R5_TAU_D if tau_D is None else tau_D
    return log_eci_smooth(E, DR) - kappa * (_softplus((E - E_ZERO_GJ) / tau_E) + _softplus((DR - DR_ZERO) / tau_D))


def r5_slopes(E, DR, kappa=None, tau_E=None, tau_D=None):
    """R-5 の (α, β) と内訳。評価点は基準点 (E*, DR*)。"""
    kappa = constants.R5_KAPPA if kappa is None else kappa
    tau_E = constants.R5_TAU_E_GJ if tau_E is None else tau_E
    tau_D = constants.R5_TAU_D if tau_D is None else tau_D
    alpha_s, beta_s = slopes(E, DR)
    value = eci_smooth(E, DR)
    sig_E = _sigmoid((E - E_ZERO_GJ) / tau_E)
    sig_D = _sigmoid((DR - DR_ZERO) / tau_D)
    alpha = alpha_s / value + (kappa / tau_E) * sig_E
    beta = beta_s / value + (kappa / tau_D) * sig_D
    return {"alpha": alpha, "beta": beta, "alpha_s": alpha_s, "beta_s": beta_s, "eci_smooth": value,
            "sigma_E": sig_E, "sigma_D": sig_D, "rho_r5": beta / alpha, "rho_smooth": beta_s / alpha_s}
