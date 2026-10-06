"""rlkit.co2pred — 換気の水準ごとの次ステップ末の CO2 の予測（指示書 V7-S 第 3-3 節・C-3）。

物質収支の式（テナントごと、水準 ℓ = 停止・Low・Middle・High）
  C(t+Δ) = C_out + (C(t) − C_out)·exp(−a_ℓ·Δ) + (b·N̄/a_ℓ)·(1 − exp(−a_ℓ·Δ))
    a_ℓ = Q_ℓ/V [1/h]、b = g/V [ppm/h・人]、Δ [h] は次のステップの長さ、N̄ は次のステップのテナントの平均在室者数
係数は constants.CO2_PRED_PARAMS（runs/diag/V7S/co2_fit.json の推定値）。
観測の値は clip((C_pred − 900)/200, −1, 1)。N̄ の求め方は推定（rlkit/diag/v7s_diag.py の build_pairs）と同じにする。
"""
import datetime

import numpy as np

from rlkit import constants

TENANT_ZONE_INDICES = {t: [k for k, z in enumerate(constants.ZONE_DEFS) if z[2] == t] for t in ("South", "North")}


def predict_levels(tenant, c0, n_bar, seconds, params=None):
    """4 水準（停止・Low・Middle・High）それぞれの次ステップ末の CO2 [ppm]。"""
    p = (params or constants.CO2_PRED_PARAMS)[tenant]
    a = np.asarray(p["a"], dtype=np.float64)
    h = float(seconds) / 3600.0
    e = np.exp(-a * h)
    return p["c_out"] + (float(c0) - p["c_out"]) * e + p["b"] * float(n_bar) / a * (1.0 - e)


def tenant_mean_occupants(exo, dt, seconds, tenant):
    """次のステップ [dt, dt + seconds) の各分の、外生表のテナントの在室者数の平均。"""
    zs = TENANT_ZONE_INDICES[tenant]
    mins = [dt + datetime.timedelta(minutes=m) for m in range(int(seconds // 60))]
    return float(np.mean([exo.zone_occupants_at(m)[zs].sum() for m in mins]))


def obs_values(pred_ppm):
    return np.clip((np.asarray(pred_ppm, dtype=np.float64) - constants.CO2_PRED_CENTER_PPM) / constants.CO2_PRED_SCALE_PPM,
                   -1.0, 1.0)


def raw_values(pred_ppm):
    """クリップ前の値（クリップ率の記録に使う）。"""
    return (np.asarray(pred_ppm, dtype=np.float64) - constants.CO2_PRED_CENTER_PPM) / constants.CO2_PRED_SCALE_PPM
