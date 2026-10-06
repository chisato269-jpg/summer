"""rlkit.psychro — 湿り空気（現行 rlkit/psychro.py を移植、変更なし）。

Tetens式による飽和水蒸気圧を用いて、乾球温度・相対湿度から絶対湿度を求める。
標準大気圧101.325kPaを仮定する。
"""
import math

STANDARD_ATM_KPA = 101.325


def saturation_vapor_pressure_kpa(t_c):
    """Tetens式による飽和水蒸気分圧[kPa]。"""
    return 0.61078 * math.exp(17.27 * t_c / (t_c + 237.3))


def relative_to_absolute_humidity(t_c, rh_pct):
    """乾球温度[℃]・相対湿度[%]から絶対湿度[g/kg]を求める。"""
    p_ws = saturation_vapor_pressure_kpa(t_c)
    p_w = (rh_pct / 100.0) * p_ws
    w_kgkg = 0.622 * p_w / max(STANDARD_ATM_KPA - p_w, 1e-6)
    return w_kgkg * 1000.0
