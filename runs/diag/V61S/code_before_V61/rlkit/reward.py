"""rlkit.reward — 学習用報酬と、ゾーン別・系統別への分解（v2 第 5-1 節）。

ステップ報酬  r_t   = −T · (α·ΔE_t + β·n_t) / (α·E* + β·DR*)
ゾーン別報酬  r_k,t = −T · (α·e_k,t + β·N_kt·DR_kt·Δt_t/W) / (α·E* + β·DR*)
系統ヘッダ    r_sys = −T · α · (ΔE_t − Σ_k e_k,t) / (α·E* + β·DR*)
恒等式        r_t = Σ_k r_k,t + r_sys,t

報酬スケール c と終端補正は使わない（v2 第 5-1-1 節）。
V6-S R-5: α・β は eci_smooth ではなく、零点境界の障壁を加えた J の勾配から求める（eci.r5_slopes）。
ステップ報酬の式と R-2''・R-3'' は変えない。
"""
import math

import numpy as np

from rlkit import constants, eci as eci_mod

N_ZONES = constants.N_ZONES
SYSTEM_INDEX_OF_ZONE = np.array([ou - 1 for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS])


def dr_co2(ppm):
    """CO2 濃度[ppm]から空気質の不満足率（現行のロジスティック曲線）。"""
    return 1.0 / (1.0 + math.exp((constants.CO2_DISSATISFACTION_MIDPOINT_PPM - ppm) /
                                 constants.CO2_DISSATISFACTION_SCALE_PPM))


def combine_dr_components(dr_thermal, dr_draft, dr_vtmp, dr_co2_value):
    """DR_kt = 1 − (1−DR_thermal)(1−DR_draft)(1−DR_vtmp)(1−DR_co2)（現行を維持）。"""
    return 1.0 - (1.0 - dr_thermal) * (1.0 - dr_draft) * (1.0 - dr_vtmp) * (1.0 - dr_co2_value)


def dissatisfaction_increment(occupants, dr, step_seconds, W):
    """n = N·DR·Δt[分]/W（在室者数と経過分で重み付けした不満足率増分）。"""
    return occupants * dr * (step_seconds / 60.0) / W


def reward_context(rho, E_star, DR_star):
    """O-5（V6-S 第 3-4 節）: 価値ネットワークの文脈層へ与える (ρ_n, D_n)。

    ρ_n = clip(log10(ρ)/4, 0, 1)、D_n = clip(log10(E* + ρ·DR*)/3, 0, 1)。ρ はその更新で使う β/α（2 倍制限の後）。
    """
    rho_n = min(max(math.log10(rho) / constants.RHO_N_LOG10_DIVISOR, 0.0), 1.0)
    D_n = min(max(math.log10(E_star + rho * DR_star) / constants.D_N_LOG10_DIVISOR, 0.0), 1.0)
    return rho_n, D_n


class ReferencePoint:
    """基準点 (E*, DR*) と傾き (α, β)（指示書 V4-S 第 2-1 節 R-2''〜R-5''、V6-S R-5）。

    R-1 : 初期値は新規初期化直後の方策の決定論的評価 1 回の (E, DR) で作る（呼び出し側。V2-S から不変）。
    R-2'': 毎更新、訓練 12 エピソードの (E, DR) の算術平均で指数移動平均を取る（update() 内）。
           決定論的評価の点は基準点に使わない。零点条件の内外は区別しない。
    R-3'': ρ = β/α の変化を毎更新 [ρ/2, 2ρ] に制限する（更新 1 は制限なし）。
    R-4'': 更新した基準点は次の採取から使う（呼び出し側）。
    R-5'': 毎更新 (Ē, DR̄, E*, DR*, α, β, β/α, 制限の有無) を history に残す。
    V6-S R-5: 傾きは eci.r5_slopes（J の勾配）。2 倍制限は R-5 で求めた比に掛ける。history には α_s・β_s、
    σ((E*−E_z)/τ_E)、σ((DR*−DR_z)/τ_D)、R-5 の比（制限前）、使った比（制限後）、ρ_n・D_n も残す（第 3-8 節）。
    """

    def __init__(self, E_star, DR_star, alpha=None, beta=None, history=None):
        self.E_star = float(E_star)
        self.DR_star = float(DR_star)
        if alpha is None or beta is None:
            s = eci_mod.r5_slopes(self.E_star, self.DR_star)
            alpha, beta = s["alpha"], s["beta"]
        self.alpha = float(alpha)
        self.beta = float(beta)
        if history is None:
            s = eci_mod.r5_slopes(self.E_star, self.DR_star)
            rho_n, D_n = reward_context(self.rho, self.E_star, self.DR_star)
            history = [{"update": None, "E_bar": None, "DR_bar": None,
                        "E_star": self.E_star, "DR_star": self.DR_star, "alpha": self.alpha,
                        "beta": self.beta, "rho_raw": self.rho, "rho": self.rho, "rho_clipped": False,
                        "first": None, "alpha_s": s["alpha_s"], "beta_s": s["beta_s"], "sigma_E": s["sigma_E"],
                        "sigma_D": s["sigma_D"], "rho_n": rho_n, "D_n": D_n}]
        self.history = list(history)

    @property
    def rho(self):
        return self.beta / self.alpha

    def denominator(self):
        return self.alpha * self.E_star + self.beta * self.DR_star

    def value_context(self):
        """O-5 の (ρ_n, D_n)。"""
        return reward_context(self.rho, self.E_star, self.DR_star)

    def update(self, E_bar, DR_bar, update_index=None, first=False):
        """訓練エピソードの平均 (Ē, DR̄) で基準点を動かす（R-2''・R-3''）。

        first: 更新 1。平均をそのまま基準点にし、β/α の 2 倍制限を課さない。
        それ以降は (E*, DR*) ← (1−c)·(E*, DR*) + c·(Ē, DR̄)（c = REFERENCE_EMA_COEF）とし、
        新しい基準点で R-5 の勾配を取り直したうえで、β/α の変化を 2 倍以内に抑える。
        """
        c = constants.REFERENCE_EMA_COEF
        if first:
            E_new, DR_new = float(E_bar), float(DR_bar)
        else:
            E_new = (1.0 - c) * self.E_star + c * float(E_bar)
            DR_new = (1.0 - c) * self.DR_star + c * float(DR_bar)
        s = eci_mod.r5_slopes(E_new, DR_new)
        alpha_new, beta_new = s["alpha"], s["beta"]
        rho_raw = beta_new / alpha_new
        rho_old = self.rho
        if first:
            rho_new = rho_raw
        else:
            k = constants.REFERENCE_RATIO_MAX_CHANGE
            rho_new = min(max(rho_raw, rho_old / k), rho_old * k)
        self.E_star, self.DR_star = E_new, DR_new
        self.alpha = float(alpha_new)
        self.beta = float(rho_new * alpha_new)
        rho_n, D_n = reward_context(rho_new, self.E_star, self.DR_star)
        entry = {"update": update_index, "E_bar": float(E_bar), "DR_bar": float(DR_bar),
                 "E_star": self.E_star, "DR_star": self.DR_star, "alpha": self.alpha, "beta": self.beta,
                 "rho_raw": rho_raw, "rho": rho_new, "rho_clipped": rho_new != rho_raw, "first": bool(first),
                 "alpha_s": s["alpha_s"], "beta_s": s["beta_s"], "sigma_E": s["sigma_E"], "sigma_D": s["sigma_D"],
                 "rho_smooth": s["rho_smooth"], "rho_n": rho_n, "D_n": D_n}
        self.history.append(entry)
        return entry

    def to_dict(self):
        return {"E_star": self.E_star, "DR_star": self.DR_star, "alpha": self.alpha, "beta": self.beta,
                "history": self.history}

    @classmethod
    def from_dict(cls, d):
        return cls(d["E_star"], d["DR_star"], alpha=d["alpha"], beta=d["beta"], history=d.get("history"))


def zone_energy_split(delta_E_gj, p_outdoor_kw, p_indoor_kw, p_hex_kw, heatload_kw, zone_on, step_seconds):
    """ゾーン k の一次エネルギー e_k,t [GJ]（指示書 V2-S 第 2-2 節の比率法。v2 第 5-1-2 節の改訂）。

    実測電力は比率の決定にだけ使い、総量は建物全体の実測 ΔE_t（累積計器の差分、積分値）に合わせる。
      q_k   = P_hex,k + share_k(P_ou,s(k)) [+ P_iu,k]   [kW]
      q_hdr = Σ_{s:|Z_s|=0} P_ou,s                       [kW]（運転ゾーンの無い系統の待機電力）
      Q     = Σ_k q_k + q_hdr
      Q > 0 のとき  e_k = ΔE·q_k/Q、e_hdr = ΔE·q_hdr/Q      Q = 0 のとき  e_k = 0、e_hdr = ΔE
      恒等式 Σ_k e_k + e_hdr = ΔE は構成上つねに成立する
      raw_resid = ΔE − Q·Δt_h·9.77/1000 [GJ]（瞬時値の標本化誤差の診断。記録のみで門にしない）

    Args:
        delta_E_gj: このステップの建物全体の一次エネルギー増分 ΔE_t [GJ]
        p_outdoor_kw: (4,) 室外機電力 P_ou,s（毎ステップ実測）
        p_indoor_kw: (18,) 室内機電力 P_iu,k の実測。None なら加えない（J-1 の判定で決める）
        p_hex_kw: (18,) 全熱交換器電力 P_hex,k（校正モデル）
        heatload_kw: (18,) 室内機熱負荷 L_k（按分の比率にのみ使う）
        zone_on: (18,) 運転中ゾーン（集合 Z_s の判定）
        step_seconds: ステップ長[秒]
    Returns:
        dict: e_gj (18,)、e_header_gj、q_kw (18,)、q_header_kw、Q_kw、raw_resid_gj、
              outdoor_share_kw (18,)、standby_kw (4,)
    按分規則（系統 s 内、v2 第 5-1-2 節のまま）:
        F_s = min(P_ou,s, a_s)
        share_k = F_s/|Z_s| + (P_ou,s − F_s)·|L_k|/Σ_{j∈Z_s}|L_j|   （Σ|L| = 0 のときは等分）
        |Z_s| = 0 のときは全額を系統ヘッダへ
    """
    p_outdoor_kw = np.asarray(p_outdoor_kw, dtype=np.float64)
    p_hex_kw = np.asarray(p_hex_kw, dtype=np.float64)
    abs_load = np.abs(np.asarray(heatload_kw, dtype=np.float64))
    zone_on = np.asarray(zone_on, dtype=bool)

    share = np.zeros(N_ZONES, dtype=np.float64)
    standby = np.zeros(4, dtype=np.float64)
    for s in range(4):
        members = np.flatnonzero((SYSTEM_INDEX_OF_ZONE == s) & zone_on)
        P = p_outdoor_kw[s]
        if members.size == 0:
            standby[s] = P
            continue
        F = min(P, constants.OUTDOOR_UNIT_A_KW[s])
        variable = P - F
        load_sum = abs_load[members].sum()
        if load_sum > 0.0:
            share[members] = F / members.size + variable * abs_load[members] / load_sum
        else:
            share[members] = (F + variable) / members.size

    q = p_hex_kw + share
    if p_indoor_kw is not None:
        q = q + np.asarray(p_indoor_kw, dtype=np.float64)
    q_header = float(standby.sum())
    Q = float(q.sum()) + q_header
    if Q > 0.0:
        e_gj = delta_E_gj * q / Q
        e_header_gj = delta_E_gj * q_header / Q
    else:
        e_gj = np.zeros(N_ZONES, dtype=np.float64)
        e_header_gj = delta_E_gj
    step_hours = step_seconds / 3600.0
    raw_resid_gj = delta_E_gj - Q * step_hours * constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0
    return {"e_gj": e_gj, "e_header_gj": e_header_gj, "q_kw": q, "q_header_kw": q_header, "Q_kw": Q,
            "raw_resid_gj": raw_resid_gj, "outdoor_share_kw": share, "standby_kw": standby}


class RewardCalculator:
    """基準点を固定した 1 回の採取ぶんの報酬計算。"""

    def __init__(self, reference, T, W):
        self.ref = reference
        self.T = T
        self.W = W

    @property
    def scale(self):
        return self.T / self.ref.denominator()

    def step_reward(self, delta_E_gj, n_total):
        return -self.scale * (self.ref.alpha * delta_E_gj + self.ref.beta * n_total)

    def zone_rewards(self, e_gj, n_zone):
        return -self.scale * (self.ref.alpha * np.asarray(e_gj) + self.ref.beta * np.asarray(n_zone))

    def system_reward(self, delta_E_gj, e_gj):
        return -self.scale * self.ref.alpha * (delta_E_gj - float(np.sum(e_gj)))
