"""rlkit.learner — 学習アルゴリズム（v2 第 5-5 節、指示書 V6-S 第 3-5 節・第 3-6 節）。sb3-contrib MaskablePPO を継承する。

1 回の更新（v2 第 5-5-1 節）のうち、本モジュールが受け持つのは次の手順である。
  1 採取        collect_rollouts: N 組 × 2 周のエピソードを集め、info の zone_rewards・step_seconds と、
                SIL の保存庫に入れる ΔE_t・e_k,t・n_k,t を記録
  1' R-8'       エピソードの終端の info の r8p_zone_n・r8p_zone_rewards で、そのエピソードの n_k,t とゾーン別報酬を
                置き換える（V6.4-S。GAE・価値の目標・SIL の保存庫より前）
  3 GAE         19 出力について、エピソード境界で区切った時間基準 λ の GAE（λ_hour は C-3 で 0.99）
  4 優位度      家族ごとに K-14' で採った水準で行動次元へ割り当て（C-1）、水準ごとに GC-1 で除かない要素だけで
                正規化する（C-2）
  5 方策        最大 4 エポック。各エポックの開始時に全ステップの approx_kl（GC-1 で除かない次元の合計・
                ステップ平均）を計算し、1.5 × target_kl を超えたらそのエポックの勾配を適用せずに打ち切る
  6 学習率      更新後の approx_kl に応じて方策の学習率を調整（範囲 [1e-5, 1e-3]、L-2）
  7 価値        分離した最適化器で 8 エポック（KL 制約なし）。価値の文脈には (ρ_n, D_n) を与える（O-5）
  7' SIL        更新 301 以降（中間判定に合格した場合）だけ、保存庫全体で 1 エポック（SIL-3〜SIL-5）
  8 記録        last_update_record
GC-1: 停止を標本化したゾーンの設定温度・風量、運転ゾーンの無い系統の蒸発温度、マスクで 1 択の要素を、
方策の損失・エントロピー・approx_kl・優位度の正規化の母集団・SIL の損失のすべてから除く。
L-1: 群別エントロピー床と双対上昇は廃止し、c_g = c0 / ln K_g（c0 は計画の 60% まで 0.002、以後 0 まで線形）。
手順 2（決定論的評価）と基準点の更新、SIL の保存庫の管理は rlkit/train_summer.py が行う。
"""
import copy
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from rlkit import action as action_mod
from rlkit import advantage as adv_mod
from rlkit import constants
from rlkit import policy as policy_mod
from rlkit.policy import PolicyNetwork, ValueNetwork, transfer_policy_stage1_to_stage2
from rlkit.reward import RewardCalculator

N_ZONES = constants.N_ZONES


# =====================================================================
# SB3 の方策クラスへの接続
# =====================================================================

class _Extractor(nn.Module):
    """SB3 の MlpExtractor の形（forward/forward_actor/forward_critic）で方策網と価値網を持つ。"""

    def __init__(self, stage):
        super().__init__()
        self.policy_net = PolicyNetwork(stage)
        self.value_net = ValueNetwork()
        self.latent_dim_pi = constants.ACTION_LOGITS[stage]
        self.latent_dim_vf = N_ZONES + 1

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class SummerPolicy(MaskableActorCriticPolicy):
    """方策網と価値網を分離し、それぞれに Adam を持たせた MaskableActorCriticPolicy。

    SB3 標準経路（ロールアウトバッファの価値など）へは 19 出力の合計を渡す（v2 第 5-9 節）。
    """

    def __init__(self, observation_space, action_space, lr_schedule, stage, **kwargs):
        self._stage = stage
        kwargs.setdefault("ortho_init", False)  # 初期化は rlkit.policy 内で行う
        kwargs.setdefault("net_arch", [])
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = _Extractor(self._stage)

    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.optimizer = th.optim.Adam(self.policy_network.parameters(), lr=constants.POLICY_LR_INIT, eps=constants.ADAM_EPS)
        self.value_optimizer = th.optim.Adam(self.value_network.parameters(), lr=constants.VALUE_LR, eps=constants.ADAM_EPS)

    @property
    def stage(self):
        return self._stage

    @property
    def policy_network(self):
        return self.mlp_extractor.policy_net

    @property
    def value_network(self):
        return self.mlp_extractor.value_net

    def set_reward_context(self, rho_n, D_n):
        """O-5: 価値ネットワークの文脈に与える (ρ_n, D_n)。その更新の全標本に同じ値を使う。"""
        self.value_network.set_reward_context(rho_n, D_n)

    @staticmethod
    def _reduce_value(value):
        return value.sum(dim=-1, keepdim=True) if value.dim() > 1 and value.shape[-1] > 1 else value

    def forward(self, obs, deterministic=False, action_masks=None):
        actions, values, log_prob = super().forward(obs, deterministic=deterministic, action_masks=action_masks)
        return actions, self._reduce_value(values), log_prob

    def predict_values(self, obs):
        return self._reduce_value(super().predict_values(obs))

    def evaluate_actions(self, obs, actions, action_masks=None):
        values, log_prob, entropy = super().evaluate_actions(obs, actions, action_masks=action_masks)
        return self._reduce_value(values), log_prob, entropy

    def masked_distribution(self, obs, action_masks):
        dist = self.action_dist.proba_distribution(action_logits=self.policy_network(self.extract_features(obs)))
        dist.apply_masking(action_masks)
        return dist

    def values19(self, obs):
        return self.value_network(self.extract_features(obs))


# =====================================================================
# 更新の部品
# =====================================================================

def _grad_norm(grads):
    total = 0.0
    for g in grads:
        if g is not None:
            total += float((g.detach() ** 2).sum())
    return math.sqrt(total)


def per_dim_log_prob(dist, actions):
    return th.stack([d.log_prob(actions[:, i]) for i, d in enumerate(dist.distributions)], dim=1)


def per_dim_entropy(dist):
    return th.stack([d.entropy() for d in dist.distributions], dim=1)


def compute_n_options(action_masks, stage):
    """(T, ロジット数) のマスクから、各 (ステップ, 次元) で選べる選択肢数 (T, 62)。"""
    return th.stack([action_masks[:, s:s + w].sum(dim=1) for s, w in action_mod.dim_logit_offsets(stage)], dim=1)


def include_mask(actions, n_options):
    """GC-1（V6-S 第 3-2 節）: 更新に使う要素（GC-1 で除かず、選択肢が 2 以上）を真とする (T, 62)。

    Returns: (include, gc1_excluded, one_option) いずれも bool テンソル
    """
    gc1 = th.as_tensor(action_mod.gc1_excluded(np.asarray(actions)), device=n_options.device)
    one_option = n_options < 2
    return (~gc1) & (~one_option), gc1, one_option


def compute_group_entropy(entropy, include, group_layout):
    """GC-1 で除かない要素だけで、群ごとの平均エントロピー（L-1）。"""
    out = {}
    for g, info in group_layout.items():
        s, n = info["start"], info["n"]
        valid = include[:, s:s + n].to(entropy.dtype)
        out[g] = None if valid.sum() == 0 else (entropy[:, s:s + n] * valid).sum() / valid.sum()
    return out


def exclusion_counts(gc1, one_option, group_layout):
    """第 3-8 節: GC-1 で除外した要素数と、1 択で除外した要素数（群別）。GC-1 と 1 択が重なる要素は GC-1 に数える。"""
    out = {"gc1": {}, "one_option": {}}
    for g, info in group_layout.items():
        s, n = info["start"], info["n"]
        out["gc1"][g] = int(gc1[:, s:s + n].sum())
        out["one_option"][g] = int((one_option[:, s:s + n] & ~gc1[:, s:s + n]).sum())
    return out


def entropy_c0_at(update_index, planned_updates):
    """L-1: 更新 u の c0。計画の 60%（1,500 なら 900）まで 0.002、以後 100%（1,500）で 0 まで線形に下げる。"""
    const_until = constants.ENTROPY_CONST_UNTIL_FRACTION * planned_updates
    zero_at = constants.ENTROPY_ZERO_AT_FRACTION * planned_updates
    if update_index <= const_until:
        return constants.ENTROPY_C0
    if update_index >= zero_at:
        return 0.0
    return constants.ENTROPY_C0 * (zero_at - update_index) / (zero_at - const_until)


def entropy_coefficients(update_index, planned_updates, group_layout):
    """L-1: 群 g の係数 c_g = c0 / ln K_g。"""
    c0 = entropy_c0_at(update_index, planned_updates)
    return {g: c0 / info["lnK"] for g, info in group_layout.items()}


def adapt_learning_rate(lr, approx_kl):
    """更新後の approx_kl が 2×target_kl を超えたら 1/1.5 倍、0.5×target_kl 未満なら 1.5 倍（範囲 [1e-5, 1e-3]、L-2）。"""
    if approx_kl > constants.LR_ADAPT_HIGH * constants.TARGET_KL:
        lr = lr / constants.LR_ADAPT_FACTOR
    elif approx_kl < constants.LR_ADAPT_LOW * constants.TARGET_KL:
        lr = lr * constants.LR_ADAPT_FACTOR
    return min(max(lr, constants.POLICY_LR_MIN), constants.POLICY_LR_MAX)


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def clipped_policy_loss(log_prob, old_log_prob, adv, include):
    """PPO の比のクリップ項（次元別）。GC-1 で除く要素は寄与しない。Returns: (損失, 比)"""
    ratio = th.exp(log_prob - old_log_prob)
    clipped = th.clamp(ratio, 1 - constants.CLIP_RANGE, 1 + constants.CLIP_RANGE)
    inc = include.to(log_prob.dtype)
    return -(th.min(adv * ratio, adv * clipped) * inc).sum(dim=1).mean(), ratio


def approx_kl_from(log_prob, old_log_prob, include):
    """approx_kl（GC-1 で除かない次元の対数比の合計・ステップの平均）。"""
    log_ratio = ((log_prob - old_log_prob) * include.to(log_prob.dtype)).sum(dim=1)
    return th.mean((th.exp(log_ratio) - 1) - log_ratio)


def sil_policy_loss(log_prob, adv, sigma_A, include, c_sil):
    """SIL-3: −c_sil × mean[log π(a|s) × max(Â, 0) / σ_A]（平均は GC-1 で除かない要素について取る）。
    σ_A は要素ごとのテンソルでもよい（C-2: 要素の水準の標準偏差）。"""
    inc = include.to(log_prob.dtype)
    n_inc = inc.sum().clamp(min=1.0)
    weight = th.clamp(adv, min=0.0) / sigma_A * inc
    return -c_sil * (log_prob * weight).sum() / n_inc


def sil_value_loss(returns, values):
    """SIL-3: 0.01 × mean[max(R_o − V_o, 0)²]。"""
    return constants.SIL_VALUE_COEF * (th.clamp(returns - values, min=0.0) ** 2).mean()


def normalize_over(m, include, eps=constants.ADVANTAGE_NORM_EPS):
    """割り当て後の優位度を、GC-1 で除かない要素だけの平均・標準偏差で正規化する。Returns: (正規化後, 平均, 標準偏差)"""
    m = np.asarray(m, dtype=np.float64)
    inc = np.asarray(include, dtype=bool)
    vals = m[inc]
    if vals.size == 0:
        return np.zeros_like(m), 0.0, 1.0
    mean, std = float(vals.mean()), float(vals.std())
    return (m - mean) / (std + eps), mean, std


def sil_returns(entry, reference, T):
    """SIL-2: 保存した e_k,t・n_k,t・ΔE_t から、その更新の α・β・E*・DR* で 19 出力のステップ報酬を計算し直し、
    収益 R（γ = 1、ステップ t 以降の和）を返す (T, 19)。"""
    calc = RewardCalculator(reference, T, entry["W"])
    e = np.asarray(entry["zone_energy"], dtype=np.float64)
    n = np.asarray(entry["zone_n"], dtype=np.float64)
    dE = np.asarray(entry["delta_E"], dtype=np.float64)
    zone_r = calc.zone_rewards(e, n)                                            # (T,18)
    sys_r = -calc.scale * reference.alpha * (dE - e.sum(axis=1))                # (T,)
    r19 = np.concatenate([zone_r, sys_r[:, None]], axis=1)
    return np.flip(np.cumsum(np.flip(r19, axis=0), axis=0), axis=0).copy()


def apply_r8p(zone_rewards_tn, zone_n_tn, terminal_infos):
    """R-8'（V6.4-S）: 終端の info の r8p_zone_n・r8p_zone_rewards で、そのエピソードの行を置き換える（その場で書き換える）。

    zone_rewards_tn・zone_n_tn: (T, N, 18)。終端ステップ step・組 i のエピソードは [step − L + 1, step] の L 行。
    採取は reset から始まりエピソードは必ず完結するため、全行が置き換わらなければ例外にする（終端 1 点の n を残さない）。
    Returns: 記録（本数、Σn と DR の相対差の最大、処理時間）
    """
    T, N = zone_n_tn.shape[:2]
    covered = np.zeros((T, N), dtype=bool)
    rels, secs, mins, empties = [], [], [], []
    for step, i, info in terminal_infos:
        n_kt = np.asarray(info["r8p_zone_n"], dtype=np.float64)
        L = n_kt.shape[0]
        lo = int(step) - L + 1
        if lo < 0:
            raise RuntimeError(f"R-8': 組 {i} の終端 {step} のエピソードが採取の先頭より前から始まっている")
        zone_n_tn[lo:step + 1, i] = n_kt
        zone_rewards_tn[lo:step + 1, i] = np.asarray(info["r8p_zone_rewards"], dtype=np.float64)
        covered[lo:step + 1, i] = True
        rels.append(float(info["r8p_sum_rel"]))
        secs.append(float(info["r8p_seconds"]))
        mins.append(float(info["r8p_min_n"]))
        empties.append(float(info["r8p_empty_max_abs_n"]))
    if not covered.all():
        raise RuntimeError(f"R-8': 置き換えられなかった行が {int((~covered).sum())} ある（エピソードが完結していない）")
    return {"branch": constants.R8P_BRANCH, "delta_min": constants.R8P_DELTA_MIN, "episodes": len(rels),
            "sum_rel_max": max(rels) if rels else None, "seconds_sum": float(sum(secs)),
            "seconds_max": max(secs) if secs else None, "min_n": min(mins) if mins else None,
            "empty_max_abs_n": max(empties) if empties else None}


def occupied_off_fraction(actions, occupants):
    """在室者 1 人以上のゾーン・ステップのうち、運転状態に停止を選んだ割合（行動を選んだ時点の在室者数）。"""
    occ = np.asarray(occupants) >= 1
    off = ~action_mod.zone_on_flags_batch(actions)
    n = int(occ.sum())
    return {"fraction": float((off & occ).sum() / n) if n else None, "n_occupied_zone_steps": n}


def sil_sigma_matrix(level_stats, n_rows):
    """SIL の σ_A を要素ごとに並べる (n_rows, ACTION_DIMS)。各要素の水準の、その更新の PPO の標準偏差（C-2）。"""
    levels = adv_mod.dim_levels()
    row = np.array([level_stats[lv]["std"] if level_stats.get(lv, {}).get("n") else 1.0 for lv in levels])
    row = np.where(row > 0, row, 1.0)
    return np.repeat(row[None, :], n_rows, axis=0)


class _RolloutCapture(BaseCallback):
    """採取中、毎ステップの全組の info から報酬の分解と SIL の保存庫に入れる量を集める。"""

    def __init__(self):
        super().__init__()
        self.zone_rewards, self.system_rewards, self.step_seconds, self.terminal_infos = [], [], [], []
        self.delta_E, self.zone_energy, self.zone_n, self.occupants = [], [], [], []

    def _on_step(self):
        infos = self.locals["infos"]
        self.zone_rewards.append(np.stack([np.asarray(inf["zone_rewards"], dtype=np.float64) for inf in infos]))
        self.system_rewards.append(np.array([inf["system_reward"] for inf in infos], dtype=np.float64))
        self.occupants.append(np.stack([np.asarray(inf["zone_occupants_at_decision"], dtype=np.float64) for inf in infos]))
        self.step_seconds.append(np.array([inf["step_seconds"] for inf in infos], dtype=np.float64))
        self.delta_E.append(np.array([inf["delta_E_gj"] for inf in infos], dtype=np.float64))
        self.zone_energy.append(np.stack([np.asarray(inf["zone_energy_gj"], dtype=np.float64) for inf in infos]))
        self.zone_n.append(np.stack([np.asarray(inf["zone_dissatisfaction_increment"], dtype=np.float64) for inf in infos]))
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.terminal_infos.append((len(self.zone_rewards) - 1, i, infos[i]))
        return True


# =====================================================================
# 学習器
# =====================================================================

class SummerPPO(MaskablePPO):
    def __init__(self, env, stage, planned_updates, seed, n_steps_per_env, verbose=0):
        if planned_updates is None:
            raise ValueError("未確定項目 J-6: 計画更新数（planned_updates）が設定されていない")
        self.stage = stage
        self.planned_updates = int(planned_updates)
        self.group_layout = action_mod.group_layout(stage)
        self.update_index = 0
        self.policy_lr = constants.POLICY_LR_INIT
        # SIL-5 の係数の状態。V7-S C-5 で自己模倣学習を止めた（係数 0。_sil は損失を加えずに記録だけを返す）
        self.c_sil = constants.SIL_C_INIT if constants.SIL_ENABLED else 0.0
        self.sil_low_kl_streak = 0
        self.last_update_record = None
        self.last_terminal_infos = []
        self._rollout = None
        super().__init__(
            SummerPolicy, env, learning_rate=constants.POLICY_LR_INIT, n_steps=n_steps_per_env,
            batch_size=constants.POLICY_MINIBATCH_STEPS, n_epochs=constants.POLICY_EPOCHS, gamma=constants.GAMMA,
            gae_lambda=constants.LAMBDA_HOUR, clip_range=constants.CLIP_RANGE, normalize_advantage=True,
            ent_coef=0.0, vf_coef=0.0, max_grad_norm=constants.MAX_GRAD_NORM, target_kl=constants.TARGET_KL,
            policy_kwargs={"stage": stage}, seed=seed, verbose=verbose)

    # ---- 手順 1・3: 採取と GAE ----

    def collect_one(self):
        """1 回の採取（全組 n_steps ステップ）。learn() を使わずに採取だけを行う（更新との間に評価の判定を挟むため）。"""
        self._last_obs = None
        total, callback = self._setup_learn(self.num_timesteps + self.n_steps * self.n_envs, None,
                                            reset_num_timesteps=False, tb_log_name="SummerPPO", progress_bar=False)
        callback.on_training_start(locals(), globals())
        ok = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.n_steps, use_masking=True)
        callback.on_training_end()
        if not ok:
            raise RuntimeError("採取が途中で止まった")
        return self.last_terminal_infos

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        capture = _RolloutCapture()
        combined = CallbackList([callback, capture])
        combined.init_callback(self)
        ok = super().collect_rollouts(env, combined, rollout_buffer, n_rollout_steps, use_masking)
        self.last_terminal_infos = capture.terminal_infos
        if not ok:
            return ok
        T = n_rollout_steps
        flat = adv_mod.flatten_env_major
        zr_tn = np.stack(capture.zone_rewards)            # (T,N,18)
        zn_tn = np.stack(capture.zone_n)
        r8p_rec = apply_r8p(zr_tn, zn_tn, capture.terminal_infos)
        zone_rewards = flat(zr_tn)                          # (N·T,18)
        # 系統の報酬はエネルギーだけで決まり R-8' の影響を受けない。info の値（倍精度）を使う
        system_rewards = flat(np.stack(capture.system_rewards)).reshape(-1)
        step_seconds = flat(np.stack(capture.step_seconds)).reshape(-1)
        starts = flat(rollout_buffer.episode_starts).reshape(-1).copy()
        starts[::T] = True  # 組をまたいで収益をつながない
        segments = adv_mod.episode_boundaries_from_starts(starts)
        observations = flat(rollout_buffer.observations)
        with th.no_grad():
            values19 = self.policy.values19(th.as_tensor(observations, device=self.device).float()).cpu().numpy()
        rewards19 = np.concatenate([zone_rewards, system_rewards[:, None]], axis=1)
        adv19, ret19 = adv_mod.segmented_gae(rewards19, values19, step_seconds, segments)
        self._rollout = {
            "observations": observations, "actions": flat(rollout_buffer.actions),
            "action_masks": flat(rollout_buffer.action_masks), "values19": values19,
            "rewards19": rewards19, "advantages19": adv19, "returns19": ret19,
            "step_seconds": step_seconds, "segments": segments,
            # SIL の保存庫に入れる量（SIL-1）
            "delta_E": flat(np.stack(capture.delta_E)).reshape(-1),
            "zone_energy": flat(np.stack(capture.zone_energy)), "zone_n": flat(zn_tn),
            "occupants": flat(np.stack(capture.occupants)), "r8p": r8p_rec,
        }
        return ok

    # ---- 手順 4〜8 ----

    def _approx_kl_by_group(self, obs, actions, masks, old_log_prob, include):
        """群ごとの KL 寄与（その群の、GC-1 で除かない次元の対数比の合計で測る）。"""
        with th.no_grad():
            log_prob = per_dim_log_prob(self.policy.masked_distribution(obs, masks), actions)
            diff = (log_prob - old_log_prob) * include.to(log_prob.dtype)
            out = {}
            for g, info in self.group_layout.items():
                s, n = info["start"], info["n"]
                lr_g = diff[:, s:s + n].sum(dim=1)
                out[g] = float(th.mean((th.exp(lr_g) - 1) - lr_g))
            return out

    def _approx_kl(self, obs, actions, masks, old_log_prob, include):
        """approx_kl（GC-1 で除かない次元の合計・ステップの平均）。"""
        with th.no_grad():
            log_prob = per_dim_log_prob(self.policy.masked_distribution(obs, masks), actions)
            return float(approx_kl_from(log_prob, old_log_prob, include))

    def train(self, sil_buffer=None, sil_mode="off", reference=None):
        """1 回の更新の学習。sil_mode: "off"（何もしない）、"stats"（SIL-7: 損失は適用せず記録だけ）、
        "apply"（SIL-4: PPO の方策エポックと価値エポックの後に保存庫全体で 1 エポック）。"""
        ro = self._rollout
        if ro is None:
            raise RuntimeError("train() の前に採取が必要")
        dev = self.device
        policy = self.policy
        policy.set_training_mode(True)
        obs = th.as_tensor(ro["observations"], device=dev).float()
        actions = th.as_tensor(ro["actions"], device=dev).long()
        masks = th.as_tensor(ro["action_masks"], device=dev).bool()
        n = obs.shape[0]
        u = self.update_index + 1

        n_options = compute_n_options(masks, self.stage)
        include, gc1, one_option = include_mask(ro["actions"], n_options)
        per_dim_raw = adv_mod.assign_per_dim_advantages(ro["advantages19"][:, :N_ZONES], ro["advantages19"][:, N_ZONES],
                                                        ro["actions"])
        # C-2: 水準ごとの正規化（SIL の σ_A も水準ごとの標準偏差を使う）
        adv_norm, level_stats, _sigma = adv_mod.normalize_by_level(per_dim_raw, include.cpu().numpy())
        per_dim_adv = th.as_tensor(adv_norm, device=dev).float()
        returns19 = th.as_tensor(ro["returns19"], device=dev).float()
        inc_f = include.float()

        with th.no_grad():
            dist0 = policy.masked_distribution(obs, masks)
            old_log_prob = per_dim_log_prob(dist0, actions)
            entropy0 = per_dim_entropy(dist0)
        H_g = compute_group_entropy(entropy0, include, self.group_layout)
        c0 = entropy_c0_at(u, self.planned_updates)
        c_g = entropy_coefficients(u, self.planned_updates, self.group_layout)

        kl_per_epoch, pg_losses, ent_losses, clip_fractions = [], [], [], []
        epochs_applied, early_stop = 0, False
        # 指示書 V3-S E-2（V5 から引き継ぐ記録項目）: 指定した更新番号の第 1 エポックの最初のミニバッチで、
        # 方策損失とエントロピー損失（係数込み）の勾配ノルムの比を計測して記録する（判定には使わない）
        measure_grad_ratio = u in constants.GRAD_NORM_RATIO_MEASURE_AT
        grad_norm_policy = grad_norm_entropy = None
        for _epoch in range(constants.POLICY_EPOCHS):
            kl = self._approx_kl(obs, actions, masks, old_log_prob, include)
            kl_per_epoch.append(kl)
            if kl > constants.KL_STOP_FACTOR * constants.TARGET_KL:
                early_stop = True
                break
            perm = np.random.permutation(n)
            for start in range(0, n, constants.POLICY_MINIBATCH_STEPS):
                idx = th.as_tensor(perm[start:start + constants.POLICY_MINIBATCH_STEPS], device=dev)
                dist = policy.masked_distribution(obs[idx], masks[idx])
                inc = inc_f[idx]
                policy_loss, ratio = clipped_policy_loss(per_dim_log_prob(dist, actions[idx]), old_log_prob[idx],
                                                         per_dim_adv[idx], include[idx])
                H_mb = compute_group_entropy(per_dim_entropy(dist), include[idx], self.group_layout)
                terms = [c_g[g] * H for g, H in H_mb.items() if H is not None]
                entropy_loss = -th.stack(terms).sum() if terms else th.zeros((), device=dev)
                loss = policy_loss + entropy_loss
                if measure_grad_ratio:
                    params = [p for p in policy.policy_network.parameters() if p.requires_grad]
                    grad_norm_policy = _grad_norm(th.autograd.grad(policy_loss, params, retain_graph=True,
                                                                   allow_unused=True))
                    grad_norm_entropy = (_grad_norm(th.autograd.grad(entropy_loss, params, retain_graph=True,
                                                                      allow_unused=True))
                                         if entropy_loss.requires_grad else 0.0)
                    measure_grad_ratio = False
                policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(policy.policy_network.parameters(), constants.MAX_GRAD_NORM)
                policy.optimizer.step()
                pg_losses.append(policy_loss.item())
                ent_losses.append(entropy_loss.item())
                n_inc = float(inc.sum())
                clip_fractions.append(float(((th.abs(ratio - 1) > constants.CLIP_RANGE).float() * inc).sum()) / max(n_inc, 1.0))
            epochs_applied += 1

        kl_after = self._approx_kl(obs, actions, masks, old_log_prob, include)
        kl_by_group = self._approx_kl_by_group(obs, actions, masks, old_log_prob, include)
        lr_before = self.policy_lr
        self.policy_lr = adapt_learning_rate(self.policy_lr, kl_after)   # SIL の前に測った approx_kl を使う（SIL-5）
        set_optimizer_lr(policy.optimizer, self.policy_lr)

        value_losses = []
        for _epoch in range(constants.VALUE_EPOCHS):
            perm = np.random.permutation(n)
            for start in range(0, n, constants.VALUE_MINIBATCH_STEPS):
                idx = th.as_tensor(perm[start:start + constants.VALUE_MINIBATCH_STEPS], device=dev)
                value_loss = F.mse_loss(policy.values19(obs[idx]), returns19[idx])
                policy.value_optimizer.zero_grad()
                value_loss.backward()
                th.nn.utils.clip_grad_norm_(policy.value_network.parameters(), constants.MAX_GRAD_NORM)
                policy.value_optimizer.step()
                value_losses.append(value_loss.item())

        sil_rec = self._sil(sil_buffer, sil_mode, reference, level_stats, obs, actions, masks, include)

        self.update_index += 1
        self._n_updates += epochs_applied
        G, V = ro["returns19"], ro["values19"]
        excl = exclusion_counts(gc1, one_option, self.group_layout)
        losses = pg_losses + ent_losses + value_losses
        self.last_update_record = {
            "update": self.update_index, "stage": self.stage,
            "approx_kl_per_epoch": kl_per_epoch, "approx_kl_after": kl_after, "kl_early_stop": early_stop,
            "epochs_applied": epochs_applied, "policy_lr_before": lr_before, "policy_lr_after": self.policy_lr,
            "entropy_c0": c0, "c_g": dict(c_g),
            "approx_kl_by_group": kl_by_group,
            "H_g": {g: (float(v) if v is not None else None) for g, v in H_g.items()},
            "H_g_over_lnK": {g: (float(v) / self.group_layout[g]["lnK"] if v is not None else None) for g, v in H_g.items()},
            "gc1_excluded_by_group": excl["gc1"], "one_option_excluded_by_group": excl["one_option"],
            "n_included_elements": int(include.sum()),
            "policy_loss_mean": float(np.mean(pg_losses)) if pg_losses else None,
            "entropy_loss_mean": float(np.mean(ent_losses)) if ent_losses else None,
            "clip_fraction_mean": float(np.mean(clip_fractions)) if clip_fractions else None,
            "grad_norm_policy": grad_norm_policy, "grad_norm_entropy": grad_norm_entropy,
            "grad_norm_ratio": (grad_norm_entropy / grad_norm_policy
                                if grad_norm_policy not in (None, 0.0) and grad_norm_entropy is not None else None),
            # V5 の記録項目のうち廃止した仕組みのもの（L-1 でエントロピー床を廃止）
            "floor_fraction": None, "floor_fraction_by_group": None,
            "value_loss_first": value_losses[0], "value_loss_last": value_losses[-1],
            "explained_variance_19": float(1.0 - np.var(G - V) / np.var(G)) if np.var(G) > 0 else None,
            "per_dim_advantage_raw_std": float(per_dim_raw.std()),
            # V6.2-S 第 5 章の記録: 採った水準、λ_hour、水準ごとの正規化の平均と標準偏差、在室ゾーン・ステップで停止を選んだ割合
            "credit_levels": dict(constants.CREDIT_LEVELS), "lambda_hour": constants.LAMBDA_HOUR,
            "advantage_norm_by_level": level_stats,
            "advantage_norm_mean": None, "advantage_norm_std": None,     # C-2 で水準ごとに替えた（全体の値は使わない）
            "occupied_zone_step_off_fraction": occupied_off_fraction(ro["actions"], ro["occupants"]),
            "r8p": ro["r8p"],
            "n_samples": int(n), "n_segments": len(ro["segments"]),
            "sil": sil_rec,
            "nonfinite": (not all(math.isfinite(x) for x in losses)) or (not self.parameters_finite()),
        }
        self._rollout = None
        return self.last_update_record

    # ---- SIL（V6-S 第 3-6 節） ----

    def _sil_batch(self, sil_buffer, reference, level_stats):
        """保存庫を 1 本の標本にまとめ、SIL-2 の収益と SIL-3 の優位度（次元への割り当て後、正規化前）を作る。"""
        dev = self.device
        obs = np.concatenate([e["obs"] for e in sil_buffer]).astype(np.float32)
        acts = np.concatenate([e["actions"] for e in sil_buffer]).astype(np.int64)
        masks = np.concatenate([e["masks"] for e in sil_buffer]).astype(bool)
        R = np.concatenate([sil_returns(e, reference, e["actions"].shape[0]) for e in sil_buffer])
        obs_t = th.as_tensor(obs, device=dev)
        with th.no_grad():
            V = self.policy.values19(obs_t).cpu().numpy()
        A19 = R - V
        per_dim = adv_mod.assign_per_dim_advantages(A19[:, :N_ZONES], A19[:, N_ZONES], acts)
        masks_t = th.as_tensor(masks, device=dev)
        include, _, _ = include_mask(acts, compute_n_options(masks_t, self.stage))
        sigma = sil_sigma_matrix(level_stats, per_dim.shape[0])
        return {"obs": obs_t, "actions": th.as_tensor(acts, device=dev), "masks": masks_t, "include": include,
                "R": th.as_tensor(R, device=dev).float(), "adv": th.as_tensor(per_dim, device=dev).float(),
                "sigma": th.as_tensor(sigma, device=dev).float()}

    def _sil(self, sil_buffer, sil_mode, reference, level_stats, ppo_obs, ppo_actions, ppo_masks, ppo_include):
        rec = {"mode": sil_mode, "buffer_size": len(sil_buffer) if sil_buffer else 0, "c_sil": self.c_sil}
        if sil_mode == "off" or not sil_buffer or reference is None:
            return rec
        b = self._sil_batch(sil_buffer, reference, level_stats)
        inc = b["include"]
        rec["positive_advantage_fraction"] = float(((b["adv"] > 0) & inc).sum()) / max(float(inc.sum()), 1.0)
        rec["sigma_A"] = {lv: v["std"] for lv, v in level_stats.items()}      # C-2: 水準ごと
        if sil_mode != "apply" or self.c_sil <= 0.0:
            return rec
        policy = self.policy
        dev = self.device
        # 取り消しに備えて SIL の直前の方策の重みと最適化器の状態を写す（SIL-5）
        saved_policy = copy.deepcopy(policy.policy_network.state_dict())
        saved_opt = copy.deepcopy(policy.optimizer.state_dict())
        with th.no_grad():
            ppo_old = per_dim_log_prob(policy.masked_distribution(ppo_obs, ppo_masks), ppo_actions)
        N = b["obs"].shape[0]
        perm = np.random.permutation(N)
        pol_losses, val_losses = [], []
        c_sil_used = self.c_sil
        for chunk in np.array_split(perm, constants.SIL_MINIBATCHES):
            idx = th.as_tensor(chunk, device=dev)
            if bool(b["include"][idx].any()):
                dist = policy.masked_distribution(b["obs"][idx], b["masks"][idx])
                logp = per_dim_log_prob(dist, b["actions"][idx])
                pol_loss = sil_policy_loss(logp, b["adv"][idx], b["sigma"][idx], b["include"][idx], c_sil_used)
                policy.optimizer.zero_grad()
                pol_loss.backward()
                th.nn.utils.clip_grad_norm_(policy.policy_network.parameters(), constants.MAX_GRAD_NORM)
                policy.optimizer.step()
                pol_losses.append(pol_loss.item())
            val_loss = sil_value_loss(b["R"][idx], policy.values19(b["obs"][idx]))
            policy.value_optimizer.zero_grad()
            val_loss.backward()
            th.nn.utils.clip_grad_norm_(policy.value_network.parameters(), constants.MAX_GRAD_NORM)
            policy.value_optimizer.step()
            val_losses.append(val_loss.item())
        kl_sil = self._approx_kl(ppo_obs, ppo_actions, ppo_masks, ppo_old, ppo_include)
        reverted = False
        if kl_sil > constants.SIL_KL_REVERT:
            policy.policy_network.load_state_dict(saved_policy)
            policy.optimizer.load_state_dict(saved_opt)
            set_optimizer_lr(policy.optimizer, self.policy_lr)
            reverted = True
            self.c_sil = max(self.c_sil / 2.0, constants.SIL_C_MIN)
            self.sil_low_kl_streak = 0
        elif kl_sil > constants.SIL_KL_HALVE:
            self.c_sil = max(self.c_sil / 2.0, constants.SIL_C_MIN)
            self.sil_low_kl_streak = 0
        elif kl_sil < constants.SIL_KL_DOUBLE_BELOW:
            self.sil_low_kl_streak += 1
            if self.sil_low_kl_streak >= constants.SIL_KL_DOUBLE_STREAK:
                self.c_sil = min(self.c_sil * 2.0, constants.SIL_C_MAX)
                self.sil_low_kl_streak = 0
        else:
            self.sil_low_kl_streak = 0
        rec.update({"c_sil_used": c_sil_used, "c_sil_after": self.c_sil, "approx_kl_sil": kl_sil,
                    "reverted": reverted, "low_kl_streak": self.sil_low_kl_streak,
                    "policy_loss_mean": float(np.mean(pol_losses)) if pol_losses else None,
                    "value_loss_mean": float(np.mean(val_losses)) if val_losses else None,
                    "n_samples": int(N)})
        return rec

    def parameters_finite(self):
        return all(bool(th.isfinite(p).all()) for p in self.policy.parameters())

    # ---- 重み・状態の受け渡し ----

    def weights(self):
        return policy_weights(self.policy)

    def load_weights(self, weights):
        """重みの読み込み。第 1 段階の重みを第 2 段階の学習器へ読むときはゾーンヘッドを拡張する（A-7 により使わない）。"""
        self.policy.value_network.load_state_dict(weights["value_net"])
        if weights["stage"] == self.stage:
            self.policy.policy_network.load_state_dict(weights["policy_net"])
        elif weights["stage"] == 1 and self.stage == 2:
            src = PolicyNetwork(1)
            src.load_state_dict(weights["policy_net"])
            transfer_policy_stage1_to_stage2(src, self.policy.policy_network)
        else:
            raise ValueError(f"段階 {weights['stage']} の重みを段階 {self.stage} の学習器へは読めない")

    def load_warm_start(self, path):
        """V7-S C-8: 走行 v2s_trial07 の保存点から、方策と価値の重み、最適化器の状態、方策学習率を引き継ぐ（読むだけ）。

        ゾーン頭部の設定温度の出力は 22〜28 ℃の 7 行だけを残し（C-2）、全体の符号化器の最初の層の新しい入力（CO2 の予測、
        C-3）の重みは 0 にする。Adam の状態も同じ写し方で写す。更新番号・エントロピーの予定は新しい走行で数え直す。
        """
        state = th.load(path, weights_only=False)
        m = state["model"]
        if m["stage"] != self.stage:
            raise ValueError("段階の異なる保存点からは引き継がない")
        specs = policy_mod.warm_start_specs()
        pol = self.policy.policy_network
        val = self.policy.value_network
        pol.load_state_dict(policy_mod.convert_state_dict(m["weights"]["policy_net"], specs["policy"]))
        val.load_state_dict(policy_mod.convert_state_dict(m["weights"]["value_net"], specs["value"]))
        self.policy.optimizer.load_state_dict(policy_mod.convert_optimizer_state(
            m["policy_optimizer"], [n for n, _ in pol.named_parameters()], specs["policy"]))
        self.policy.value_optimizer.load_state_dict(policy_mod.convert_optimizer_state(
            m["value_optimizer"], [n for n, _ in val.named_parameters()], specs["value"]))
        self.policy_lr = float(m["policy_lr"])
        set_optimizer_lr(self.policy.optimizer, self.policy_lr)
        return {"source": path, "source_update_index": int(m["update_index"]), "policy_lr": self.policy_lr,
                "specs": {k: {n: [sp[0], (sp[1] if isinstance(sp[1], int) else list(sp[1]))] for n, sp in v.items()}
                          for k, v in specs.items()}}

    def training_state(self):
        return {"stage": self.stage, "weights": self.weights(),
                "policy_optimizer": self.policy.optimizer.state_dict(),
                "value_optimizer": self.policy.value_optimizer.state_dict(),
                "policy_lr": self.policy_lr, "update_index": self.update_index,
                "num_timesteps": self.num_timesteps, "n_updates": self._n_updates,
                "c_sil": self.c_sil, "sil_low_kl_streak": self.sil_low_kl_streak}

    def load_training_state(self, state):
        if state["stage"] != self.stage:
            raise ValueError("段階の異なる学習状態は読めない")
        self.load_weights(state["weights"])
        self.policy.optimizer.load_state_dict(state["policy_optimizer"])
        self.policy.value_optimizer.load_state_dict(state["value_optimizer"])
        self.policy_lr = state["policy_lr"]
        set_optimizer_lr(self.policy.optimizer, self.policy_lr)
        self.update_index = state["update_index"]
        self.num_timesteps = state["num_timesteps"]
        self._n_updates = state["n_updates"]
        self.c_sil = state["c_sil"]
        self.sil_low_kl_streak = state["sil_low_kl_streak"]


def policy_weights(policy):
    """SummerPolicy の方策網・価値網の重みの写し（段階を添える）。"""
    return {"stage": policy.stage,
            "policy_net": {k: v.detach().clone() for k, v in policy.policy_network.state_dict().items()},
            "value_net": {k: v.detach().clone() for k, v in policy.value_network.state_dict().items()}}


def transfer_to_stage2(src, dst):
    """第 1 段階の学習器 src の状態を第 2 段階の学習器 dst へ移す（A-7 により V6 では呼ばれない）。"""
    dst.load_weights(src.weights())
    dst.policy.value_optimizer.load_state_dict(src.policy.value_optimizer.state_dict())
    dst.policy_lr = src.policy_lr
    set_optimizer_lr(dst.policy.optimizer, dst.policy_lr)
    dst.update_index = src.update_index
    dst.num_timesteps = src.num_timesteps
    dst._n_updates = src._n_updates


def run_deterministic_episode(policy, vec_env, n_steps=constants.ZONE_STEPS_PER_WEEK):
    """決定論的（argmax）方策で 1 エピソードを評価する（1 組のベクトル環境）。Returns: (終端の info, 行動列 (T,62))"""
    obs = vec_env.reset()
    actions = []
    for _ in range(n_steps):
        masks = np.stack(vec_env.env_method("action_masks"))
        act, _ = policy.predict(obs, deterministic=True, action_masks=masks)
        obs, _rewards, dones, infos = vec_env.step(act)
        actions.append(np.asarray(act[0], dtype=np.int64))
        if dones[0]:
            return infos[0], np.stack(actions)
    raise RuntimeError(f"決定論的評価が {n_steps} ステップで終端しなかった")


def replay_action_sequence(vec_env, actions):
    """保存した行動列をそのまま書き込んで 1 エピソード再生する（指示書 V4-S H-2、V6-S F-1・B-1）。

    行動列は (T, 62) の整数配列。マスクは方策側の仕組みで環境は行動を書き換えないため、
    再生では方策もマスクも使わない。Returns: 終端の info
    """
    actions = np.asarray(actions, dtype=np.int64)
    vec_env.reset()
    for t in range(actions.shape[0]):
        _obs, _rewards, dones, infos = vec_env.step(actions[t][None, :])
        if dones[0]:
            return infos[0]
    raise RuntimeError(f"行動列の再生が {actions.shape[0]} ステップで終端しなかった")
