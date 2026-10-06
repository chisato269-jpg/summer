"""rlkit.learner — 学習アルゴリズム（v2 第 5-5 節）。sb3-contrib MaskablePPO を継承する（現行の基礎を維持）。

1 回の更新（v2 第 5-5-1 節）のうち、本モジュールが受け持つのは次の手順である。
  1 採取        collect_rollouts: N 組 × 2 周のエピソードを集め、info の zone_rewards・step_seconds を記録
  3 GAE         19 出力について、エピソード境界で区切った時間基準 λ の GAE
  4 優位度      信用割当の階層で 60 次元へ割り当て、行列の全要素で正規化
  5 方策        最大 4 エポック。各エポックの開始時に全ステップの approx_kl（次元合計・ステップ平均）を
                計算し、1.5 × target_kl を超えたらそのエポックの勾配を適用せずに打ち切る
  6 学習率      更新後の approx_kl に応じて方策の学習率を調整
  7 価値        分離した最適化器で 8 エポック（KL 制約なし）
  8 記録        last_update_record
手順 2（決定論的評価）と基準点の更新は rlkit/train_summer.py が行う。
"""
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
from rlkit.policy import PolicyNetwork, ValueNetwork, transfer_policy_stage1_to_stage2

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
    """(T, ロジット数) のマスクから、各 (ステップ, 次元) で選べる選択肢数 (T, 60)。"""
    return th.stack([action_masks[:, s:s + w].sum(dim=1) for s, w in action_mod.dim_logit_offsets(stage)], dim=1)


def compute_group_entropy(entropy, n_options, group_layout):
    """非縮退（選択肢 2 以上）の (ステップ, 次元) だけで群ごとの平均エントロピー（現行と同じ）。"""
    out = {}
    for g, info in group_layout.items():
        s, n = info["start"], info["n"]
        valid = (n_options[:, s:s + n] >= 2).to(entropy.dtype)
        out[g] = None if valid.sum() == 0 else (entropy[:, s:s + n] * valid).sum() / valid.sum()
    return out


def floor_fraction_at(update_index, planned_updates, group=None):
    """floor_fraction: 0.30 から 0.05 へ、計画更新数の 60% までに線形減衰し、以後 0.05（v2 第 5-5-5 節）。

    group を渡すと、S-6（指示書 V5-S 第 2-5 節）で群別に定めた床があればそれを返す。
    換気（tenant_vent）は状態に応じて決まるべき量なので 0.10 で一定とする。
    """
    if group is not None and group in constants.ENTROPY_FLOOR_BY_GROUP:
        return constants.ENTROPY_FLOOR_BY_GROUP[group]
    horizon = constants.FLOOR_DECAY_END_FRACTION * planned_updates
    progress = min(update_index / horizon, 1.0) if horizon > 0 else 1.0
    return constants.FLOOR_FRACTION_START + (constants.FLOOR_FRACTION_END - constants.FLOOR_FRACTION_START) * progress


def dual_ascent_update(c_g, H_g, group_layout, floor_fraction, c_min, c_max, eta=constants.DUAL_ASCENT_ETA):
    """群別エントロピー係数の双対上昇（現行 learner_per_zone.dual_ascent_update_c_g と同じ式）。
    log c_g ← clip(log c_g + η·(φ_g − H_g)/lnK_g, log c_min, log c_max)、φ_g = floor_fraction·lnK_g"""
    new = dict(c_g)
    for g, info in group_layout.items():
        H = H_g.get(g)
        if H is None:
            continue
        H = float(H)
        phi = floor_fraction[g] if isinstance(floor_fraction, dict) else floor_fraction
        log_c = math.log(c_g[g]) + eta * (phi * info["lnK"] - H) / info["lnK"]
        new[g] = math.exp(min(max(log_c, math.log(c_min)), math.log(c_max)))
    return new


def adapt_learning_rate(lr, approx_kl):
    """更新後の approx_kl が 2×target_kl を超えたら 1/1.5 倍、0.5×target_kl 未満なら 1.5 倍（範囲 [1e-5, 1e-3]）。"""
    if approx_kl > constants.LR_ADAPT_HIGH * constants.TARGET_KL:
        lr = lr / constants.LR_ADAPT_FACTOR
    elif approx_kl < constants.LR_ADAPT_LOW * constants.TARGET_KL:
        lr = lr * constants.LR_ADAPT_FACTOR
    return min(max(lr, constants.POLICY_LR_MIN), constants.POLICY_LR_MAX)


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


class _RolloutCapture(BaseCallback):
    """採取中、毎ステップの全組の info["zone_rewards"]・info["step_seconds"] と、終端した組の info を集める。"""

    def __init__(self):
        super().__init__()
        self.zone_rewards, self.step_seconds, self.terminal_infos = [], [], []

    def _on_step(self):
        infos = self.locals["infos"]
        self.zone_rewards.append(np.stack([np.asarray(inf["zone_rewards"], dtype=np.float64) for inf in infos]))
        self.step_seconds.append(np.array([inf["step_seconds"] for inf in infos], dtype=np.float64))
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.terminal_infos.append((len(self.zone_rewards) - 1, i, infos[i]))
        return True


# =====================================================================
# 学習器
# =====================================================================

class SummerPPO(MaskablePPO):
    def __init__(self, env, stage, entropy_coef_init, planned_updates, seed, n_steps_per_env, verbose=0):
        if entropy_coef_init is None:
            raise ValueError("未確定項目 U-1: 群別エントロピー係数の初期値（entropy_coef_init）が設定されていない")
        if planned_updates is None:
            raise ValueError("未確定項目 J-6: 計画更新数（planned_updates）が設定されていない")
        self.stage = stage
        self.entropy_coef_init = float(entropy_coef_init)
        self.planned_updates = int(planned_updates)
        self.c_min = self.entropy_coef_init / constants.C_MIN_DIVISOR
        self.c_max = self.entropy_coef_init * constants.C_MAX_MULTIPLIER
        self.group_layout = action_mod.group_layout(stage)
        self.c_g = {g: self.entropy_coef_init for g in constants.ENTROPY_GROUPS}
        self.update_index = 0
        self.policy_lr = constants.POLICY_LR_INIT
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
        zone_rewards = flat(np.stack(capture.zone_rewards))  # (N·T,18)
        rewards = flat(rollout_buffer.rewards).reshape(-1)
        system_rewards = rewards - zone_rewards.sum(axis=1)
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
        }
        return ok

    # ---- 手順 4〜8 ----

    def _approx_kl_by_group(self, obs, actions, masks, old_log_prob):
        """群ごとの KL 寄与（その群の次元の対数比の合計で測る。指示書 V5-S S-7）。"""
        with th.no_grad():
            log_prob = per_dim_log_prob(self.policy.masked_distribution(obs, masks), actions)
            diff = log_prob - old_log_prob
            out = {}
            for g, info in self.group_layout.items():
                s, n = info["start"], info["n"]
                lr_g = diff[:, s:s + n].sum(dim=1)
                out[g] = float(th.mean((th.exp(lr_g) - 1) - lr_g))
            return out

    def _approx_kl(self, obs, actions, masks, old_log_prob):
        with th.no_grad():
            log_prob = per_dim_log_prob(self.policy.masked_distribution(obs, masks), actions).sum(dim=1)
            log_ratio = log_prob - old_log_prob.sum(dim=1)
            return float(th.mean((th.exp(log_ratio) - 1) - log_ratio))

    def train(self):
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

        per_dim_raw = adv_mod.assign_per_dim_advantages(ro["advantages19"][:, :N_ZONES], ro["advantages19"][:, N_ZONES],
                                                        ro["actions"])
        per_dim_adv = th.as_tensor(adv_mod.normalize_matrix(per_dim_raw), device=dev).float()
        returns19 = th.as_tensor(ro["returns19"], device=dev).float()

        with th.no_grad():
            dist0 = policy.masked_distribution(obs, masks)
            old_log_prob = per_dim_log_prob(dist0, actions)
            entropy0 = per_dim_entropy(dist0)
        n_options = compute_n_options(masks, self.stage)
        H_g = compute_group_entropy(entropy0, n_options, self.group_layout)
        floor = floor_fraction_at(self.update_index, self.planned_updates)
        # S-6: 群ごとの床（tenant_vent だけ 0.10 で一定、他は既定の減衰）
        floors = {g: floor_fraction_at(self.update_index, self.planned_updates, group=g) for g in self.group_layout}
        self.c_g = dual_ascent_update(self.c_g, H_g, self.group_layout, floors, self.c_min, self.c_max)

        kl_per_epoch, pg_losses, ent_losses, clip_fractions = [], [], [], []
        epochs_applied, early_stop = 0, False
        # 指示書 V3-S E-2: 指定した更新番号の第 1 エポックの最初のミニバッチで、方策損失とエントロピー損失
        # （係数込み）の勾配ノルムの比を計測して記録する（報告用の量であり、判定には使わない）。
        measure_grad_ratio = (self.update_index + 1) in constants.GRAD_NORM_RATIO_MEASURE_AT
        grad_norm_policy = grad_norm_entropy = None
        for _epoch in range(constants.POLICY_EPOCHS):
            kl = self._approx_kl(obs, actions, masks, old_log_prob)
            kl_per_epoch.append(kl)
            if kl > constants.KL_STOP_FACTOR * constants.TARGET_KL:
                early_stop = True
                break
            perm = np.random.permutation(n)
            for start in range(0, n, constants.POLICY_MINIBATCH_STEPS):
                idx = th.as_tensor(perm[start:start + constants.POLICY_MINIBATCH_STEPS], device=dev)
                dist = policy.masked_distribution(obs[idx], masks[idx])
                ratio = th.exp(per_dim_log_prob(dist, actions[idx]) - old_log_prob[idx])
                adv = per_dim_adv[idx]
                clipped = th.clamp(ratio, 1 - constants.CLIP_RANGE, 1 + constants.CLIP_RANGE)
                policy_loss = -th.min(adv * ratio, adv * clipped).sum(dim=1).mean()
                H_mb = compute_group_entropy(per_dim_entropy(dist), n_options[idx], self.group_layout)
                terms = [self.c_g[g] * H for g, H in H_mb.items() if H is not None]
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
                clip_fractions.append(float(th.mean((th.abs(ratio - 1) > constants.CLIP_RANGE).float())))
            epochs_applied += 1

        kl_after = self._approx_kl(obs, actions, masks, old_log_prob)
        lr_before = self.policy_lr
        self.policy_lr = adapt_learning_rate(self.policy_lr, kl_after)
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

        self.update_index += 1
        self._n_updates += epochs_applied
        G, V = ro["returns19"], ro["values19"]
        self.last_update_record = {
            "update": self.update_index, "stage": self.stage,
            "approx_kl_per_epoch": kl_per_epoch, "approx_kl_after": kl_after, "kl_early_stop": early_stop,
            "epochs_applied": epochs_applied, "policy_lr_before": lr_before, "policy_lr_after": self.policy_lr,
            "floor_fraction": floor, "floor_fraction_by_group": dict(floors), "c_g": dict(self.c_g),
            "approx_kl_by_group": self._approx_kl_by_group(obs, actions, masks, old_log_prob),
            "H_g": {g: (float(v) if v is not None else None) for g, v in H_g.items()},
            "H_g_over_lnK": {g: (float(v) / self.group_layout[g]["lnK"] if v is not None else None) for g, v in H_g.items()},
            "policy_loss_mean": float(np.mean(pg_losses)) if pg_losses else None,
            "entropy_loss_mean": float(np.mean(ent_losses)) if ent_losses else None,
            "clip_fraction_mean": float(np.mean(clip_fractions)) if clip_fractions else None,
            "grad_norm_policy": grad_norm_policy, "grad_norm_entropy": grad_norm_entropy,
            "grad_norm_ratio": (grad_norm_entropy / grad_norm_policy
                                if grad_norm_policy not in (None, 0.0) and grad_norm_entropy is not None else None),
            "value_loss_first": value_losses[0], "value_loss_last": value_losses[-1],
            "explained_variance_19": float(1.0 - np.var(G - V) / np.var(G)) if np.var(G) > 0 else None,
            "per_dim_advantage_raw_std": float(per_dim_raw.std()),
            "n_samples": int(n), "n_segments": len(ro["segments"]),
        }
        self._rollout = None
        return self.last_update_record

    # ---- 重み・状態の受け渡し ----

    def weights(self):
        return policy_weights(self.policy)

    def load_weights(self, weights):
        """最良の重みの再読み込み（v2 第 5-5-6 節）。第 1 段階の重みを第 2 段階の学習器へ読むときはゾーンヘッドを拡張する。"""
        self.policy.value_network.load_state_dict(weights["value_net"])
        if weights["stage"] == self.stage:
            self.policy.policy_network.load_state_dict(weights["policy_net"])
        elif weights["stage"] == 1 and self.stage == 2:
            src = PolicyNetwork(1)
            src.load_state_dict(weights["policy_net"])
            transfer_policy_stage1_to_stage2(src, self.policy.policy_network)
        else:
            raise ValueError(f"段階 {weights['stage']} の重みを段階 {self.stage} の学習器へは読めない")

    def training_state(self):
        return {"stage": self.stage, "weights": self.weights(),
                "policy_optimizer": self.policy.optimizer.state_dict(),
                "value_optimizer": self.policy.value_optimizer.state_dict(),
                "policy_lr": self.policy_lr, "c_g": dict(self.c_g), "update_index": self.update_index,
                "num_timesteps": self.num_timesteps, "n_updates": self._n_updates}

    def load_training_state(self, state):
        if state["stage"] != self.stage:
            raise ValueError("段階の異なる学習状態は読めない")
        self.load_weights(state["weights"])
        self.policy.optimizer.load_state_dict(state["policy_optimizer"])
        self.policy.value_optimizer.load_state_dict(state["value_optimizer"])
        self.policy_lr = state["policy_lr"]
        set_optimizer_lr(self.policy.optimizer, self.policy_lr)
        self.c_g = dict(state["c_g"])
        self.update_index = state["update_index"]
        self.num_timesteps = state["num_timesteps"]
        self._n_updates = state["n_updates"]


def policy_weights(policy):
    """SummerPolicy の方策網・価値網の重みの写し（段階を添える）。"""
    return {"stage": policy.stage,
            "policy_net": {k: v.detach().clone() for k, v in policy.policy_network.state_dict().items()},
            "value_net": {k: v.detach().clone() for k, v in policy.value_network.state_dict().items()}}


def transfer_to_stage2(src, dst):
    """第 1 段階の学習器 src の状態を第 2 段階の学習器 dst へ移す（A-1）。

    方策網はゾーンヘッドを拡張して写す。価値網と価値の最適化器の状態、方策の学習率、群別係数、更新番号は引き継ぐ。
    方策の最適化器は、ゾーンヘッドの形が変わるため新しく作り直した状態から始める（設計書に規定が無いための補完）。
    """
    dst.load_weights(src.weights())
    dst.policy.value_optimizer.load_state_dict(src.policy.value_optimizer.state_dict())
    dst.policy_lr = src.policy_lr
    set_optimizer_lr(dst.policy.optimizer, dst.policy_lr)
    dst.c_g = dict(src.c_g)
    dst.update_index = src.update_index
    dst.num_timesteps = src.num_timesteps
    dst._n_updates = src._n_updates


def run_deterministic_episode(policy, vec_env, n_steps=constants.ZONE_STEPS_PER_WEEK):
    """決定論的（argmax）方策で 1 エピソードを評価する（1 組のベクトル環境）。Returns: (終端の info, 行動列 (T,60))"""
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
    """保存した行動列をそのまま書き込んで 1 エピソード再生する（指示書 V4-S H-2）。

    行動列は (T, 60) の整数配列。マスクは方策側の仕組みで環境は行動を書き換えないため、
    再生では方策もマスクも使わない。Returns: 終端の info
    """
    actions = np.asarray(actions, dtype=np.int64)
    vec_env.reset()
    for t in range(actions.shape[0]):
        _obs, _rewards, dones, infos = vec_env.step(actions[t][None, :])
        if dones[0]:
            return infos[0]
    raise RuntimeError(f"行動列の再生が {actions.shape[0]} ステップで終端しなかった")
