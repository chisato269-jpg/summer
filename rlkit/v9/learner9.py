"""rlkit.v9.learner9 — V9-S 段階 A の PPO（指示書 C9-2〜C9-4、第 5-2 節）。

trial08 の学習器 rlkit.learner.SummerPPO（sb3-contrib MaskablePPO を継承）を、V8 の行動空間（80 次元・372 ロジット）と観測（704 次元）に
当てはめたもの。学習則の数値（方策 4 エポック・価値 8 エポック・ミニバッチ・KL の目標と打ち切り・学習率の範囲と調整・クリップ・
勾配ノルム・λ_hour）は rlkit.constants の trial08 の値のまま使う。変えたのは次の点だけ。
  網        方策網・価値網は rlkit.v9.nets（新規の初期化。新しい選択肢の偏り −4）
  信用割当  rlkit.v9.credit（C9-4）
  GC-1      rlkit.v8.action.gc1_excluded（割合 0 のゾーンの設定温度・風量、運転ゾーンの無い系統の蒸発温度）と、1 択の判断を除く
  エントロピー 群ごとに c_g = 0.002 / ln K_g を、段階 A の全期間で一定（trial08 の減衰を使わない）
  SIL       使わない（trial08 と同じ）
採取（collect_rollouts）・GAE・R-8'・価値の学習は SummerPPO のものを継承する。
"""
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from rlkit import advantage as adv_mod
from rlkit import constants
from rlkit.learner import (SummerPPO, SummerPolicy, _grad_norm, adapt_learning_rate, approx_kl_from, clipped_policy_loss,
                           compute_group_entropy, exclusion_counts, per_dim_entropy, per_dim_log_prob, set_optimizer_lr)
from rlkit.v8 import action as act8
from rlkit.v9 import credit, nets
from sb3_contrib import MaskablePPO

ENTROPY_C0 = constants.ENTROPY_C0                  # 0.002（段階 A の全期間で一定）


class _ExtractorV9(nn.Module):
    """SB3 の MlpExtractor の形で、V9 の方策網と価値網を持つ。"""

    def __init__(self):
        super().__init__()
        self.policy_net = nets.build_policy_net()
        self.value_net = nets.build_value_net()
        self.latent_dim_pi = act8.ACTION_LOGITS
        self.latent_dim_vf = constants.N_ZONES + 1

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class SummerPolicyV9(SummerPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = _ExtractorV9()


def occupied_off_fraction(actions, occupants):
    """在室者 1 人以上のゾーン・ステップのうち、割合 0（停止）を選んだ割合（行動を選んだ時点の在室者数）。"""
    occ = np.asarray(occupants) >= 1
    off = act8.duty_indices(np.asarray(actions)) == act8.DUTY_OFF
    n = int(occ.sum())
    return {"fraction": float((off & occ).sum() / n) if n else None, "n_occupied_zone_steps": n}


def entropy_coefficients(group_layout):
    """c_g = c0 / ln K_g（c0 = 0.002 で一定）。"""
    return {g: ENTROPY_C0 / info["lnK"] for g, info in group_layout.items()}


class SummerPPOV9(SummerPPO):
    def __init__(self, env, planned_updates, seed, n_steps_per_env, verbose=0):
        if planned_updates is None:
            raise ValueError("planned_updates が設定されていない")
        self.stage = 1
        self.planned_updates = int(planned_updates)
        self.group_layout = credit.group_layout()
        self.update_index = 0
        self.policy_lr = constants.POLICY_LR_INIT
        self.c_sil = 0.0
        self.sil_low_kl_streak = 0
        self.last_update_record = None
        self.last_terminal_infos = []
        self._rollout = None
        MaskablePPO.__init__(
            self, SummerPolicyV9, env, learning_rate=constants.POLICY_LR_INIT, n_steps=n_steps_per_env,
            batch_size=constants.POLICY_MINIBATCH_STEPS, n_epochs=constants.POLICY_EPOCHS, gamma=constants.GAMMA,
            gae_lambda=constants.LAMBDA_HOUR, clip_range=constants.CLIP_RANGE, normalize_advantage=True,
            ent_coef=0.0, vf_coef=0.0, max_grad_norm=constants.MAX_GRAD_NORM, target_kl=constants.TARGET_KL,
            policy_kwargs={"stage": 1}, seed=seed, verbose=verbose)

    def train(self, sil_buffer=None, sil_mode="off", reference=None):
        """1 回の更新の学習（SummerPPO.train の V8 版。SIL は使わない）。"""
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

        n_options = th.as_tensor(act8.n_valid_options(ro["action_masks"]), device=dev)
        gc1 = th.as_tensor(act8.gc1_excluded(np.asarray(ro["actions"])), device=dev)
        one_option = n_options < 2
        include = (~gc1) & (~one_option)
        per_dim_raw = credit.assign_per_dim_advantages(ro["advantages19"][:, :constants.N_ZONES],
                                                       ro["advantages19"][:, constants.N_ZONES], ro["actions"])
        adv_norm, level_stats, _sigma = adv_mod.normalize_by_level(per_dim_raw, include.cpu().numpy(),
                                                                   level_of_dim=credit.dim_levels())
        per_dim_adv = th.as_tensor(adv_norm, device=dev).float()
        returns19 = th.as_tensor(ro["returns19"], device=dev).float()
        inc_f = include.float()

        with th.no_grad():
            dist0 = policy.masked_distribution(obs, masks)
            old_log_prob = per_dim_log_prob(dist0, actions)
            entropy0 = per_dim_entropy(dist0)
        H_g = compute_group_entropy(entropy0, include, self.group_layout)
        c0 = ENTROPY_C0
        c_g = entropy_coefficients(self.group_layout)

        kl_per_epoch, pg_losses, ent_losses, clip_fractions = [], [], [], []
        epochs_applied, early_stop = 0, False
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
        excl = exclusion_counts(gc1, one_option, self.group_layout)
        losses = pg_losses + ent_losses + value_losses
        self.last_update_record = {
            "update": self.update_index, "stage": self.stage, "phase": "A",
            "approx_kl_per_epoch": kl_per_epoch, "approx_kl_after": kl_after, "kl_early_stop": early_stop,
            "epochs_applied": epochs_applied, "policy_lr_before": lr_before, "policy_lr_after": self.policy_lr,
            "entropy_c0": c0, "c_g": dict(c_g), "approx_kl_by_group": kl_by_group,
            "H_g": {g: (float(v) if v is not None else None) for g, v in H_g.items()},
            "H_g_over_lnK": {g: (float(v) / self.group_layout[g]["lnK"] if v is not None else None) for g, v in H_g.items()},
            "gc1_excluded_by_group": excl["gc1"], "one_option_excluded_by_group": excl["one_option"],
            "n_included_elements": int(include.sum()),
            "policy_loss_mean": float(np.mean(pg_losses)) if pg_losses else None,
            "entropy_loss_mean": float(np.mean(ent_losses)) if ent_losses else None,
            "clip_fraction_mean": float(np.mean(clip_fractions)) if clip_fractions else None,
            "grad_norm_policy": None, "grad_norm_entropy": None, "grad_norm_ratio": None,
            "floor_fraction": None, "floor_fraction_by_group": None,
            "value_loss_first": value_losses[0], "value_loss_last": value_losses[-1],
            "explained_variance_19": float(1.0 - np.var(G - V) / np.var(G)) if np.var(G) > 0 else None,
            "per_dim_advantage_raw_std": float(per_dim_raw.std()),
            "credit_levels": dict(credit.LEVEL_OF_GROUP), "lambda_hour": constants.LAMBDA_HOUR,
            "advantage_norm_by_level": level_stats,
            "advantage_norm_mean": None, "advantage_norm_std": None,
            "occupied_zone_step_off_fraction": occupied_off_fraction(ro["actions"], ro["occupants"]),
            "r8p": ro["r8p"], "n_samples": int(n), "n_segments": len(ro["segments"]),
            "sil": {"mode": "off"},
            "nonfinite": (not all(math.isfinite(x) for x in losses)) or (not self.parameters_finite()),
        }
        self._rollout = None
        return self.last_update_record
