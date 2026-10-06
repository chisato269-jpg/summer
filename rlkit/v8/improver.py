"""rlkit.v8.improver — 優位度と方策の更新（指示書 第 7 章 L-5〜L-8）。

PPO の代理目的、価値ネットワークによる優位度、エントロピーの項は使わない（L-11）。
  L-5 優位度: C = E + ρ·DR。兄弟 s の優位度 A_s = (C_基準 − C_s) / (E_ref + ρ·DR_ref)。A_s > 0 は区間の変更で良くなったこと。
  L-6 更新: A_s > 0 の兄弟の区間の中の判断（GC-1 で除く次元と選択肢が 1 つの次元を除く）の選択の対数確率を、重み w_s = A_s / max(A_s) で
      増やす。損失は −Σ_s w_s × (区間の判断の対数確率の平均)。基準の軌跡の全判断について、更新前の方策との KL（次元ごとの KL の和の
      ステップ平均）を毎エポックの後に測り、0.01 を超えたら以後のエポックを行わず打ち切る（超えたエポックの更新は残す。PPO の早期終了と同じ）。Adam・学習率 1e-4・最大 8 エポック。
      A_s > 0 の兄弟が無い更新では方策を更新しない。
  L-7 採用率: 更新の後、改善した兄弟の区間の判断の状態で、新しい方策の確率最大の選択が兄弟の選択と一致する割合。
"""
import copy

import numpy as np
import torch

from rlkit.v8 import action as act8
from rlkit.v8 import policy as pol8

LEARNING_RATE = 1e-4
MAX_EPOCHS = 8
KL_LIMIT = 0.01


def advantages(c_base, c_siblings, denominator):
    """A_s = (C_基準 − C_s) / (E_ref + ρ·DR_ref)。"""
    return (float(c_base) - np.asarray(c_siblings, dtype=np.float64)) / float(denominator)


def block_decision_mask(actions, flat_masks, steps, dims):
    """区間の中の判断の bool 配列 (len(steps), 80)。dims の次元、GC-1 で除かれない、選択肢が 2 つ以上、のものだけ True。"""
    steps = np.asarray(steps, dtype=np.int64)
    a = np.asarray(actions)[steps]
    n_opt = act8.n_valid_options(np.asarray(flat_masks)[steps])
    excl = act8.gc1_excluded(a, n_opt)
    in_block = np.zeros(act8.ACTION_DIMS, dtype=bool)
    in_block[list(dims)] = True
    return in_block[None, :] & ~excl


def base_kl(logp_old, logp_new, masks):
    """基準の軌跡の全判断について、次元ごとの KL(旧‖新) の和のステップ平均。logp は (T,372)、無効な選択肢は −inf。"""
    valid = masks
    p_old = torch.where(valid, logp_old.exp(), torch.zeros_like(logp_old))
    diff = torch.where(valid, logp_old - logp_new, torch.zeros_like(logp_old))
    return float((p_old * diff).sum(dim=1).mean())


class PairedImprover:
    def __init__(self, policy, lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, kl_limit=KL_LIMIT):
        self.policy = policy
        self.lr = lr
        self.max_epochs = max_epochs
        self.kl_limit = kl_limit
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def state(self):
        return {"optimizer": self.optimizer.state_dict()}

    def load_state(self, st):
        self.optimizer.load_state_dict(st["optimizer"])
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr

    def update(self, base, siblings, adv):
        """base: dict(obs (T,704), masks (T,372) bool)。siblings: 長さ 12 のリスト。各要素は dict(obs, masks, actions (T,80), steps, dims)。
        adv: 長さ 12 の優位度。戻り値は記録用の辞書。"""
        adv = np.asarray(adv, dtype=np.float64)
        improved = [s for s in range(len(siblings)) if adv[s] > 0.0]
        out = {"n_improved": len(improved), "updated": False, "epochs_applied": 0, "kl_cutoff": False, "kl_per_epoch": [],
               "loss_per_epoch": [], "adoption_rate": None, "n_decisions": 0, "weights": {}, "nonfinite": False}
        if not improved:
            return out
        amax = float(adv[improved].max())
        weights = {s: float(adv[s] / amax) for s in improved}
        out["weights"] = {str(s): w for s, w in weights.items()}
        rows = []
        for s in improved:
            sib = siblings[s]
            dm = block_decision_mask(sib["actions"], sib["masks"], sib["steps"], sib["dims"])
            steps = np.asarray(sib["steps"], dtype=np.int64)
            rows.append({"obs": torch.as_tensor(np.asarray(sib["obs"])[steps], dtype=torch.float32),
                         "masks": torch.as_tensor(np.asarray(sib["masks"])[steps], dtype=torch.bool),
                         "actions": torch.as_tensor(np.asarray(sib["actions"])[steps], dtype=torch.long),
                         "dm": torch.as_tensor(dm), "w": weights[s], "n": int(dm.sum())})
        out["n_decisions"] = int(sum(r["n"] for r in rows))
        base_obs = torch.as_tensor(np.asarray(base["obs"]), dtype=torch.float32)
        base_masks = torch.as_tensor(np.asarray(base["masks"]), dtype=torch.bool)
        self.policy.train()
        with torch.no_grad():
            logp_old = pol8.per_dim_log_softmax(self.policy(base_obs), base_masks)

        def loss_fn():
            total = 0.0
            for r in rows:
                if r["n"] == 0:
                    continue
                lp = pol8.chosen_log_prob(pol8.per_dim_log_softmax(self.policy(r["obs"]), r["masks"]), r["actions"])
                total = total - r["w"] * (lp * r["dm"]).sum() / r["n"]
            return total

        for epoch in range(1, self.max_epochs + 1):
            before = copy.deepcopy(self.policy.state_dict())
            loss = loss_fn()
            if not torch.is_tensor(loss):
                break
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                kl = base_kl(logp_old, pol8.per_dim_log_softmax(self.policy(base_obs), base_masks), base_masks)
            lv = float(loss.detach())
            out["loss_per_epoch"].append(lv)
            out["kl_per_epoch"].append(kl)
            if not np.isfinite(kl) or not np.isfinite(lv):
                self.policy.load_state_dict(before)
                out["nonfinite"] = True
                break
            out["epochs_applied"] = epoch
            if kl > self.kl_limit:                              # このエポックで KL が 0.01 を超えた: 以後のエポックを行わず打ち切る（PPO の早期終了と同じ）
                out["kl_cutoff"] = True
                break
        out["updated"] = out["epochs_applied"] > 0
        # L-7 採用率: 改善した兄弟の区間の判断の状態で、新しい方策の確率最大の選択が兄弟の選択と一致する割合
        self.policy.eval()
        agree, total = 0, 0
        with torch.no_grad():
            for r in rows:
                if r["n"] == 0:
                    continue
                mine = torch.as_tensor(pol8.masked_argmax(self.policy(r["obs"]).numpy(), r["masks"].numpy()))
                agree += int(((mine == r["actions"]) & r["dm"]).sum())
                total += int(r["dm"].sum())
        out["adoption_rate"] = agree / total if total else None
        return out
