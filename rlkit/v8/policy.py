"""rlkit.v8.policy — V8-S の方策ネットワークと、V7 の重みからの中立な初期化（指示書 第 8 章 N-1〜N-3）。

構造は V7（rlkit/policy.py）と同じ符号化器・文脈・共有の頭部で、入力の次元と頭部の出力の数だけが違う。
  観測 704 = 全体 92 ＋ ゾーン 34 × 18
  ゾーン頭部   [zone_emb ‖ ctx] 224 → 18（割合 4 ＋ 設定温度 9 ＋ 風量 3 ＋ リモコン 2）… 18 ゾーン共有
  系統頭部     224 → 9（蒸発温度 14〜22 ℃）… 4 系統共有
  テナント頭部 224 → 6（換気 4 ＋ バイパス 2）… 2 テナント共有
出力は action.NVEC の並び（割合 | 設定温度 | 風量 | 蒸発温度 | 換気 | リモコン | バイパス）に連結した 372 ロジット。
V7 の重みから、確率最大の選択が V7 の方策と同じになるように初期化する（新しい選択肢の行は最も近い既存の行の複写で偏り −4、
新しい頭部は重み 0 で偏り「許可」・「使わない」0、「禁止」・「使う」−4、新しい入力の列は 0）。
"""
import numpy as np
import torch
import torch.nn as nn

from rlkit import constants, policy as p7
from rlkit.v8 import action as act8
from rlkit.v8 import observation as obs8

N_ZONES = constants.N_ZONES
ZONE_DIM = obs8.ZONE_DIM
GLOBAL_DIM = obs8.GLOBAL_DIM
ZONE_EMBED_DIM = p7.ZONE_EMBED_DIM
GLOBAL_EMBED_DIM = p7.GLOBAL_EMBED_DIM
ZONE_AGG_DIM = p7.ZONE_AGG_DIM
CONTEXT_DIM = p7.CONTEXT_DIM
HEAD_IN_DIM = p7.HEAD_IN_DIM
ZONE_HEAD_OUT = 4 + 9 + 3 + 2
SYSTEM_HEAD_OUT = 9
TENANT_HEAD_OUT = 4 + 2
NEW_ROW_BIAS = -4.0                       # 新しい選択肢の、複写元に対する偏り（相対の確率 e^-4 ≈ 0.018）
SYSTEM_OF_ZONE = list(act8.ZONE_SYSTEM)
TENANT_OF_ZONE = list(act8.ZONE_TENANT)


class _Trunk:
    def _build_trunk(self, extra_context_dim=0):
        self.zone_encoder = p7._encoder(ZONE_DIM, 64, ZONE_EMBED_DIM)
        self.global_encoder = p7._encoder(GLOBAL_DIM, 96, GLOBAL_EMBED_DIM)
        self.context = nn.Sequential(nn.Linear(ZONE_AGG_DIM + GLOBAL_EMBED_DIM + extra_context_dim, CONTEXT_DIM), nn.Tanh())
        p7._init_linear(self.context[0], p7.TANH_GAIN)
        self.register_buffer("system_of_zone", torch.tensor(SYSTEM_OF_ZONE, dtype=torch.long), persistent=False)
        self.register_buffer("tenant_of_zone", torch.tensor(TENANT_OF_ZONE, dtype=torch.long), persistent=False)

    def _trunk(self, obs, extra_context=None):
        B = obs.shape[0]
        zone_x = obs[:, GLOBAL_DIM:].reshape(B * N_ZONES, ZONE_DIM)
        zone_emb = self.zone_encoder(zone_x).reshape(B, N_ZONES, ZONE_EMBED_DIM)
        global_emb = self.global_encoder(obs[:, :GLOBAL_DIM])
        parts = [zone_emb.mean(dim=1), zone_emb.max(dim=1).values, global_emb]
        if extra_context is not None:
            parts.append(extra_context)
        return zone_emb, self.context(torch.cat(parts, dim=-1))

    @staticmethod
    def _with_context(per_item, ctx):
        return torch.cat([per_item, ctx.unsqueeze(1).expand(-1, per_item.shape[1], -1)], dim=-1)


class PolicyNetworkV8(_Trunk, nn.Module):
    def __init__(self):
        super().__init__()
        self._build_trunk()
        self.zone_head = nn.Linear(HEAD_IN_DIM, ZONE_HEAD_OUT)
        self.system_head = nn.Linear(HEAD_IN_DIM, SYSTEM_HEAD_OUT)
        self.tenant_head = nn.Linear(HEAD_IN_DIM, TENANT_HEAD_OUT)
        for head in (self.zone_head, self.system_head, self.tenant_head):
            p7._init_linear(head, p7.POLICY_HEAD_GAIN)

    def forward(self, obs):
        """obs (B,704) → NVEC の並びのロジット (B, 372)。"""
        B = obs.shape[0]
        zone_emb, ctx = self._trunk(obs)
        zl = self.zone_head(self._with_context(zone_emb, ctx))                                  # (B,18,18)
        system_mean = torch.stack([zone_emb[:, self.system_of_zone == s, :].mean(dim=1) for s in range(4)], dim=1)
        sl = self.system_head(self._with_context(system_mean, ctx))                             # (B,4,9)
        tenant_mean = torch.stack([zone_emb[:, self.tenant_of_zone == t, :].mean(dim=1) for t in range(2)], dim=1)
        tl = self.tenant_head(self._with_context(tenant_mean, ctx))                             # (B,2,6)
        return torch.cat([zl[:, :, 0:4].reshape(B, -1), zl[:, :, 4:13].reshape(B, -1), zl[:, :, 13:16].reshape(B, -1),
                          sl.reshape(B, -1), tl[:, :, 0:4].reshape(B, -1), zl[:, :, 16:18].reshape(B, -1),
                          tl[:, :, 4:6].reshape(B, -1)], dim=-1)


class ValueNetworkV8(_Trunk, nn.Module):
    """価値ネットワーク（V8 では更新しない。L-11）。V7 と同じ構造で、入力の次元だけが違う。重みは V7 から読み込んだまま。"""

    def __init__(self):
        super().__init__()
        self._build_trunk(extra_context_dim=constants.VALUE_REWARD_CONTEXT_DIM)
        self.zone_value_head = nn.Linear(HEAD_IN_DIM, 1)
        self.system_value_head = nn.Linear(CONTEXT_DIM, 1)
        self.register_buffer("reward_context", torch.zeros(constants.VALUE_REWARD_CONTEXT_DIM), persistent=False)

    def forward(self, obs):
        extra = self.reward_context.to(obs.dtype).unsqueeze(0).expand(obs.shape[0], -1)
        zone_emb, ctx = self._trunk(obs, extra_context=extra)
        zone_values = self.zone_value_head(self._with_context(zone_emb, ctx)).squeeze(-1)
        return torch.cat([zone_values, self.system_value_head(ctx)], dim=-1)


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


# =====================================================================
# 中立な初期化（N-1〜N-3）
# =====================================================================

def _pad_cols(w, n_new):
    return torch.cat([w, torch.zeros(w.shape[0], n_new, dtype=w.dtype)], dim=1)


def _trunk_state(old):
    """符号化器・文脈の重み。入力層の新しい列は 0（N-3）。"""
    new = {k: v.clone() for k, v in old.items()}
    new["zone_encoder.0.weight"] = _pad_cols(old["zone_encoder.0.weight"], ZONE_DIM - obs8.ZONE_DIM_V7)
    new["global_encoder.0.weight"] = _pad_cols(old["global_encoder.0.weight"], GLOBAL_DIM - obs8.GLOBAL_DIM_V7)
    return new


def convert_policy_state(old):
    """V7 の PolicyNetwork の state_dict → V8 の PolicyNetworkV8 の state_dict。"""
    new = _trunk_state({k: v for k, v in old.items() if not k.startswith(("zone_head", "system_head", "tenant_head"))})
    zw, zb = old["zone_head.weight"], old["zone_head.bias"]          # 行: 運転状態 2（0 停止・1 冷房）、設定温度 7、風量 3
    n_state, n_sp = 2, len(act8.V7_SETPOINT_GRID)
    W = torch.zeros(ZONE_HEAD_OUT, HEAD_IN_DIM, dtype=zw.dtype)
    b = torch.zeros(ZONE_HEAD_OUT, dtype=zb.dtype)
    # 割合 0・1/3・2/3・1（行 0〜3）: 0 は旧「停止」、1 は旧「運転」、1/3・2/3 は旧「運転」の複写（偏り −4）
    for j, (src, shift) in enumerate([(0, 0.0), (1, NEW_ROW_BIAS), (1, NEW_ROW_BIAS), (1, 0.0)]):
        W[j], b[j] = zw[src], zb[src] + shift
    # 設定温度（行 4〜12、22・22.5・23・23.5・24・25・26・27・28 ℃）: V7 の格子の値は同じ温度の行、22.5・23.5 は 22・23 の複写（偏り −4）
    for j, g in enumerate(act8.SETPOINT_GRID):
        if g in act8.V7_SETPOINT_GRID:
            src, shift = n_state + act8.V7_SETPOINT_GRID.index(g), 0.0
        else:
            src, shift = n_state + act8.V7_SETPOINT_GRID.index(float(int(g))), NEW_ROW_BIAS
        W[4 + j], b[4 + j] = zw[src], zb[src] + shift
    for j in range(3):                                               # 風量（行 13〜15）
        W[13 + j], b[13 + j] = zw[n_state + n_sp + j], zb[n_state + n_sp + j]
    # リモコン（行 16・17）: 重み 0、偏り 許可 0・禁止 −4（N-2）
    b[17] = NEW_ROW_BIAS
    new["zone_head.weight"], new["zone_head.bias"] = W, b
    sw, sb = old["system_head.weight"], old["system_head.bias"]       # 蒸発温度 15〜22 ℃（8 行）
    W = torch.zeros(SYSTEM_HEAD_OUT, HEAD_IN_DIM, dtype=sw.dtype)
    b = torch.zeros(SYSTEM_HEAD_OUT, dtype=sb.dtype)
    W[0], b[0] = sw[0], sb[0] + NEW_ROW_BIAS                          # 14 ℃ は 15 ℃の複写（偏り −4）
    W[1:], b[1:] = sw, sb
    new["system_head.weight"], new["system_head.bias"] = W, b
    tw, tb = old["tenant_head.weight"], old["tenant_head.bias"]       # 換気 4 行
    W = torch.zeros(TENANT_HEAD_OUT, HEAD_IN_DIM, dtype=tw.dtype)
    b = torch.zeros(TENANT_HEAD_OUT, dtype=tb.dtype)
    W[:4], b[:4] = tw, tb
    b[5] = NEW_ROW_BIAS                                               # バイパス: 重み 0、偏り 使わない 0・使う −4（N-2）
    new["tenant_head.weight"], new["tenant_head.bias"] = W, b
    return new


def convert_value_state(old):
    """V7 の ValueNetwork の state_dict → V8。入力層の新しい列は 0。"""
    return _trunk_state(old)


def build_from_v7(entry):
    """trial08 の best/best_official.pt の entry（weights に policy_net・value_net）から、初期化した (方策網, 価値網) を作る。"""
    pol, val = PolicyNetworkV8(), ValueNetworkV8()
    pol.load_state_dict(convert_policy_state(entry["weights"]["policy_net"]))
    val.load_state_dict(convert_value_state(entry["weights"]["value_net"]), strict=False)
    return pol, val


# =====================================================================
# 確率最大の選択（マスク付き）と対数確率
# =====================================================================

def masked_argmax(logits, masks):
    """logits (B,372)、masks (B,372) bool → (B,80) 次元ごとの有効な選択肢の中の確率最大の索引（同点は先頭）。"""
    logits = np.asarray(logits)
    masks = np.asarray(masks, dtype=bool)
    out = np.zeros((logits.shape[0], act8.ACTION_DIMS), dtype=np.int64)
    for i, (o, n) in enumerate(zip(act8.OFFSETS, act8.NVEC)):
        seg = np.where(masks[:, o:o + n], logits[:, o:o + n], -np.inf)
        out[:, i] = seg.argmax(axis=1)
    return out


def per_dim_log_softmax(logits, masks):
    """logits (B,372)（torch）、masks (B,372) bool（torch）→ 無効な選択肢を除いた次元ごとの対数確率 (B,372)（無効は −inf）。"""
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(masks, logits, torch.full_like(logits, neg))
    out = torch.empty_like(masked)
    for o, n in zip(act8.OFFSETS.tolist(), act8.NVEC.tolist()):
        out[:, o:o + n] = torch.log_softmax(masked[:, o:o + n], dim=-1)
    return torch.where(masks, out, torch.full_like(out, float("-inf")))


def chosen_log_prob(logp, actions):
    """logp (B,372)、actions (B,80) long → 選んだ選択肢の対数確率 (B,80)。"""
    idx = torch.as_tensor(act8.OFFSETS, dtype=torch.long).unsqueeze(0) + actions
    return torch.gather(logp, 1, idx)
