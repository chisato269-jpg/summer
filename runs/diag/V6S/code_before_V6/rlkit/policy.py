"""rlkit.policy — 方策ネットワークと価値ネットワーク（v2 第 5-7 節・第 5-9 節・第 5-10 節）。

方策側
  全体ブロック (B,80) → 全体符号化器 80→96→128 [tanh] → global_emb (B,128)
  ゾーンブロック (B,18,31) → ゾーン共有符号化器 31→64→96 [tanh] → zone_emb (B,18,96)
  [mean(zone_emb) ‖ max(zone_emb) ‖ global_emb] (B,320) → 文脈 320→128 [tanh] → ctx_pi
  ゾーンヘッド [zone_emb ‖ ctx_pi] (B,18,224) → 13（第 1 段階）／19（第 2 段階）… 18 ゾーン共有
  系統ヘッド   [系統内ゾーン平均 ‖ ctx_pi] (B,4,224) → 8 … 4 系統共有
  換気ヘッド   ctx_pi 128 → 4 → (B,2,2)
価値側（方策と重みを共有しない、同じ構造の符号化器と文脈）
  ゾーン価値ヘッド [zone_emb_v ‖ ctx_v] 224 → 1 … 18 ゾーン共有 → 18 出力
  系統価値ヘッド   ctx_v 128 → 1 → 1 出力

初期化（v2 第 5-7 節）: 符号化器・文脈は直交 gain=5/3、方策ヘッドは直交 gain=0.01、価値ヘッドは直交 gain=1.0、
バイアスは 0。正規化層・ドロップアウトは使わない。
"""
import torch
import torch.nn as nn

from rlkit import action as action_mod
from rlkit import constants

GLOBAL_DIM = constants.GLOBAL_BLOCK_DIM
ZONE_DIM = constants.ZONE_BLOCK_DIM
N_ZONES = constants.N_ZONES
N_SYSTEMS = 4
ZONE_EMBED_DIM = 96
GLOBAL_EMBED_DIM = 128
ZONE_AGG_DIM = 2 * ZONE_EMBED_DIM
CONTEXT_DIM = 128
HEAD_IN_DIM = ZONE_EMBED_DIM + CONTEXT_DIM  # 224

TANH_GAIN = nn.init.calculate_gain("tanh")  # 5/3
POLICY_HEAD_GAIN = 0.01
VALUE_HEAD_GAIN = 1.0

SYSTEM_OF_ZONE_INDEX = [ou - 1 for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS]


def _init_linear(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def _encoder(in_dim, hidden_dim, out_dim):
    net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, out_dim), nn.Tanh())
    for m in net:
        if isinstance(m, nn.Linear):
            _init_linear(m, TANH_GAIN)
    return net


class _Trunk:
    """符号化器と文脈（方策側と価値側でそれぞれ別の重みを持つ）。"""

    def _build_trunk(self):
        self.zone_encoder = _encoder(ZONE_DIM, 64, ZONE_EMBED_DIM)
        self.global_encoder = _encoder(GLOBAL_DIM, 96, GLOBAL_EMBED_DIM)
        self.context = nn.Sequential(nn.Linear(ZONE_AGG_DIM + GLOBAL_EMBED_DIM, CONTEXT_DIM), nn.Tanh())
        _init_linear(self.context[0], TANH_GAIN)
        self.register_buffer("system_of_zone", torch.tensor(SYSTEM_OF_ZONE_INDEX, dtype=torch.long), persistent=False)

    def _trunk(self, obs):
        B = obs.shape[0]
        zone_x = obs[:, GLOBAL_DIM:].reshape(B * N_ZONES, ZONE_DIM)
        zone_emb = self.zone_encoder(zone_x).reshape(B, N_ZONES, ZONE_EMBED_DIM)
        global_emb = self.global_encoder(obs[:, :GLOBAL_DIM])
        agg = torch.cat([zone_emb.mean(dim=1), zone_emb.max(dim=1).values, global_emb], dim=-1)
        return zone_emb, self.context(agg)

    def _with_context(self, per_item, ctx):
        return torch.cat([per_item, ctx.unsqueeze(1).expand(-1, per_item.shape[1], -1)], dim=-1)


class PolicyNetwork(_Trunk, nn.Module):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.n_state = len(action_mod.zone_states(stage))   # S-5: 第 1 段階は 2 択
        self.n_setpoint = len(action_mod.setpoint_grid(stage))
        self.n_fan = len(constants.ZONE_FAN_SPEED)
        self._build_trunk()
        self.zone_head = nn.Linear(HEAD_IN_DIM, self.n_state + self.n_setpoint + self.n_fan)
        self.system_head = nn.Linear(HEAD_IN_DIM, len(constants.SYSTEM_EVAPORATING_C))
        self.vent_head = nn.Linear(CONTEXT_DIM, action_mod.N_TENANTS * len(constants.TENANT_VENT_ON))
        for head in (self.zone_head, self.system_head, self.vent_head):
            _init_linear(head, POLICY_HEAD_GAIN)

    def forward(self, obs):
        """obs (B,638) → action.nvec(stage) の並びに連結したロジット (B, 270 または 378)。"""
        B = obs.shape[0]
        zone_emb, ctx = self._trunk(obs)
        zl = self.zone_head(self._with_context(zone_emb, ctx))  # (B,18,K)
        system_mean = torch.stack([zone_emb[:, self.system_of_zone == s, :].mean(dim=1) for s in range(N_SYSTEMS)], dim=1)
        sl = self.system_head(self._with_context(system_mean, ctx))  # (B,4,8)
        vl = self.vent_head(ctx)  # (B,4) = (B,2,2) を平坦化した並び
        a, b = self.n_state, self.n_state + self.n_setpoint
        return torch.cat([zl[:, :, :a].reshape(B, -1), zl[:, :, a:b].reshape(B, -1), zl[:, :, b:].reshape(B, -1),
                          sl.reshape(B, -1), vl.reshape(B, -1)], dim=-1)


class ValueNetwork(_Trunk, nn.Module):
    def __init__(self):
        super().__init__()
        self._build_trunk()
        self.zone_value_head = nn.Linear(HEAD_IN_DIM, 1)
        self.system_value_head = nn.Linear(CONTEXT_DIM, 1)
        _init_linear(self.zone_value_head, VALUE_HEAD_GAIN)
        _init_linear(self.system_value_head, VALUE_HEAD_GAIN)

    def forward(self, obs):
        """obs (B,638) → 19 出力（ゾーン 18 ＋ 系統 1）。"""
        zone_emb, ctx = self._trunk(obs)
        zone_values = self.zone_value_head(self._with_context(zone_emb, ctx)).squeeze(-1)  # (B,18)
        return torch.cat([zone_values, self.system_value_head(ctx)], dim=-1)


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def expand_zone_head_to_stage2(src_head, dst_head):
    """第 2 段階へ移るときのゾーンヘッドの拡張（A-1、および指示書 V5-S S-5）。

    運転状態 2 択 → 3 択。停止・冷房の行はそのまま写し、新しい「送風」の行は「停止」の行から複写する
    （診断 K-6 で送風と停止の E・DR が 16 進一致したため、停止の確率を引き継ぐのが正しい事前分布になる）。
    設定温度 7 択 → 13 択。偶数番の新しい選択肢は同じ温度の既存の行をそのまま写し、奇数番（0.5 ℃）の
    新しい選択肢は隣接する既存の 2 行の平均で初期化する（ロジットは行の一次式なので、ロジットも平均になる）。
    """
    s1 = constants.ZONE_STATE_BY_STAGE[1]
    s2 = constants.ZONE_STATE_BY_STAGE[2]
    n1 = len(constants.ZONE_SETPOINT_C[1])
    n2 = len(constants.ZONE_SETPOINT_C[2])
    n_fan = len(constants.ZONE_FAN_SPEED)
    assert src_head.out_features == len(s1) + n1 + n_fan
    assert dst_head.out_features == len(s2) + n2 + n_fan
    with torch.no_grad():
        W1, b1 = src_head.weight, src_head.bias
        W2 = torch.empty_like(dst_head.weight)
        b2 = torch.empty_like(dst_head.bias)
        for j, name in enumerate(s2):
            src = s1.index(name) if name in s1 else s1.index("off")   # 送風は停止から複写（S-5）
            W2[j], b2[j] = W1[src], b1[src]
        for j in range(n2):
            lo, hi = len(s1) + (j // 2), len(s1) + ((j + 1) // 2)
            W2[len(s2) + j] = 0.5 * (W1[lo] + W1[hi])
            b2[len(s2) + j] = 0.5 * (b1[lo] + b1[hi])
        W2[len(s2) + n2:], b2[len(s2) + n2:] = W1[len(s1) + n1:], b1[len(s1) + n1:]
        dst_head.weight.copy_(W2)
        dst_head.bias.copy_(b2)


def transfer_policy_stage1_to_stage2(src_net, dst_net):
    """第 1 段階の方策ネットワークの重みを第 2 段階のネットワークへ写す（ゾーンヘッドのみ拡張）。"""
    state = {k: v for k, v in src_net.state_dict().items() if not k.startswith("zone_head.")}
    missing, unexpected = dst_net.load_state_dict(state, strict=False)
    assert not unexpected and sorted(missing) == ["zone_head.bias", "zone_head.weight"], (missing, unexpected)
    expand_zone_head_to_stage2(src_net.zone_head, dst_net.zone_head)
