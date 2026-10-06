"""rlkit.v9.nets — V9-S の方策網・価値網の新規の初期化（指示書 V9-S C9-3）。

方策網は V8 の PolicyNetworkV8（符号化器・文脈は直交 gain=5/3、頭部は直交 gain=0.01、偏り 0）をそのまま使い、
新しい選択肢の出力の偏りだけを −4 にする（相対の確率 e^−4 ≈ 0.018）。
  ゾーン頭部（18 出力）  運転の割合 [0:4] の 1/3・2/3（索引 1・2）、設定温度 [4:13] の 22.5・23.5 ℃（格子の索引 1・3）、
                         リモコン [16:18] の「禁止」（索引 1）
  系統頭部（9 出力）     蒸発温度 14 ℃（索引 0）
  テナント頭部（6 出力） バイパス [4:6] の「使う」（索引 1）
それ以外の偏りは 0。頭部の重みが gain=0.01 と小さいので、初期のロジットはほぼ偏りだけで決まる。
価値網は V8 の ValueNetworkV8 に、V7 の ValueNetwork と同じ初期化（価値頭部は直交 gain=1.0、偏り 0）と、
文脈 (ρ_n, D_n) を与える set_reward_context を足したもの（V8 の価値網は V7 の重みを読むので初期化を持たない）。
"""
import numpy as np
import torch
import torch.nn as nn

from rlkit import policy as p7
from rlkit.v8 import action as act8
from rlkit.v8 import policy as p8

NEW_OPTION_BIAS = -4.0
ZONE_HEAD_OFFSET = {"duty": 0, "setpoint": 4, "fan": 13, "remote": 16}      # ゾーン頭部の出力の並び（PolicyNetworkV8.forward の切り出し）
TENANT_HEAD_OFFSET = {"vent": 0, "bypass": 4}


def new_option_rows():
    """偏り −4 を置く出力の行: {頭部名: [行の索引]}。"""
    sp = act8.SETPOINT_GRID
    zone = ([ZONE_HEAD_OFFSET["duty"] + 1, ZONE_HEAD_OFFSET["duty"] + 2]
            + [ZONE_HEAD_OFFSET["setpoint"] + sp.index(22.5), ZONE_HEAD_OFFSET["setpoint"] + sp.index(23.5)]
            + [ZONE_HEAD_OFFSET["remote"] + act8.REMOTE_PROHIBIT])
    system = [act8.EVAP_GRID.index(14)]
    tenant = [TENANT_HEAD_OFFSET["bypass"] + act8.BYPASS_ON]
    return {"zone_head": zone, "system_head": system, "tenant_head": tenant}


def apply_new_option_bias(policy, bias=NEW_OPTION_BIAS):
    rows = new_option_rows()
    with torch.no_grad():
        for head_name, idx in rows.items():
            getattr(policy, head_name).bias[idx] = bias
    return rows


def new_option_logit_indices():
    """372 ロジットの中で、新しい選択肢に当たる位置（試験・記録用）。bool (372,)。"""
    out = np.zeros(act8.ACTION_LOGITS, dtype=bool)
    sp = act8.SETPOINT_GRID
    for g, offs in (("zone_duty", [1, 2]), ("zone_setpoint", [sp.index(22.5), sp.index(23.5)]),
                    ("zone_remote", [act8.REMOTE_PROHIBIT]), ("system_evap", [act8.EVAP_GRID.index(14)]),
                    ("tenant_bypass", [act8.BYPASS_ON])):
        s = act8.LAYOUT[g][0]
        for d in range(s, s + act8.LAYOUT[g][1]):
            for o in offs:
                out[int(act8.OFFSETS[d]) + o] = True
    return out


class ValueNetworkV9(p8.ValueNetworkV8):
    def __init__(self):
        super().__init__()
        p7._init_linear(self.zone_value_head, p7.VALUE_HEAD_GAIN)
        p7._init_linear(self.system_value_head, p7.VALUE_HEAD_GAIN)

    def set_reward_context(self, rho_n, D_n):
        self.reward_context.copy_(torch.tensor([rho_n, D_n], dtype=self.reward_context.dtype))


def build_policy_net():
    net = p8.PolicyNetworkV8()
    apply_new_option_bias(net)
    return net


def build_value_net():
    net = ValueNetworkV9()
    for prm in net.parameters():
        prm.requires_grad_(True)
    return net
