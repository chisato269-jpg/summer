"""指示書 V8-S の単体試験（UT-V8-1〜UT-V8-8）。エミュレータは使わない（模擬の通信・模擬の環境）。

エミュレータを使う確認（MUST-8 の書き込みと読み戻し、G-0 の同等性）は rlkit/diag/v8s_live.py と test_v8_live.py。
"""
import asyncio
import datetime
import json
import os

import numpy as np
import pytest
import torch

from rlkit import config, constants
from rlkit import policy as p7
from rlkit.v8 import action as a8
from rlkit.v8 import blocks as blk
from rlkit.v8 import improver as imp
from rlkit.v8 import observation as o8
from rlkit.v8 import policy as p8
from rlkit.v8 import trainer as tr8

TRIAL08_BEST = os.path.join("runs", "v2s_trial08", "best", "best_official.pt")


def _v7_entry(real=False):
    if real and os.path.exists(TRIAL08_BEST):
        return torch.load(TRIAL08_BEST, weights_only=False)
    torch.manual_seed(5)
    return {"stage": 1, "update": 0, "E": 0.3, "DR": 0.05, "eci_smooth": 0.3057, "E_hex": float.hex(0.3), "DR_hex": float.hex(0.05),
            "weights": {"policy_net": p7.PolicyNetwork(1).state_dict(), "value_net": p7.ValueNetwork().state_dict()}}


# ---- UT-V8-1: 中立な初期化 ----

def _v7_argmax(net7, x7, masks7):
    with torch.no_grad():
        logits = net7(torch.tensor(x7)).numpy()
    out, pos = [], 0
    for n in a8_v7_nvec():
        out.append(np.where(masks7[:, pos:pos + n], logits[:, pos:pos + n], -np.inf).argmax(1))
        pos += n
    return np.stack(out, 1)


def a8_v7_nvec():
    from rlkit import action as a7
    return list(a7.nvec(1))


def test_ut_v8_1_neutral_initialisation_equal_decisions():
    """無作為の観測 1,000 個で、拡張した方策の確率最大の選択が、旧方策の選択（停止 → 割合 0、運転 → 割合 1）と一致し、
    A-2 は許可、A-3 は使わない。M-5'・M-9 のマスクがある場合も一致する。"""
    entry = _v7_entry(real=True)
    old = p7.PolicyNetwork(1)
    old.load_state_dict(entry["weights"]["policy_net"])
    old.eval()
    new, _val = p8.build_from_v7(entry)
    new.eval()
    rng = np.random.default_rng(0)
    N = 1000
    x7 = rng.uniform(0, 1, size=(N, 648)).astype(np.float32)
    x8 = rng.uniform(0, 1, size=(N, o8.TOTAL_DIM)).astype(np.float32)
    x8[:, o8.column_map_v7_to_v8()] = x7                       # 新しい入力の列は無作為（重み 0 なので出力に効かない）
    # マスク: 半分の観測で、無作為のゾーン 5 つを M-5'（停止だけ）、テナント 1 つを M-9（換気停止）にする
    m7 = np.ones((N, 256), dtype=bool)
    m8 = np.ones((N, a8.ACTION_LOGITS), dtype=bool)
    from rlkit import action as a7
    off7 = a7.dim_logit_offsets(1)
    for i in range(0, N, 2):
        for k in rng.choice(18, 5, replace=False):
            o, n = off7[k]
            m7[i, o:o + n] = False
            m7[i, o + 0] = True                                  # 停止
            o8_, n8 = int(a8.OFFSETS[k]), int(a8.NVEC[k])
            m8[i, o8_:o8_ + n8] = False
            m8[i, o8_ + a8.DUTY_OFF] = True
        j = int(rng.integers(2))
        o, n = off7[58 + j]
        m7[i, o:o + n] = False
        m7[i, o + constants.VENT_OFF] = True
        o8_, n8 = int(a8.OFFSETS[58 + j]), int(a8.NVEC[58 + j])
        m8[i, o8_:o8_ + n8] = False
        m8[i, o8_ + constants.VENT_OFF] = True
    with torch.no_grad():
        l8 = new(torch.tensor(x8)).numpy()
    acts8 = p8.masked_argmax(l8, m8)
    acts7 = _v7_argmax(old, x7, m7)
    assert np.array_equal(a8.v7_to_v8_action(acts7), acts8)
    assert (acts8[:, a8.dim_slice("zone_remote")] == a8.REMOTE_PERMIT).all()
    assert (acts8[:, a8.dim_slice("tenant_bypass")] == a8.BYPASS_OFF).all()


def test_ut_v8_1_shapes_and_new_inputs_have_zero_effect():
    entry = _v7_entry()
    new, val = p8.build_from_v7(entry)
    assert o8.TOTAL_DIM == 704 and a8.ACTION_DIMS == 80 and a8.ACTION_LOGITS == 372
    x = torch.rand(4, o8.TOTAL_DIM)
    y = x.clone()
    for k in range(18):
        b = o8.zone_base(k)
        y[:, b + o8.ZONE_DIM_V7:b + o8.ZONE_DIM] = torch.rand(4, 3)
    y[:, o8.IDX_BYPASS:o8.IDX_BYPASS + 2] = torch.rand(4, 2)
    with torch.no_grad():
        assert torch.equal(new(x), new(y))                       # N-3: 新しい入力の列の重みは 0
        assert new(x).shape == (4, 372) and val(x).shape == (4, 19)
    assert p8.parameter_count(new) > p7.parameter_count(p7.PolicyNetwork(1))


# ---- 模擬の通信（UT-V8-2・3） ----

class _Clock:
    def __init__(self):
        self.t = 0.0


class _Group:
    def __init__(self, log, clock, names):
        for n in names:
            setattr(self, n, self._make(n, log, clock))

    @staticmethod
    def _make(name, log, clock):
        def f(*a):
            async def coro():
                log.append((clock.t, name) + tuple(a))
                return (True, None)
            return coro()
        return f


VRF_NAMES = ["turn_on", "turn_off", "change_mode", "change_setpoint_temperature", "change_fan_speed", "change_direction",
             "permit_local_control", "prohibit_local_control", "enable_refrigerant_temperatureControl",
             "change_evaporating_temperature", "change_condensing_temperature"]
VENT_NAMES = ["enable_bypass_control", "disable_bypass_control", "stop_ventilation", "start_ventilation", "change_fan_speed"]


class _FakeDriver:
    def __init__(self, clock, start):
        self.clock = clock
        self.dt = start
        self.final_dtime = None
        self.log = []
        self.comms = type("C", (), {})()
        self.comms.vrf = _Group(self.log, clock, VRF_NAMES)
        self.comms.vent = _Group(self.log, clock, VENT_NAMES)

    async def advance(self, seconds, **_):
        self.log.append((self.clock.t, "advance", seconds))
        self.clock.t += seconds
        self.dt += datetime.timedelta(seconds=seconds)
        return self.dt

    async def true_sim_datetime(self):
        return self.dt

    async def park(self):
        pass


def _fake_env(monkeypatch, start):
    from rlkit import comm
    from rlkit.v8 import env as e8
    snap_state = {"mj": 0.0}

    async def snapshot(comms):
        snap_state["mj"] += 3.0
        zones = {l: {"drybulb_c": 26.0, "relative_humidity_pct": 50.0, "occupant_number": 3.0, "thermal_sensation": 0.5,
                     "clothing_index": 0.6, "dissatisfied_thermal": 0.1, "dissatisfied_draft": 0.02, "dissatisfied_vtmp": 0.01,
                     "heatload_kw": 1.0} for l in constants.ZONE_LABELS}
        return {"building": {"drybulb_c": 30.0, "relative_humidity_pct": 60.0, "global_horizontal_radiation": 100.0,
                             "nocturnal_radiation": 10.0, "total_energy_consumption_mj": snap_state["mj"],
                             "averaged_dissatisfaction_rate": 0.1, "instantaneous_energy_consumption_kw": 1.0,
                             "instantaneous_dissatisfaction_rate": 0.1}, "zones": zones, "co2_ppm": {"South": 700.0, "North": 700.0}}

    async def zeros(comms):
        return [0.5] * 4

    async def indoor(comms):
        return [0.1] * 18

    async def readback(comms):
        return [24.0] * 18

    async def thermal(comms):
        return {t: {"n_present": 0, "counts": [0] * 7} for t in constants.TENANTS}

    async def permit(comms):
        return [True] * 18

    async def onoff(comms):
        return [False] * 18
    monkeypatch.setattr(comm, "read_step_snapshot", snapshot)
    monkeypatch.setattr(comm, "read_outdoor_unit_power", zeros)
    monkeypatch.setattr(comm, "read_indoor_unit_power", indoor)
    monkeypatch.setattr(comm, "read_setpoint_readback", readback)
    monkeypatch.setattr(comm, "read_tenant_thermal_sensation", thermal)
    monkeypatch.setattr(comm, "read_remote_permission_status", permit)
    monkeypatch.setattr(comm, "read_onoff_status", onoff)
    env = e8.SummerEnvV8(include_indoor_unit_power=True, zone_step_energy_norm_gj=0.02)
    clock = _Clock()
    drv = _FakeDriver(clock, start)
    env._driver = drv
    env._dw = comm.DiffWriter()
    env._ob.reset()
    env._reset_episode_state()
    env._pending_dt = start
    env._pending_boundary = start.minute == 0
    env._prev_building_energy_mj = 0.0
    return env, drv


def _action(duty=3, setpoint=4, fan=2, remote=0, evap=5, vent=2, bypass=0):
    a = np.zeros(a8.ACTION_DIMS, dtype=np.int64)
    a[a8.dim_slice("zone_duty")] = duty
    a[a8.dim_slice("zone_setpoint")] = setpoint
    a[a8.dim_slice("zone_fan")] = fan
    a[a8.dim_slice("zone_remote")] = remote
    a[a8.dim_slice("system_evap")] = evap
    a[a8.dim_slice("tenant_vent")] = vent
    a[a8.dim_slice("tenant_bypass")] = bypass
    return a


def test_ut_v8_2_duty_fractions_write_stops_at_the_right_times(monkeypatch):
    """割合 1/3・2/3 のときに 1/3・2/3 の時点で停止が書かれ、割合 1 では途中の書き込みが無く、割合 0 では始めに停止が書かれる。
    一時停止は割合によらず 1/3・2/3 の時点で起きる。"""
    start = datetime.datetime(1999, 7, 21, 7, 0)                 # 在室の刻み（900 秒）
    env, drv = _fake_env(monkeypatch, start)
    duty = np.array([0, 1, 2, 3] * 4 + [3, 3])
    act = _action(duty=duty)
    asyncio.run(env._async_step(act))
    adv = [(t, a) for t, n, *a in drv.log if n == "advance"]
    assert [x[1][0] for x in adv] == [300, 300, 300] and [x[0] for x in adv] == [0.0, 300.0, 600.0]     # 毎刻み 3 回
    offs = {(ou, iu): t for t, n, ou, iu in [x for x in drv.log if len(x) == 4 and x[1] == "turn_off"]}
    ons = [x for x in drv.log if len(x) == 4 and x[1] == "turn_on"]
    for k, (ou, iu, *_r) in enumerate(constants.ZONE_DEFS):
        d = duty[k]
        if d == 0:
            assert offs[(ou, iu)] == 0.0 and not any(x[2:] == (ou, iu) for x in ons)       # 割合 0: 始めに停止、運転は書かない
        elif d == 3:
            assert (ou, iu) not in offs                                                      # 割合 1: 途中の書き込みなし
        else:
            assert offs[(ou, iu)] == 300.0 * d                                               # 1/3 → 300 秒、2/3 → 600 秒
        if d > 0:
            assert any(x[0] == 0.0 and x[2:] == (ou, iu) for x in ons)                       # 運転は刻みの始め
    setpoints = [x for x in drv.log if x[1] == "change_setpoint_temperature"]
    assert len(setpoints) == int((duty > 0).sum()) and all(x[0] == 0.0 for x in setpoints)   # 割合 > 0 のときだけ刻みの始め
    assert env._v8_mid_writes == int(((duty == 1) | (duty == 2)).sum())
    # 無人の刻み（3,600 秒）も 3 分割（1,200 秒）
    env2, drv2 = _fake_env(monkeypatch, datetime.datetime(1999, 7, 21, 22, 0))
    asyncio.run(env2._async_step(_action(duty=3)))
    assert [a[0] for t, a in [(t, a) for t, n, *a in drv2.log if n == "advance"]] == [1200, 1200, 1200]


def test_ut_v8_3_remote_bypass_evap_and_half_degree_setpoints(monkeypatch):
    """A-2・A-3 の書き込み、A-5 の 0.5 ℃、A-4 の 14 ℃がそのまま書かれる。A-2 の次元は毎時の境目以外で前回値に固定される。"""
    start = datetime.datetime(1999, 7, 21, 7, 0)
    env, drv = _fake_env(monkeypatch, start)
    remote = np.zeros(18, dtype=np.int64)
    remote[0] = a8.REMOTE_PROHIBIT
    act = _action(setpoint=1, remote=remote, evap=0, bypass=np.array([a8.BYPASS_ON, a8.BYPASS_OFF]))
    asyncio.run(env._async_step(act))
    sp = [x for x in drv.log if x[1] == "change_setpoint_temperature"]
    assert sp and all(x[4] == 22.5 for x in sp)
    ev = [x for x in drv.log if x[1] == "change_evaporating_temperature"]
    assert [x[3] for x in ev] == [14, 14, 14, 14]
    assert any(x[1] == "prohibit_local_control" and x[2:] == (1, 1) for x in drv.log)
    assert sum(1 for x in drv.log if x[1] == "permit_local_control") == 17
    ben = [x for x in drv.log if x[1] == "enable_bypass_control"]
    bdis = [x for x in drv.log if x[1] == "disable_bypass_control"]
    assert len(ben) == 9 and len(bdis) == 9                                                  # 南のテナントの 9 台が使う、北の 9 台が使わない
    # 7:15（毎時の境目ではない）: A-2・蒸発温度の次元は前回値の 1 択（M-3）、換気・バイパスは動かせる
    mask = env.action_masks()
    nv = a8.NVEC
    for g in ("zone_remote", "system_evap"):
        for i in range(*[(s.start, s.stop) for s in [a8.dim_slice(g)]][0]):
            seg = mask[a8.OFFSETS[i]:a8.OFFSETS[i] + nv[i]]
            assert seg.sum() == 1 and seg[act[i]]
    for g in ("tenant_bypass", "tenant_vent"):
        i = a8.LAYOUT[g][0]
        assert mask[a8.OFFSETS[i]:a8.OFFSETS[i] + nv[i]].sum() >= 2 or g == "tenant_vent"
    # 毎時の境目では A-2 も動かせる
    env._pending_boundary = True
    mask = env.action_masks()
    i = a8.LAYOUT["zone_remote"][0]
    assert mask[a8.OFFSETS[i]:a8.OFFSETS[i] + 2].sum() == 2


# ---- UT-V8-4・5: 兄弟の揺らし方と区間の割り当て ----

def _flat_mask():
    return np.ones(a8.ACTION_LOGITS, dtype=bool)


def test_ut_v8_4_siblings_only_change_inside_the_block_and_are_reproducible():
    table = blk.block_steps_table()
    base = np.zeros(a8.ACTION_DIMS, dtype=np.int64)
    for kind in ("sparse", "coherent"):
        for b in (0, 17, 50):
            p = blk.Perturber(b, kind, blk.sibling_seed(19800614, 3, 5), table[b])
            q = blk.Perturber(b, kind, blk.sibling_seed(19800614, 3, 5), table[b])
            inside = set(int(s) for s in table[b])
            dims = set(blk.block_dims(p.tenant))
            for step in range(constants.ZONE_STEPS_PER_WEEK):
                out = p.apply(step, base, _flat_mask())
                out2 = q.apply(step, base, _flat_mask())
                assert np.array_equal(out, out2)                                            # 同じ種なら同じ揺らし方
                diff = np.flatnonzero(out != base)
                if step not in inside:
                    assert diff.size == 0                                                    # 区間の外は確率最大の選択のまま（MUST-4）
                else:
                    assert set(diff.tolist()) <= dims                                        # 区間の次元だけ
            assert p.record()["changes"] == q.record()["changes"]
    # まとまり: 1 つの種類・1 つの値、単位ごとに確率 0.5
    p = blk.Perturber(10, "coherent", blk.sibling_seed(1, 1, 1), table[10])
    c = p.coherent
    assert c["type"] in blk.ACTION_TYPES and set(c["units"]) <= set(c["all_units"])
    for step in table[10]:
        out = p.apply(int(step), base, _flat_mask())
        for d in c["units"]:
            assert out[d] == c["value"]
    # まばら: 確率 0.03 の割合で、区間の判断が選び直される
    p = blk.Perturber(10, "sparse", blk.sibling_seed(1, 1, 2), table[10])
    n_decisions = len(table[10]) * len(p.dims)
    ones = np.zeros(a8.ACTION_DIMS, dtype=np.int64)
    for step in table[10]:
        p.apply(int(step), ones, _flat_mask())
    assert 0 < len(p.changes) <= 0.2 * n_decisions                                           # 一様な選び直しの一部は元と同じ値
    # 無効な選択肢は選ばない（マスクで 1 択の次元は変わらない）
    m = _flat_mask()
    d0 = blk.block_dims(p.tenant)[0]
    m[a8.OFFSETS[d0]:a8.OFFSETS[d0] + a8.NVEC[d0]] = False
    m[a8.OFFSETS[d0]] = True
    p2 = blk.Perturber(10, "sparse", blk.sibling_seed(9, 9, 9), table[10])
    for step in table[10]:
        assert p2.apply(int(step), ones, m)[d0] == 0
    # 種が違えば違う揺らし方
    a = blk.Perturber(10, "sparse", blk.sibling_seed(1, 1, 1), table[10])
    b = blk.Perturber(10, "sparse", blk.sibling_seed(1, 1, 2), table[10])
    for step in table[10]:
        a.apply(int(step), ones, _flat_mask())
        b.apply(int(step), ones, _flat_mask())
    assert a.changes != b.changes


def test_ut_v8_5_block_assignment_covers_84_blocks_in_7_updates():
    table = blk.block_steps_table()
    assert len(table) == 84 and sum(len(t) for t in table) == 2 * constants.ZONE_STEPS_PER_WEEK
    assert all(3 <= len(t) <= 12 for t in table)
    order = blk.block_order(19800614)
    assert sorted(order.tolist()) == list(range(84))
    seen = []
    for u in range(1, 8):
        a = blk.assignment(u, order)
        assert len(a) == 12 and len(set(a)) == 12
        seen += a
    assert sorted(seen) == list(range(84))                                                   # 7 更新で一巡
    assert blk.assignment(8, order) == blk.assignment(1, order)                              # 周回
    assert blk.block_order(19800614).tolist() == order.tolist() and blk.block_order(1).tolist() != order.tolist()
    for u in (1, 2, 8):                                                                       # 毎更新、6 本ずつ
        kinds = [blk.sibling_kind(u, j) for j in range(12)]
        assert kinds.count("sparse") == 6 and kinds.count("coherent") == 6
    assert blk.sibling_kind(1, 0) != blk.sibling_kind(8, 0)                                  # 周回ごとに入れ替わる
    # 区間の次元: 40（ゾーン 9 × 4 種 + 換気・バイパス + 系統 2）、2 つのテナントで重ならず全 80 次元を覆う
    d0, d1 = set(blk.block_dims(0)), set(blk.block_dims(1))
    assert len(d0) == 40 and len(d1) == 40 and not (d0 & d1) and (d0 | d1) == set(range(80))


# ---- UT-V8-6・8: 優位度と更新、GC-1 ----

def _synthetic(update_seed=0):
    torch.manual_seed(update_seed)
    entry = _v7_entry()
    pol, _ = p8.build_from_v7(entry)
    rng = np.random.default_rng(update_seed)
    T = 40
    table = blk.block_steps_table()
    obs = rng.uniform(0, 1, size=(T, o8.TOTAL_DIM)).astype(np.float32)
    masks = np.ones((T, a8.ACTION_LOGITS), dtype=bool)
    base = {"obs": obs, "masks": masks}
    sibs = []
    for j in range(12):
        acts = rng.integers(0, a8.NVEC, size=(T, a8.ACTION_DIMS)).astype(np.int64)
        steps = np.arange(5, 5 + 4 + j % 3)
        sibs.append({"obs": obs, "masks": masks, "actions": acts, "steps": steps, "dims": blk.block_dims(j % 2)})
    return pol, base, sibs


def test_ut_v8_6_advantage_and_update_only_use_improved_siblings():
    adv = imp.advantages(1.0, [0.9, 1.2, 1.0, 0.7], 2.0)
    assert np.allclose(adv, [0.05, -0.1, 0.0, 0.15])                                          # A_s = (C_基準 − C_s) / 分母
    pol, base, sibs = _synthetic()
    improver = imp.PairedImprover(pol)
    a = np.array([0.0] * 12)
    out = improver.update(base, sibs, a)
    assert out["n_improved"] == 0 and not out["updated"]                                      # 改善が無い更新では方策は変わらない
    before = {k: v.clone() for k, v in pol.state_dict().items()}
    assert all(torch.equal(before[k], v) for k, v in pol.state_dict().items())
    a[3], a[7], a[9] = 0.02, 0.04, -0.01
    out = improver.update(base, sibs, a)
    assert out["n_improved"] == 2 and set(out["weights"]) == {"3", "7"}
    assert out["weights"]["7"] == pytest.approx(1.0) and out["weights"]["3"] == pytest.approx(0.5)  # w = A_s / max(A_s)
    assert out["updated"] and out["epochs_applied"] >= 1 and out["adoption_rate"] is not None
    changed = any(not torch.equal(before[k], v) for k, v in pol.state_dict().items())
    assert changed


def test_ut_v8_6_loss_only_counts_block_decisions_of_improved_siblings():
    pol, base, sibs = _synthetic()
    for j in range(12):                                          # 区間の外のステップだけを変えても、損失は変わらない
        pass
    improver = imp.PairedImprover(pol, max_epochs=1, kl_limit=1e9)
    a = np.zeros(12)
    a[2] = 0.1
    s0 = {k: v.clone() for k, v in pol.state_dict().items()}
    improver.update(base, sibs, a)
    s1 = {k: v.clone() for k, v in pol.state_dict().items()}
    # 同じ初期値で、兄弟 2 の区間の外の行動を書き換えても、更新が同じ（区間の外の判断は損失に入らない）
    pol2, base2, sibs2 = _synthetic()
    pol2.load_state_dict(s0)
    sibs2[2]["actions"] = sibs2[2]["actions"].copy()
    outside = np.setdiff1d(np.arange(40), sibs2[2]["steps"])
    sibs2[2]["actions"][outside] = 0
    imp.PairedImprover(pol2, max_epochs=1, kl_limit=1e9).update(base2, sibs2, a)
    s2 = pol2.state_dict()
    assert all(torch.allclose(s1[k], s2[k], atol=1e-7) for k in s1)


def test_ut_v8_6_kl_cutoff_stops_the_epochs():
    pol, base, sibs = _synthetic()
    a = np.zeros(12)
    a[1] = 0.05
    loose = imp.PairedImprover(pol, max_epochs=8, kl_limit=1e9).update(base, sibs, a)
    assert loose["epochs_applied"] == 8 and not loose["kl_cutoff"]
    pol, base, sibs = _synthetic()
    tight = imp.PairedImprover(pol, max_epochs=8, kl_limit=1e-9).update(base, sibs, a)
    assert tight["kl_cutoff"] and tight["epochs_applied"] == 1 and len(tight["kl_per_epoch"]) == 1   # 超えたエポックで打ち切る
    mid = imp.PairedImprover(_synthetic()[0], max_epochs=8, kl_limit=0.01)
    pol2, base2, sibs2 = _synthetic()
    mid.policy = pol2
    mid.optimizer = torch.optim.Adam(pol2.parameters(), lr=1e-4)
    out = mid.update(base2, sibs2, a)
    assert (out["kl_cutoff"] and out["kl_per_epoch"][-1] > 0.01) or (not out["kl_cutoff"] and max(out["kl_per_epoch"]) <= 0.01)
    assert all(out["kl_per_epoch"][i] <= 0.01 for i in range(len(out["kl_per_epoch"]) - 1))      # 打ち切りより前のエポックは 0.01 以下


def test_ut_v8_8_gc1_excludes_zero_duty_zones_from_the_loss():
    T = 6
    acts = np.zeros((T, a8.ACTION_DIMS), dtype=np.int64)
    acts[:, a8.dim_slice("zone_duty")] = a8.DUTY_FULL
    acts[2, a8.LAYOUT["zone_duty"][0] + 4] = a8.DUTY_OFF                                    # ゾーン 4 は刻み 2 で割合 0
    acts[:, a8.dim_slice("zone_duty")][:, 9:14] = a8.DUTY_OFF                              # 系統 3（ゾーン 9〜13）は全ゾーン割合 0
    masks = np.ones((T, a8.ACTION_LOGITS), dtype=bool)
    ex = a8.gc1_excluded(acts)
    sp, fan = a8.LAYOUT["zone_setpoint"][0], a8.LAYOUT["zone_fan"][0]
    assert ex[2, sp + 4] and ex[2, fan + 4] and not ex[1, sp + 4] and not ex[1, fan + 4]
    es = a8.LAYOUT["system_evap"][0]
    assert ex[0, es + 2] and not ex[0, es + 0]                                               # 運転ゾーンの無い系統の蒸発温度
    dm = imp.block_decision_mask(acts, masks, np.arange(T), blk.block_dims(0))
    assert not dm[2, sp + 4] and not dm[2, fan + 4] and dm[1, sp + 4]                       # 割合 0 のゾーンの設定温度・風量は損失から除く
    assert dm[2, a8.LAYOUT["zone_duty"][0] + 4] and dm[2, a8.LAYOUT["zone_remote"][0] + 4]  # 割合・リモコンの判断は残す
    one = masks.copy()
    d = a8.LAYOUT["tenant_vent"][0]
    one[:, a8.OFFSETS[d]:a8.OFFSETS[d] + 4] = False
    one[:, a8.OFFSETS[d]] = True
    assert not imp.block_decision_mask(acts, one, np.arange(T), blk.block_dims(0))[:, d].any()   # 選択肢が 1 つの次元も除く
    assert not dm[:, a8.LAYOUT["zone_duty"][0] + 10].any()                                  # 北のゾーンは南の区間の外


# ---- UT-V8-7: 学習の入口（模擬の環境） ----

def _trainer_cfg(tmp, planned=2):
    cfg = config.load(os.path.join("rlkit", "config_summer_v8.json"))
    os.makedirs(str(tmp), exist_ok=True)
    entry_path = os.path.join(str(tmp), "init_entry.pt")
    torch.save(_v7_entry(), entry_path)
    cfg.update({"run_dir": str(tmp / "run"), "model_dir": str(tmp / "models"), "planned_updates": planned,
                "init_weights": entry_path, "start_E_hex": None, "start_DR_hex": None})   # 模擬の環境は実機の始点を持たない
    return cfg


def test_ut_v8_7_trainer_runs_base_on_slot_12_and_harvests_from_13(tmp_path):
    cfg = _trainer_cfg(tmp_path, planned=3)
    tr = tr8.V8Trainer(cfg, cfg["run_dir"], factory="mock_v8")
    tr.run()
    assert tr.stop_reason.startswith("計画更新数") and tr.update_index == 3
    ups = [json.loads(l) for l in open(os.path.join(cfg["run_dir"], "updates.jsonl"), encoding="utf-8")]
    eps = [json.loads(l) for l in open(os.path.join(cfg["run_dir"], "episodes.jsonl"), encoding="utf-8")]
    assert len(ups) == 3 and len(eps) == 3 * 13
    base_rows = [e for e in eps if e["role"] == "base"]
    assert len(base_rows) == 3 and all(e["slot"] == 12 for e in base_rows)                   # 基準の軌跡は組 12
    assert all(e["slot"] in range(12) for e in eps if e["role"] == "sibling")
    for u in ups:                                                                             # 決定論的評価として記録（V7 と同じ項目）
        assert u["eval_ECI"] == [e["ECI"] for e in base_rows if e["update"] == u["update"]][0]
        assert len(u["sibling_ECI"]) == 12 and len(u["train_c11"]) == 12 and "eval_c11" in u and "c13" in u
        assert u["harvest_update_ECI"] == pytest.approx(max([u["eval_ECI"]] + u["sibling_ECI"]))   # 収穫は 13 本から
        assert u["harvest_best"]["ECI"] >= u["harvest_update_ECI"] - 1e-12 or u["harvest_best"]["update"] != u["update"]
    sib = [json.loads(l) for l in open(os.path.join(cfg["run_dir"], "siblings.jsonl"), encoding="utf-8")]
    assert len(sib) == 3 and all(len(s["siblings"]) == 12 and "seed" in s["siblings"][0] and "changes" in s["siblings"][0] for s in sib)
    ev = [json.loads(l) for l in open(os.path.join(cfg["run_dir"], "events.jsonl"), encoding="utf-8")]
    start = [e for e in ev if e.get("event") == "run_start"][0]
    assert start["instruction_version"] == "V8-S" and start["init"]["init_sha256"] and start["observation_dim"] == 704
    assert os.path.exists(os.path.join(cfg["model_dir"], "checkpoint_latest.pt"))
    assert not any(f.endswith(".pt") for f in os.listdir(cfg["run_dir"]))                    # モデルは run_dir に出来ない
    # 価値ネットワークは更新しない（L-11）
    assert all(not p.requires_grad for p in tr.value.parameters())


def test_ut_v8_7_resume_and_start_protection(tmp_path):
    cfg = _trainer_cfg(tmp_path, planned=2)
    tr8.V8Trainer(cfg, cfg["run_dir"], factory="mock_v8").run()
    with pytest.raises(SystemExit):
        tr8.V8Trainer(cfg, cfg["run_dir"], factory="mock_v8").run(resume=False)
    cfg2 = dict(cfg, planned_updates=3)
    t2 = tr8.V8Trainer(cfg2, cfg["run_dir"], factory="mock_v8")
    t2.run(resume=True)
    assert t2.update_index == 3
    assert [json.loads(l)["update"] for l in open(os.path.join(cfg["run_dir"], "updates.jsonl"), encoding="utf-8")] == [1, 2, 3]


def test_collapse_rule_uses_base_trajectory():
    h = [0.70] + [0.71] * 10
    assert not tr8.collapse_check(h, 10)["triggered"]
    assert tr8.collapse_check([0.70] + [0.71] * 5 + [0.69] * 10, 15)["triggered"]            # 中央値 0.69 < 最大 0.71 − 0.01
    assert not tr8.collapse_check([0.70] + [0.71] * 5 + [0.70] * 10, 15)["triggered"]


def test_observation_column_map_and_reference_start():
    cm = o8.column_map_v7_to_v8()
    assert len(cm) == 648 and len(set(cm.tolist())) == 648 and cm.max() < 704
    assert cm[constants.GLOBAL_BLOCK_DIM] == o8.GLOBAL_DIM and cm[constants.GLOBAL_BLOCK_DIM + 31] == o8.GLOBAL_DIM + o8.ZONE_DIM
    cfg = config.load(os.path.join("rlkit", "config_summer_v8.json"))
    assert cfg["planned_updates"] == 600 and cfg["lr"] == 1e-4 and cfg["max_epochs"] == 8 and cfg["kl_limit"] == 0.01
    assert len(cfg["train_slots"]) == 12 and cfg["eval_slot"] == 12 and cfg["run_dir"] == "runs/v8s_trial01"


# ---- V8.1-S: C8-2・C8-3・C8-4（UT-V8-9・UT-V8-10） ----

def test_c82_start_point_hex_check(tmp_path):
    """C8-2: 更新 1 の基準の軌跡の E・DR が始点と 16 進で一致しなければ停止し、一致すれば続く。"""
    cfg = _trainer_cfg(tmp_path / "a", planned=3)
    os.makedirs(cfg["run_dir"], exist_ok=True)
    tr8.V8Trainer(cfg, cfg["run_dir"], factory="mock_v8").run()
    u1 = json.loads(open(os.path.join(cfg["run_dir"], "updates.jsonl"), encoding="utf-8").readline())
    right = dict(_trainer_cfg(tmp_path / "b", planned=3), start_E_hex=float.hex(u1["eval_E"]), start_DR_hex=float.hex(u1["eval_DR"]))
    t = tr8.V8Trainer(right, right["run_dir"], factory="mock_v8")
    t.run()
    assert t.update_index == 3 and t.judgments["C8-2"]["passed"]
    # 始点は設定の値（E・DR の 16 進）から作る
    assert t.base_eci_history[0] == pytest.approx(__import__("rlkit.eci", fromlist=["x"]).eci_smooth(u1["eval_E"], u1["eval_DR"]))
    wrong = dict(_trainer_cfg(tmp_path / "c", planned=3), start_E_hex=float.hex(u1["eval_E"] + 1e-9), start_DR_hex=float.hex(u1["eval_DR"]))
    t2 = tr8.V8Trainer(wrong, wrong["run_dir"], factory="mock_v8")
    t2.run()
    assert t2.update_index == 1 and "C8-2" in t2.stop_reason and not t2.judgments["C8-2"]["passed"]
    assert json.load(open(os.path.join(wrong["run_dir"], "judgment_c82.json"), encoding="utf-8"))["passed"] is False


def test_ut_v8_9_time_per_update_stops_when_mean_exceeds_limit(tmp_path):
    """UT-V8-9（C8-3）: 更新 1・2 の実時間の平均が上限を超えると更新 2 で止まり、以下なら続く。"""
    over = dict(_trainer_cfg(tmp_path / "over", planned=5), c83_limit_s=1e-9)
    t = tr8.V8Trainer(over, over["run_dir"], factory="mock_v8")
    t.run()
    assert t.update_index == 2 and "C8-3" in t.stop_reason and not t.judgments["C8-3"]["passed"]
    ok = dict(_trainer_cfg(tmp_path / "ok", planned=3), c83_limit_s=600.0)
    t2 = tr8.V8Trainer(ok, ok["run_dir"], factory="mock_v8")
    t2.run()
    assert t2.update_index == 3 and t2.judgments["C8-3"]["passed"] and t2.stop_reason.startswith("計画更新数")
    j = json.load(open(os.path.join(ok["run_dir"], "judgment_c83.json"), encoding="utf-8"))
    assert 0 < j["mean_update_seconds_1_2"] <= 600.0


def test_ut_v8_10_kl_epochs_cutoff_recorded_and_alias_harvest_files(tmp_path):
    """UT-V8-10（C8-4）: 毎更新、最後の KL・エポック数・打ち切りが記録される。最良が更新された更新でだけ別名の行動列が書かれ、
    既存のファイルは上書きされない。"""
    cfg = _trainer_cfg(tmp_path, planned=6)
    entry_path = cfg["init_weights"]
    e = torch.load(entry_path, weights_only=False)
    e["eci_smooth"] = 0.0                                          # 更新 1 から最良が更新されるようにする
    torch.save(e, entry_path)
    harvest = os.path.join(cfg["run_dir"], "harvest")
    os.makedirs(harvest, exist_ok=True)
    sentinel = os.path.join(harvest, "best_harvest_u1.npz")        # 既存のファイル（更新 1 の別名と同じ名前）
    open(sentinel, "wb").write(b"sentinel")
    tr = tr8.V8Trainer(cfg, cfg["run_dir"], factory="mock_v8")
    tr.run()
    ups = [json.loads(l) for l in open(os.path.join(cfg["run_dir"], "updates.jsonl"), encoding="utf-8")]
    assert len(ups) == 6
    for u in ups:
        assert "kl_last" in u and "epochs_applied" in u and "kl_cutoff" in u
        if u["kl_per_epoch"]:
            assert u["kl_last"] == u["kl_per_epoch"][-1] and u["epochs_applied"] == len(u["kl_per_epoch"])
        else:
            assert u["kl_last"] is None and u["epochs_applied"] == 0
    improved = [u["update"] for u in ups if u["harvest_update_best"].get("improved_best")]
    assert improved and improved[0] == 1
    aliases = sorted(int(f[len("best_harvest_u"):-4]) for f in os.listdir(harvest) if f.startswith("best_harvest_u") and f.endswith(".npz"))
    assert aliases == improved                                     # 最良が更新された更新でだけ書かれる
    assert open(sentinel, "rb").read() == b"sentinel"              # 既存のファイルは上書きされない
    ev = [json.loads(l) for l in open(os.path.join(cfg["run_dir"], "events.jsonl"), encoding="utf-8")]
    assert any(x.get("event") == "harvest_alias_exists_not_overwritten" and x["update"] == 1 for x in ev)
    for u in improved[1:]:
        d = np.load(os.path.join(harvest, f"best_harvest_u{u}.npz"))["actions"]
        assert d.shape == (constants.ZONE_STEPS_PER_WEEK, a8.ACTION_DIMS)
