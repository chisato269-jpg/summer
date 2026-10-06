"""ネットワーク構成・観測・外生表・設定の試験（v2 第 5-2 節、第 5-9 節、第 5-10 節、指示書 V6-S 第 3-4 節・第 3-7 節）。

V6-S で差し替えた試験: 母数（第 3-7 節の表の K-9 結果 B の行、UT-11）、出力の形（296 ロジット）、第 2 段階の変換
（格子 9 → 17。第 2 段階は呼ばれないがコードは残すため形だけ合わせた）、観測の並び（642 次元、O-1・O-3、+19 は 9 択）、
設定の雛形（config_summer_v6.json）。
V6.1-S で差し替え: 観測の並びの試験で、42・43 を CO2 の現在値の複写から O-6（変化量）へ。設定の雛形を config_summer_v61.json へ。
"""
import datetime
import json
import os

import numpy as np
import pytest
import torch

from rlkit import action, config, constants, exogenous, observation, policy


def test_ut11_parameter_counts_table_3_7():
    """UT-11: 第 3-7 節の表の K-9 結果 B の行（観測 642、方策 76,252、価値 70,562、総計 146,814）。"""
    assert constants.K9_RESULT == "B"
    p1, v = policy.PolicyNetwork(1), policy.ValueNetwork()
    assert policy.parameter_count(p1) == 76252
    assert policy.parameter_count(v) == 70562
    assert policy.parameter_count(p1) + policy.parameter_count(v) == 146814
    shapes = {n: tuple(m.weight.shape) for n, m in p1.named_modules() if isinstance(m, torch.nn.Linear)}
    assert shapes == {"zone_encoder.0": (64, 31), "zone_encoder.2": (96, 64), "global_encoder.0": (96, 84),
                      "global_encoder.2": (128, 96), "context.0": (128, 320), "zone_head": (14, 224),
                      "system_head": (8, 224), "tenant_head": (6, 224)}
    assert tuple(v.context[0].weight.shape) == (128, 322)          # O-5
    assert tuple(v.zone_value_head.weight.shape) == (1, 224) and tuple(v.system_value_head.weight.shape) == (1, 128)


def test_forward_shapes_and_init_gains():
    obs = torch.rand(5, 642)
    p = policy.PolicyNetwork(1)
    assert p(obs).shape == (5, 296)
    assert policy.ValueNetwork()(obs).shape == (5, 19)
    assert torch.allclose(p.zone_head.weight.norm(dim=1), torch.full((14,), 0.01), atol=1e-6)
    assert torch.allclose(p.tenant_head.weight.norm(dim=1), torch.full((6,), 0.01), atol=1e-6)   # 第 3-7 節
    assert torch.all(p.zone_head.bias == 0) and torch.all(p.tenant_head.bias == 0)
    # テナントヘッドは 2 テナントで重みを共有し、テナント内ゾーンの埋め込みの平均から出す
    with torch.no_grad():
        p.tenant_head.weight.normal_()
        logits = p(obs)
    vent, remote = logits[:, 284:292].reshape(5, 2, 4), logits[:, 292:296].reshape(5, 2, 2)
    assert not torch.allclose(vent[:, 0], vent[:, 1]) and remote.shape == (5, 2, 2)


def test_stage2_expansion_logits():
    """B-4: 第 2 段階への変換。設定温度は 7 → 13 へ内挿し、送風の行は停止の行から複写する（S-5）。"""
    torch.manual_seed(0)
    p1, p2 = policy.PolicyNetwork(1), policy.PolicyNetwork(2)
    with torch.no_grad():
        p1.zone_head.weight.normal_()
        p1.zone_head.bias.normal_()
    policy.transfer_policy_stage1_to_stage2(p1, p2)
    obs = torch.rand(3, 642)
    l1, l2 = p1(obs), p2(obs)
    n1, n2 = 18 * 2, 18 * 3                       # 運転状態のロジット数（2 択 → 3 択）
    st1 = l1[:, :n1].reshape(3, 18, 2)
    st2 = l2[:, :n2].reshape(3, 18, 3)
    assert torch.allclose(st2[:, :, :2], st1, atol=1e-5)            # 停止・冷房はそのまま
    assert torch.allclose(st2[:, :, 2], st1[:, :, 0], atol=1e-5)    # 送風は停止から複写（S-5）
    sp1 = l1[:, n1:n1 + 18 * 9].reshape(3, 18, 9)
    sp2 = l2[:, n2:n2 + 18 * 17].reshape(3, 18, 17)
    assert torch.allclose(sp2[:, :, 0::2], sp1, atol=1e-5)
    assert torch.allclose(sp2[:, :, 1::2], 0.5 * (sp1[:, :, :-1] + sp1[:, :, 1:]), atol=1e-5)
    # 風量・系統・換気は同じ重み・同じ入力だが、ゾーンヘッドの出力数が違うため float32 の丸めの差が残る
    assert torch.allclose(l1[:, n1 + 162:], l2[:, n2 + 306:], atol=1e-4)


def test_exogenous_total_and_occupancy_helpers():
    exo = exogenous.load()
    assert exo.total_occupant_minutes() == 159497.0
    occupied = exo.zone_occupants[:, 0] > 0
    j = int(np.flatnonzero(occupied)[0])
    t = lambda m: exo.t0 + datetime.timedelta(minutes=m)
    assert exo.hours_until_next_occupancy(t(j - 30))[0] == pytest.approx(0.5)
    assert exo.hours_until_next_occupancy(t(j))[0] == 0.0
    assert exo.zone_occupied_within(t(j - 60), 2.0)[0]
    assert not exo.zone_occupied_within(t(j - 121), 2.0)[0]
    assert np.isinf(exo.hours_since_last_occupancy_end(t(0))[0])
    end = j + int(np.argmin(occupied[j:]))  # 最初の在室区間が終わった分
    assert exo.hours_since_last_occupancy_end(t(end + 90))[0] == pytest.approx(1.5)


def _fake_snapshot(drybulb_zone=26.0):
    zones = {l: {"drybulb_c": drybulb_zone, "relative_humidity_pct": 50.0, "occupant_number": 2, "thermal_sensation": 0.5,
                 "clothing_index": 0.6, "dissatisfied_thermal": 0.1, "dissatisfied_draft": 0.05,
                 "dissatisfied_vtmp": 0.02, "heatload_kw": -1.0} for l in constants.ZONE_LABELS}
    b = {"drybulb_c": 30.0, "relative_humidity_pct": 60.0, "global_horizontal_radiation": 300.0,
         "nocturnal_radiation": 20.0, "total_energy_consumption_mj": 0.0, "averaged_dissatisfaction_rate": 0.0}
    return {"building": b, "zones": zones, "co2_ppm": {"South": 800.0, "North": 900.0}}


def test_observation_layout():
    exo = exogenous.load()
    with pytest.raises(ValueError):
        observation.ObservationBuilder(exo, None)
    ob = observation.ObservationBuilder(exo, zone_step_energy_norm_gj=0.01)
    dt = datetime.datetime(1999, 7, 21, 0, 0)
    a = action.initial_action(1)
    a[0], a[18] = 1, 6  # S1 冷房 26 ℃（20〜28 ℃の格子の索引 6）
    a[58], a[59] = 1, 3  # 換気 南 Low、北 High
    decoded = action.decode_action(a, 1)
    tenant = {"South": {"counts": [0, 0, 1, 2, 1, 0, 0], "n_present": 4}, "North": {"counts": [0] * 7, "n_present": 0}}
    readback = [24.0] * 18
    obs, raw = ob.build(dt, 0, 1, _fake_snapshot(), decoded, zone_energy_gj := np.full(18, 0.005),
                        np.full(18, 0.2), tenant, readback, remote_status={"South": 1.0, "North": 0.0})
    assert obs.shape == (642,) and obs.dtype == np.float32 and obs.min() >= 0 and obs.max() <= 1
    assert raw[80] == pytest.approx(1 / 3) and raw[81] == pytest.approx(1.0)      # O-1: 水準 ÷ 3
    assert raw[82] == 1.0 and raw[83] == 0.0                                      # O-3: 許可 1、禁止 0
    assert raw[2] == 1.0  # 水曜
    assert raw[40] == pytest.approx(0.2) and raw[41] == pytest.approx(0.225)
    assert raw[42] == 0.0 and raw[43] == 0.0      # O-6（V6.1-S）: 最初のステップの CO2 の変化量は 0（V6-S までは現在値の複写）
    assert raw[64] == 1.0 and raw[65] == 1.0
    assert raw[66:73].tolist() == pytest.approx([0, 0, 0.25, 0.5, 0.25, 0, 0]) and raw[73:80].sum() == 0
    base = 84
    assert raw[base + 17] == 1.0 and raw[base + 19] == pytest.approx(6 / 8)       # A-2: 9 択の格子で正規化
    assert raw[base + 22] == pytest.approx(0.005 / 3.0) and raw[base + 25] == pytest.approx(0.5)
    assert raw[base + 26] == pytest.approx(0.2)
    assert raw[base + 29] == pytest.approx((24.0 - 16.0) / 16.0)
    assert raw[base + 30] == 1.0  # 26 ℃を書いたのに読み戻しは 24 ℃
    assert raw[base + 31 + 30] == 0.0  # 停止中のゾーンは上書き判定しない
    obs2, raw2 = ob.build(dt, 1, 1, _fake_snapshot(drybulb_zone=60.0), decoded, None, None, tenant, readback)
    assert observation.clipped_entries(raw2)[base + 4]


def test_observation_normalization_v3s():
    """B-3: 指示書 V3-S 第 2-4 節の 4 種。境界値でクリップされず、進行率は最終ステップで 1 になる。"""
    exo = exogenous.load()
    ob = observation.ObservationBuilder(exo, zone_step_energy_norm_gj=0.01)
    dt = datetime.datetime(1999, 7, 21, 0, 0)
    decoded = action.decode_action(action.initial_action(1), 1)
    snap = _fake_snapshot()
    snap["co2_ppm"] = {"South": 0.0, "North": 4000.0}
    for label in constants.ZONE_LABELS:  # 系統内の室内機容量の合計まで熱負荷を上げた状態
        snap["zones"][label]["heatload_kw"] = -constants.INDOOR_UNIT_CAPACITY_KW[label]
    readback = [16.0] * 9 + [32.0] * 9
    obs, raw = ob.build(dt, constants.ZONE_STEPS_PER_WEEK, 1, snap, decoded, None, None, None, readback)
    assert raw[10] == 1.0                                              # 進行率 step/T
    assert raw[40] == 0.0 and raw[41] == pytest.approx(1.0)            # CO2 ÷ 4000 ppm
    for s in range(4):                                                 # 負荷率 ÷ 室内機容量の合計
        assert raw[observation.IDX_SYSTEM + 5 * s + 3] == pytest.approx(1.0)
    assert raw[observation.zone_base(0) + 29] == 0.0                   # 16 ℃
    assert raw[observation.zone_base(17) + 29] == pytest.approx(1.0)   # 32 ℃
    clipped = observation.clipped_entries(raw)
    idx = [10, 40, 41, 42, 43] + [observation.IDX_SYSTEM + 5 * s + 3 for s in range(4)] + \
          [observation.zone_base(k) + 29 for k in range(constants.N_ZONES)]
    assert not clipped[idx].any()


def test_config_unresolved_and_template():
    cfg = config.load()
    assert [k for k, _, _ in config.unresolved_items(cfg)] == list(config.UNRESOLVED)
    template = os.path.join(os.path.dirname(config.__file__), "config_summer_v61.json")
    cfg_t = config.load(template)
    # 値は指示書 V6.1-S 第 12 章の ASCII 数値参照リストと同じ（走行 2 は runs/v2s_trial06。V6-S の設定は残す）
    assert config.unresolved_items(cfg_t) == []
    assert cfg_t["run_dir"] == "runs/v2s_trial06" and cfg_t["include_indoor_unit_power"] is True
    assert config.load(os.path.join(os.path.dirname(config.__file__), "config_summer_v6.json"))["run_dir"] == "runs/v2s_trial05"
    assert cfg_t["zone_step_energy_norm_gj"] == 0.020 and cfg_t["seed"] == 19800614
    assert cfg_t["planned_updates"] == 1500 and cfg_t["m9_enabled"] is True and cfg_t["k9_result"] == "B"
    assert cfg_t["max_collection_retries"] == 4       # S-8: やり直しが 3 回を超えたら停止
    with open(template, encoding="utf-8") as f:
        assert set(json.load(f)) == set(config.DEFAULTS)
