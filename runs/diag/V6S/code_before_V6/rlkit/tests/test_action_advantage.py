"""行動空間・マスク・GAE・信用割当の試験（v2 第 5-3 節、第 5-4 節、第 5-5-2 節、第 5-6 節）。"""
import numpy as np
import pytest

from rlkit import action, advantage, constants


def test_action_space_sizes():
    assert len(action.nvec(1)) == 60 and sum(action.nvec(1)) == 252   # S-5: 第 1 段階は運転状態 2 択
    assert len(action.nvec(2)) == 60 and sum(action.nvec(2)) == 378
    assert action.zone_states(1) == ["off", "cool"] and action.zone_states(2) == ["off", "cool", "fan"]
    assert action.setpoint_grid(1) == [22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]
    assert action.setpoint_grid(2)[1] == 22.5 and len(action.setpoint_grid(2)) == 13
    gl = action.group_layout(1)
    assert gl["zone_state"]["K"] == 2 and gl["zone_setpoint"]["K"] == 7
    assert gl["system_evap"]["K"] == 8 and gl["tenant_vent"]["K"] == 2
    assert action.group_layout(2)["zone_state"]["K"] == 3


def test_decode_and_convert():
    a = np.zeros(60, dtype=int)
    a[0] = 1  # S1 冷房
    a[18] = 4  # 26 ℃
    a[36] = 2  # High
    a[54] = 3  # 系統1 蒸発温度 18 ℃
    a[59] = 1  # 北テナント換気 ON
    d = action.decode_action(a, 1)
    assert d["zones"]["S1"] == {"state": "cool", "setpoint": 26.0, "fan_speed": "High"}
    assert d["zones"]["S2"]["setpoint"] is None
    assert d["systems"][1]["evap_c"] == 18
    assert d["tenants"] == {"South": False, "North": True}
    b = action.convert_action_stage1_to_stage2(a)
    assert action.decode_action(b, 2)["zones"]["S1"]["setpoint"] == 26.0


def test_masks_m1_m3_stage1():
    """M-1（停止中の退化）と M-3（境界以外の退化）。第 1 段階に送風は無いため M-2 は第 2 段階で試す。"""
    prev = np.zeros(60, dtype=int)
    prev[2] = action.zone_states(1).index("cool")
    prev[18 + 0], prev[36 + 0] = 3, 1
    prev[54], prev[58] = 6, 1
    m = action.compute_mask(prev, is_boundary=False, stage=1)
    assert m[18].tolist() == [i == 3 for i in range(7)] and m[36].tolist() == [False, True, False]  # M-1
    assert m[20].all() and m[38].all()                                   # 冷房中のゾーンは退化しない
    assert m[54].tolist() == [i == 6 for i in range(8)] and m[58].tolist() == [False, True]  # M-3
    m_b = action.compute_mask(prev, is_boundary=True, stage=1)
    assert m_b[54].all() and m_b[58].all()
    assert action.mask_to_flat(m).shape == (252,)


def test_mask_m2_stage2():
    """M-2: 前ステップで送風だったゾーンの設定温度を前回値に退化させる（送風は第 2 段階のみ）。"""
    prev = np.zeros(60, dtype=int)
    prev[1] = action.zone_states(2).index("fan")
    prev[18 + 1], prev[36 + 1] = 5, 2
    m = action.compute_mask(prev, is_boundary=True, stage=2)
    assert m[19].tolist() == [i == 5 for i in range(13)] and m[37].all()   # 風量は制約しない
    assert action.mask_to_flat(m).shape == (378,)


def test_mask_m5_follows_flag_not_stage():
    """指示書 V2-S G-2・G-3: M-5 の適用は段階ではなく m5_enabled で決める。"""
    prev = np.zeros(60, dtype=int)
    no_cool = np.zeros(18, dtype=bool)
    no_cool[4] = True
    for stage in (1, 2):
        allowed = [s != "cool" for s in action.zone_states(stage)]
        on = action.compute_mask(prev, True, stage, no_cool, m5_enabled=True)
        assert on[4].tolist() == allowed          # 冷房だけを除く（第 1 段階は停止のみが残る）
        assert on[3].all()
        off = action.compute_mask(prev, True, stage, no_cool, m5_enabled=False)
        assert off[4].all()


def test_mask_m8_occupied_zone_must_cool():
    """B-2: 在室ゾーンは冷房だけ。M-5 と排他で、第 2 段階（解除後）では働かない。"""
    prev = np.zeros(60, dtype=int)
    occupied = np.zeros(18, dtype=bool); occupied[[0, 1]] = True
    no_cool = np.zeros(18, dtype=bool); no_cool[[5, 6]] = True
    m = action.compute_mask(prev, True, 1, no_cool, m5_enabled=True, occupied_zones=occupied, m8_enabled=True)
    cool = action.zone_states(1).index("cool")
    off = action.zone_states(1).index("off")
    assert m[0].tolist() == [i == cool for i in range(2)]      # M-8: 停止を除く
    assert m[5].tolist() == [i == off for i in range(2)]       # M-5: 冷房を除く
    assert m[2].all()                                          # どちらの対象でもないゾーンは自由
    # M-8 の在室ゾーンでは M-1 の退化が働かない（前ステップが停止でも設定温度を選べる）
    assert m[18 + 0].all() and m[36 + 0].all()
    assert m[18 + 2].sum() == 1                                # 対象外のゾーンは M-1 どおり退化する
    # 排他性: 在室かつ M-5 の対象という矛盾した入力でも、冷房だけが残る（M-8 が優先）
    both = action.compute_mask(prev, True, 1, occupied, m5_enabled=True, occupied_zones=occupied, m8_enabled=True)
    assert both[0].tolist() == [i == cool for i in range(2)]
    # 解除（第 2 段階）では 3 択がすべて残る
    m2 = action.compute_mask(action.convert_action_stage1_to_stage2(prev), True, 2, no_cool,
                             m5_enabled=False, occupied_zones=occupied, m8_enabled=False)
    assert m2[0].all() and m2[5].all()


def test_mask_m9_ventilation():
    """B-3: 採用時のみ、在室 0 かつ 1 時間先も 0 のテナントの換気を停止に固定する。"""
    prev = np.zeros(60, dtype=int)
    prev[58] = prev[59] = constants.TENANT_VENT_ON.index(True)
    idle = np.array([True, False])
    off = action.compute_mask(prev, True, 1, None, idle_tenants=idle, m9_enabled=False)
    assert off[58].all() and off[59].all()
    on = action.compute_mask(prev, True, 1, None, idle_tenants=idle, m9_enabled=True)
    assert on[58].tolist() == [True, False]        # 南: 停止に固定
    assert on[59].all()                            # 北: 1 時間先に在室があるため固定しない
    # 境界でないステップでも停止に倒す（M-3 の退化より後に適用するため選べる値が残る）
    mid = action.compute_mask(prev, False, 1, None, idle_tenants=idle, m9_enabled=True)
    assert mid[58].tolist() == [True, False] and mid[59].sum() == 1


def test_lambda_per_step():
    lam = advantage.lambda_per_step([900, 3600])
    assert lam[0] == pytest.approx(0.95 ** 0.25) and round(lam[0], 3) == 0.987
    assert lam[1] == pytest.approx(0.95)


def _classic_gae(r, v, gamma, lam):
    T = len(r)
    adv = np.zeros(T)
    run = 0.0
    for t in reversed(range(T)):
        nv = v[t + 1] if t + 1 < T else 0.0
        run = r[t] + gamma * nv - v[t] + gamma * lam * run
        adv[t] = run
    return adv


def test_variable_lambda_gae_reduces_to_classic():
    rng = np.random.default_rng(1)
    r, v = rng.normal(size=20), rng.normal(size=20)
    adv, ret = advantage.gae_variable_lambda(r, v, np.full(20, 0.9), gamma=1.0)
    assert np.allclose(adv, _classic_gae(r, v, 1.0, 0.9))
    assert np.allclose(ret, adv + v)


def test_segmented_gae_no_leak_and_additivity():
    rng = np.random.default_rng(2)
    T = 12
    r19 = rng.normal(size=(T, 19))
    v19 = rng.normal(size=(T, 19))
    secs = np.where(np.arange(T) % 3 == 0, 3600, 900)
    segs = advantage.episode_boundaries_from_starts([True] + [False] * 5 + [True] + [False] * 5)
    assert segs == [(0, 6), (6, 12)]
    adv, _ = advantage.segmented_gae(r19, v19, secs, segs)
    r_mod = r19.copy()
    r_mod[6:] += 100.0
    adv_mod, _ = advantage.segmented_gae(r_mod, v19, secs, segs)
    assert np.allclose(adv[:6], adv_mod[:6])
    total, _ = advantage.segmented_gae(r19.sum(axis=1, keepdims=True), v19.sum(axis=1, keepdims=True), secs, segs)
    assert np.allclose(adv.sum(axis=1), total[:, 0])


def test_assign_per_dim_advantages():
    T = 2
    zone_adv = np.arange(18, dtype=float)[None, :].repeat(T, axis=0)
    sys_adv = np.array([10.0, 20.0])
    actions = np.zeros((T, 60), dtype=int)
    actions[0, [0, 1, 9]] = 1  # t=0: S1, S2（系統1）と N1（系統3）が運転
    out = advantage.assign_per_dim_advantages(zone_adv, sys_adv, actions)
    assert np.allclose(out[:, 0:18], zone_adv) and np.allclose(out[:, 18:36], zone_adv) and np.allclose(out[:, 36:54], zone_adv)
    s1 = zone_adv[0, 0:5].sum()
    assert out[0, 54] == pytest.approx(s1 + 10.0 * 2 / 3)
    assert out[0, 55] == pytest.approx(zone_adv[0, 5:9].sum())
    assert out[0, 56] == pytest.approx(zone_adv[0, 9:14].sum() + 10.0 / 3)
    a = np.asarray(constants.OUTDOOR_UNIT_A_KW)
    assert out[1, 54] == pytest.approx(s1 + 20.0 * a[0] / a.sum())  # 運転ゾーン無し
    assert out[0, 58] == pytest.approx(zone_adv[0, 0:9].sum()) and out[0, 59] == pytest.approx(zone_adv[0, 9:18].sum())
    n = advantage.normalize_matrix(out)
    assert n.mean() == pytest.approx(0.0, abs=1e-12) and n.std() == pytest.approx(1.0, rel=1e-6)
