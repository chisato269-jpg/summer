"""行動空間・マスク・GAE・信用割当の試験（v2 第 5-3 節、第 5-4 節、第 5-5-2 節、第 5-6 節、指示書 V6-S 第 3-1 節・第 3-2 節）。

V6-S で差し替えた試験（報告書に理由を記す）: 行動空間の大きさ（60 次元・252 ロジット → 62 次元・296 ロジット）、
展開（換気は水準、リモコンを追加、設定温度 20〜28 ℃）、M-1・M-2・M-8 の試験（規則の廃止により GC-1・M-3・M-5'・M-9 の試験へ）。
"""
import numpy as np
import pytest

from rlkit import action, advantage, constants


def test_action_space_sizes():
    assert len(action.nvec(1)) == 62 and sum(action.nvec(1)) == 296    # V6-S 第 3-1 節
    assert action.zone_states(1) == ["off", "cool"]
    assert action.setpoint_grid(1) == [20.0 + i for i in range(9)]    # A-2
    gl = action.group_layout(1)
    assert gl["zone_state"]["K"] == 2 and gl["zone_setpoint"]["K"] == 9 and gl["zone_fan"]["K"] == 3
    assert gl["system_evap"]["K"] == 8 and gl["tenant_vent"]["K"] == 4 and gl["tenant_remote"]["K"] == 2
    assert action.LAYOUT["tenant_vent"] == (58, 2) and action.LAYOUT["tenant_remote"] == (60, 2)
    assert constants.ACTION_LOGITS[1] == 296 and constants.ACTION_DIMS == 62


def test_decode_and_v5_mapping():
    a = np.zeros(62, dtype=int)
    a[0] = 1   # S1 冷房
    a[18] = 6  # 26 ℃
    a[36] = 2  # High
    a[54] = 3  # 系統1 蒸発温度 18 ℃
    a[58], a[59] = 1, 3   # 南 Low、北 High
    a[60], a[61] = 0, 1   # 南 禁止、北 許可
    d = action.decode_action(a, 1)
    assert d["zones"]["S1"] == {"state": "cool", "setpoint": 26.0, "fan_speed": "High"}
    assert d["zones"]["S2"]["setpoint"] is None
    assert d["systems"][1]["evap_c"] == 18
    assert d["tenants"] == {"South": 1, "North": 3}
    assert d["remote"] == {"South": False, "North": True}
    # B-1 の写像: 換気の運転 → Middle、停止 → 停止、設定温度は物理値のまま、リモコンは許可
    v5 = np.zeros((2, 60), dtype=int)
    v5[0, 0], v5[0, 18], v5[0, 58], v5[0, 59] = 1, 4, 1, 0     # V5 の索引 4 = 26 ℃
    v6 = action.v5_to_v6_actions(v5)
    d6 = action.decode_action(v6[0], 1)
    assert d6["zones"]["S1"]["setpoint"] == 26.0 and d6["tenants"] == {"South": 2, "North": 0}
    assert d6["remote"] == {"South": True, "North": True}
    assert action.v5_to_v6_actions(v5[1]).shape == (62,)


def test_ut2_mask_m3_system_and_remote_only():
    """UT-2: M-3 は系統とリモコンだけ。換気は M-3 の対象外（境界でないステップでも 4 択が残る）。"""
    prev = np.zeros(62, dtype=int)
    prev[54], prev[58], prev[60] = 6, 2, 1
    m = action.compute_mask(prev, is_boundary=False, stage=1)
    assert m[54].tolist() == [i == 6 for i in range(8)]            # 系統: 前回値に退化
    assert m[60].tolist() == [False, True] and m[61].tolist() == [True, False]   # リモコン: 前回値に退化
    assert m[58].all() and m[59].all()                              # 換気: M-3 の対象外
    # M-1 は廃止: 前ステップで停止していたゾーンでも設定温度・風量は全選択肢が残る
    assert m[18].all() and m[36].all()
    m_b = action.compute_mask(prev, is_boundary=True, stage=1)
    assert m_b[54].all() and m_b[60].all()
    assert action.mask_to_flat(m).shape == (296,)


def test_ut2_mask_m9_ventilation():
    """UT-2: M-9 は在室 0 かつ 1 時間先も 0 のテナントの換気を停止（0）に固定する。境界でないステップでも効く。"""
    prev = np.zeros(62, dtype=int)
    prev[58] = prev[59] = 2
    idle = np.array([True, False])
    for boundary in (True, False):
        on = action.compute_mask(prev, boundary, 1, None, idle_tenants=idle, m9_enabled=True)
        assert on[58].tolist() == [True, False, False, False]      # 南: 停止に固定
        assert on[59].all()                                        # 北: 固定しない
    off = action.compute_mask(prev, True, 1, None, idle_tenants=idle, m9_enabled=False)
    assert off[58].all()


def test_mask_m5_prime_allows_only_off():
    """M-5': 対象ゾーンは停止だけ。M-8 は廃止され、在室ゾーンでも停止を選べる。"""
    prev = np.zeros(62, dtype=int)
    no_cool = np.zeros(18, dtype=bool)
    no_cool[4] = True
    m = action.compute_mask(prev, True, 1, no_cool, m5_enabled=True)
    assert m[4].tolist() == [True, False]
    assert m[3].all()
    assert action.compute_mask(prev, True, 1, no_cool, m5_enabled=False)[4].all()


def _synthetic_exo(tmp_path, starts_min):
    """ゾーン k の在室が starts_min[k] 分後から始まる外生表（UT-3 用。その他のゾーンは 1 週間在室なし）。"""
    import datetime
    from rlkit import exogenous
    n = 10080
    t0 = datetime.datetime(1999, 7, 21, 0, 0)
    occ = np.zeros((n, 18))
    for k, m in starts_min.items():
        occ[m:m + 60, k] = 1.0
    path = tmp_path / "exo.npz"
    np.savez(path, datetime=np.array([(t0 + datetime.timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n)]),
             zone_labels=np.array(constants.ZONE_LABELS), zone_occupants=occ, total_occupants=occ.sum(axis=1),
             outdoor_drybulb=np.zeros(n), outdoor_rh=np.zeros(n), outdoor_abs_humidity=np.zeros(n))
    return exogenous.ExogenousData(str(path)), t0


def test_ut3_m5_prime_boundary(tmp_path):
    """UT-3: 在室の開始が 11.9 時間後なら運転を許し、12.1 時間後以降だけなら禁止する（環境の判定で確かめる）。"""
    from rlkit.env import SummerEnv
    exo, t0 = _synthetic_exo(tmp_path, {0: int(11.9 * 60), 1: int(12.1 * 60)})
    env = SummerEnv(include_indoor_unit_power=False, zone_step_energy_norm_gj=0.02)
    env.exo = exo
    snap = {"zones": {l: {"occupant_number": 0.0} for l in constants.ZONE_LABELS}}
    env._update_masks(t0, snap)
    m = env.action_masks()
    off, cool = 0, 1
    assert constants.M5_FORECAST_WINDOW_H == 12.0
    assert m[0 * 2 + cool]                 # ゾーン 0: 11.9 時間後に在室 → 運転を許す
    assert not m[1 * 2 + cool] and m[1 * 2 + off]    # ゾーン 1: 12.1 時間後 → 停止だけ
    assert not m[2 * 2 + cool]             # 在室なし → 停止だけ
    # 現在の在室者が 1 人以上なら先読みにかかわらず運転を許す
    snap["zones"]["S2"]["occupant_number"] = 1.0
    env._update_masks(t0, snap)
    assert env.action_masks()[1 * 2 + cool]


def test_gc1_excluded_elements():
    """GC-1: 停止を標本化したゾーンの設定温度・風量と、運転ゾーンの無い系統の蒸発温度を除く。"""
    a = np.zeros((2, 62), dtype=int)
    a[0, [0, 1, 9]] = 1          # t=0: S1・S2（系統1）と N1（系統3）が運転
    ex = action.gc1_excluded(a)
    assert ex.shape == (2, 62)
    assert not ex[0, 18] and not ex[0, 36] and not ex[0, 27]        # 運転ゾーンの設定温度・風量は残す
    assert ex[0, 20] and ex[0, 38]                                  # 停止ゾーン S3 は除く
    assert not ex[0, 54] and ex[0, 55] and not ex[0, 56] and ex[0, 57]   # 系統 2・4 は運転ゾーンが無い
    assert ex[1, 18:58].all()                                       # t=1: 全停止
    assert not ex[:, :18].any() and not ex[:, 58:].any()            # 運転状態・換気・リモコンは GC-1 の対象外


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
    actions = np.zeros((T, 62), dtype=int)
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
    # V6-S: リモコン許可（60・61）も換気と同じくテナント内ゾーンの和
    assert out.shape == (T, 62)
    assert out[0, 60] == pytest.approx(zone_adv[0, 0:9].sum()) and out[0, 61] == pytest.approx(zone_adv[0, 9:18].sum())
    n = advantage.normalize_matrix(out)
    assert n.mean() == pytest.approx(0.0, abs=1e-12) and n.std() == pytest.approx(1.0, rel=1e-6)
