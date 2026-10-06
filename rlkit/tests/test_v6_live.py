"""指示書 V6-S の単体試験のうち、エミュレータへの書き込みと読み戻しを確かめるもの（UT-1・UT-15・UT-16）。

エミュレータ（組 6）を 1 回起動する。環境変数 RLKIT_LIVE=1 のときだけ実行する（通常の試験では飛ばす）:
  RLKIT_LIVE=1 python -m pytest rlkit/tests/test_v6_live.py -q
K-9 は結果 B のため、UT-16 はリモコン許可の書き込みと読み戻しだけを試す（発停の毎ステップ書き込みと O-2 は結果 A 用）。
V6.4-S（V6.2-S C-5）で差し替え: リモコンの次元を廃止したため、UT-16 は「許可に固定され、読み戻しが全台許可」を試す。
UT-25 (b)（リモコン禁止の書き込み）は行動で禁止を指令できなくなったため削除した。
V7-S（C-2）で差し替え: 設定温度の格子が 22〜28 ℃になったため、UT-15 は格子の下限（22 ℃）の受理を試し、
UT-25 の設定温度は索引ではなく格子の値で期待値を作る（索引 4 は 24 ℃ → 26 ℃）。
"""
import os

import numpy as np
import pytest

from rlkit import action, comm, constants, driver

pytestmark = pytest.mark.skipif(os.environ.get("RLKIT_LIVE") != "1", reason="エミュレータを使う試験（RLKIT_LIVE=1 で実行）")

LIVE_SLOT = 6


@pytest.fixture(scope="module")
def live_env():
    from rlkit.env import SummerEnv
    from rlkit.parallel import launch, slots
    # MUST-2: 既存の診断の記録に書き足さない。記録の置き場所は RLKIT_LIVE_DIR で変えられる（V7-S では runs/diag/V7S/UT_live_prev）
    run_dir = os.path.join(os.environ.get("RLKIT_LIVE_DIR", os.path.join("runs", "diag", "V64S")), "UT_live_v6")
    os.makedirs(run_dir, exist_ok=True)
    launch.preflight(run_dir, [LIVE_SLOT])
    os.environ[launch.LAUNCH_LOG_ENV] = os.path.join(run_dir, launch.LAUNCH_DIR_NAME)
    slot = slots.make_slot(LIVE_SLOT)
    slots.prepare_replica(slot)
    env = SummerEnv(include_indoor_unit_power=True, zone_step_energy_norm_gj=0.02, slot=slot)
    env.set_diag_info(True)
    env.reset()
    yield env
    env.close()


def _act(vent=(0, 0), s1_on=False, sp_idx=0):
    a = np.zeros(constants.ACTION_DIMS, dtype=np.int64)
    a[action.LAYOUT["tenant_vent"][0]:action.LAYOUT["tenant_vent"][0] + 2] = vent
    if s1_on:
        a[0] = 1
        a[action.LAYOUT["zone_setpoint"][0]] = sp_idx
    return a


def _vent_state(env):
    return driver.run_coro(env._driver, comm.read_ventilation_state(env._driver.comms))


def _remote_state(env):
    return driver.run_coro(env._driver, comm.read_remote_permission_status(env._driver.comms))


def test_ut1_ventilation_levels_written_and_read_back(live_env):
    """UT-1: 換気の 4 水準それぞれについて、1 ステップの書き込みの後、テナントの全ゾーンの全熱交換器の発停と風量を読み戻して一致する。"""
    env = live_env
    for south, north in ((1, 3), (2, 0), (3, 1), (0, 2)):
        env.step(_act(vent=(south, north)))
        st = _vent_state(env)
        for k, (ou, iu, tenant, zn, label, area) in enumerate(constants.ZONE_DEFS):
            level = south if tenant == "South" else north
            onoff, fan = st[k]
            if level == constants.VENT_OFF:
                assert onoff == 0, (label, st[k])
            else:
                assert onoff == 1 and fan == level, (label, level, st[k])   # 風量 1:Low 2:Middle 3:High


def test_ut16_remote_permission_fixed_permit(live_env):
    """UT-16（C-5 で差し替え）: リモコンは許可に固定され、全室内機の 1114 の読み戻しが許可になる。"""
    env = live_env
    for _ in range(2):
        _obs, _r, _d, _t, info = env.step(_act())
        assert all(_remote_state(env))
        assert info["diag"]["remote_status"] == {"South": 1.0, "North": 1.0}


def test_ut15_setpoint_20c_accepted(live_env):
    """UT-15: 設定温度の格子の下限（V6-S は 20 ℃、V7-S C-2 で 22 ℃）の書き込みがエミュレータに受理され、1 ステップ後に読み戻せる。"""
    env = live_env
    lo = action.setpoint_grid(1)[0]
    assert lo == 22.0
    _obs, _r, _d, _t, info = env.step(_act(s1_on=True, sp_idx=0))
    assert info["diag"]["setpoint_readback"][0] == pytest.approx(lo, abs=constants.SETPOINT_OVERRIDE_TOL_C)
    assert info["diag"]["onoff_status"][0] is True


def test_ut25_k13_tools(live_env):
    """UT-25（V6.1-S）: K-13 の道具。(a) 読み戻しが指令と違うときだけ設定温度を書き直す、(b) リモコン禁止の書き込みと読み戻し。"""
    env = live_env
    env.set_setpoint_rewrite_on_mismatch(True)
    try:
        act = _act(s1_on=True, sp_idx=4)                     # S1 冷房（22〜28 ℃の格子の索引 4 = 26 ℃）
        sp_cmd = action.setpoint_grid(1)[4]
        env.step(act)
        n0 = env._setpoint_rewrites
        # 執務者の操作を模して、設定温度を 27 ℃に直接書き換える
        driver.run_coro(env._driver, comm.call_write(env._driver.comms.vrf.change_setpoint_temperature(1, 1, 27.0), "sp"))
        # 状態値（1106）はシミュレーションが進んだ後に反映されるため、60 秒だけ進める（執務者の操作はステップ内に起こり、
        # 次のステップの書き込み前の読み戻しで見えるのと同じ状況を作る）
        driver.run_coro(env._driver, env._driver.advance(60))
        rb = driver.run_coro(env._driver, comm.read_setpoint_readback(env._driver.comms))
        assert rb[0] == pytest.approx(27.0)
        _o, _r, _d, _t, info = env.step(act)                   # 読み戻し 27 ≠ 指令 26 → 書き直す
        assert env._setpoint_rewrites == n0 + 1
        assert info["diag"]["setpoint_readback"][0] == pytest.approx(sp_cmd)
        env.step(act)                                          # 読み戻し 24 = 指令 → 書かない
        assert env._setpoint_rewrites == n0 + 1
    finally:
        env.set_setpoint_rewrite_on_mismatch(False)
