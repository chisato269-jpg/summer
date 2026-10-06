"""指示書 V6-S の単体試験のうち、エミュレータへの書き込みと読み戻しを確かめるもの（UT-1・UT-15・UT-16）。

エミュレータ（組 6）を 1 回起動する。環境変数 RLKIT_LIVE=1 のときだけ実行する（通常の試験では飛ばす）:
  RLKIT_LIVE=1 python -m pytest rlkit/tests/test_v6_live.py -q
K-9 は結果 B のため、UT-16 はリモコン許可の書き込みと読み戻しだけを試す（発停の毎ステップ書き込みと O-2 は結果 A 用）。
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
    run_dir = os.path.join("runs", "diag", "V6S", "UT_live")
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


def _act(vent=(0, 0), remote=(1, 1), s1_on=False, sp_idx=0):
    a = np.zeros(constants.ACTION_DIMS, dtype=np.int64)
    a[action.LAYOUT["tenant_vent"][0]:action.LAYOUT["tenant_vent"][0] + 2] = vent
    a[action.LAYOUT["tenant_remote"][0]:action.LAYOUT["tenant_remote"][0] + 2] = remote
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


def test_ut16_remote_permission_written_and_read_back(live_env):
    """UT-16（結果 B）: リモコン許可の書き込みと読み戻し。テナントの全室内機に同じ値が入る。"""
    env = live_env
    for south, north in ((0, 1), (1, 0), (1, 1)):
        assert env._pending_boundary     # 6 時までは 3,600 秒刻みで、毎ステップが境界（M-3 に触れない）
        _obs, _r, _d, _t, info = env.step(_act(remote=(south, north)))
        st = _remote_state(env)
        for k, (ou, iu, tenant, zn, label, area) in enumerate(constants.ZONE_DEFS):
            assert st[k] == bool(south if tenant == "South" else north), (label, st[k])
        assert info["diag"]["remote_status"] == {"South": float(south), "North": float(north)}   # O-3 の元の値


def test_ut15_setpoint_20c_accepted(live_env):
    """UT-15: 設定温度 20 ℃の書き込みがエミュレータに受理され、1 ステップ後に読み戻せる。"""
    env = live_env
    assert action.setpoint_grid(1)[0] == 20.0
    _obs, _r, _d, _t, info = env.step(_act(s1_on=True, sp_idx=0))
    assert info["diag"]["setpoint_readback"][0] == pytest.approx(20.0, abs=constants.SETPOINT_OVERRIDE_TOL_C)
    assert info["diag"]["onoff_status"][0] is True
