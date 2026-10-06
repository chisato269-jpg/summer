"""指示書 V8-S の MUST-8（新しい書き込みの読み戻しの一致）を実機で確かめる試験。環境変数 RLKIT_LIVE=1 のときだけ実行する。

エミュレータ（組 12）を 1 回起動する。確かめること:
  A-5 設定温度 22.5・23.5 ℃が丸められず読み戻せる
  A-4 蒸発温度 14 ℃が受理され読み戻せる
  A-2 リモコンの禁止・許可の書き込みが読み戻し（1114）と一致する
  A-3 バイパスの有効・無効の書き込みが読み戻し（binaryOutput の xxx4）と一致する
  A-1 割合 1/3・2/3 のゾーンが刻みの終端で停止している（発停の読み戻し 1102）、割合 1 のゾーンは運転している
記録の置き場所は環境変数 RLKIT_LIVE_DIR（既定は runs/diag/V8S）。
"""
import os

import numpy as np
import pytest

from rlkit import comm, constants, driver
from rlkit.v8 import action as a8
from rlkit.v8 import observation as o8

pytestmark = pytest.mark.skipif(os.environ.get("RLKIT_LIVE") != "1", reason="エミュレータを使う試験（RLKIT_LIVE=1 で実行）")

LIVE_SLOT = 12


@pytest.fixture(scope="module")
def live_env():
    from rlkit.parallel import launch, slots
    from rlkit.v8.env import SummerEnvV8
    run_dir = os.path.join(os.environ.get("RLKIT_LIVE_DIR", os.path.join("runs", "diag", "V8S")), "UT_live_v8")
    os.makedirs(run_dir, exist_ok=True)
    launch.preflight(run_dir, [LIVE_SLOT], limit=13, min_free_gb=1.0)
    os.environ[launch.LAUNCH_LOG_ENV] = os.path.join(run_dir, launch.LAUNCH_DIR_NAME)
    slot = slots.make_slot(LIVE_SLOT)
    slots.prepare_replica(slot)
    env = SummerEnvV8(include_indoor_unit_power=True, zone_step_energy_norm_gj=0.02, slot=slot)
    obs, _info = env.reset()
    assert obs.shape == (o8.TOTAL_DIM,)
    yield env
    env.close()


def _act(duty=0, setpoint=0, fan=2, remote=0, evap=1, vent=0, bypass=0):
    a = np.zeros(a8.ACTION_DIMS, dtype=np.int64)
    a[a8.dim_slice("zone_duty")] = duty
    a[a8.dim_slice("zone_setpoint")] = setpoint
    a[a8.dim_slice("zone_fan")] = fan
    a[a8.dim_slice("zone_remote")] = remote
    a[a8.dim_slice("system_evap")] = evap
    a[a8.dim_slice("tenant_vent")] = vent
    a[a8.dim_slice("tenant_bypass")] = bypass
    return a


def _run(env, coro):
    return driver.run_coro(env._driver, coro)


def _step_to_boundary(env):
    """毎時の境目まで進める（A-2 は 1 時間ごとに書かれる）。無人の時間帯は刻みが 3,600 秒なので、始まりが正時なら常に境目。"""
    assert env._pending_boundary, f"毎時の境目でない: {env._pending_dt}"


def test_must8_setpoint_half_degree_read_back(live_env):
    """A-5: 設定温度 22.5 ℃・23.5 ℃（格子の索引 1・3）が丸められずに読み戻せる。"""
    env = live_env
    _step_to_boundary(env)
    sp = np.zeros(18, dtype=np.int64)
    sp[0], sp[1], sp[2] = 1, 3, 0                                              # S1 22.5、S2 23.5、S3 22.0
    duty = np.full(18, a8.DUTY_FULL)
    env.step(_act(duty=duty, setpoint=sp))
    rb = _run(env, comm.read_setpoint_readback(env._driver.comms))
    assert rb[0] == pytest.approx(22.5, abs=1e-6) and rb[1] == pytest.approx(23.5, abs=1e-6) and rb[2] == pytest.approx(22.0, abs=1e-6)


def test_must8_evaporating_temperature_14_read_back(live_env):
    """A-4: 蒸発温度 14 ℃（格子の索引 0）が受理され、読み戻しが 14 ℃になる。"""
    env = live_env
    env.step(_act(duty=a8.DUTY_FULL, evap=np.array([0, 1, 2, 3])))
    vals = [_run(env, comm.call_write(env._driver.comms.vrf.get_evaporating_temperature(ou), f"evap {ou}")) for ou in range(1, 5)]
    assert [float(v) for v in vals] == [14.0, 15.0, 16.0, 17.0]


def test_must8_remote_prohibit_and_permit_read_back(live_env):
    """A-2: ゾーン S1 のリモコンの禁止が読み戻し（1114）で偽になり、許可に戻すと真になる。他のゾーンは許可のまま。"""
    env = live_env
    while not env._pending_boundary:
        env.step(_act(duty=a8.DUTY_FULL))
    remote = np.zeros(18, dtype=np.int64)
    remote[0] = a8.REMOTE_PROHIBIT
    env.step(_act(duty=a8.DUTY_FULL, remote=remote))
    st = _run(env, comm.read_remote_permission_status(env._driver.comms))
    assert st[0] is False and all(st[1:])
    while not env._pending_boundary:                                          # 次の毎時の境目で許可に戻す
        env.step(_act(duty=a8.DUTY_FULL, remote=remote))
    env.step(_act(duty=a8.DUTY_FULL, remote=np.zeros(18, dtype=np.int64)))
    st = _run(env, comm.read_remote_permission_status(env._driver.comms))
    assert all(st)


def _bypass_state(env):
    vent = env._driver.comms.vent
    out = []
    for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS:
        inst = "binaryOutput:" + vent._get_instance_number(ou, iu, vent._member.HexBypassEnabled.value)
        out.append(int(_run(env, comm.call_write(vent.read_present_value(vent.target_ip, inst), f"{label} bypass"))))
    return out


def test_must8_bypass_enable_and_disable_read_back(live_env):
    """A-3: 南のテナントのバイパスを有効にすると南の全熱交換器の読み戻しが 1、北は 0。無効に戻すと全て 0。"""
    env = live_env
    env.step(_act(duty=a8.DUTY_FULL, vent=2, bypass=np.array([a8.BYPASS_ON, a8.BYPASS_OFF])))
    st = _bypass_state(env)
    south = [k for k, z in enumerate(constants.ZONE_DEFS) if z[2] == "South"]
    assert all(st[k] == 1 for k in south) and all(st[k] == 0 for k in range(18) if k not in south)
    env.step(_act(duty=a8.DUTY_FULL, vent=2, bypass=0))
    assert _bypass_state(env) == [0] * 18


def test_must8_duty_fractions_end_state_read_back(live_env):
    """A-1: 割合 1/3・2/3・0 のゾーンは刻みの終端で停止、割合 1 のゾーンは運転（発停の読み戻し）。途中の停止の書き込みの数が合う。"""
    env = live_env
    duty = np.full(18, a8.DUTY_FULL)
    duty[0], duty[1], duty[2] = 1, 2, 0
    n0 = env._v8_mid_writes
    mism0 = env._onoff_mismatch
    _o, _r, _d, _t, info = env.step(_act(duty=duty))
    on = _run(env, comm.read_onoff_status(env._driver.comms))
    assert on[0] is False and on[1] is False and on[2] is False and all(on[3:])
    assert env._v8_mid_writes - n0 == 2                                         # 割合 1/3 と 2/3 のゾーンの途中の停止（前の刻みは全ゾーン運転）
    assert env._onoff_mismatch == mism0                                         # 終端の状態は想定どおり
    _o, _r, _d, _t, info = env.step(_act(duty=duty))
    assert env._v8_mid_writes - n0 == 4                                         # 次の刻みも、始めに運転を書き、途中で 2 つ停止する
