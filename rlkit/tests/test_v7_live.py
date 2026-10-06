"""指示書 V7-S の単体試験のうち、エミュレータを使うもの（UT-35）。RLKIT_LIVE=1 のときだけ実行する:
  RLKIT_LIVE=1 python -m pytest rlkit/tests/test_v7_live.py -q
"""
import os

import numpy as np
import pytest

from rlkit import action, comm, constants, driver

pytestmark = pytest.mark.skipif(os.environ.get("RLKIT_LIVE") != "1", reason="エミュレータを使う試験（RLKIT_LIVE=1 で実行）")

LIVE_SLOT = 6
RUN_DIR = os.path.join("runs", "diag", "V7S", "UT_live")


def test_ut35_direction_fixed_horizontal():
    """UT-35: 1 ステップ後に全室内機の風向を読み戻し、1（水平）である（全ゾーンを冷房にして書き込む）。"""
    from rlkit.env import SummerEnv
    from rlkit.parallel import launch, slots
    os.makedirs(RUN_DIR, exist_ok=True)
    report = launch.preflight(RUN_DIR, [LIVE_SLOT])
    os.environ[launch.LAUNCH_LOG_ENV] = report["log_dir"]
    slot = slots.make_slot(LIVE_SLOT)
    slots.prepare_replica(slot)
    env = SummerEnv(include_indoor_unit_power=True, zone_step_energy_norm_gj=0.02, slot=slot)
    try:
        env.reset()
        before = driver.run_coro(env._driver, comm.read_direction_status(env._driver.comms))
        a = action.initial_action(1)
        st0, _ = action.LAYOUT["zone_state"]
        a[st0:st0 + constants.N_ZONES] = constants.ZONE_STATE.index("cool")
        env.step(a)
        after = driver.run_coro(env._driver, comm.read_direction_status(env._driver.comms))
    finally:
        env.close()
    assert after == [1] * constants.N_ZONES, after
    assert constants.ZONE_DIRECTION_FIXED == "Horizontal"
    print("読み戻し（書き込み前）", before, "（書き込み後）", after)
