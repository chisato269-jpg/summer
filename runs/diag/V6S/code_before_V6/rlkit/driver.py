"""rlkit.driver — Shizuku2.exe の起動と二段刻みのエピソード駆動（現行 rlkit/driver.py を移植）。

冬季の再生とスケジュールは移植していない。規則ベース最良軌跡（夏季）の再生は、J-1・J-5 の診断
（v2 第 8 章）で使うため残した。COV 購読は使わず、日時は毎回 BACnet で直接読む（現行と同じ）。
"""
import asyncio
import datetime
import json
import os
import platform
import sys
import threading
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rlkit import comm, constants
from rlkit.timing import is_system_boundary, zone_step_seconds  # noqa: F401（現行の呼び出し元との互換）

_SHIZUKU2_DIR = os.path.join(_REPO_ROOT, "Shizuku2")

# 加速度（現行と同じ 10 日/秒）
ACCELERATION_RATE = 86400 * 10
BOOT_READY_MARKER = 'Press "Enter" key to continue.'


def _launch_log(slot, event, **fields):
    """組ごとの起動結果を記録する（指示書 V4-S 第 2-7 節）。組を使わない走行では何もしない。"""
    if slot is None:
        return
    from rlkit.parallel import launch
    launch.log_slot_event(slot.index, event, **fields)


class EpisodeDriver:
    """Shizuku2.exe を BACnet 経由で起動し、二段刻みでエピソードを駆動する。

    slot: rlkit.parallel.slots.SlotSpec。None は原本の Shizuku2 ディレクトリを使う。
    """

    def __init__(self, slot=None):
        self.slot = slot
        self.comms = None
        self.process = None
        self.loop = None
        self._thread = None
        self._loop_ready = threading.Event()
        self.final_dtime = None

    # ---- ライフサイクル ----

    def start(self):
        if self._thread is not None:
            raise RuntimeError("既に起動済み")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._loop_ready.wait(timeout=30):
            raise RuntimeError("イベントループの起動に失敗した")
        fut = asyncio.run_coroutine_threadsafe(self._async_start(), self.loop)
        fut.result(timeout=90)

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._loop_ready.set()
        self.loop.run_forever()

    async def _drain(self, stream):
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                line = await stream.read(1 << 16)
            if not line:
                return

    async def _async_start(self):
        if platform.system() != "Windows":
            raise RuntimeError("本実装は Windows 環境のみを対象としている（現行と同じ）")
        runtime = None
        if self.slot is None:
            executable, cwd = os.path.join(_SHIZUKU2_DIR, "Shizuku2.exe"), _SHIZUKU2_DIR
        else:
            from rlkit.parallel import slots
            runtime = slots.runtime_for(self.slot)
            runtime.before_launch()
            executable, cwd = self.slot.executable, self.slot.replica_dir
        self.process = await asyncio.create_subprocess_exec(
            executable, cwd=cwd,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        if runtime is not None:
            runtime.after_launch(self.process.pid)
        _launch_log(self.slot, "emulator_started", pid=self.process.pid, executable=executable)
        self.loop.create_task(self._drain(self.process.stderr))

        while True:
            line = await self.process.stdout.readline()
            if not line:
                raise RuntimeError("Shizuku2.exe が起動プロンプトを出す前に終了した")
            if BOOT_READY_MARKER in line.decode(errors="ignore"):
                break
        self.loop.create_task(self._drain(self.process.stdout))
        await asyncio.sleep(2.0)

        self.comms = comm.Communicators(slot=self.slot)

        async def _retry(coro_fn, timeout=30.0):
            t0 = time.monotonic()
            while True:
                ok, val = await coro_fn()
                if ok:
                    return val
                if time.monotonic() - t0 > timeout:
                    raise comm.BacnetError(f"起動シーケンスがタイムアウトした: {val}")
                await asyncio.sleep(0.1)

        await _retry(lambda: self.comms.vrf.change_pause_timer(0))
        self.final_dtime = await _retry(lambda: self.comms.vrf.get_end_dateTime())
        self.process.stdin.write(b"\r\n")
        await self.process.stdin.drain()
        await _retry(lambda: self.comms.vrf.change_acceleration_rate(ACCELERATION_RATE))
        _launch_log(self.slot, "bacnet_ready", pid=self.process.pid, final_dtime=self.final_dtime)

    def close(self):
        if self.loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(self._async_close(), self.loop)
        try:
            fut.result(timeout=30)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=10)
            self._thread = None
            self.loop = None

    async def _async_close(self):
        if self.comms is not None:
            await self.comms.close()
            self.comms = None
        if self.process is not None and self.process.returncode is None:
            self.process.kill()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        if self.slot is not None and self.process is not None:
            from rlkit.parallel import slots
            slots.runtime_for(self.slot).after_exit(self.process.pid, self.process.returncode is not None)
            _launch_log(self.slot, "emulator_exited", pid=self.process.pid, returncode=self.process.returncode)
        self.process = None

    # ---- 時刻駆動 ----

    async def true_sim_datetime(self):
        return await comm.true_sim_datetime(self.comms)

    async def advance(self, seconds, poll_interval=0.02, timeout=60.0):
        """目標のシミュレーション秒数だけ進め、到達した日時を返す（現行と同じ）。"""
        last_dt = await self.true_sim_datetime()
        target = last_dt + datetime.timedelta(seconds=seconds)
        await comm.call_write(self.comms.vrf.change_pause_timer(seconds), "change_pause_timer")
        elapsed = 0.0
        while True:
            now = await self.true_sim_datetime()
            if now >= target:
                return now
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            if elapsed > timeout:
                raise TimeoutError(f"advance({seconds}) がタイムアウト。現在{now}, 目標{target}")

    async def park(self):
        """エピソード終端で一時停止タイマーを遠くへ置き、成績ファイルの書き出しを待つ（現行 env の終端処理と同じ）。"""
        await comm.call_write(self.comms.vrf.change_pause_timer(86400 * 10), "park pause timer")
        await asyncio.sleep(1.0)


def run_coro(drv, coro, timeout=None):
    return asyncio.run_coroutine_threadsafe(coro, drv.loop).result(timeout=timeout)


# =====================================================================
# 規則ベース最良軌跡（夏季）の再生。現行 driver.py の同名関数を移植（ロジックは同一）。
# verify/raw/H1/best_summer_trajectory_v2.json は行動列ではなく規則のパラメータを保存している。
# =====================================================================

SUMMER_EVAP_BASE_C = 18.0
SUMMER_SETPOINT_BASE_C = 24.0
SUMMER_CLOSE_HOUR_MIN = 19 * 60
SUMMER_PEAK_START_HOUR = 14
SUMMER_PEAK_END_HOUR = 16
SOUTH_ZONES_OU_IU = [(1, iu) for iu in range(1, 6)] + [(2, iu) for iu in range(1, 5)]


def _is_occupied_weekday_hour_8_19(dt):
    """規則ベース軌跡の在室判定窓（平日 08:00-19:00）。駆動刻みの 06:00-20:00 とは別の概念。"""
    return dt.weekday() < 5 and 8 <= dt.hour < 19


async def _w(diff_writer, key, value, coro):
    if diff_writer is None:
        return await comm.call_write(coro, description=str(key))
    return await diff_writer.write(key, value, coro)


def _minutes(dt):
    return dt.hour * 60 + dt.minute


def summer_best_zone_state(dt, zone, p2_offset_min):
    """規則ベース最良軌跡の、あるゾーン (ou, iu) のある時刻の運転内容。"""
    on = _is_occupied_weekday_hour_8_19(dt)
    state_votes = []
    if _is_occupied_weekday_hour_8_19(dt) and _minutes(dt) >= SUMMER_CLOSE_HOUR_MIN - p2_offset_min:
        state_votes.append(False)  # p2
    if dt.weekday() < 5 and not _is_occupied_weekday_hour_8_19(dt) and dt.hour == 7 and dt.minute >= 30:
        state_votes.append(True)  # p4
    if False in state_votes:
        on = False
    elif True in state_votes:
        on = True
    setpoint_delta = 0.0
    if zone in SOUTH_ZONES_OU_IU:  # p5
        setpoint_delta += 1.0
    if SUMMER_PEAK_START_HOUR <= dt.hour < SUMMER_PEAK_END_HOUR:  # p10
        setpoint_delta += 1.0
    setpoint = (SUMMER_SETPOINT_BASE_C + setpoint_delta) if on else None
    return {"on": on, "setpoint": setpoint, "mode": "Cooling" if on else None,
            "fan_speed": "High" if on else None, "direction": "Horizontal" if on else None}


def make_summer_best_schedule(p2_offset_min, diff_writer=None):
    """規則ベース最良軌跡の再生スケジュール（p2, p4, p5, p10 すべて有効）。"""
    from szklib.VRFSystemCommunicator import VRFSystemCommunicator as vrc

    async def apply(driver, dt, is_boundary):
        comms = driver.comms
        for i in range(len(constants.I_UNIT_NUM)):
            ou = i + 1
            await _w(diff_writer, ("sys_refrig_ctrl", ou), True, comms.vrf.enable_refrigerant_temperatureControl(ou))
            await _w(diff_writer, ("sys_evap", ou), SUMMER_EVAP_BASE_C, comms.vrf.change_evaporating_temperature(ou, SUMMER_EVAP_BASE_C))
            assert constants.SUMMER_COND_TEMP_C < constants.CONDENSING_TEMP_MAX_C
            await _w(diff_writer, ("sys_cond", ou), constants.SUMMER_COND_TEMP_C,
                     comms.vrf.change_condensing_temperature(ou, constants.SUMMER_COND_TEMP_C))
            for j in range(constants.I_UNIT_NUM[i]):
                iu = j + 1
                st = summer_best_zone_state(dt, (ou, iu), p2_offset_min)
                if st["on"]:
                    await _w(diff_writer, ("zone_on", ou, iu), True, comms.vrf.turn_on(ou, iu))
                    await _w(diff_writer, ("zone_mode", ou, iu), "Cooling", comms.vrf.change_mode(ou, iu, vrc.Mode.Cooling))
                    await _w(diff_writer, ("zone_sp", ou, iu), st["setpoint"], comms.vrf.change_setpoint_temperature(ou, iu, st["setpoint"]))
                    await _w(diff_writer, ("zone_fan", ou, iu), "High", comms.vrf.change_fan_speed(ou, iu, vrc.FanSpeed.High))
                    await _w(diff_writer, ("zone_dir", ou, iu), "Horizontal", comms.vrf.change_direction(ou, iu, vrc.Direction.Horizontal))
                    await _w(diff_writer, ("zone_permit", ou, iu), True, comms.vrf.permit_local_control(ou, iu))
                else:
                    await _w(diff_writer, ("zone_on", ou, iu), False, comms.vrf.turn_off(ou, iu))
        await _apply_standard_ventilation(driver, dt, diff_writer)
    return apply


async def _apply_standard_ventilation(driver, dt, diff_writer=None):
    """換気の既定ポリシー（平日 08:00-19:00 のみ、テナント全台で中風量・バイパス無効）。"""
    from szklib.VentilationSystemCommunicator import VentilationSystemCommunicator as vsc
    comms = driver.comms
    occupied = _is_occupied_weekday_hour_8_19(dt)
    for ou in range(1, 5):
        for iu in range(1, constants.I_UNIT_NUM[ou - 1] + 1):
            if occupied:
                await _w(diff_writer, ("vent_on", ou, iu), True, comms.vent.start_ventilation(ou, iu))
                await _w(diff_writer, ("vent_bypass", ou, iu), False, comms.vent.disable_bypass_control(ou, iu))
                await _w(diff_writer, ("vent_fan", ou, iu), "Middle", comms.vent.change_fan_speed(ou, iu, vsc.FanSpeed.Middle))
            else:
                await _w(diff_writer, ("vent_on", ou, iu), False, comms.vent.stop_ventilation(ou, iu))


def load_summer_best_trajectory(path=None):
    path = path or os.path.join(_REPO_ROOT, "verify", "raw", "H1", "best_summer_trajectory_v2.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
