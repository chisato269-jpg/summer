"""rlkit.comm — BACnet 通信を逐次で行う薄い層（現行 rlkit/comm.py を移植し、v2 の読み出しを追加）。

szklib の各 Communicator を直接使う。読み書きの失敗は再試行せず BacnetError として送出する（現行と同じ）。

v2 で追加した読み出し:
  read_outdoor_unit_power       室外機電力 P_ou,s ×4（v2 第 5-1-2 節、毎ステップ実測）
  read_indoor_unit_power        室内機電力 P_iu,k ×18（J-1 で「加える」と判定した場合のみ使う）
  read_setpoint_readback        設定温度の読み戻し ×18（観測 +29・+30）
  read_tenant_thermal_sensation テナント別の執務者温冷感 7 段階の人数（観測 66–79）
"""
import asyncio
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from szklib.VRFSystemCommunicator import VRFSystemCommunicator
from szklib.VentilationSystemCommunicator import VentilationSystemCommunicator
from szklib.EnvironmentCommunicator import EnvironmentCommunicator
from szklib.OccupantCommunicator import OccupantCommunicator

from rlkit import constants

Tenant = OccupantCommunicator.Tenant

# クライアント側の BACnet デバイス ID（現行と同じ。エミュレータ側の 1〜6・9 とは別物）
CLIENT_DEVICE_IDS = {"vrf": 12, "vent": 13, "env": 14, "occ": 15}

# DateTimeController のアドレス（組を指定しないとき）
DATETIME_CONTROLLER_ADDR = "127.0.0.1:47809"

COMM_STATS = {"datetime_read_failures": 0}


class BacnetError(RuntimeError):
    pass


class Communicators:
    """4 つの Communicator を束ねる。slot は rlkit.parallel.slots.SlotSpec（None は原本のエミュレータ）。"""

    def __init__(self, slot=None):
        if slot is None:
            device_ip, emulator_ip, ids = "127.0.0.1", "127.0.0.1", CLIENT_DEVICE_IDS
            self.datetime_addr = DATETIME_CONTROLLER_ADDR
        else:
            device_ip, emulator_ip, ids = slot.device_ip, slot.emulator_ip, slot.client_ids
            self.datetime_addr = slot.datetime_addr
        self.vrf = VRFSystemCommunicator(id=ids["vrf"], device_ip=device_ip, emulator_ip=emulator_ip)
        self.vent = VentilationSystemCommunicator(id=ids["vent"], device_ip=device_ip, emulator_ip=emulator_ip)
        self.env = EnvironmentCommunicator(id=ids["env"], device_ip=device_ip, emulator_ip=emulator_ip)
        self.occ = OccupantCommunicator(id=ids["occ"], device_ip=device_ip, emulator_ip=emulator_ip)

    def all(self):
        return [self.vrf, self.vent, self.env, self.occ]

    async def close(self):
        """BACnet デバイスを閉じる（現行と同じく、ソケット解放の競合を避けるため sleep を挟む）。"""
        for c in self.all():
            try:
                c.bacdevice.close()
            except Exception:
                pass
        await asyncio.sleep(0.25)


async def _check(result, description):
    ok, value = result
    if not ok:
        raise BacnetError(f"{description} に失敗: {value}")
    return value


async def call_write(coro, description=""):
    """szklib の読み書きメソッド呼び出し（コルーチン）を 1 回 await し、失敗なら送出する。"""
    return await _check(await coro, description)


async def true_sim_datetime(comms):
    """is_paused キャッシュに依存せず、現在のシミュレーション日時を BACnet で直接読む（現行と同じ）。"""
    last_err = None
    for _ in range(30):
        ok, val = await comms.vrf.read_present_value(comms.datetime_addr, "datetimeValue:1")
        if ok:
            return val
        last_err = val
        COMM_STATS["datetime_read_failures"] += 1
        await asyncio.sleep(1.0)
    raise BacnetError(f"CurrentDateTimeInSimulation の読み取りに失敗（30回リトライ後）: {last_err}")


async def read_step_snapshot(comms):
    """建物全体・18 ゾーン・2 テナントの物理量（現行 read_step_snapshot と同じ 172 点）。"""
    snap = {}
    b = {}
    b["drybulb_c"] = await call_write(comms.env.get_drybulb_temperature(), "building drybulb")
    b["relative_humidity_pct"] = await call_write(comms.env.get_relative_humidity(), "building RH")
    b["global_horizontal_radiation"] = await call_write(comms.env.get_global_horizontal_radiation(), "GHR")
    b["nocturnal_radiation"] = await call_write(comms.env.get_nocturnal_radiation(), "nocturnal radiation")
    b["total_energy_consumption_mj"] = await call_write(comms.env.get_total_energy_consumption(), "total energy")
    b["averaged_dissatisfaction_rate"] = await call_write(comms.env.get_averaged_dissatisfaction_rate(), "avg DR")
    b["instantaneous_energy_consumption_kw"] = await call_write(comms.env.get_instantaneous_energy_consumption(), "inst energy")
    b["instantaneous_dissatisfaction_rate"] = await call_write(comms.env.get_instantaneous_dissatisfaction_rate(), "inst DR")
    snap["building"] = b

    zones = {}
    for ou, iu, tenant_name, zn, label, area in constants.ZONE_DEFS:
        tenant = Tenant.South if tenant_name == "South" else Tenant.North
        z = {}
        z["drybulb_c"] = await call_write(comms.env.get_zone_drybulb_temperature(ou, iu), f"{label} drybulb")
        z["relative_humidity_pct"] = await call_write(comms.env.get_zone_relative_humidity(ou, iu), f"{label} RH")
        z["occupant_number"] = await call_write(comms.occ.get_zone_occupant_number(tenant, zn), f"{label} occupants")
        z["thermal_sensation"] = await call_write(comms.occ.get_averaged_thermal_sensation(tenant, zn), f"{label} TS")
        z["clothing_index"] = await call_write(comms.occ.get_averaged_clothing_index(tenant, zn), f"{label} clo")
        z["dissatisfied_thermal"] = await call_write(comms.occ.get_thermally_dissatisfied_rate(tenant, zn), f"{label} DR_thermal")
        z["dissatisfied_draft"] = await call_write(comms.occ.get_dissatisfied_rate_caused_by_draft(tenant, zn), f"{label} DR_draft")
        z["dissatisfied_vtmp"] = await call_write(
            comms.occ.get_dissatisfied_rate_caused_by_vertical_temperature_distribution(tenant, zn), f"{label} DR_vtmp")
        z["heatload_kw"] = await call_write(comms.vrf.get_indoor_unit_heatload(ou, iu), f"{label} heatload")
        zones[label] = z
    snap["zones"] = zones

    south_co2 = await call_write(comms.vent.get_south_tenant_CO2_level(), "south CO2")
    north_co2 = await call_write(comms.vent.get_north_tenant_CO2_level(), "north CO2")
    snap["co2_ppm"] = {"South": south_co2, "North": north_co2}
    return snap


async def read_outdoor_unit_power(comms):
    """室外機電力[kW]（系統 1〜4）。"""
    return [float(await call_write(comms.vrf.get_outdoor_unit_electricity(ou), f"outdoor unit {ou} electricity"))
            for ou in range(1, 5)]


async def read_indoor_unit_power(comms):
    """室内機電力[kW]（constants.ZONE_DEFS の並び）。"""
    out = []
    for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS:
        out.append(float(await call_write(comms.vrf.get_indoor_unit_electricity(ou, iu), f"{label} indoor unit electricity")))
    return out


async def read_setpoint_readback(comms):
    """室内機の設定温度の状態値[℃]（constants.ZONE_DEFS の並び）。執務者のリモコン操作を反映した実際の値。"""
    out = []
    for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS:
        out.append(float(await call_write(comms.vrf.get_setpoint_temperature(ou, iu), f"{label} setpoint status")))
    return out


async def read_tenant_thermal_sensation(comms):
    """テナント別に、在室している執務者の温冷感 7 段階（Cold〜Hot）の人数を数える（観測 66–79）。

    執務者番号で引く API（get_thermal_sensation）しかなく、在室者の番号は分からないため、
    テナントの在室者数が 1 人以上のときに全執務者の在室状況（is_occupant_stay_in_office）を読み、
    在室している執務者についてだけ温冷感を読む。温冷感は最も近い整数に丸めて −3〜3 に収める。
    Returns: {"South": {"counts": [7 個], "n_present": int}, "North": {...}}
    """
    out = {}
    for tenant_name in constants.TENANTS:
        tenant = Tenant.South if tenant_name == "South" else Tenant.North
        counts = [0] * len(constants.THERMAL_SENSATION_LEVELS)
        n_present = 0
        n_tenant = await call_write(comms.occ.get_occupant_number(tenant), f"{tenant_name} occupant number")
        if n_tenant > 0:
            for i in range(1, constants.TENANT_OCCUPANT_COUNT[tenant_name] + 1):
                stay = await call_write(comms.occ.is_occupant_stay_in_office(tenant, i), f"{tenant_name} occupant {i} availability")
                if not stay:
                    continue
                ts = await call_write(comms.occ.get_thermal_sensation(tenant, i), f"{tenant_name} occupant {i} thermal sensation")
                level = int(math.floor(float(ts) + 0.5))
                level = min(max(level, constants.THERMAL_SENSATION_LEVELS[0]), constants.THERMAL_SENSATION_LEVELS[-1])
                counts[level - constants.THERMAL_SENSATION_LEVELS[0]] += 1
                n_present += 1
        out[tenant_name] = {"counts": counts, "n_present": n_present}
    return out


class DiffWriter:
    """前回書き込んだ値を記憶し、同一なら書き込みを省く（現行と同じ）。"""

    def __init__(self, write_all=False):
        self._last = {}
        self.write_all = write_all
        self.n_writes = 0
        self.n_skipped = 0

    async def write(self, key, value, coro):
        """key: 書き込み対象を一意に表すタプル。coro: 実際に書き込むコルーチン（省くときは close する）。"""
        if not self.write_all and key in self._last and self._last[key] == value:
            coro.close()
            self.n_skipped += 1
            return False
        await call_write(coro, description=str(key))
        self._last[key] = value
        self.n_writes += 1
        return True
