"""rlkit.comm_rpm — BACnet の一括読み取り（ReadPropertyMultiple）による 1 ステップの読み取り（並列数拡張マニュアルの第 5 節）。

冬季機（21(w)_Shizuku2-RL-kit）の rlkit/comm_rpm.py を夏季機の読み取り関数に合わせて移植したもの。

目的: 1 ステップに約 390 回ある読み取りは、1 回あたり約 0.6 ms のうちほぼ全部がクライアントの CPU（bacpypes3）なので、
  1 つの要求に複数のオブジェクトをまとめて、CPU と実時間を減らす。
方式: 読む点（宛先・オブジェクト）の一覧は、既存の逐次の読み取り関数をそのまま「計画モード」で走らせて作る。
  計画モードでは各通信の read_present_value を、通信せずに宛先とオブジェクトを記録して 0 を返す関数に差し替える。
  実行では宛先ごとに CHUNK 点ずつ一括で読み、値を表（宛先, オブジェクト → 値）に入れる。
  最後に、同じ逐次の関数を「再生モード」で走らせる（read_present_value を表の引き当てに差し替える）。
  これで、返す値の形・キーの順・値の変換は逐次版と同一になる（コードを複写しない）。
  執務者ごとの読み取り（read_tenant_thermal_sensation）は 3 段（在室者数 → 全執務者の在室確認 → 在室者の温冷感）で、どの段も一括で読む。
失敗（時間切れ・個別のエラー）は、その塊だけを 1 回ずつの読み取りに戻す。戻した回数は COMM_RPM_STATS に数える。

夏季機の 1 ステップで通る読み取り関数（rlkit/env.py）: read_step_snapshot、read_outdoor_unit_power、read_indoor_unit_power、
read_setpoint_readback、read_tenant_thermal_sensation、read_remote_permission_status、read_onoff_status。install() は
これらを一括版に差し替え、元の逐次版は comm.<名前>_sequential に残す。
"""
import asyncio
import contextlib

from rlkit import comm, constants

CHUNK = 40                 # 1 要求にまとめる点数（冬季機の探査で 50 は成功、100 は無応答。夏季機でも探査で確かめる）
RPM_TIMEOUT_S = 2.0
COMM_RPM_STATS = {"requests": 0, "points": 0, "fallback_chunks": 0, "fallback_points": 0}

# 差し替える読み取り関数（comm の属性名）
PATCHED = ("read_step_snapshot", "read_outdoor_unit_power", "read_indoor_unit_power", "read_setpoint_readback",
           "read_remote_permission_status", "read_onoff_status")
THERMAL = "read_tenant_thermal_sensation"


@contextlib.contextmanager
def _patched_reads(comms, fn):
    """4 つの通信（vrf・vent・env・occ）の read_present_value を fn(owner, addr, obj_id) に差し替える。"""
    owners = [getattr(comms, n) for n in ("vrf", "vent", "env", "occ")]
    saved = []
    for o in owners:
        had = "read_present_value" in o.__dict__
        saved.append((o, had, o.__dict__.get("read_present_value")))

        async def _read(addr, obj_id, _o=o):
            return await fn(_o, addr, obj_id)
        o.read_present_value = _read
    try:
        yield
    finally:
        for o, had, old in saved:
            if had:
                o.read_present_value = old
            else:
                del o.__dict__["read_present_value"]


async def _plan(comms, coro_fn, value=0.0):
    """coro_fn(comms) を計画モードで走らせ、読む点 [(owner, addr, obj_id)] を重複なしの順序つきで返す。
    value: すべての読み取りに返す値。0 なら在室者数が 0 に見えるので、執務者ごとの読み取りは計画に入らない。"""
    seen, order = set(), []

    async def fake(owner, addr, obj_id):
        key = (id(owner), addr, obj_id)
        if key not in seen:
            seen.add(key)
            order.append((owner, addr, obj_id))
        return True, value

    with _patched_reads(comms, fake):
        await coro_fn(comms)
    return order


async def _rpm(owner, addr, objs):
    """宛先 addr の objs（オブジェクト ID の文字列のリスト）の present-value を 1 回の要求で読む。値のリストを返す。"""
    from bacpypes3.pdu import Address
    from bacpypes3.primitivedata import ObjectIdentifier
    plist = []
    for obj_id in objs:
        plist += [ObjectIdentifier(obj_id), ["present-value"]]
    resp = await asyncio.wait_for(owner.bacdevice.read_property_multiple(Address(addr), plist), timeout=RPM_TIMEOUT_S)
    values = [r[3] for r in resp]
    if len(values) != len(objs):
        raise RuntimeError(f"一括読み取りの返り値の数が違う: {len(values)} != {len(objs)}")
    for v in values:
        if v is None or type(v).__name__ in ("ErrorType", "ErrorRejectAbortNack") or isinstance(v, Exception):
            raise RuntimeError(f"一括読み取りに個別のエラー: {v!r}")
    return values


def _single(owner, addr, obj_id):
    """1 回ずつの読み取り（戻し先）。元の read_present_value を使う。"""
    cls = next(c for c in type(owner).__mro__ if c.__name__ == "PresentValueReadWriter")
    return cls.read_present_value(owner, addr, obj_id)


async def fetch(points):
    """points: [(owner, addr, obj_id)]。宛先ごと・CHUNK 点ずつ一括で読み、{(id(owner), addr, obj_id): 値} を返す。
    失敗した塊は 1 回ずつの読み取りに戻す。"""
    groups = {}
    for owner, addr, obj_id in points:
        groups.setdefault((id(owner), addr), (owner, []))[1].append(obj_id)
    jobs = []
    for (oid, addr), (owner, objs) in groups.items():
        for i in range(0, len(objs), CHUNK):
            jobs.append((owner, addr, objs[i:i + CHUNK]))

    async def one(owner, addr, objs):
        COMM_RPM_STATS["requests"] += 1
        COMM_RPM_STATS["points"] += len(objs)
        try:
            vals = await _rpm(owner, addr, objs)
        except Exception:
            COMM_RPM_STATS["fallback_chunks"] += 1
            COMM_RPM_STATS["fallback_points"] += len(objs)
            vals = []
            for obj_id in objs:
                vals.append(await comm.call_write(_single(owner, addr, obj_id), f"{addr} {obj_id}"))
        return owner, addr, objs, vals

    # 通信（宛先のデバイス）ごとに同時に出す要求は 1 つ。デバイスの間は並行にする
    per_owner = {}
    for j in jobs:
        per_owner.setdefault(id(j[0]), []).append(j)

    async def chain(job_list):
        out = []
        for owner, addr, objs in job_list:
            out.append(await one(owner, addr, objs))
        return out

    table = {}
    results = await asyncio.gather(*[chain(v) for v in per_owner.values()])
    for res in results:
        for owner, addr, objs, vals in res:
            for obj_id, v in zip(objs, vals):
                table[(id(owner), addr, obj_id)] = v
    return table


async def _replay(comms, coro_fn, table):
    """coro_fn(comms) を再生モードで走らせる。読み取りは表の引き当て（表に無ければ誤り）。"""
    async def lookup(owner, addr, obj_id):
        key = (id(owner), addr, obj_id)
        if key not in table:
            raise KeyError(f"一括読み取りの表に無い点: {addr} {obj_id}")
        return True, table[key]

    with _patched_reads(comms, lookup):
        return await coro_fn(comms)


def _make_batched(seq_fn):
    """逐次の読み取り関数 seq_fn(comms, ...) を、計画 → 一括の取得 → 再生 の 3 手で実行する関数にする。"""
    async def batched(comms, *args, **kwargs):
        async def run(c):
            return await seq_fn(c, *args, **kwargs)
        pts = await _plan(comms, run)
        table = await fetch(pts)
        return await _replay(comms, run, table)
    batched.__name__ = seq_fn.__name__ + "_rpm"
    return batched


async def _occupant_points(comms, tenant_name, idxs, what):
    tenant = comm.Tenant.South if tenant_name == "South" else comm.Tenant.North
    occ = comms.occ
    pts = []

    async def fake(owner, addr, obj_id):
        pts.append((owner, addr, obj_id))
        return True, 0.0

    with _patched_reads(comms, fake):
        for i in idxs:
            if what == "presence":
                await occ.is_occupant_stay_in_office(tenant, i)
            else:
                await occ.get_thermal_sensation(tenant, i)
    return pts


async def read_tenant_thermal_sensation_rpm(comms):
    """comm.read_tenant_thermal_sensation（逐次版）と同じ値を、3 段の一括読み取りで作る。"""
    seq = comm.read_tenant_thermal_sensation_sequential

    async def run(c):
        return await seq(c)

    # 段 1: テナントごとの在室者数（0 を返す計画では執務者ごとの読み取りは入らない）
    pts = await _plan(comms, run, value=0.0)
    table = await fetch(pts)
    tenants = {"South": comm.Tenant.South, "North": comm.Tenant.North}
    counts = {}
    for name, tenant in tenants.items():
        got = []

        async def fake(owner, addr, obj_id, got=got):
            got.append((owner, addr, obj_id))
            return True, 0.0

        with _patched_reads(comms, fake):
            await comms.occ.get_occupant_number(tenant)
        counts[name] = float(table[(id(got[0][0]), got[0][1], got[0][2])])
    # 段 2: 在室者がいるテナントの全執務者の在室確認
    presence_pts, idx_by = [], {}
    for name in tenants:
        if counts[name] > 0:
            idxs = list(range(1, constants.TENANT_OCCUPANT_COUNT[name] + 1))
            idx_by[name] = idxs
            presence_pts += await _occupant_points(comms, name, idxs, "presence")
    if presence_pts:
        table.update(await fetch(presence_pts))
    # 段 3: 在室者の温冷感
    thermal_pts = []
    for name, idxs in idx_by.items():
        pres = await _occupant_points(comms, name, idxs, "presence")
        present_idxs = [i for i, p in zip(idxs, pres) if table[(id(p[0]), p[1], p[2])]]
        thermal_pts += await _occupant_points(comms, name, present_idxs, "thermal")
    if thermal_pts:
        table.update(await fetch(thermal_pts))
    return await _replay(comms, run, table)


_INSTALLED = {}


def install():
    """comm の読み取り関数を一括読み取り版に差し替える（元の逐次版は comm.<名前>_sequential に残す）。
    同じプロセスで 2 回呼んでも二重に差し替えない。戻り値: 元に戻す関数。"""
    if _INSTALLED:
        return _restore
    for name in PATCHED + (THERMAL,):
        orig = getattr(comm, name)
        setattr(comm, name + "_sequential", orig)
        _INSTALLED[name] = orig
    for name in PATCHED:
        setattr(comm, name, _make_batched(_INSTALLED[name]))
    setattr(comm, THERMAL, read_tenant_thermal_sensation_rpm)
    return _restore


def _restore():
    for name, orig in list(_INSTALLED.items()):
        setattr(comm, name, orig)
    _INSTALLED.clear()


def is_installed():
    return bool(_INSTALLED)
