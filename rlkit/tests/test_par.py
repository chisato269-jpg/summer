"""並列数拡張（並列数拡張マニュアル第 5・7・8 節）の単体試験。エミュレータは使わない（模擬の通信と模擬の環境）。

エミュレータを使う確認（D-4 の探査、D-5 の逐次との全点一致、組数ごとの 16 進一致）は rlkit/diag/v7s_par.py で行い、
結果は runs/diag/V7S/par/ にある。測定・学習・試験を同時に走らせない（組のロックを壊す。マニュアル B-1）。
"""
import asyncio
import json
import math
import os

import numpy as np
import pytest

from rlkit import comm, comm_rpm, config, constants
from rlkit import train_summer as ts


# ---- 模擬の通信 ----

class PresentValueReadWriter:        # comm_rpm._single がクラス名でこの基底を探す
    def __init__(self, id_, log):
        self.id = id_
        self.log = log
        self.bacdevice = _FakeApp(self)

    @staticmethod
    def value(addr, key):
        h = sum(ord(c) * (i + 3) for i, c in enumerate(addr + key))
        return float(h % 997) / 7.0

    async def read_present_value(self, addr, obj_id):
        from bacpypes3.primitivedata import ObjectIdentifier
        self.log["single"] += 1
        return True, self.value(addr, str(ObjectIdentifier(obj_id)))


class _FakeApp:
    def __init__(self, owner):
        self.owner = owner
        self.fail = False

    async def read_property_multiple(self, address, plist):
        self.owner.log["rpm"] += 1
        self.owner.log["points"] += len(plist) // 2
        if self.fail:
            raise asyncio.TimeoutError()
        out = []
        for i in range(0, len(plist), 2):
            out.append((plist[i], "present-value", None, self.owner.value(str(address), str(plist[i]))))
        return out


class FakeVrf(PresentValueReadWriter):
    async def get_a(self, i):
        return await self.read_present_value("127.0.0.1:47810", f"analogInput:{i}")


class FakeEnv(PresentValueReadWriter):
    async def get_b(self, i):
        return await self.read_present_value("127.0.0.1:47811", f"analogInput:{i}")


class FakeOcc(PresentValueReadWriter):
    """執務者ごとの読み取りの 3 段を再現する（在室者数 → 在室確認 → 温冷感）。"""
    N = {"South": 5, "North": 4}

    async def get_occupant_number(self, tenant):
        ok, v = await self.read_present_value("127.0.0.1:47812", f"analogInput:{100 + int(tenant.value)}")
        return ok, 3.0 if v >= 0 else 0.0

    async def is_occupant_stay_in_office(self, tenant, i):
        ok, v = await self.read_present_value("127.0.0.1:47812", f"binaryInput:{int(tenant.value) * 1000 + i}")
        return ok, int(v) % 2 == 0

    async def get_thermal_sensation(self, tenant, i):
        return await self.read_present_value("127.0.0.1:47812", f"analogInput:{int(tenant.value) * 1000 + i}")


class FakeComms:
    def __init__(self):
        self.log = {"single": 0, "rpm": 0, "points": 0}
        self.vrf = FakeVrf(1, self.log)
        self.vent = FakeEnv(2, self.log)
        self.env = FakeEnv(3, self.log)
        self.occ = FakeOcc(4, self.log)


async def seq_many(comms, n=95):
    out = {}
    for i in range(n):
        ok, v = await comms.vrf.get_a(i)
        out[("a", i)] = v
    for i in range(7):
        ok, v = await comms.env.get_b(i)
        out[("b", i)] = v
    return out


def run(coro):
    return asyncio.run(coro)


def test_rpm_matches_sequential_and_chunks():
    """一括版の返り値が逐次版と同一で、要求の数が宛先ごとに ceil(点数 / CHUNK) になる。"""
    c1, c2 = FakeComms(), FakeComms()
    seq = run(seq_many(c1))
    batched = comm_rpm._make_batched(seq_many)
    stats0 = dict(comm_rpm.COMM_RPM_STATS)
    rpm = run(batched(c2))
    assert rpm == seq
    assert c2.log["single"] == 0 and c2.log["rpm"] == math.ceil(95 / comm_rpm.CHUNK) + 1     # vrf 95 点（3 要求）+ env 7 点（1 要求）
    assert c2.log["points"] == 95 + 7
    assert comm_rpm.COMM_RPM_STATS["requests"] - stats0["requests"] == c2.log["rpm"]
    assert comm_rpm.COMM_RPM_STATS["fallback_chunks"] == stats0["fallback_chunks"]
    assert c1.log["single"] == 102                                                         # 逐次版は 1 点ずつ 102 回


def test_rpm_failure_falls_back_to_single_reads():
    """一括読み取りが失敗した塊は 1 回ずつの読み取りに戻り、値は変わらず、戻した数が数えられる。"""
    c1, c2 = FakeComms(), FakeComms()
    seq = run(seq_many(c1))
    c2.vrf.bacdevice.fail = True
    stats0 = dict(comm_rpm.COMM_RPM_STATS)
    rpm = run(comm_rpm._make_batched(seq_many)(c2))
    assert rpm == seq
    assert comm_rpm.COMM_RPM_STATS["fallback_chunks"] - stats0["fallback_chunks"] == 3     # vrf の 3 つの塊すべて
    assert comm_rpm.COMM_RPM_STATS["fallback_points"] - stats0["fallback_points"] == 95
    assert c2.log["single"] == 95


def test_rpm_thermal_sensation_three_stage_matches_sequential(monkeypatch):
    """執務者ごとの 3 段の一括読み取りが、逐次版（comm.read_tenant_thermal_sensation）と同じ値を返す。"""
    monkeypatch.setattr(constants, "TENANT_OCCUPANT_COUNT", {"South": 5, "North": 4})
    seq_fn = comm.read_tenant_thermal_sensation
    monkeypatch.setattr(comm, "read_tenant_thermal_sensation_sequential", seq_fn, raising=False)

    class OccNum(FakeOcc):
        async def get_occupant_number(self, tenant):         # 在室者数は常に正（執務者ごとの読み取りが走る）
            await self.read_present_value("127.0.0.1:47812", f"analogInput:{100 + int(tenant.value)}")
            return True, 3.0
    c1, c2 = FakeComms(), FakeComms()
    c1.occ, c2.occ = OccNum(4, c1.log), OccNum(4, c2.log)
    seq = run(seq_fn(c1))
    rpm = run(comm_rpm.read_tenant_thermal_sensation_rpm(c2))
    assert rpm == seq and rpm["South"]["n_present"] + rpm["North"]["n_present"] > 0
    assert c2.log["single"] == 0 and c2.log["rpm"] <= 6 and c1.log["single"] > c2.log["rpm"]


def test_install_is_idempotent_and_restores():
    names = comm_rpm.PATCHED + (comm_rpm.THERMAL,)
    before = {n: getattr(comm, n) for n in names}
    assert not comm_rpm.is_installed()
    restore = comm_rpm.install()
    try:
        assert comm_rpm.is_installed()
        again = comm_rpm.install()                            # 二重に差し替えない
        for n in names:
            assert getattr(comm, n + "_sequential") is before[n]
            assert getattr(comm, n) is not before[n]
        assert again is restore
    finally:
        restore()
    assert not comm_rpm.is_installed()
    assert all(getattr(comm, n) is before[n] for n in names)


# ---- 設定の検査 ----

def _cfg(**kw):
    c = config.load()
    c.update({"include_indoor_unit_power": False, "zone_step_energy_norm_gj": 0.01, "planned_updates": 10})
    c.update(kw)
    return c


def test_config_slot_layouts_keep_twelve_episodes_per_update():
    config.validate(_cfg())                                                        # 従来: 6 組 × 2 本
    config.validate(_cfg(train_slots=list(range(12)), eval_slot=12, episodes_per_slot=1, slot_limit=13))   # 案 B
    config.validate(_cfg(train_slots=list(range(4)), eval_slot=4, episodes_per_slot=3, slot_limit=7))      # 4 × 3 も 12 本
    with pytest.raises(ValueError):
        config.validate(_cfg(train_slots=list(range(12)), eval_slot=12, episodes_per_slot=2, slot_limit=13))  # 24 本（バッチが変わる）
    with pytest.raises(ValueError):
        config.validate(_cfg(train_slots=list(range(12)), eval_slot=12, episodes_per_slot=1, slot_limit=12))  # 評価の組が上限を超える
    with pytest.raises(ValueError):
        config.validate(_cfg(train_slots=list(range(12)), eval_slot=3, episodes_per_slot=1, slot_limit=13))   # 評価が訓練と重なる


def test_slot_numbering_accepts_high_slots():
    from rlkit.parallel import slots
    assert slots.N_SLOTS >= 32
    s = slots.make_slot(24)
    assert s.address == "127.0.0.25" and s.client_ids["vrf"] == 12 + 240
    assert len({slots.make_slot(k).client_ids["vrf"] for k in range(32)}) == 32
    assert not set(slots.FORBIDDEN_IDS) & {i for k in range(32) for i in slots.make_slot(k).client_ids.values()}


def test_rename_with_retries(tmp_path, monkeypatch):
    from rlkit.parallel import slots
    calls = {"n": 0}
    real = os.rename

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "拒否")
        return real(a, b)
    monkeypatch.setattr(os, "rename", flaky)
    src, dst = tmp_path / "a.partial", tmp_path / "a"
    src.mkdir()
    assert slots.rename_with_retries(str(src), str(dst), attempts=5, delay=0.0) == 3 and dst.is_dir()
    monkeypatch.setattr(os, "rename", lambda a, b: (_ for _ in ()).throw(PermissionError(5, "拒否")))
    with pytest.raises(slots.SlotError):
        slots.rename_with_retries(str(dst), str(tmp_path / "b"), attempts=2, delay=0.0)


# ---- 学習の入口（模擬の環境）で 12 組 × 1 本 ----

def test_trainer_twelve_slots_one_episode_each(tmp_path):
    cfg = _cfg(train_slots=list(range(12)), eval_slot=12, episodes_per_slot=1, slot_limit=13, planned_updates=2,
               run_dir=str(tmp_path))
    tr = ts.Trainer(cfg, str(tmp_path), factory="mock")
    assert tr.eps_per_slot == 1 and tr.n_episodes == 12
    tr.run()
    status = json.load(open(tmp_path / "status.json", encoding="utf-8"))
    assert status["status"] == "stopped", status
    ups = [json.loads(l) for l in open(tmp_path / "updates.jsonl", encoding="utf-8")]
    eps = [json.loads(l) for l in open(tmp_path / "episodes.jsonl", encoding="utf-8")]
    assert len(ups) == 2 and all(u["train_episodes"] == 12 for u in ups)
    assert sorted({e["slot"] for e in eps}) == list(range(12)) and len(eps) == 24
    assert all(k in ups[0] for k in ts.REQUIRED_UPDATE_KEYS)
    best = ups[0]["harvest_update_best"]
    assert 0 <= best["slot"] < 12 and best["step_in_collection"] == 377                    # 1 組 1 本の終端は採取の最後のステップ
    d = np.load(tmp_path / "harvest" / "best_harvest.npz")
    assert d["actions"].shape == (378, 60)
    meta = [json.loads(l) if False else json.load(open(p, encoding="utf-8")) for p in
            [os.path.join(tmp_path, f) for f in os.listdir(tmp_path) if f.startswith("meta_start")]][0]
    assert meta["episodes_per_slot"] == 1 and len(meta["train_slots"]) == 12


def test_trainer_default_layout_unchanged(tmp_path):
    cfg = _cfg(planned_updates=1, run_dir=str(tmp_path))
    tr = ts.Trainer(cfg, str(tmp_path), factory="mock")
    assert tr.eps_per_slot == 2 and tr.n_episodes == 12 and len(tr.cfg["train_slots"]) == 6
    assert tr.env_kwargs(1)["comm_rpm"] is False
