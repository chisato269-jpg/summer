"""段階 Z' の診断 K-9〜K-12（指示書 V6-S 第 6 章）。学習は行わない。

基準の軌跡（runs/v2s_trial04 更新 16 の収穫、第 1 段階の索引）を V5 の書き込み層で再生し、1 条件だけ変える。
規則ベースの条件は再生の道具で指令を上書きして作り、マスクは掛けない。

  Z'-1 使う組ごとに基準の軌跡をそのまま再生し、E・DR が P-2 と 16 進で一致することを確かめる
  K-9  組 6 の Z'-1 の再生で、毎ステップ 18 ゾーンの発停・運転モード・設定温度・リモコン許可の状態を読み戻す
  K-10 既存の走行の分単位の出力（runs/diag/V6S/K10/source）で、運転を続けたステップ内の運転モードを調べる
  K-11 換気の水準（a〜d）。全熱交換器の電力は組の出力 vent.csv（テナント合計、分単位）から求める
  K-12 在室中の停止（a〜c）

使い方（リポジトリルートから）:
  python -m rlkit.diag.v6s_diag --step z1k9     # 7 組で Z'-1（組 6 は K-9 の読み戻しつき）
  python -m rlkit.diag.v6s_diag --step k11k12   # 7 組で K-11 (a)〜(d) と K-12 (a)〜(c)
  python -m rlkit.diag.v6s_diag --step k10      # エミュレータを使わない
"""
import argparse
import collections
import csv
import datetime
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from rlkit import action as action_mod
from rlkit import comm, config as config_mod, constants, driver, env as env_mod, eci as eci_mod, exogenous, runlog, timing
from rlkit import reward as reward_mod
from rlkit.diag import v5s_diag
from rlkit.parallel import launch as launch_mod
from rlkit.parallel import slots

OUT_ROOT = os.path.join("runs", "diag", "V6S")
BASELINE_DIR = os.path.join("runs", "v2s_trial04", "harvest")
BASELINE_E_HEX = "0x1.a228e147ae148p+2"
BASELINE_DR_HEX = "0x1.863c140000000p-3"
ALL_SLOTS = [0, 1, 2, 3, 4, 5, 6]
K9_SLOT = 6
KWH_TO_GJ = constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0
TENANT_ZONES = v5s_diag.TENANT_ZONES
VENT_LEVEL_NAMES = {1: "Low", 2: "Middle", 3: "High"}
# K-11 (b)・(c) の CO2 による切替（取扱説明書 Code 4.7。指示書 V6-S 第 12 章 K11_co2_thresholds_ppm）
K11_CO2_LOW_BELOW_PPM = 600.0
K11_CO2_MIDDLE_BELOW_PPM = 800.0
# K-9 の判定（指示書 V6-S 第 12 章）
K9_ONOFF_MISMATCH_FRAC = 0.01
K9_ONOFF_ENERGY_FRAC = 0.05
K9_SETPOINT_ENERGY_FRAC = 0.10


def load_baseline():
    actions = np.load(os.path.join(BASELINE_DIR, "best_harvest.npz"))["actions"].astype(np.int64)
    meta = json.load(open(os.path.join(BASELINE_DIR, "best_harvest.json"), encoding="utf-8"))
    assert meta["E_hex"] == BASELINE_E_HEX and meta["DR_hex"] == BASELINE_DR_HEX and meta["stage"] == 1
    return actions, meta


# =====================================================================
# 診断用の環境（V5 の書き込み層に、換気の水準の上書きと K-9 の読み戻しを足す）
# =====================================================================

class DiagEnv(env_mod.SummerEnv):
    """vent_rule(env) が {テナント: 水準 0〜3 または None} を返すと、そのステップの換気をその水準で書く。
    None のテナントは行動の換気（運転 → Middle、停止 → 停止）に従う。vent_rule が None なら V5 の書き込みそのもの。"""

    def __init__(self, *args, vent_rule=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vent_rule = vent_rule
        self._vent_levels = None
        self.last_co2 = None

    def _update_m5(self, dt, snap):
        super()._update_m5(dt, snap)
        self.last_co2 = dict(snap["co2_ppm"])

    def step(self, act):
        self._vent_levels = self.vent_rule(self) if self.vent_rule is not None else None
        return super().step(act)

    async def _write_action(self, decoded):
        if self._vent_levels is None:
            return await super()._write_action(decoded)
        # 室内機と系統は V5 と同じ（換気だけを差し替えるため、換気の書き込みを飛ばして呼ぶ）
        saved = decoded["tenants"]
        decoded = dict(decoded)
        decoded["tenants"] = {}
        await super()._write_action(decoded)
        vrc, vsc = env_mod._szk()
        comms = self._driver.comms
        dw = self._dw
        for tenant_name, on in saved.items():
            level = self._vent_levels.get(tenant_name)
            if level is None:
                level = 2 if on else 0
            for label in [l for l in constants.ZONE_LABELS if env_mod.ZONE_INFO[l][2] == tenant_name]:
                ou, iu, tenant, zn, area = env_mod.ZONE_INFO[label]
                await dw.write(("vent_bypass", ou, iu), constants.HEX_BYPASS_ENABLED, comms.vent.disable_bypass_control(ou, iu))
                if level == 0:
                    await dw.write(("vent_on", ou, iu), False, comms.vent.stop_ventilation(ou, iu))
                else:
                    name = VENT_LEVEL_NAMES[level]
                    await dw.write(("vent_fan", ou, iu), name, comms.vent.change_fan_speed(ou, iu, getattr(vsc.FanSpeed, name)))
                    await dw.write(("vent_on", ou, iu), True, comms.vent.start_ventilation(ou, iu))

    @property
    def applied_vent_levels(self):
        return dict(self._vent_levels) if self._vent_levels is not None else None


async def read_k9_status(comms):
    """K-9: 18 ゾーンの発停状態（xx02）・運転モード状態（xx04）・設定温度状態（xx06）・リモコン許可状態（xx14）。"""
    out = {"onoff": [], "mode": [], "sp": [], "remote": []}
    for ou, iu, tenant, zn, label, area in constants.ZONE_DEFS:
        out["onoff"].append(bool(await comm.call_write(comms.vrf.is_turned_on(ou, iu), f"{label} onoff status")))
        mode = await comm.call_write(comms.vrf.get_mode(ou, iu), f"{label} mode status")
        out["mode"].append(mode.name)
        out["sp"].append(float(await comm.call_write(comms.vrf.get_setpoint_temperature(ou, iu), f"{label} sp status")))
        out["remote"].append(bool(await comm.call_write(comms.vrf.is_local_control_permitted(ou, iu), f"{label} remote status")))
    return out


# =====================================================================
# 換気の規則（K-11）
# =====================================================================

def _co2_level(ppm):
    if ppm < K11_CO2_LOW_BELOW_PPM:
        return 1
    if ppm < K11_CO2_MIDDLE_BELOW_PPM:
        return 2
    return 3


def vent_rule_factory(kind):
    """K-11 の条件ごとの換気の規則。M-9 の対象（在室 0 かつ 1 時間先も 0）のテナントは停止。"""
    def rule(env):
        dt = env._pending_dt
        out = {}
        for j, tenant in enumerate(action_mod.VENT_TENANT_ORDER):
            if env._idle_tenants[j]:
                out[tenant] = 0
            elif kind == "a":
                out[tenant] = 1
            elif kind == "d":
                out[tenant] = 3
            elif kind in ("b", "c"):
                if kind == "c" and dt.weekday() >= 5:
                    out[tenant] = 0
                else:
                    out[tenant] = _co2_level(float(env.last_co2[tenant]))
            else:
                raise ValueError(kind)
        return out
    return rule


# =====================================================================
# 行動列の組み立て（K-12）
# =====================================================================

def k12_actions(base, kind):
    """K-12 の条件の行動列（第 1 段階の索引）。"""
    a = np.asarray(base, dtype=np.int64).copy()
    dts = v5s_diag.step_datetimes()
    st0 = action_mod.LAYOUT["zone_state"][0]
    sp0 = action_mod.LAYOUT["zone_setpoint"][0]
    vent0 = action_mod.LAYOUT["tenant_vent"][0]
    off = constants.ZONE_STATE.index("off")
    cool = constants.ZONE_STATE.index("cool")
    vent_off = constants.TENANT_VENT_ON.index(False)
    changed = {"zone_steps_changed": 0, "vent_steps_changed": 0}
    if kind == "a":
        win = np.array([dt.weekday() >= 5 for dt in dts])
        changed["zone_steps_changed"] = int((a[win, st0:st0 + 18] != off).sum())
        changed["vent_steps_changed"] = int((a[win, vent0:vent0 + 2] != vent_off).sum())
        a[win, st0:st0 + 18] = off
        a[win, vent0:vent0 + 2] = vent_off
    elif kind == "b":
        win = np.array([dt.weekday() < 5 and dt.hour >= 19 for dt in dts])
        changed["zone_steps_changed"] = int((a[win, st0:st0 + 18] != off).sum())
        a[win, st0:st0 + 18] = off
        idle = v5s_diag.tenant_idle_mask(constants.M9_FORECAST_WINDOW_H)
        for j, t in enumerate(action_mod.VENT_TENANT_ORDER):
            target = win & idle[t] & (a[:, vent0 + j] != vent_off)
            changed["vent_steps_changed"] += int(target.sum())
            a[target, vent0 + j] = vent_off
    elif kind == "c":
        win_idx = [i for i, dt in enumerate(dts) if dt.weekday() < 5 and 9 <= dt.hour < 18]
        sp22 = constants.ZONE_SETPOINT_C[1].index(22.0)
        first_of_day = {}
        for i in win_idx:
            day = dts[i].date()
            first_of_day.setdefault(day, i)
            on = (i - first_of_day[day]) % 2 == 0
            before = a[i, st0:st0 + 18].copy()
            if on:
                a[i, st0:st0 + 18] = cool
                a[i, sp0:sp0 + 18] = sp22
            else:
                a[i, st0:st0 + 18] = off
            changed["zone_steps_changed"] += int((before != a[i, st0:st0 + 18]).sum())
        changed["window_steps"] = len(win_idx)
    else:
        raise ValueError(kind)
    return a, changed


# =====================================================================
# 1 本の再生（子プロセス）
# =====================================================================

def _dr_components(steps, W):
    """在室者数と経過分で重み付けした不満足率の成分（Σ N·DR_c·Δt / W）。"""
    out = collections.OrderedDict((k, 0.0) for k in ("thermal", "draft", "vtmp", "co2", "combined"))
    for s in steps:
        occ = np.asarray(s["occ_zone"])
        m = s["seconds"] / 60.0
        for key in ("thermal", "draft", "vtmp", "co2", "combined"):
            out[key] += float(np.sum(occ * np.asarray(s["dr_" + key])) * m / W)
    return out


def replay_job(job):
    """job: {slot, label, actions(list), vent_kind, k9}. 終端の成績とステップごとの内訳を返す。"""
    cfg = config_mod.load(os.path.join("rlkit", "config_summer_v2.json"))
    slot = slots.make_slot(job["slot"])
    slots.prepare_replica(slot)
    rule = vent_rule_factory(job["vent_kind"]) if job.get("vent_kind") else None
    env = DiagEnv(stage=1, include_indoor_unit_power=cfg["include_indoor_unit_power"],
                  zone_step_energy_norm_gj=cfg["zone_step_energy_norm_gj"], slot=slot, vent_rule=rule)
    env.set_diag_info(True)
    actions = np.asarray(job["actions"], dtype=np.int64)
    rec = {"label": job["label"], "slot": job["slot"], "vent_kind": job.get("vent_kind"), "steps": [], "k9": []}
    t0 = time.time()
    try:
        env.reset()
        for t in range(actions.shape[0]):
            dt = env._pending_dt
            _obs, _r, done, _trunc, info = env.step(actions[t])
            d = info["diag"]
            co2 = d["co2_ppm"]
            rec["steps"].append({
                "step": t, "dt": dt.isoformat(), "seconds": info["step_seconds"], "delta_E_gj": info["delta_E_gj"],
                "zone_energy_gj": [float(x) for x in info["zone_energy_gj"]],
                "hex_kw_model": [float(x) for x in d["hex_power_kw"]], "q_kw": [float(x) for x in d["q_kw"]],
                "Q_kw": float(np.sum(d["q_kw"]) + d["q_header_kw"]),
                "occ_zone": [float(x) for x in d["occupants"]],
                "dr_thermal": [float(x) for x in d["dissatisfied_thermal"]],
                "dr_draft": [float(x) for x in d["dissatisfied_draft"]],
                "dr_vtmp": [float(x) for x in d["dissatisfied_vtmp"]],
                "dr_co2": [float(reward_mod.dr_co2(co2[constants.ZONE_DEFS[k][2]])) for k in range(constants.N_ZONES)],
                "dr_combined": [float(x) for x in d["zone_dr"]],
                "co2_ppm": co2, "vent_levels": env.applied_vent_levels,
                "idle_tenants": [bool(x) for x in env._idle_tenants],
            })
            if job.get("k9"):
                st = driver.run_coro(env._driver, read_k9_status(env._driver.comms))
                rec["k9"].append(st)
            if done:
                rec["final"] = {k: info[k] for k in ("E_rep", "DR_rep", "eci_official", "eci_smooth", "sum_delta_E_gj",
                                                     "raw_resid_rel", "header_energy_fraction", "setpoint_override_rate",
                                                     "n_steps")}
                break
    finally:
        env.close()
    if "final" not in rec:
        raise RuntimeError(f"{job['label']}: 再生が終端しなかった")
    rec["final"]["E_hex"] = float.hex(rec["final"]["E_rep"])
    rec["final"]["DR_hex"] = float.hex(rec["final"]["DR_rep"])
    rec["elapsed_s"] = time.time() - t0
    # 組の分単位の出力を保全する（次の再生で上書きされるため）
    src = slot.data_dir
    dst = os.path.join(job["out_dir"], "slot_output", job["label"])
    os.makedirs(dst, exist_ok=True)
    for name in ("vent.csv", "vrf.csv", "result.txt"):
        if os.path.exists(os.path.join(src, name)):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    return rec


def _child(job, q):
    try:
        os.environ[launch_mod.LAUNCH_LOG_ENV] = job["launch_dir"]
        rec = replay_job(job)
        raw_dir = os.path.join(job["out_dir"], "raw")
        os.makedirs(raw_dir, exist_ok=True)
        runlog.write_json(os.path.join(raw_dir, job["label"] + ".json"), rec)
        q.put(("ok", job["label"], rec))
    except Exception:
        q.put(("error", job["label"], traceback.format_exc()))


def run_parallel(jobs, out_dir):
    """組ごとに 1 プロセスで並列に再生する（親が 1 回だけ起動前の検査を行う）。"""
    slot_ids = [j["slot"] for j in jobs]
    report = launch_mod.preflight(out_dir, slot_ids)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for j in jobs:
        j = dict(j, out_dir=out_dir, launch_dir=report["log_dir"])
        p = ctx.Process(target=_child, args=(j, q))
        p.start()
        procs.append(p)
    results, errors = {}, {}
    for _ in jobs:
        status, label, payload = q.get()
        (results if status == "ok" else errors)[label] = payload
        print(f"[v6s_diag] {label}: {status}", flush=True)
    for p in procs:
        p.join(timeout=60)
    if errors:
        runlog.write_json(os.path.join(out_dir, "errors.json"), errors)
        raise RuntimeError(f"再生に失敗した組がある: {sorted(errors)}\n" + "\n".join(errors.values()))
    return results


# =====================================================================
# 集計
# =====================================================================

def read_vent_csv(path):
    """組の vent.csv（分単位）→ 日時、CO2（南北）、換気電力（南北）の配列。"""
    rows = list(csv.reader(open(path, encoding="utf-8")))
    head = rows[0]
    i_co2 = [head.index("CO2 level (South)[ppm]"), head.index("CO2 level (North)[ppm]")]
    i_fan = [head.index("Fan electricity (South)[kW]"), head.index("Fan electricity (North)[kW]")]
    dts, co2, fan = [], [], []
    for r in rows[1:]:
        dts.append(datetime.datetime.strptime(r[0] + " " + r[1], "%Y/%m/%d %H:%M:%S"))
        co2.append([float(r[i]) for i in i_co2])
        fan.append([float(r[i]) for i in i_fan])
    return dts, np.array(co2), np.array(fan)


def _minute_step_index(minute_dts, step_starts):
    """分の行（時刻 t）がどのステップに属するか。ステップ i は (start_i, start_i+1] の分を持つ。"""
    starts = np.array([s.timestamp() for s in step_starts])
    out = []
    for d in minute_dts:
        i = int(np.searchsorted(starts, d.timestamp(), side="left")) - 1
        out.append(i)
    return np.array(out)


def summarize_run(rec, out_dir, W):
    steps = rec["steps"]
    f = rec["final"]
    comps = _dr_components(steps, W)
    summ = {"E": f["E_rep"], "DR": f["DR_rep"], "E_hex": f["E_hex"], "DR_hex": f["DR_hex"],
            "eci_official": f["eci_official"], "eci_smooth": f["eci_smooth"], "raw_resid_rel": f["raw_resid_rel"],
            "dr_components": comps, "elapsed_s": rec["elapsed_s"], "slot": rec["slot"]}
    vent_path = os.path.join(out_dir, "slot_output", rec["label"], "vent.csv")
    if os.path.exists(vent_path):
        mdts, co2, fan = read_vent_csv(vent_path)
        step_starts = [datetime.datetime.fromisoformat(s["dt"]) for s in steps]
        sidx = _minute_step_index(mdts, step_starts)
        valid = sidx >= 0
        summ["hex_energy_gj_from_vent_csv"] = float(fan[valid].sum() / 60.0 * KWH_TO_GJ)
        summ["hex_energy_gj_by_tenant"] = {t: float(fan[valid, j].sum() / 60.0 * KWH_TO_GJ)
                                           for j, t in enumerate(constants.TENANTS)}
        # テナント別の CO2（全時間と在室中）
        exo = exogenous.load()
        occ = np.array([exo.zone_occupants_at(d) for d in mdts])
        co2_stats = {}
        for j, t in enumerate(constants.TENANTS):
            occupied = occ[:, TENANT_ZONES[t]].sum(axis=1) > 0
            co2_stats[t] = {"mean_all": float(co2[:, j].mean()), "max_all": float(co2[:, j].max()),
                            "mean_occupied": float(co2[occupied, j].mean()) if occupied.any() else None,
                            "minutes_over_1000_occupied": int((occupied & (co2[:, j] > 1000)).sum())}
        summ["co2"] = co2_stats
        # 水準ごとのテナント合計の換気電力（分の行を、その分が属するステップの水準で分ける）
        levels = collections.defaultdict(list)
        for m in np.flatnonzero(valid):
            st = steps[sidx[m]]
            for j, t in enumerate(constants.TENANTS):
                if st["vent_levels"] is not None:
                    lv = st["vent_levels"][t]
                else:
                    lv = None
                if lv is None:
                    continue
                levels[(t, int(lv))].append(fan[m, j])
        summ["hex_tenant_kw_by_level"] = {f"{t}:{lv}": {"median": float(np.median(v)), "mean": float(np.mean(v)),
                                                        "min": float(np.min(v)), "max": float(np.max(v)), "n_minutes": len(v)}
                                          for (t, lv), v in sorted(levels.items())}
        use = collections.Counter()
        for st in steps:
            if st["vent_levels"] is not None:
                for t, lv in st["vent_levels"].items():
                    use[f"{t}:{lv}"] += st["seconds"] / 3600.0
        summ["vent_level_hours"] = dict(sorted(use.items()))
    return summ


def k9_analysis(rec, base_actions):
    """K-9 の集計（発停・運転モード・設定温度・リモコン許可の、指令との不一致）。"""
    steps, k9 = rec["steps"], rec["k9"]
    n_steps = len(steps)
    sp_grid = constants.ZONE_SETPOINT_C[1]
    cool = constants.ZONE_STATE.index("cool")
    onoff_mis = np.zeros((n_steps, 18), dtype=bool)
    mode_mis = np.zeros((n_steps, 18), dtype=bool)
    sp_mis = np.zeros((n_steps, 18), dtype=bool)
    remote_off = np.zeros((n_steps, 18), dtype=bool)
    cmd_on = np.zeros((n_steps, 18), dtype=bool)
    e_vrf = np.zeros((n_steps, 18))
    total_vrf = 0.0
    total_hex_model = 0.0
    for t in range(n_steps):
        s = steps[t]
        st = k9[t]
        a = base_actions[t]
        for k in range(18):
            on = int(a[k]) == cool
            cmd_on[t, k] = on
            onoff_mis[t, k] = (st["onoff"][k] != on)
            if on:
                mode_mis[t, k] = st["mode"][k] != "Cooling"
                sp_cmd = sp_grid[int(a[18 + k])]
                sp_mis[t, k] = abs(st["sp"][k] - sp_cmd) > constants.SETPOINT_OVERRIDE_TOL_C
            remote_off[t, k] = not st["remote"][k]
        q = np.asarray(s["q_kw"])
        hexk = np.asarray(s["hex_kw_model"])
        Q = s["Q_kw"]
        dE = s["delta_E_gj"]
        if Q > 0:
            e_vrf[t] = dE * (q - hexk) / Q
            e_hex = dE * hexk.sum() / Q
        else:
            e_hex = 0.0
        total_vrf += dE - e_hex
        total_hex_model += e_hex
    n = n_steps * 18
    onoff_frac = float(onoff_mis.sum() / n)
    onoff_energy = float(e_vrf[onoff_mis].sum())
    sp_energy = float(e_vrf[sp_mis].sum())
    out = {
        "zone_steps": n, "cooling_zone_steps": int(cmd_on.sum()),
        "total_vrf_energy_gj": total_vrf, "total_hex_energy_gj_model": total_hex_model,
        "zone_vrf_energy_gj": float(e_vrf.sum()),
        "onoff_mismatch": int(onoff_mis.sum()), "onoff_mismatch_frac": onoff_frac,
        "onoff_mismatch_cmd_on_status_off": int((onoff_mis & cmd_on).sum()),
        "onoff_mismatch_cmd_off_status_on": int((onoff_mis & ~cmd_on).sum()),
        "onoff_mismatch_vrf_energy_gj": onoff_energy, "onoff_mismatch_energy_frac": onoff_energy / total_vrf,
        "mode_mismatch_while_cmd_cool": int(mode_mis.sum()),
        "setpoint_mismatch": int(sp_mis.sum()),
        "setpoint_mismatch_frac_of_cooling": float(sp_mis.sum() / max(cmd_on.sum(), 1)),
        "setpoint_mismatch_vrf_energy_gj": sp_energy, "setpoint_mismatch_energy_frac": sp_energy / total_vrf,
        "remote_prohibited_zone_steps": int(remote_off.sum()),
        "mode_status_counts": dict(collections.Counter(m for st in k9 for m in st["mode"])),
        "onoff_mismatch_by_zone": {constants.ZONE_LABELS[k]: int(onoff_mis[:, k].sum()) for k in range(18)},
        "setpoint_readback_values_when_mismatch": dict(collections.Counter(
            round(k9[t]["sp"][k], 3) for t in range(n_steps) for k in range(18) if sp_mis[t, k])),
    }
    if out["onoff_mismatch_frac"] > K9_ONOFF_MISMATCH_FRAC or out["onoff_mismatch_energy_frac"] > K9_ONOFF_ENERGY_FRAC:
        result = "A"
    elif out["setpoint_mismatch_energy_frac"] > K9_SETPOINT_ENERGY_FRAC:
        result = "B"
    else:
        result = "C"
    out["result"] = result
    out["thresholds"] = {"onoff_mismatch_frac": K9_ONOFF_MISMATCH_FRAC, "onoff_energy_frac": K9_ONOFF_ENERGY_FRAC,
                         "setpoint_energy_frac": K9_SETPOINT_ENERGY_FRAC}
    out["energy_definition"] = ("VRF 電力量 = ΔE_t − 全熱交換器の模型分（ΔE_t·ΣP_hex/Q_t）。ゾーン別の VRF 電力量は "
                                "e_k,t から全熱交換器分を除いた ΔE_t·(q_k − P_hex,k)/Q_t。系統ヘッダ（待機）は全体にだけ含む")
    out["diff_writer_skips_onoff"] = {
        "answer": True,
        "evidence": ("rlkit/env.py の _write_action は発停を dw.write(('zone_on', ou, iu), 値, ...) で書き、"
                     "rlkit/comm.py の DiffWriter.write は同じキーに同じ値を前回書いていれば書き込みを省く"
                     "（constants.USE_DIFF_WRITER = True）。執務者が発停を変えても、指令が変わるまで書き直さない")}
    return out


# =====================================================================
# 手順
# =====================================================================

def run_z1k9(out_dir):
    base, meta = load_baseline()
    jobs = [{"slot": k, "label": f"z1_slot{k}", "actions": base.tolist(), "k9": (k == K9_SLOT)} for k in ALL_SLOTS]
    results = run_parallel(jobs, out_dir)
    W = exogenous.load().total_occupant_minutes()
    z1 = {}
    for k in ALL_SLOTS:
        r = results[f"z1_slot{k}"]
        f = r["final"]
        z1[k] = {"E": f["E_rep"], "DR": f["DR_rep"], "E_hex": f["E_hex"], "DR_hex": f["DR_hex"],
                 "eci_official": f["eci_official"], "raw_resid_rel": f["raw_resid_rel"],
                 "match": f["E_hex"] == BASELINE_E_HEX and f["DR_hex"] == BASELINE_DR_HEX,
                 "elapsed_s": r["elapsed_s"]}
    out = {"baseline": meta, "slots": z1, "all_match": all(v["match"] for v in z1.values())}
    runlog.write_json(os.path.join(out_dir, "Z1_replay.json"), out)
    base_rec = results[f"z1_slot{K9_SLOT}"]
    runlog.write_json(os.path.join(out_dir, "baseline_summary.json"), summarize_run(base_rec, out_dir, W))
    np.savez_compressed(os.path.join(out_dir, "baseline_steps.npz"),
                        steps=np.array([json.dumps(s) for s in base_rec["steps"]]),
                        k9=np.array([json.dumps(s) for s in base_rec["k9"]]))
    k9 = k9_analysis(base_rec, base)
    runlog.write_json(os.path.join(out_dir, "K9_detail.json"), k9)
    decision = {"result": k9["result"], "time": runlog.now_iso(),
                "onoff_mismatch_frac": k9["onoff_mismatch_frac"], "onoff_mismatch_energy_frac": k9["onoff_mismatch_energy_frac"],
                "setpoint_mismatch_energy_frac": k9["setpoint_mismatch_energy_frac"], "thresholds": k9["thresholds"],
                "implementation": {"A": "発停を毎ステップ必ず書き込む。O-2・O-3、A-6 を加える",
                                   "B": "O-3、A-6 を加える", "C": "どれも加えない"}[k9["result"]]}
    runlog.write_json(os.path.join(out_dir, "K9_decision.json"), decision)
    print("Z'-1 全組一致:", out["all_match"])
    for k, v in z1.items():
        print(f"  組{k}: E {v['E_hex']} DR {v['DR_hex']} 一致 {v['match']} ({v['elapsed_s']:.0f} 秒)")
    print("K-9:", json.dumps({k: v for k, v in k9.items() if not isinstance(v, dict)}, ensure_ascii=False, indent=1))
    return out


def run_k11k12(out_dir):
    base, meta = load_baseline()
    jobs = []
    for i, kind in enumerate(("a", "b", "c", "d")):
        jobs.append({"slot": ALL_SLOTS[i], "label": f"k11_{kind}", "actions": base.tolist(), "vent_kind": kind})
    changes = {}
    for i, kind in enumerate(("a", "b", "c")):
        acts, ch = k12_actions(base, kind)
        changes[f"k12_{kind}"] = ch
        jobs.append({"slot": ALL_SLOTS[4 + i], "label": f"k12_{kind}", "actions": acts.tolist()})
    results = run_parallel(jobs, out_dir)
    W = exogenous.load().total_occupant_minutes()
    base_summary = json.load(open(os.path.join(out_dir, "baseline_summary.json"), encoding="utf-8"))
    out = {"baseline": base_summary, "conditions": {}, "k12_action_changes": changes}
    for label, rec in sorted(results.items()):
        s = summarize_run(rec, out_dir, W)
        s["dE"] = s["E"] - base_summary["E"]
        s["dDR"] = s["DR"] - base_summary["DR"]
        s["dECI"] = s["eci_official"] - base_summary["eci_official"]
        s["dDR_components"] = {k: s["dr_components"][k] - base_summary["dr_components"][k] for k in s["dr_components"]}
        out["conditions"][label] = s
    b = out["conditions"]["k11_b"]
    c = out["conditions"]["k11_c"]
    out["k11_c_minus_b"] = {"dE": c["E"] - b["E"], "dDR": c["DR"] - b["DR"]}
    runlog.write_json(os.path.join(out_dir, "K11_K12.json"), out)
    for label, s in sorted(out["conditions"].items()):
        print("%-6s E=%.4f dE=%+.4f DR=%.5f dDR=%+.5f ECI=%.4f hex=%.4f GJ" % (
            label, s["E"], s["dE"], s["DR"], s["dDR"], s["eci_official"], s.get("hex_energy_gj_from_vent_csv", float("nan"))))
    return out


def run_k10(out_dir):
    """K-10: 運転を 2 ステップ以上続けて指令したゾーン・ステップの中で、分単位の運転モードが交互になるか。"""
    base, meta = load_baseline()
    src = os.path.join(out_dir, "source", "slot6")
    info = open(os.path.join(src, "result.txt"), encoding="utf-8").read()
    rows = list(csv.reader(open(os.path.join(src, "vrf.csv"), encoding="utf-8")))
    head = rows[0]
    mdts = [datetime.datetime.strptime(r[0] + " " + r[1], "%Y/%m/%d %H:%M:%S") for r in rows[1:]]
    # ステップの開始時刻は再生の記録（エミュレータの日時）を使う。外生表の時刻は 00:01 起点で 1 分ずれるため
    base_steps = np.load(os.path.join(OUT_ROOT, "baseline_steps.npz"))["steps"]
    step_starts = [datetime.datetime.fromisoformat(json.loads(s)["dt"]) for s in base_steps]
    sidx = _minute_step_index(mdts, step_starts)
    cool = constants.ZONE_STATE.index("cool")
    labels = [f"VRF{ou}-{iu} Mode" for ou, iu, *_ in constants.ZONE_DEFS]
    cols = [head.index(l) for l in labels]
    modes = np.array([[r[c] for c in cols] for r in rows[1:]])
    res = {"daytime": collections.Counter(), "all": collections.Counter()}
    examples = []
    for k in range(18):
        on = base[:, k] == cool
        for i in range(len(step_starts)):
            # 前後どちらかのステップも運転（＝運転を 2 ステップ以上続けて指令した）
            if not on[i] or not ((i > 0 and on[i - 1]) or (i + 1 < len(on) and on[i + 1])):
                continue
            mins = modes[sidx == i, k]
            if mins.size == 0:
                continue
            seq = list(mins)
            trans = collections.Counter((a, b) for a, b in zip(seq[:-1], seq[1:]) if a != b)
            kinds = set(seq)
            dt = step_starts[i]
            scopes = ["all"] + (["daytime"] if dt.weekday() < 5 and 9 <= dt.hour < 18 else [])
            for sc in scopes:
                r = res[sc]
                r["zone_steps"] += 1
                r["minutes"] += len(seq)
                for m in seq:
                    r["min_" + m] += 1
                if "ShutOff" in kinds and ("Cooling" in kinds or "ThermoOff" in kinds):
                    r["steps_with_shutoff_and_running"] += 1
                sc_trans = sum(v for (a, b), v in trans.items() if "ShutOff" in (a, b))
                if sc_trans >= 2:
                    r["steps_shutoff_alternating(>=2 transitions)"] += 1
                th_trans = sum(v for (a, b), v in trans.items() if {a, b} == {"Cooling", "ThermoOff"})
                if th_trans >= 1:
                    r["steps_with_cooling_thermooff_transition"] += 1
                if th_trans >= 2:
                    r["steps_cooling_thermooff_alternating(>=2 transitions)"] += 1
            if len(examples) < 6 and dt.weekday() < 5 and 9 <= dt.hour < 18 and len(trans) > 0:
                examples.append({"zone": constants.ZONE_LABELS[k], "step_start": dt.isoformat(), "minutes": seq})
    out = {"source": os.path.join(src, "vrf.csv"), "source_result_txt": info,
           "note": ("組 6 の出力は V5 の走行終了時の収穫の再生（基準の軌跡を第 2 段階の索引に写したもの。物理量は同じ）。"
                    "運転モード列の値は ShutOff（停止）・Cooling・ThermoOff（運転中の圧縮機の休止）・Heating（開始時刻のみ）"),
           "daytime_definition": "平日 9:00 以上 18:00 未満のステップ",
           "results": {k: dict(v) for k, v in res.items()}, "examples": examples}
    runlog.write_json(os.path.join(out_dir, "K10.json"), out)
    print(json.dumps(out["results"], ensure_ascii=False, indent=1))
    return out


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", required=True, choices=["z1k9", "k11k12", "k10"])
    a = p.parse_args(argv)
    runlog.check_thread_env()
    out_dir = {"z1k9": OUT_ROOT, "k11k12": OUT_ROOT, "k10": os.path.join(OUT_ROOT, "K10")}[a.step]
    os.makedirs(out_dir, exist_ok=True)
    {"z1k9": run_z1k9, "k11k12": run_k11k12, "k10": run_k10}[a.step](out_dir)


if __name__ == "__main__":
    main()
