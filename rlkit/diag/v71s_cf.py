"""指示書 V7.1-S 第 4-1 節: 反事実の再生 CF-0〜CF-8（報告のみ。学習には使わない。NOT-1）。

更新 330 の収穫の最良の行動列（60 次元 × 378 ステップ）を土台に、冷房中のゾーン・ステップの設定温度と風量の索引だけを
変えて 9 本を 9 組で同時に再生し、E・DR・P_E・P_DR・ECI と内訳を記録する。走行ディレクトリ runs/v2s_trial08 には何も書かない
（MUST-16。読むのは行動列の保存ファイルと保存点だけ）。結果は runs/diag/V71S/cf/ に書く。

  python -m rlkit.diag.v71s_cf

変える対象は設定温度の索引（格子 22〜28 ℃ の 7 択）と風量の索引（Low / Middle / High）に限る。運転状態、停止ゾーン、蒸発温度、
換気の水準は変えない。
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rlkit import action as action_mod
from rlkit import constants, reward as reward_mod

RUN_DIR = os.path.join("runs", "v2s_trial08")
OUT_DIR = os.path.join("runs", "diag", "V71S", "cf")
CONFIG = os.path.join("rlkit", "config_summer_v7p.json")
SETPOINT_GRID = constants.ZONE_SETPOINT_C[1]
FAN_NAMES = constants.ZONE_FAN_SPEED
COOL = constants.ZONE_STATE.index("cool")
SP0, FAN0, STATE0 = (action_mod.LAYOUT["zone_setpoint"][0], action_mod.LAYOUT["zone_fan"][0], action_mod.LAYOUT["zone_state"][0])
N_Z = constants.N_ZONES

# 名前: (説明, 設定温度の変更, 風量の変更)。設定温度の変更は (対象の設定温度 [℃] の集合 または "all", 変更後 [℃])。
CF_DEFS = {
    0: ("変更なし（更新 330 の収穫と 16 進一致するはず）", None, None),
    1: ("冷房中で 22 ℃を選んだゾーン・ステップだけ 23 ℃へ", ({22.0}, 23.0), None),
    2: ("同 22 ℃ → 24 ℃", ({22.0}, 24.0), None),
    3: ("同 22 ℃ → 26 ℃", ({22.0}, 26.0), None),
    4: ("冷房中の全ゾーン・ステップを 26 ℃へ", ("all", 26.0), None),
    5: ("冷房中の High を Middle へ（全ゾーン・ステップ）", None, ("High", "Middle")),
    6: ("冷房中の High を Low へ（全ゾーン・ステップ）", None, ("High", "Low")),
    7: ("CF-2 と CF-5 の合成", ({22.0}, 24.0), ("High", "Middle")),
    8: ("CF-3 と CF-6 の合成", ({22.0}, 26.0), ("High", "Low")),
}


def make_cf_actions(base, cf):
    """base: (T, 60) の整数配列。CF の番号に従って、冷房中のゾーン・ステップの設定温度・風量の索引だけを変えた複製と、
    変えたゾーン・ステップ数 {"setpoint": n, "fan": n, "cooling_zone_steps": n} を返す。"""
    base = np.asarray(base, dtype=np.int64)
    out = base.copy()
    _desc, sp_rule, fan_rule = CF_DEFS[cf]
    cooling = base[:, STATE0:STATE0 + N_Z] == COOL                      # (T, 18)
    n_sp = n_fan = 0
    if sp_rule is not None:
        targets, new_c = sp_rule
        new_idx = SETPOINT_GRID.index(new_c)
        cur = base[:, SP0:SP0 + N_Z]
        if targets == "all":
            sel = cooling & (cur != new_idx)
        else:
            idx_set = [SETPOINT_GRID.index(c) for c in targets]
            sel = cooling & np.isin(cur, idx_set)
        sp = out[:, SP0:SP0 + N_Z]
        sp[sel] = new_idx
        out[:, SP0:SP0 + N_Z] = sp
        n_sp = int(sel.sum())
    if fan_rule is not None:
        old_i, new_i = FAN_NAMES.index(fan_rule[0]), FAN_NAMES.index(fan_rule[1])
        cur = base[:, FAN0:FAN0 + N_Z]
        sel = cooling & (cur == old_i)
        fan = out[:, FAN0:FAN0 + N_Z]
        fan[sel] = new_i
        out[:, FAN0:FAN0 + N_Z] = fan
        n_fan = int(sel.sum())
    # 変えてはならない次元（運転状態・蒸発温度・換気）が変わっていないことの確認
    keep = np.ones(base.shape[1], dtype=bool)
    keep[SP0:SP0 + N_Z] = False
    keep[FAN0:FAN0 + N_Z] = False
    assert np.array_equal(out[:, keep], base[:, keep])
    # 冷房でないゾーン・ステップの設定温度・風量も変わっていない
    notc = ~cooling
    assert np.array_equal(out[:, SP0:SP0 + N_Z][notc], base[:, SP0:SP0 + N_Z][notc])
    assert np.array_equal(out[:, FAN0:FAN0 + N_Z][notc], base[:, FAN0:FAN0 + N_Z][notc])
    return out, {"setpoint": n_sp, "fan": n_fan, "cooling_zone_steps": int(cooling.sum())}


class Acc:
    """1 本の再生で、ステップごとの内訳を足し上げる（標本電力はステップ終端の 1 点を時間で積分した推定）。"""

    def __init__(self, W):
        self.W = W
        self.n_total = 0.0
        self.n_comp = {"thermal": 0.0, "draft": 0.0, "vtmp": 0.0, "co2": 0.0}
        self.e_outdoor = self.e_indoor = self.e_hex = 0.0
        self.cooling_steps = 0
        self.setpoint_hist = {}
        self.fan_hist = {}
        self.seconds = 0.0

    def add(self, info, act):
        d = info["diag"]
        sec = float(info["step_seconds"])
        self.seconds += sec
        occ = d["occupants"]
        for name, key in (("thermal", "dissatisfied_thermal"), ("draft", "dissatisfied_draft"), ("vtmp", "dissatisfied_vtmp")):
            self.n_comp[name] += float(reward_mod.dissatisfaction_increment(occ, d[key], sec, self.W).sum())
        tenant_of = [z[2] for z in constants.ZONE_DEFS]
        co2 = np.array([reward_mod.dr_co2(d["co2_ppm"][t]) for t in tenant_of])
        self.n_comp["co2"] += float(reward_mod.dissatisfaction_increment(occ, co2, sec, self.W).sum())
        self.n_total += float(np.sum(info["zone_dissatisfaction_increment"]))
        k = sec / 3600.0 * (constants.ENERGY_CONVERSION_MJ_PER_KWH / 1000.0)
        self.e_outdoor += float(np.sum(d["outdoor_power_kw"])) * k
        if d["indoor_power_kw"] is not None:
            self.e_indoor += float(np.sum(d["indoor_power_kw"])) * k
        self.e_hex += float(np.sum(d["hex_power_kw"])) * k
        cool = act[STATE0:STATE0 + N_Z] == COOL
        self.cooling_steps += int(cool.sum())
        for z in np.flatnonzero(cool):
            s = SETPOINT_GRID[int(act[SP0 + z])]
            f = FAN_NAMES[int(act[FAN0 + z])]
            self.setpoint_hist[s] = self.setpoint_hist.get(s, 0) + 1
            self.fan_hist[f] = self.fan_hist.get(f, 0) + 1


def run_all(cfs, base, slot_indices, out_dir):
    import torch as th
    from rlkit import config as config_mod
    from rlkit.parallel import launch, slots
    from rlkit.parallel.batched import SlotVecEnv
    from rlkit.reward import R7Reference
    cfg = config_mod.load(CONFIG)
    state = th.load(os.path.join(config_mod.model_dir_of(cfg), "checkpoint_latest.pt"), weights_only=False)
    ref = R7Reference.from_dict(state["reference"]).to_dict()
    report = launch.preflight(os.path.abspath(out_dir), slot_indices, limit=cfg.get("slot_limit", launch.SLOT_LIMIT),
                              min_free_gb=cfg.get("min_free_gb", launch.MIN_FREE_MEM_GB))
    os.environ[launch.LAUNCH_LOG_ENV] = report["log_dir"]
    checks = [slots.prepare_replica(slots.make_slot(k)) for k in slot_indices]
    env_kwargs = {"stage": 1, "include_indoor_unit_power": cfg["include_indoor_unit_power"],
                  "zone_step_energy_norm_gj": cfg["zone_step_energy_norm_gj"], "m9_enabled": cfg["m9_enabled"],
                  "comm_rpm": bool(cfg.get("comm_rpm", False))}
    T = base.shape[0]
    acts = [make_cf_actions(base, cf)[0] for cf in cfs]
    vec = SlotVecEnv(slot_indices, env_kwargs, T, factory="real")
    accs = [None] * len(cfs)
    finals = [None] * len(cfs)
    t0 = time.time()
    try:
        vec.env_method("set_diag_info", True)
        vec.env_method("set_reference", ref)
        W = vec.get_attr("W")[0]
        accs = [Acc(W) for _ in cfs]
        vec.reset()
        for t in range(T):
            _o, _r, dones, infos = vec.step(np.stack([a[t] for a in acts]))
            for i in range(len(cfs)):
                accs[i].add(infos[i], acts[i][t])
                if dones[i]:
                    finals[i] = infos[i]
        overrides = [(vec.get_attr("_override_count", indices=[i])[0], vec.get_attr("_override_opportunities", indices=[i])[0])
                     for i in range(len(cfs))]
    finally:
        vec.close()
    return {"finals": finals, "accs": accs, "overrides": overrides, "seconds": time.time() - t0, "preflight": report,
            "replica_checks": checks, "W": W}


def build_rows(cfs, base, res, saved_hex):
    rows = []
    for i, cf in enumerate(cfs):
        f, a = res["finals"][i], res["accs"][i]
        _act, changed = make_cf_actions(base, cf)
        n_ov, n_opp = res["overrides"][i]
        rows.append({
            "cf": cf, "description": CF_DEFS[cf][0], "changed_zone_steps": changed,
            "E": f["E_rep"], "DR": f["DR_rep"], "E_hex": float.hex(f["E_rep"]), "DR_hex": float.hex(f["DR_rep"]),
            "P_E": f["P_E"], "P_DR": f["P_DR"], "ECI": f["ECI"],
            "n_sum": a.n_total, "n_by_component_alone_sum": a.n_comp,
            "energy_est_gj": {"outdoor": a.e_outdoor, "indoor": a.e_indoor, "hex": a.e_hex,
                              "sum": a.e_outdoor + a.e_indoor + a.e_hex},
            "hex_energy_gj_env": f["hex_energy_gj"],
            "setpoint_override_count": n_ov, "setpoint_override_opportunities": n_opp,
            "setpoint_override_rate": f["setpoint_override_rate"],
            "cooling_zone_steps": a.cooling_steps, "setpoint_hist_c": {str(k): v for k, v in sorted(a.setpoint_hist.items())},
            "fan_hist": a.fan_hist,
            "raw_resid_rel": f["raw_resid_rel"], "raw_resid_sum_gj": f["raw_resid_sum_gj"],
            "energy_identity_rel": abs(f["sum_zone_energy_gj"] + f["sum_header_energy_gj"] - f["sum_delta_E_gj"]) / f["sum_delta_E_gj"],
            "diss_identity_rel": f["r8p_sum_rel"]})
    base_row = next(r for r in rows if r["cf"] == 0) if any(r["cf"] == 0 for r in rows) else None
    for r in rows:
        if base_row is not None:
            r["dE"] = r["E"] - base_row["E"]
            r["dDR"] = r["DR"] - base_row["DR"]
            r["dECI"] = r["ECI"] - base_row["ECI"]
    cf0 = base_row
    if cf0 is not None:
        cf0["hex_match_saved"] = (cf0["E_hex"] == saved_hex[0] and cf0["DR_hex"] == saved_hex[1])
    return rows


def main(argv=None):
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser()
    p.add_argument("--cf", default="0,1,2,3,4,5,6,7,8")
    a = p.parse_args(argv)
    from rlkit import runlog
    runlog.check_thread_env()
    cfs = [int(x) for x in a.cf.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(os.path.join(RUN_DIR, "best_harvest.npz") if False else os.path.join(RUN_DIR, "harvest", "best_harvest.npz"))
    base = d["actions"].astype(np.int64)
    meta = json.load(open(os.path.join(RUN_DIR, "harvest", "best_harvest.json"), encoding="utf-8"))
    saved_hex = (meta.get("E_hex"), meta.get("DR_hex"))
    print("base:", base.shape, "saved hex", saved_hex, flush=True)
    res = run_all(cfs, base, list(range(len(cfs))), OUT_DIR)
    rows = build_rows(cfs, base, res, saved_hex)
    out = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "base_run": RUN_DIR, "base_meta": meta,
           "saved_hex": saved_hex, "episode_seconds": res["seconds"], "W": res["W"], "rows": rows,
           "setpoint_grid_c": SETPOINT_GRID}
    runlog.write_json(os.path.join(OUT_DIR, "cf_results.json"), out)
    for r in rows:
        print(json.dumps({k: r[k] for k in ("cf", "E", "DR", "ECI", "dE", "dDR", "dECI") if k in r}, ensure_ascii=False), flush=True)
    cf0 = next((r for r in rows if r["cf"] == 0), None)
    if cf0 is not None and not cf0["hex_match_saved"]:
        print("CF-0 の E・DR が更新 330 の収穫と 16 進で一致しない（停止規則）", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
