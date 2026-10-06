"""指示書 V6.4-S の単体試験のうち、エミュレータを使うもの（UT-29・UT-31・UT-32'・UT-34）。

環境変数 RLKIT_LIVE=1 のときだけ実行する（通常の試験では飛ばす）:
  RLKIT_LIVE=1 python -m pytest rlkit/tests/test_v64_live.py -q
（test_v6_live.py とは別のプロセスで実行する。起動前の検査が組のロックを見るため）
組 6 で基準の軌跡と反事実を 1 本ずつ再生し（各約 2 分）、7 組で異なる 7 本を同時に再生する（約 3 分）。
"""
import json
import os

import numpy as np
import pytest

from rlkit import constants, r8p

pytestmark = pytest.mark.skipif(os.environ.get("RLKIT_LIVE") != "1", reason="エミュレータを使う試験（RLKIT_LIVE=1 で実行）")

RUN_DIR = os.path.join("runs", "diag", "V64S", "UT_live")
BASE_E_HEX = "0x1.a228e147ae148p+2"
BASE_DR_HEX = "0x1.863c140000000p-3"
RHO_EVAL = 72.3
_CACHE = {}


def _launch(slot_ids):
    """起動前の検査は 1 つの試験プロセスで 1 回、7 組まとめて行う（一度使った組のロックはこのプロセスが持つため、
    2 回目の検査は「使用中」として拒否される）。"""
    from rlkit.parallel import launch, slots
    if "launched" not in _CACHE:
        os.makedirs(RUN_DIR, exist_ok=True)
        report = launch.preflight(RUN_DIR, [0, 1, 2, 3, 4, 5, 6])
        os.environ[launch.LAUNCH_LOG_ENV] = report["log_dir"]
        _CACHE["launched"] = True
    for k in slot_ids:
        slots.prepare_replica(slots.make_slot(k))


def _env_kwargs():
    from rlkit import config
    cfg = config.load(os.path.join("rlkit", "config_summer_v64.json"))
    return {"stage": 1, "include_indoor_unit_power": cfg["include_indoor_unit_power"],
            "zone_step_energy_norm_gj": cfg["zone_step_energy_norm_gj"], "m9_enabled": cfg["m9_enabled"]}


def _replay(actions, slot_index=6):
    from rlkit.env import SummerEnv
    from rlkit.parallel import slots
    _launch([slot_index])
    env = SummerEnv(slot=slots.make_slot(slot_index), **_env_kwargs())
    env.set_diag_info(True)
    T = actions.shape[0]
    e, dE = np.zeros((T, 18)), np.zeros(T)
    remote_ok = []
    try:
        env.reset()
        for t in range(T):
            _o, _r, done, _tr, info = env.step(actions[t])
            e[t], dE[t] = info["zone_energy_gj"], info["delta_E_gj"]
            remote_ok.append(info["diag"]["remote_status"] == {"South": 1.0, "North": 1.0})
            if done:
                break
    finally:
        env.close()
    minute = r8p.load_minute_data(env.minute_output_dir())
    return {"info": info, "e": e, "dE": dE, "remote_ok": remote_ok, "A": list(env._r8p_A),
            "reads": list(env._r8p_reads), "starts": list(env._r8p_starts), "minute": minute, "exo": env.exo}


def _baseline():
    if "base" not in _CACHE:
        from rlkit.diag import v62s_diag
        _CACHE["base"] = _replay(v62s_diag.baseline_actions())
    return _CACHE["base"]


def _jobs():
    from rlkit.diag import v62s_diag, v64s_diag
    base = v62s_diag.baseline_actions()
    dts = v62s_diag.step_datetimes()
    return base, {j["label"]: j for j in v62s_diag.k14_jobs(base, dts) + v64s_diag.s2_jobs(base, dts)}


def test_ut29_ut32p_live_baseline_replay():
    """UT-29（リモコンが許可に固定され、1114 の読み戻しが毎ステップ全台許可）と UT-32'（R-8' の (1)〜(4)）。
    基準の軌跡は C-5 の後の 60 次元の行動で再生し、E・DR が 16 進で一致する。"""
    rec = _baseline()
    info = rec["info"]
    assert float.hex(info["E_rep"]) == BASE_E_HEX and float.hex(info["DR_rep"]) == BASE_DR_HEX
    assert all(rec["remote_ok"]) and len(rec["remote_ok"]) == constants.ZONE_STEPS_PER_WEEK
    assert info["r8p_branch"] == "R8'-A" and info["r8p_delta_min"] == -1
    assert info["r8p_sum_rel"] <= 1e-9                                                   # (1)
    assert info["r8p_zone_sum_max_abs"] <= 1e-15                                         # (2)
    assert info["r8p_empty_max_abs_n"] <= 1e-7 and info["r8p_empty_steps"] > 0          # (4)
    assert info["r8p_min_n"] >= -1e-7 and info["r8p_minute_B_week_rel"] <= 1e-6
    ev = r8p.evaluate(rec["A"], rec["reads"], rec["starts"], rec["minute"], rec["exo"])
    assert ev["branch"] == "R8'-A" and ev["delta"] == -1 and ev["by_delta"]["-1"]["L1"] <= 1e-3   # (3)
    n_kt = np.asarray(info["r8p_zone_n"])
    assert n_kt.shape == (constants.ZONE_STEPS_PER_WEEK, 18)
    assert np.abs(n_kt.sum() - info["DR_rep"]) / info["DR_rep"] <= 1e-9


def test_ut31_live_counterfactual_identity():
    """UT-31: 反事実の再生で、反転前は基準と同一、ΔC_true が ΔE_true + ρ_eval·ΔDR_true と 1% 以内（R8'-A では 1e-9 GJ 相当）。"""
    rows = json.load(open(os.path.join("runs", "diag", "V64S", "K14", "K14p_rows.json"), encoding="utf-8"))
    pick = max((r for r in rows if r["family"] == "S" and r["to_off"]), key=lambda r: abs(r["dC_true"]))
    _base_actions, jobs = _jobs()
    job = jobs[pick["label"]]
    b = _baseline()
    r = _replay(job["actions"])
    i = job["start"]
    assert np.array_equal(r["e"][:i], b["e"][:i]) and np.array_equal(r["dE"][:i], b["dE"][:i])
    assert r["A"][:i] == b["A"][:i]
    nb, nr = np.asarray(b["info"]["r8p_zone_n"]), np.asarray(r["info"]["r8p_zone_n"])
    de, dn, ddE = r["e"] - b["e"], nr - nb, r["dE"] - b["dE"]
    dC_true = float(de[i:].sum() + RHO_EVAL * dn[i:].sum() + (ddE[i:] - de[i:].sum(axis=1)).sum())
    official = (r["info"]["E_rep"] - b["info"]["E_rep"]) + RHO_EVAL * (r["info"]["DR_rep"] - b["info"]["DR_rep"])
    assert abs(dC_true - official) <= 0.01 * abs(official)
    assert abs(dC_true - official) <= 1e-9
    assert dC_true == pytest.approx(pick["dC_true"], abs=1e-12)                        # K-14' の記録と同じ


def test_ut34_live_seven_slots_read_own_output():
    """UT-34: 並列の 7 組で、各組のエピソードが自分の分単位の出力を読む（7 組で異なる軌跡を同時に再生する）。"""
    from rlkit.parallel import slots
    from rlkit.parallel.batched import SlotVecEnv
    rows = json.load(open(os.path.join("runs", "diag", "V64S", "K14", "K14p_rows.json"), encoding="utf-8"))
    picks = sorted(rows, key=lambda r: -abs(r["dDR_true"]))
    chosen, drs = [], [0.0]
    for r in picks:                                  # DR が互いに相対 1e-5 以上離れた 6 本
        if all(abs(r["dDR_true"] - d) > 1e-5 * 0.19 for d in drs):
            chosen.append(r["label"])
            drs.append(r["dDR_true"])
        if len(chosen) == 6:
            break
    base, jobs = _jobs()
    trajs = [base] + [jobs[l]["actions"] for l in chosen]
    ids = [0, 1, 2, 3, 4, 5, 6]
    _launch(ids)
    T = base.shape[0]
    vec = SlotVecEnv(ids, _env_kwargs(), steps_per_collection=T)
    try:
        vec.reset()
        for t in range(T):
            _obs, _r, dones, infos = vec.step(np.stack([a[t] for a in trajs]))
        assert all(dones)
    finally:
        vec.close()
    DR = np.array([inf["DR_rep"] for inf in infos])
    # 分単位の出力（小数 4 桁）から作る R8-B の週の和の誤差は、基準の軌跡で 2.5e-7、反事実で 3e-6 程度まで出る
    # （初回の実行で確認。当初の 1e-6 の閾値は基準の軌跡だけを見て置いたため改めた）。取り違えの判定は下の行列で行う
    for k, inf in enumerate(infos):
        assert inf["r8p_minute_B_week_rel"] <= 1e-5, (k, inf["r8p_minute_B_week_rel"])
        assert inf["r8p_sum_rel"] <= 1e-9
    # 組ごとの出力を読み直し、他の組の DR とは一致しないこと（試験が取り違えを検出できること）を確かめる
    from rlkit import exogenous
    exo = exogenous.load()
    for k in ids:
        minute = r8p.load_minute_data(slots.make_slot(k).data_dir)
        total = sum(minute["rate_all"][m] * exo.total_occupants_at(d) for m, d in enumerate(minute["dt"])) / exo.total_occupant_minutes()
        rel = np.abs(total - DR) / DR
        assert rel[k] == pytest.approx(infos[k]["r8p_minute_B_week_rel"], rel=1e-6)     # 環境が読んだのと同じ出力
        assert int(np.argmin(rel)) == k and np.all(np.delete(rel, k) > 3 * rel[k]), (k, rel.tolist())
