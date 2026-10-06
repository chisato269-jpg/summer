"""V71S_report_u1500.md の集計表を作る（読むだけ）。出力: runs/diag/V71S/u1500_tables.md と、数値の要約 u1500_summary.json。"""
import collections
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN = os.path.join(ROOT, "runs", "v2s_trial08")
OUT = os.path.join(ROOT, "runs", "diag", "V71S")
LNK = {"zone_state": math.log(2), "zone_setpoint": math.log(7), "zone_fan": math.log(3), "system_evap": math.log(8),
       "tenant_vent": math.log(4)}
GROUPS = ["zone_state", "zone_setpoint", "zone_fan", "system_evap", "tenant_vent"]
sys.stdout.reconfigure(encoding="utf-8")

ups = [json.loads(l) for l in open(os.path.join(RUN, "updates.jsonl"), encoding="utf-8")]
by = {u["update"]: u for u in ups}
assert [u["update"] for u in ups] == list(range(1, 1501))
ev = [json.loads(l) for l in open(os.path.join(RUN, "events.jsonl"), encoding="utf-8")]
cw = [json.loads(l) for l in open(os.path.join(RUN, "collapse_watch.jsonl"), encoding="utf-8")]
med = st.median


def hl(u):
    return "/".join(f"{u['H_g'][g] / LNK[g]:.2f}" for g in GROUPS)


def vent(u, t):
    f = u["train_vent_level_fraction_mean"][t]
    return "/".join(f"{f[k]:.2f}" for k in ("off", "Low", "Middle", "High"))


def blocks(lo, hi, step, first_end=None):
    out, a = [], lo
    if first_end:
        out.append((a, first_end))
        a = first_end + 1
    while a <= hi:
        out.append((a, min(a + step - 1, hi)))
        a += step
    return out


rows_a, rows_b = [], []
for a, b in blocks(685, 1500, 20, first_end=700):
    U = [by[k] for k in range(a, b + 1)]
    last = U[-1]
    harv = [u["harvest_update_ECI"] for u in U]
    c13 = [u["c13"] for u in U]
    rows_a.append(
        f"| {a}〜{b} | {med(harv):.4f} | {max(harv):.4f} | {last['harvest_best']['ECI']:.4f}（更新 {last['harvest_best']['update']}） | "
        f"{med(u['train_E_median'] for u in U):.3f} | {med(u['train_DR_median'] for u in U):.4f} | {med(u['eval_ECI'] for u in U):.4f} | "
        f"{U[0]['reference_used']['rho']:.1f}→{last['reference']['rho']:.1f} | {last['reference']['E_ref']:.3f}/{last['reference']['DR_ref']:.4f} | "
        f"{med(u['policy_lr_after'] for u in U):.2e} | {med(u['approx_kl_after'] for u in U):.4f} | {sum(bool(u['kl_early_stop']) for u in U)} | "
        f"{med(u['occupied_zone_step_off_fraction']['fraction'] for u in U):.3f} | {med(u['update_seconds'] for u in U):.0f} |")
    rows_b.append(
        f"| {a}〜{b} | {vent(last, 'South')} | {vent(last, 'North')} | {med(u['train_hex_energy_gj_mean'] for u in U):.3f} | {hl(last)} | "
        f"{max(abs(u['eval_raw_resid_sum_gj']) for u in U):.4f} | {max(u['train_raw_resid_a_gj_abs_max'] for u in U):.4f} | "
        f"{max(abs(u['eval_raw_resid_rel']) for u in U) * 100:.2f}% | {min(c['min_free_memory_gb'] for c in c13):.2f} | "
        f"{sum(c['rpm_fallback_chunks'] for c in c13)} | {med(c['parent_wait_median_s'] for c in c13):.3f}/{max(c['parent_wait_p99_s'] for c in c13):.2f} |")

# 更新 900〜1500（エントロピー係数が下がる区間）の 50 更新ごと
rows_e = []
for a, b in blocks(901, 1500, 50):
    U = [by[k] for k in range(a, b + 1)]
    harv = [u["harvest_update_ECI"] for u in U]
    last = U[-1]
    rows_e.append(f"| {a}〜{b} | {last['entropy_c0']:.5f} | {med(harv):.4f} | {max(harv):.4f} | {med(u['eval_ECI'] for u in U):.4f} | {hl(last)} | "
                  f"{med(u['train_E_median'] for u in U):.3f} | {med(u['train_DR_median'] for u in U):.4f} |")
h900 = by[900]

# 最良の更新履歴
best_hist, cur = [], 0.6712562046627591
for k in range(685, 1501):
    b = by[k]["harvest_best"]
    if b["ECI"] > cur:
        cur = b["ECI"]
        best_hist.append((b["update"], b["ECI"], b["E"], b["DR"]))

# 崩れの見張り
ratios = [(c["update"], c["median_recent"] / c["best_previous"]) for c in cw if c.get("best_previous")]
cw_after = [c for c in cw if c["update"] >= 700]
# 停止規則の値
all_after = [by[k] for k in range(685, 1501)]
secs = [u["update_seconds"] for u in all_after]
ev_after = [e for e in ev if e["time"] >= "2026-10-01T14:20"]
cnt = collections.Counter(e["event"] for e in ev_after)
fails = [e for e in ev_after if e["event"] in ("collection_failed_discarded", "eval_failed")]
rho_stuck = max((sum(1 for _ in g) for key, g in __import__("itertools").groupby(all_after, key=lambda u: u["reference_used"]["rho"] in (30.0, 200.0)) if key), default=0)
c11 = {"eval_energy_identity_max": max(u["eval_c11"]["energy_identity_rel"] for u in all_after),
       "eval_diss_identity_max": max(u["eval_c11"]["diss_identity_rel"] for u in all_after),
       "train_energy_identity_max": max(u["train_energy_identity_rel_max"] for u in all_after),
       "train_diss_identity_max": max(r["diss_identity_rel"] for u in all_after for r in u["train_c11"]),
       "eval_abs_r_max": max(abs(u["eval_c11"]["r"]) for u in all_after),
       "eval_abs_a_gj_max": max(abs(u["eval_c11"]["a_gj"]) for u in all_after),
       "train_abs_r_max": max(abs(r["r"]) for u in all_after for r in u["train_c11"]),
       "train_abs_a_gj_max": max(abs(r["a_gj"]) for u in all_after for r in u["train_c11"]),
       "both_exceeded_count": sum(1 for u in all_after for r in [u["eval_c11"]] + u["train_c11"]
                                  if abs(r["r"]) > 0.05 and abs(r["a_gj"]) > 0.15)}
summary = {
    "updates_685_1500": len(all_after), "update_seconds_median": med(secs), "update_seconds_p99": sorted(secs)[int(0.99 * len(secs))],
    "update_seconds_max": max(secs), "over_250s": sum(s > 250 for s in secs), "over_600s": sum(s > 600 for s in secs),
    "median_by_period": {"685-768（再開 1 回目）": med(by[k]["update_seconds"] for k in range(685, 769)),
                         "769-1500（再開 2 回目）": med(by[k]["update_seconds"] for k in range(769, 1501))},
    "rpm_fallback_total": sum(u["c13"]["rpm_fallback_chunks"] for u in all_after),
    "min_free_gb": min(u["c13"]["min_free_memory_gb"] for u in all_after),
    "min_free_gb_update": min(all_after, key=lambda u: u["c13"]["min_free_memory_gb"])["update"],
    "parent_wait_median_s": med(u["c13"]["parent_wait_median_s"] for u in all_after),
    "parent_wait_p99_s_max": max(u["c13"]["parent_wait_p99_s"] for u in all_after),
    "events_after_resume": dict(cnt), "collection_failed": sum(e["event"] == "collection_failed_discarded" for e in fails),
    "eval_failed": sum(e["event"] == "eval_failed" for e in fails),
    "bacnet_no_response": sum("no-response" in e.get("error", "") for e in fails),
    "collapse_checks_total": len(cw), "collapse_triggered": sum(c["triggered"] for c in cw),
    "collapse_min_ratio": min(ratios, key=lambda x: x[1]), "collapse_after_resume_min_ratio": min(
        (c["update"], c["median_recent"] / c["best_previous"]) for c in cw_after),
    "best_history_after_684": best_hist, "c11": c11, "rho_stuck_max_streak": rho_stuck,
    "rho_range": [min(u["reference_used"]["rho"] for u in all_after), max(u["reference_used"]["rho"] for u in all_after)],
    "lr_at_max_fraction": sum(u["policy_lr_after"] >= 1e-3 for u in all_after) / len(all_after),
    "kl_early_stop_count": sum(bool(u["kl_early_stop"]) for u in all_after),
    "h900": {g: h900["H_g"][g] / LNK[g] for g in GROUPS}, "h1500": {g: by[1500]["H_g"][g] / LNK[g] for g in GROUPS},
    "h684": {g: by[684]["H_g"][g] / LNK[g] for g in GROUPS},
    "harvest_median_by_period": {"601-684": med(by[k]["harvest_update_ECI"] for k in range(601, 685)),
                                 "685-900": med(by[k]["harvest_update_ECI"] for k in range(685, 901)),
                                 "901-1500": med(by[k]["harvest_update_ECI"] for k in range(901, 1501)),
                                 "1401-1500": med(by[k]["harvest_update_ECI"] for k in range(1401, 1501)),
                                 "1481-1500": med(by[k]["harvest_update_ECI"] for k in range(1481, 1501))},
    "final": {"E": by[1500]["harvest_best"]["E"], "DR": by[1500]["harvest_best"]["DR"], "ECI": by[1500]["harvest_best"]["ECI"],
              "update": by[1500]["harvest_best"]["update"], "reference_rho_1500": by[1500]["reference"]["rho"]},
    "sil_positive_adv_1500": by[1500]["sil"]["positive_advantage_fraction"],
}
json.dump(summary, open(os.path.join(OUT, "u1500_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

hdr_a = ("| 更新 | 収穫 ECI 中央値 | 収穫 ECI 最大 | 走行の最良 ECI（末） | 訓練 E 中央値 | 訓練 DR 中央値 | 評価 ECI 中央値 | ρ 始→末 | 評価点 E/DR（末） | "
         "学習率 中央値 | KL 後 中央値 | 打切 | 停止割合 中央値 | 実時間 中央値 [秒] |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
hdr_b = ("| 更新 | 換気 南 停/L/M/H（末） | 換気 北 停/L/M/H（末） | 換気 GJ 中央値 | H/lnK 発停/設定/風量/蒸発/換気（末） | 評価の \\|a\\| 最大 [GJ] | 訓練の \\|a\\| 最大 [GJ] | "
         "評価の \\|r\\| 最大 | 空きメモリ最小 [GB] | 戻し 合計 | 親の待ち 中央値/p99最大 [秒] |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
hdr_e = ("| 更新 | c0（末） | 収穫 ECI 中央値 | 収穫 ECI 最大 | 評価 ECI 中央値 | H/lnK 発停/設定/風量/蒸発/換気（末） | 訓練 E 中央値 | 訓練 DR 中央値 |\n"
         "|---|---|---|---|---|---|---|---|\n")
with open(os.path.join(OUT, "u1500_tables.md"), "w", encoding="utf-8") as f:
    f.write("## 表 A\n\n" + hdr_a + "\n".join(rows_a) + "\n\n## 表 B\n\n" + hdr_b + "\n".join(rows_b) + "\n\n## 表 E\n\n" + hdr_e + "\n".join(rows_e) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=1)[:6000])
