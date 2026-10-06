"""V8S_report_u50.md の集計（読むだけ）。出力: runs/diag/V8S/u50_tables.md、u50_summary.json。"""
import collections
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN = os.path.join(ROOT, "runs", "v8s_trial01")
OUT = os.path.join(ROOT, "runs", "diag", "V8S")
sys.stdout.reconfigure(encoding="utf-8")
med = st.median
ups = [json.loads(l) for l in open(os.path.join(RUN, "updates.jsonl"), encoding="utf-8")]
eps = [json.loads(l) for l in open(os.path.join(RUN, "episodes.jsonl"), encoding="utf-8")]
sibs = [json.loads(l) for l in open(os.path.join(RUN, "siblings.jsonl"), encoding="utf-8")]
N = len(ups)
START = 0.7093194881516841
base_eci = [u["eval_ECI"] for u in ups]
hdr = ("| 更新 | 基準の ECI 中央値 | 基準の ECI 最大 | 基準の E 中央値 | 基準の DR 中央値 | 収穫の最大 | 改善した兄弟の割合 | Σ A（改善分）平均 | 採用率 | "
       "基準の C の変化（L-8）平均 | 最後の KL 中央値 | エポック数 中央値 | 打ち切り | 実時間 中央値 [秒] | 空きメモリ最小 [GB] |\n" + "|---|" * 1 + "---|" * 14 + "\n")
rows = []
for a in range(1, N + 1, 10):
    U = ups[a - 1:a + 9]
    dc = [u["delta_C_base_norm"] for u in U if u["delta_C_base_norm"] is not None]
    ad = [u["adoption_rate"] for u in U if u["adoption_rate"] is not None]
    rows.append(f"| {U[0]['update']}〜{U[-1]['update']} | {med(u['eval_ECI'] for u in U):.4f} | {max(u['eval_ECI'] for u in U):.4f} | "
                f"{med(u['eval_E'] for u in U):.4f} | {med(u['eval_DR'] for u in U):.5f} | {max(u['harvest_update_ECI'] for u in U):.4f} | "
                f"{st.mean(u['frac_improved'] for u in U):.3f} | {st.mean(u['sum_A_improved'] for u in U):.5f} | "
                f"{(st.mean(ad) if ad else float('nan')):.3f} | {(st.mean(dc) if dc else float('nan')):+.5f} | "
                f"{med(u['kl_last'] for u in U if u['kl_last'] is not None):.4f} | {med(u['epochs_applied'] for u in U):.0f} | "
                f"{sum(bool(u['kl_cutoff']) for u in U)} | {med(u['update_seconds'] for u in U):.0f} | "
                f"{min(u['c13']['min_free_memory_gb'] for u in U):.2f} |")
t1 = hdr + "\n".join(rows) + "\n"
# 兄弟の区間の時間帯別・揺らし方別
sib_rows = [e for e in eps if e["role"] == "sibling"]
win_of = {}
for s in sibs:
    for sb in s["siblings"]:
        win_of[(s["update"], sb["slot"])] = (sb["window"], sb["kind"], sb["tenant"], sb["day"], sb["n_changed"])
agg_w = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
agg_k = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
for e in sib_rows:
    w, k, t, d, nc = win_of[(e["update"], e["slot"])]
    for agg, key in ((agg_w, w), (agg_k, k)):
        agg[key][0] += 1
        agg[key][1] += int(e["improved"])
        agg[key][2] += e["A"]
        agg[key][3] += e["A"] if e["improved"] else 0.0
names = ["0〜7 時", "7〜10 時", "10〜13 時", "13〜16 時", "16〜19 時", "19〜24 時"]
t2 = "| 時間帯 | 兄弟の本数 | 改善した割合 | 平均の A_s（全兄弟） | 改善した兄弟の平均の A_s |\n|---|---|---|---|---|\n"
for w in range(6):
    n, im, sa, sai = agg_w[w]
    t2 += f"| {names[w]} | {n} | {im / n:.3f} | {sa / n:+.6f} | {(sai / im if im else float('nan')):.6f} |\n"
t3 = "| 揺らし方 | 兄弟の本数 | 改善した割合 | 平均の A_s（全兄弟） | 改善した兄弟の平均の A_s |\n|---|---|---|---|---|\n"
for k, label in (("sparse", "まばら"), ("coherent", "まとまり")):
    n, im, sa, sai = agg_k[k]
    t3 += f"| {label} | {n} | {im / n:.3f} | {sa / n:+.6f} | {(sai / im if im else float('nan')):.6f} |\n"
# 変更の無かった兄弟（n_changed = 0）の数
unchanged = sum(1 for s in sibs for sb in s["siblings"] if sb["n_changed"] == 0)
total_sib = sum(len(s["siblings"]) for s in sibs)
chg_improved = collections.Counter()
for e in sib_rows:
    nc = win_of[(e["update"], e["slot"])][4]
    chg_improved[(nc == 0, e["improved"])] += 1
# 新しい選択肢の使用（基準の軌跡）
use = [u["v8_use_base"] for u in ups]
def share(u):
    occ = u["duty_counts_occupied"]
    on = sum(occ[1:])
    return {"frac_partial_of_operating_occupied": (occ[1] + occ[2]) / on if on else None,
            "frac_partial_of_all_occupied": (occ[1] + occ[2]) / sum(occ), "duty_occupied": occ,
            "prohibit_zone_steps": u["prohibit_zone_steps"], "bypass_tenant_steps": u["bypass_tenant_steps"],
            "setpoint_counts": u["setpoint_counts"], "evap_counts": u["evap_counts"]}
first, last = share(use[0]), share(use[-1])
avg_partial = st.mean(share(u)["frac_partial_of_operating_occupied"] for u in use)
tot_prohibit = sum(u["prohibit_zone_steps"] for u in use)
tot_bypass = sum(u["bypass_tenant_steps"] for u in use)
# KL・エポック数の分布
epochs = collections.Counter(u["epochs_applied"] for u in ups)
kls = [u["kl_last"] for u in ups if u["kl_last"] is not None]
no_update = sum(1 for u in ups if not u["updated"])
summary = {
    "n_updates": N, "start_eci": START, "rho_start": ups[0]["rho"], "rho_range": [min(u["rho"] for u in ups), max(u["rho"] for u in ups)],
    "base_eci_first10": base_eci[:10], "base_eci_last10": base_eci[-10:], "base_eci_max": max(base_eci),
    "base_eci_max_update": base_eci.index(max(base_eci)) + 1, "g50_threshold": START + 0.002,
    "frac_improved_mean": st.mean(u["frac_improved"] for u in ups), "n_improved_total": sum(u["n_improved"] for u in ups),
    "adoption_mean": st.mean(u["adoption_rate"] for u in ups if u["adoption_rate"] is not None),
    "kl_last_median": med(kls), "kl_last_max": max(kls), "epochs_counter": dict(epochs), "no_update_count": no_update,
    "cutoff_count": sum(bool(u["kl_cutoff"]) for u in ups),
    "update_seconds_median": med(u["update_seconds"] for u in ups), "update_seconds_max": max(u["update_seconds"] for u in ups),
    "update_seconds_mean_1_2": st.mean(u["update_seconds"] for u in ups[:2]),
    "min_free_gb": min(u["c13"]["min_free_memory_gb"] for u in ups), "rpm_fallback": sum(u["c13"]["rpm_fallback_chunks"] for u in ups),
    "parent_wait_median": med(u["c13"]["parent_wait_median_s"] for u in ups),
    "retries": sum(u["collection_retries"] for u in ups),
    "harvest_best": ups[-1]["harvest_best"], "harvest_improved_updates": [u["update"] for u in ups if u["harvest_update_best"].get("improved_best")],
    "siblings_unchanged_n0": unchanged, "siblings_total": total_sib,
    "improved_by_changed": {f"no_change={k[0]},improved={k[1]}": v for k, v in chg_improved.items()},
    "use_first": first, "use_last": last, "avg_partial_share_operating_occupied": avg_partial,
    "prohibit_zone_steps_total_base": tot_prohibit, "bypass_tenant_steps_total_base": tot_bypass,
    "sib_eci_best_update": max(((max(u["sibling_ECI"]), u["update"]) for u in ups)),
    "sum_A_improved_total": sum(u["sum_A_improved"] for u in ups),
    "delta_C_base_total": sum(u["delta_C_base_norm"] for u in ups if u["delta_C_base_norm"] is not None),
    "c11_max": {"energy_identity": max(max(c["energy_identity_rel"] for c in [u["eval_c11"]] + u["train_c11"]) for u in ups),
                "diss_identity": max(max(c["diss_identity_rel"] for c in [u["eval_c11"]] + u["train_c11"]) for u in ups),
                "abs_r": max(max(abs(c["r"]) for c in [u["eval_c11"]] + u["train_c11"]) for u in ups),
                "abs_a_gj": max(max(abs(c["a_gj"]) for c in [u["eval_c11"]] + u["train_c11"]) for u in ups)},
}
json.dump(summary, open(os.path.join(OUT, "u50_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(OUT, "u50_tables.md"), "w", encoding="utf-8") as f:
    f.write("## 表 1\n\n" + t1 + "\n## 表 2\n\n" + t2 + "\n## 表 3\n\n" + t3)
# 全更新の一覧
lines = ["| 更新 | 基準の ECI | 収穫の ECI（13 本の最大） | 改善した兄弟 | エポック | 最後の KL | 打ち切り | 採用率 | 実時間 [秒] |", "|---|---|---|---|---|---|---|---|---|"]
for u in ups:
    lines.append(f"| {u['update']} | {u['eval_ECI']:.5f} | {u['harvest_update_ECI']:.5f} | {u['n_improved']} | {u['epochs_applied']} | "
                 f"{(u['kl_last'] if u['kl_last'] is not None else float('nan')):.4f} | {'あり' if u['kl_cutoff'] else 'なし'} | "
                 f"{(u['adoption_rate'] if u['adoption_rate'] is not None else float('nan')):.3f} | {u['update_seconds']:.0f} |")
open(os.path.join(OUT, "u50_per_update.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=1)[:5500])
