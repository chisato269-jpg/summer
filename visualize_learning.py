"""
visualize_learning.py
学習経過（不満足者率・消費電力量・ECI・エピソード報酬）を可視化する

データソース:
  saved_models/rewards.npy      … エピソードごとの累積報酬（1D配列）
  saved_models/episode_log.npy  … エピソードごとの [energy, disrate, eci]（2D配列）
  saved_models/best_eci_info.npy … [best_eci, best_eci_episode]（任意）

学習を途中で停止・再開しても単一ファイルに追記されるため、
1エピソード目から最新まで途切れなくグラフ化できる。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAVE_PATH = "./saved_models_v6/"

# ──────────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────────

def load_npy(path, allow_pickle=False):
    if os.path.exists(path):
        data = np.load(path, allow_pickle=allow_pickle)
        return data
    return None


rewards_raw = load_npy(os.path.join(SAVE_PATH, "rewards.npy"))
log_raw     = load_npy(os.path.join(SAVE_PATH, "episode_log.npy"))
eci_info    = load_npy(os.path.join(SAVE_PATH, "best_eci_info.npy"))

has_rewards = rewards_raw is not None and len(rewards_raw) > 0
has_log     = log_raw is not None and len(log_raw) > 0

if not has_rewards and not has_log:
    print("ERROR: rewards.npy も episode_log.npy も見つかりません。")
    print(f"  検索パス: {os.path.abspath(SAVE_PATH)}")
    raise SystemExit(1)

# 共通エピソード数（両ファイルが存在する場合は短い方に揃える）
n_ep_rewards = len(rewards_raw) if has_rewards else 0
n_ep_log     = len(log_raw)     if has_log     else 0
n_episodes   = max(n_ep_rewards, n_ep_log)

print(f"rewards.npy     : {n_ep_rewards} episodes")
print(f"episode_log.npy : {n_ep_log} episodes")
print(f"グラフ対象      : {n_episodes} episodes")

if has_log:
    log = np.array(log_raw, dtype=np.float64)
    energy_arr  = log[:, 0]
    disrate_arr = log[:, 1]
    eci_arr     = log[:, 2]           # NaN = ペナルティ／未計算
    valid_eci   = ~np.isnan(eci_arr)  # NaN を除いた有効マスク

# ──────────────────────────────────────────────
# ベストECI情報
# ──────────────────────────────────────────────
best_eci_val = float(eci_info[0]) if eci_info is not None else np.nan
best_eci_ep  = int(eci_info[1])   if eci_info is not None else -1

# ──────────────────────────────────────────────
# 移動平均ヘルパー
# ──────────────────────────────────────────────
def moving_avg(arr, window):
    return pd.Series(arr).rolling(window=window, min_periods=1).mean().values

MA = max(5, n_episodes // 20)   # 動的に窓幅を調整

# ──────────────────────────────────────────────
# 描画
# ──────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 100,
})

n_cols = 3
n_rows = 2 if (has_rewards or has_log) else 1
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.35)

title_suffix = f"({n_episodes} episodes"
if not np.isnan(best_eci_val):
    title_suffix += f", Best ECI={best_eci_val:.4f} @ ep{best_eci_ep}"
title_suffix += ")"
fig.suptitle(f"Shizuku2 RL Training Progress {title_suffix}", fontsize=13, fontweight="bold")


def scatter_with_ma(ax, x, y, color_raw, color_ma, ylabel, title, ma_loc="best"):
    ax.scatter(x, y, alpha=0.25, s=8, color=color_raw, label="Raw")
    ax.plot(x, moving_avg(y, MA), color=color_ma, linewidth=2, label=f"MA({MA})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=ma_loc, fontsize=8)


ep_log = np.arange(1, n_ep_log + 1) if has_log else None
ep_rwd = np.arange(1, n_ep_rewards + 1) if has_rewards else None


# ── 1. 不満足者率 ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
if has_log:
    scatter_with_ma(ax1, ep_log, disrate_arr, "#FF6B35", "#D32F2F",
                    "Dissatisfaction Rate", "Dissatisfaction Rate (↓ better)", "upper right")
else:
    ax1.text(0.5, 0.5, "No data (episode_log.npy missing)",
             ha="center", va="center", transform=ax1.transAxes)
    ax1.set_title("Dissatisfaction Rate")

# ── 2. 消費電力量 ──────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
if has_log:
    scatter_with_ma(ax2, ep_log, energy_arr, "#4FC3F7", "#1565C0",
                    "Energy [GJ]", "Energy Consumption (↓ better)", "upper right")
else:
    ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("Energy Consumption")

# ── 3. ECIスコア ───────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
if has_log and valid_eci.any():
    # NaN（ペナルティ／未計算）は除いてプロット
    ax3.scatter(ep_log[valid_eci], eci_arr[valid_eci],
                alpha=0.3, s=8, color="#81C784", label="Raw (valid)")
    ax3.plot(ep_log[valid_eci],
             moving_avg(eci_arr[valid_eci], max(3, valid_eci.sum() // 20)),
             color="#2E7D32", linewidth=2, label=f"MA")
    # ペナルティエピソードをバツ印でマーク
    penalty_mask = ~valid_eci
    if penalty_mask.any():
        ax3.scatter(ep_log[penalty_mask], np.zeros(penalty_mask.sum()),
                    color="red", s=20, marker="x", zorder=5, label="Penalty/N/A")
    if not np.isnan(best_eci_val):
        ax3.axhline(best_eci_val, color="gold", linestyle="--", linewidth=1.5,
                    label=f"Best={best_eci_val:.4f}")
    ax3.legend(loc="lower right", fontsize=8)
else:
    ax3.text(0.5, 0.5, "No ECI data", ha="center", va="center", transform=ax3.transAxes)
ax3.set_xlabel("Episode")
ax3.set_ylabel("ECI Score")
ax3.set_title("ECI Score (↑ better)")

# ── 4. エピソード報酬（rewards.npy）─────────────
ax4 = fig.add_subplot(gs[1, 0])
if has_rewards:
    ax4.scatter(ep_rwd, rewards_raw, alpha=0.25, s=8, color="#CE93D8", label="Raw")
    ax4.plot(ep_rwd, moving_avg(rewards_raw, MA), color="#7B1FA2", linewidth=2, label=f"MA({MA})")
    # 報酬 0 のエピソード（異常終了候補）をマーク
    zero_mask = np.array(rewards_raw) == 0
    if zero_mask.any():
        ax4.scatter(ep_rwd[zero_mask], np.zeros(zero_mask.sum()),
                    color="red", s=25, marker="x", zorder=5, label="Reward=0")
    ax4.legend(loc="lower right", fontsize=8)
else:
    ax4.text(0.5, 0.5, "No data (rewards.npy missing)",
             ha="center", va="center", transform=ax4.transAxes)
ax4.set_xlabel("Episode")
ax4.set_ylabel("Total Reward")
ax4.set_title("Episode Reward (↑ better)")

# ── 5. 不満足者率 vs 消費電力量（パレートフロント）─
ax5 = fig.add_subplot(gs[1, 1])
if has_log:
    sc = ax5.scatter(energy_arr, disrate_arr,
                     c=ep_log, cmap="viridis", s=12, alpha=0.7)
    cb = plt.colorbar(sc, ax=ax5)
    cb.set_label("Episode")
    ax5.set_xlabel("Energy [GJ]")
    ax5.set_ylabel("Dissatisfaction Rate")
    ax5.set_title("Pareto Front (Energy vs Discomfort)")
else:
    ax5.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax5.transAxes)
    ax5.set_title("Pareto Front")

# ── 6. ベストECI推移（episode_log から逐次 max）──
ax6 = fig.add_subplot(gs[1, 2])
if has_log and valid_eci.any():
    # NaN を 0 として置換した上で cummax
    eci_filled = np.where(valid_eci, eci_arr, 0.0)
    cummax_eci = pd.Series(eci_filled).cummax().values
    ax6.plot(ep_log, cummax_eci, color="#FF8F00", linewidth=2, label="Best ECI so far")
    if not np.isnan(best_eci_val):
        ax6.axhline(best_eci_val, color="gold", linestyle="--", linewidth=1,
                    label=f"Saved best={best_eci_val:.4f}")
    ax6.legend(loc="lower right", fontsize=8)
else:
    ax6.text(0.5, 0.5, "No ECI data", ha="center", va="center", transform=ax6.transAxes)
ax6.set_xlabel("Episode")
ax6.set_ylabel("Best ECI (cumulative)")
ax6.set_title("Best ECI Progress")

# ──────────────────────────────────────────────
# 保存・表示
# ──────────────────────────────────────────────
out_path = "learning_progress.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
