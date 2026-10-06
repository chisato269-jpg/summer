# V9-S の設定の対照表（trial08・trial03・V9-S）

trial08 は、最後の再開（`meta_resume_1790861684.json`、13 組の構成）の `config`。最初の起動（6 組）の `config` は 13 組の項目を持たないので使わない。trial03 は `meta_start` の `config`。違いのある項目に意図の理由を付け、意図の一覧に無い違いは「要確認」とする。

| 項目 | trial08（段階 A の元） | trial03（段階 B の元） | V9-S | 違い |
|---|---|---|---|---|
| c83_limit_s | "（無し）" | 600.0 | "（無し）" | V9-S は C8-3 を使わない（意図） |
| comm_rpm | true | true | true |  |
| episodes_per_slot | 1 | 1 | 1 |  |
| eval_concurrent | true | false | true | 段階 B は評価を基準の軌跡（組 12）で兼ねる（trial03 と同じ。意図） |
| eval_slot | 12 | 12 | 12 |  |
| include_indoor_unit_power | true | true | true |  |
| init_policy | "（無し）" | "C:/Users/chisa/rl_models/v8s_trial03/g0r3_policy.pt" | "（無し）" | 新規の初期化（意図） |
| init_weights | "（無し）" | "runs/v2s_trial08/best/best_official.pt" | "（無し）" | 新規の初期化（意図） |
| k9_result | "B" | "B" | "B" |  |
| l6r_loss | "（無し）" | "k3" | "（無し）" | 段階 B の設定は `v9.phase_b` に置いた（値は trial03 と同じ K-3。意図） |
| l6r_lr | "（無し）" | 0.003 | "（無し）" | 同上（3e-3） |
| l6r_max_epochs | "（無し）" | 1000 | "（無し）" | 同上（1,000） |
| m9_enabled | true | true | true |  |
| max_collection_retries | 4 | 4 | 4 |  |
| max_eval_retries | 4 | 4 | 4 |  |
| min_free_gb | 3.0 | 3.0 | 4.5 | 起動前の空きメモリ 4.5 GB（手順 9。意図） |
| model_dir | "（無し）" | "C:\\Users\\chisa\\rl_models\\v8s_trial03" | "C:/Users/chisa/rl_models/v9s_single01" | モデルの置き場所（意図。OneDrive の外。MUST-7） |
| planned_updates | 1500 | 600 | 2500 | 計画更新数（意図。C9-9） |
| run_dir | "runs/v2s_trial08" | "runs/v8s_trial03" | "C:/Users/chisa/rl_local/runs/v9s_single01" | 走行名（意図） |
| seed | 19800614 | 19800614 | 20261008 | 学習の種（意図。P-4） |
| slot_limit | 13 | 13 | 13 |  |
| start_DR_hex | "（無し）" | "0x1.6bec4c0000000p-3" | "（無し）" | 同上 |
| start_E_hex | "（無し）" | "0x1.1ac98b4395810p+2" | "（無し）" | 段階 B の始点は切り替えの時に自動で決まる（意図。C9-6） |
| train_slots | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] |  |
| v9 | "（無し）" | "（無し）" | {"reference0": {"rho": 60.0, "E_ref": 6.5, "DR_ref": 0.2}, "switch": {"min_update": 600, "window": 300, "rise": 0.002, "cap_update": 1200}, "phase_b": {"l6r_loss": "k3", "l6r_lr": 0.003, "l6r_max_epochs": 1000}, "candidate2": {"loss": "k3", "lr": 0.003, "max_epochs": 3000}, "environment_seeds": {"rseed_o": 19800614, "rseed_w": 19800614, "rseed_oprm": 1}} | V9-S の設定（切り替え・環境の種・新規初期化。意図） |
| warm_start_from | "runs/v2s_trial07/checkpoint_latest.pt" | null | "（無し）" | 新規の初期化（意図。S-1） |
| zone_step_energy_norm_gj | 0.02 | 0.02 | 0.02 |  |

## 学習則の数値（`rlkit/constants.py`。trial08 の段階 A と V9-S で同じファイルを使う）

| 項目 | V9-S 指示書の第 5-2 節 | `constants.py` の値 | trial08 の記録 | 判定 |
|---|---|---|---|---|
| 方策のエポック数 | 4 | 4 | updates.jsonl の epochs_applied 最大 4 | 一致 |
| 価値のエポック数 | 8 | 8 |  | 一致 |
| ミニバッチ（方策） | 756 | 756 |  | 一致 |
| ミニバッチ（価値） | （表に価値の値は無い。V9S-i） | 378 | trial08 と同じ 378 | 指示書に価値の値が無い（trial08 の 378 を使う） |
| 方策学習率 初期・下限・上限 | 1e-4・1e-5・1e-3 | 0.0001・1e-05・0.001 |  | 一致 |
| 学習率の調整 | KL>0.1 で 1/1.5、<0.025 で 1.5 倍 | 目標 0.05 の 2.0 倍超で 1/1.5、0.5 倍未満で ×1.5 | KL>0.1・<0.025 と同じ | 一致 |
| KL の目標・打ち切り | 0.05・0.075 | 0.05・0.07500000000000001 |  | 一致 |
| エントロピー係数 | 群ごとに 0.002/ln K、一定 | c0 0.002（V9 は減衰を使わない） | trial08 の c_g: {"zone_state": 0.00289, "zone_setpoint": 0.00103, "zone_fan": 0.00182, "system_evap": 0.00096, "tenant_vent": 0.00144} | 一致 |
| λ_hour | 0.99 | 0.99 | updates.jsonl の lambda_hour 0.99 | 一致 |
| 価値の学習率 | （記載なし。V9S-g） | 0.0005 |  | 一致 |
| クリップ範囲・最大勾配ノルム・Adam eps | （記載なし。V9S-g） | 0.2・0.5・1e-05 |  | 一致 |
| 自己模倣学習 | 使わない | SIL_ENABLED = False | trial08 の sil.c_sil は 0 | 一致 |
| 信用割当の水準 | S・P 建物、V テナント（trial08）+ C9-4 | {"S": "building", "P": "building", "V": "tenant"} | meta_start の credit_levels と同じ | 一致 |

意図した違い（指示書の手順 2）: 新規の初期化、段階 A の行動空間（80 次元・372 ロジット）と観測（704 次元）、C9-3（新しい選択肢の偏り −4）、エントロピー係数の一定化、切り替え、計画更新数 2,500、学習の種 20261008、C9-7、C9-10。上の表に意図の一覧に無い違いは無い（`min_free_gb` は手順 9 の起動前の空きメモリの確認値 4.5 GB）。
