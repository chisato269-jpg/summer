# V6.4-S 進捗

| 手順 | 状態 | 記録 |
|---|---|---|
| 1 Z 事前登録 | 完了 | prereg.md |
| 2〜3 R8'-1・R8'-2 | 完了: R8'-A、δ = −1（分の行は [開始, 終わり) に割り当て、V64S-a） | R8p_decision.json |
| 4 K-14'（112 本を再生し直し） | 完了: S 建物、P 建物、V テナント。恒等式の最大誤差 8.8e-16 | K14_decision.json |
| 5 K-15 | 完了: λ_hour = 0.99 | K15_decision.json |
| 6 分岐 | 「それ以外」→ 段階 A へ | — |
| 7 診断の短報 | 完了 | V64S_report_diag.md |
| 8 段階 A | 完了: 非実機 100 本・実機 7 本すべて通過 | ut_full.log、ut_live*.log |
| 9 段階 B | 完了: B-1・B-3 16 進一致、T-3 合格 | stageB/B1.json・B3.json |
| 10 段階 C | 完了: H-1〜H-5 合格、H-9 発動（報告のみ） | V64S_report_u20.md |
| 11 段階 D | 完了: 更新 300 で I-1〜I-3 合格、SIL を更新 301 から有効化（収穫の最良 ECI 0.6066） | V64S_report_u300.md |
| 12 段階 E | 更新 340 で SIL-6 発動なし。2026-09-27 18:25 に利用者の指示で更新 388 で中断（STOP ファイル）。F-1 未実施。再開は STOP を消して --resume | V64S_report_u388.md |

## 段階 A の作業一覧

C-1 水準（S 建物、P 建物、V テナント、system_evap は変えない）、C-2 水準ごとの正規化、C-3 λ_hour 0.99、C-4 R-7 の比を毎更新 ρ_raw へ（2 倍制限）、C-5 tenant_remote の廃止（60 次元・292 ロジット、観測 640、母数 145,980、リモコンは許可に固定）、訓練での R-8'（R8'-A、δ −1、S-11 1e-9）、記録項目、設定 `rlkit/config_summer_v64.json`（run_dir runs/v2s_trial07）、UT-26〜31・UT-32'・UT-33・UT-34。変更前のコードは `code_before_V64/`。

## 再開時の注意

`--resume` は指示書の分岐（H 合格）に従って 1 回だけ実行済み。停止規則で止まった場合は指示なしに再開しない。更新 21 で MUST-9 の引き継ぎ（events.jsonl の resumed と updates の更新 20・21）を確かめる。trial04〜06 を上書きしない。

## 段階 A の実装の所在（MUST-10、2026-09-26 時点）

| 決定 | ファイル:行 |
|---|---|
| C-1 水準 | rlkit/constants.py:269（CREDIT_LEVELS）、rlkit/advantage.py:76（level_zone_advantages）・135（assign_per_dim_advantages） |
| C-2 水準ごとの正規化・SIL の σ_A | rlkit/advantage.py:175（normalize_by_level）、rlkit/learner.py:453・305（sil_sigma_matrix） |
| C-3 λ_hour 0.99 | rlkit/constants.py:264 |
| C-4 比の毎更新の追従 | rlkit/reward.py:212（update_after_harvest） |
| C-5 リモコン廃止 | rlkit/constants.py:186・194・220、rlkit/action.py:31・158、rlkit/observation.py:29、rlkit/policy.py:89 |
| R-8' 環境 | rlkit/env.py:224（minute_output_dir）・228（_finish_r8p） |
| R-8' 学習器 | rlkit/learner.py:266（apply_r8p）・388 |
| S-7 不満足の側・S-11 | rlkit/train_summer.py:688・692 |
| 設定 | rlkit/config_summer_v64.json（run_dir runs/v2s_trial07） |
| 段階 B | rlkit/diag/v6s_stage_b.py（--out runs/diag/V64S/stageB --config rlkit/config_summer_v64.json） |
