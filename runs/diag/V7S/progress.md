# V7-S 進捗

| 手順 | 状態 | 記録 |
|---|---|---|
| 1 事前登録 | 完了 | prereg.md |
| 2 3-1 最良の軌跡の再生 | 完了: 16 進一致 | prep_prev.json |
| 3 3-2〜3-5 | 完了: 風向は既に 1、CO2 の予測を採用、影響範囲は不採用、λ 0.99 | co2_fit.json・influence_decision.json |
| 4 準備の短報 | 完了 | V7S_report_prep.md |
| 5 実装と単体試験 | 完了: 非実機 112・実機 8 すべて通過 | ut_full.log・ut_live*.log |
| 6 書き込み層の確認 | 完了: 旧設定の基準は 16 進一致、新設定の P-1 は E・DR とも旧と同じ | wcheck.json |
| 7 trial08 更新 1〜20 | 完了: 第 9 章の判定に合格（収穫の最良 0.6128）。更新 21 から再開し MUST-9 を確認 | V7S_report_u20.md・must9_u21.json |
| 8 更新 21〜300 | 更新 246 の後に例外で停止（2026-09-29 04:49、save_checkpoint の os.replace で PermissionError。保存点は更新 246 の状態で健全）。指示書に無い分岐のため再開せず報告・判断待ち | runs/v2s_trial08/status.json・events.jsonl |
| 8' 更新 300 の中間報告 | 完了（2026-10-01 作成、走行は止めず） | V7S_report_u300.md |
| 9 更新 301〜1,500 | 更新 684 で停止（2026-10-01 09:45 に外から終了。利用者の指示で STOP ファイルも設置済み）。保存点は更新 684 で健全。F-1・最終報告は未実施。再開は STOP を消して --resume（W-28 の判断待ち） | V7S_report_u684.md |

## 決まったこと・再開に要る情報

変更前のコードは `code_before_V7/`（V6.4-S の最終）。3-3 は既存の分単位の出力（runs/diag/V6S/slot_output/k11_a・k11_b・k11_d・z1_slot6 と raw/*.json）を使う。V は論文 Fig. 2・3 から 737.1 m³、隣接は論文 Fig. 2。trial07 は更新 388 で停止（STOP ファイル）、保存点は読むだけ（MUST-2）。

## 並列数拡張（2026-10-01、利用者の依頼）

一括読み取りと 12 組 × 1 本 + 評価 1 組の構成を実装し、実機の 3 更新で確認（1 更新 約 410 秒 → 約 160 秒）。報告 `runs/diag/V7S/par/parallel_diagnosis.md`。走行 trial08 は停止のまま（再開は W-28・W-29 の判断待ち。`--resume` を付け、STOP を削除、`config_summer_v7p.json` を使う）。
