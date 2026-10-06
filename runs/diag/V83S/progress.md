# V8.3-S 進捗（再開に必要な状態）

| 項目 | 内容 |
|---|---|
| 目的 | 指示書 `V83S instruction.md`（2026-10-04 発行）の実行。採用の学習（L-6R）の設定を実データで較正し、選んだ設定で `runs/v8s_trial03` を 600 更新走らせる |
| 規則 | CLAUDE.md、V8-S〜V8.3-S の MUST/NOT。停止規則に当たったら停止して報告（自己判断で続けない）。trial02 の記録・保存点は書き換えない（MUST-20）。リーダーの走行ログは読まない |
| 報告 | `V83S_report_calib.md`（較正）、`V83S_report_u50.md`（更新 50）、`V83S_report_u600.md`（更新 600）。停止したら `V83S_report_u{更新}.md` |

## 手順の状況（更新: 2026-10-04 17:40）

| 手順 | 状況 |
|---|---|
| 1 事前登録 | 完了（`prereg.md`） |
| 2 実装と試験 | 完了（非実機 188 本通過、V8 実機 5 本通過） |
| 3 教師の再生成 | 完了（41 本すべて 16 進一致、外した本数 0） |
| 4 較正と選択 | 完了。K-1 42/42、K-3 41/42、K-4 25/42。選択は K-3（学習率 3e-3、最大 1,000） |
| 5 G-0R3 | 完了。合格（49 エポック、基準の軌跡が始点と 16 進一致）。始点 E 0x1.1ac98b4395810p+2、DR 0x1.6bec4c0000000p-3、ECI 0.712315 |
| 6 較正の報告 | 完了（`V83S_report_calib.md`） |
| 7 走行の開始 | 2026-10-04 13:30 に `runs/v8s_trial03` を開始（`python -m rlkit.launch_detached --log runs/v8s_trial03/train_stdout.log --pidfile runs/v8s_trial03/train_pid.json -- python -m rlkit.v8.trainer3 --config rlkit/config_summer_v83.json`。再開は末尾に `--resume`、PC の再起動後など。許可の要否は都度確認） |
| 8〜9 | 更新 50 まで完了。L-7R 50/50 一致、C8-3・G-20・G-50 合格（基準 ECI 0.72541）。`V83S_report_u50.md` 作成済み（2026-10-04 16:50） |
| 10 | 走行継続中。更新 59 で利用者の指示により一度停止（2026-10-04 17:16、C-12 合格、`V83S_report_u59.md`）。利用者の選択 B により、記録を `C:\Users\chisa\rl_local\runs\v8s_trial03` へ移し（112 ファイルの SHA-256 一致を確認。OneDrive 側は削除して `LOCATION.txt` だけ残した）、2026-10-04 18:08 に `--resume` で再開した（設定 `rlkit/config_summer_v83_local.json`。`run_dir` のみ絶対パスに変更し、損失・学習率・エポックは同じ）。更新 60・61 は L-7R 一致で通過。更新 600 で C-12 の再生と `V83S_report_u600.md`。停止規則に当たったら `V83S_report_u{更新}.md` を書いて停止 |

## 完了（2026-10-06）

走行は更新 600 まで完了（2026-10-06 03:31）。最良 ECI 0.76130（E 4.4697 GJ、DR 0.16843）、C-12 合格、最終の方策の確認も 16 進一致（`final_policy_check.json`）。最終報告 `V83S_report_u600.md` 作成済み。未解決は W-40（続けるか）と W-41（提出の評価方法）。以下の「次の一手」は完了前のもの。

## 次の一手（完了前の記録）

走行の記録は `C:\Users\chisa\rl_local\runs\v8s_trial03`（`updates.jsonl`・`l7r.jsonl`・`status.json`）で確認する。再開が要るときは同じ設定で `--resume`（`STOP` を削除、出力は `train_stdout_resume.log`・`train_pid_resume.json`）。状況表示は `python -m rlkit.diag.status_text`（両方の置き場所を探す）。通信量対策は実施済み（`rlkit/localdata.py`、`tests/test_localruns.py`）。新しい走行は設定の `run_dir` に `~/rl_local/runs/<走行名>` の絶対パスを書く。W-39 は選択 B で解決。
