# V7.1-S 進捗（最終更新 2026-10-01 14:35）
| 手順 | 状態 |
|---|---|
| 1 事前登録 | 完了 prereg.md |
| 2 保存点の複写 | 完了（SHA-256 一致） |
| 3 外部終了の原因 | 完了 kill_cause.md（PC の再起動。起動者はわかりません） |
| 4 C-11〜C-13・C-15 と試験 | 完了（非実機 143、実機 8 通過）。tests/ |
| 5 CF-0〜8 | 完了 cf/ （CF-0 は 16 進一致、CF-1〜8 は ECI で CF-0 を上回らず） |
| 6 空きメモリ | 再開前 6.0 GB。更新 685 中の最小 1.50 GB（要注意） |
| 7 再開 | 2026-10-01 14:20:46、PID 10560、C-15 の方式 |
| 8 更新 685 の MUST-9 | 合格 must9_check_u685.json |
| 9 V71S_report_resume.md | 完了 |
| 10 更新 686〜1,500 | 実行中。停止したら C-12 の再生の結果（runs/v2s_trial08/harvest/replay_check.json と status.json）を読み、V71S_report_u{更新}.md を書く。1,500 到達なら V71S_report_u1500.md |

再開の要点: 走行は rlkit.launch_detached で切り離して起動済み。止まっていたら（updates.jsonl の更新が 20 分以上無い）、status.json と events.jsonl の末尾で原因を見る。再起動で止まった場合は STOP が無いことを確認し、同じ命令（--resume）で再開できる（C-15 (2) により --resume なしは拒否される）。再開前に logs: runs/v2s_trial08/stdout_resume_u685.log。

## 追記 2026-10-01 22:35
- 2026-10-01 18:49 に更新 768 で停止（保存点 sil_buffer.pt の置き換えが OneDrive との競合で拒否された PermissionError。更新 246 と同種）。C-12 の再生は自動で走り、16 進一致（収穫の最良は更新 740 の ECI 0.6718）。
- 対策: runlog.retrying / replace_with_retries（拒否時に 3 秒おき 10 回までやり直す）を save_checkpoint・write_json・checkpoint_uN の複写に入れた。試験 46 本通過（tests/after_retry.log）。
- 22:33 に更新 768 から再開（--resume、launch_detached）。組 9・11 の複製の error.log が元と違い起動を拒否されたため、runs/diag/V71S/replica_error_logs/ に退避して元の内容に戻した（内容はエミュレータが dissatisfaction.csv を開けなかった IOException。OneDrive の掴みと見られる）。
- 表示: /status（rlkit/diag/status_text.py）、最前面の窓（~/.claude/trial_window.py）、スマホ用ページ（Artifact。rlkit/diag/status_page.py で作り直して公開し直す）。

## 完了 2026-10-03
- 更新 1500 到達（13:15:56）。C-12 の再生は 16 進一致（収穫の最良は更新 1268、ECI 0.7071）。最終報告 V71S_report_u1500.md を作成。
- 走行後の変更: モデルの保存先を OneDrive の外にする仕組み（model_dir / resolve_model_dir）。新しい走行は ~/rl_models/<走行名> に保存。trial08 のモデルはそのまま（config_summer_v7p.json は変更なし）。全試験 151 通過。
- 次に始めるもの無し（指示書の完了条件）。
