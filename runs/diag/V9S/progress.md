# V9-S 進捗（再開に必要な状態）

| 項目 | 内容 |
|---|---|
| 状況 | **中断中**（2026-10-06 13:00 ごろ、利用者の指示で A1-S・H1-S を先に行うため）。走行 `v9s_single01` は未起動。手順 1〜2 を終え、手順 2 の実装と単体試験は終了、手順 3 以降は未着手 |
| 完了 | 手順 1 事前登録（`prereg.md`、実装前の解釈 V9S-a〜V9S-h を含む）。手順 2 の実装（`rlkit/v9/` の nets・credit・learner9・phase_a・phase_b・switch・trainer_single、`rlkit/config_summer_v9.json`・`config_summer_v9_smoke.json`、`rlkit/diag/v9s_freeze.py`、`rlkit/config.py` の任意項目 `v9`、`rlkit/tests/test_v9.py`）。V9 の単体試験 8 本（UT-V9-1〜7）は通過 |
| 未完 | (a) 既存の非実機の全試験（中断で止めた。`tests/full_nonlive.log` は途中）。(b) 実機の試験（V8 の実機 5 本ほか）。(c) trial08・trial03 の設定との対照表。(d) 手順 3（環境の種の確認は、原本と複製 13 組で 19800614・19800614・1 を確認済み）。(e) 手順 5（trial03 の最良の再生）、手順 6（コード一式を OneDrive の外 `C:\Users\chisa\rl_code\v9s` へ複製し、複製で再生をやり直す）、手順 7（試走 `v9s_smoke`）、手順 8〜13 |
| 読んで分かったこと | 利用者の準備（U-1・U-2）: 電源接続時のスリープ・休止は「なし」（AC の設定 0）、Windows Update は 2026-11-05 まで一時停止（開始 2026-10-01）。ただし、アクティブ時間は 9〜3 時、最後の起動は 2026-10-06 12:24 |
| 注意 | 模擬の環境は有効でない判断（割合 0 のゾーンの設定温度など）も成績に入れるため、採用の後の L-7R が一致しない（本物の環境では一致する）。段階 A の PPO は `rlkit/v9/learner9.py`（`SummerPPOV9`）、段階 A の 1 更新は `rlkit/v9/phase_a.py`、段階 B は `rlkit/v9/phase_b.py`、切り替えは `rlkit/v9/switch.py` |
| 次の一手 | A1-S・H1-S の後に、V9-S の手順 4（全試験の続き）から再開する |
