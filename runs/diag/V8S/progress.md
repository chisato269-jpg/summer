# V8-S 進捗（最終更新 2026-10-03）
| 手順 | 状態 |
|---|---|
| 1 事前登録 | 完了（prereg.md） |
| 2 初期の重みの再生（D-1 を含む） | 完了。候補は trial08 best/best_official.pt（更新 1461）だけ（trial09 は存在しない）。決定論的な再生で E 0x1.1e374dd2f1aa0p+2、DR 0x1.6b6d400000000p-3 が記録と 16 進一致、ECI 0.70913700374106。初期の重み＝これ。d1/d1_result.json、d1/d1_actions.npz |
| 3 観測の時刻の量・系統とテナントの対応 | 系統とテナントの対応は問題なし（系統 1・2 = 南、3・4 = 北。停止条件に当たらない）。観測の一覧は実装の報告に書く |
| 4 実装（rlkit/v8/） | 完了（action・observation・policy・env・mockenv・blocks・improver・trainer、config_summer_v8.json、slot_worker に real_v8・mock_v8） |
| 5 単体試験 UT-V8-1〜8 と全試験 | UT-V8 は 14 本通過（test_v8.py）。全試験と実機は tests/ で実行中 |
| 6 MUST-8 の実機の確認 | 完了。5 本通過（test_v8_live.py）。オブジェクト番号は参照マニュアル A1-4・A1-17 と照合（書き込みと読み戻しが一致） |
| 7 G-0 | **不一致（停止規則）**。E 0x1.1e1dcac083127p+2、DR 0x1.6b69ae0000000p-3（記録 0x1.1e374dd2f1aa0p+2、0x1.6b6d4p-3）。V7 の行動列の再生は一時停止ありでも 16 進一致（物理は同じ）。切り分け: 配分の比を 3 点の標本から作ると観測のゾーンのエネルギーの特徴が変わり、378×80 判断のうち 4 つが変わる。終端の標本だけにすると一致。回避策は講じず、設計側の判断待ち |
| 8 V8S_report_impl.md | 作成済み（G-0 の不一致で停止して報告。設計側の判断待ち: W-32 G-0 の扱い、W-33 前提 V8S-a〜g） |
| 9 走行の開始（runs/v8s_trial01） | 完了（V8.1-S: 2026-10-03 20:36 開始、PID 29140、launch_detached） |
| 10〜11 | G-20 合格、**G-50 不合格（更新 50、23:41 に停止）**。C8-2・C8-3 合格。V8S_report_u50.md 作成済み。設計側の判断待ち（W-34・W-35）。保存点は ~/rl_models/v8s_trial01（更新 50）。再開は python -m rlkit.v8.trainer --config rlkit/config_summer_v8.json --resume（STOP 無し、設計側の許可後） |

再開の要点: 判断（G-0 の案 1〜3）が出たら、(1) 案に合わせて rlkit/v8/env.py の配分または G-0 の基準を変える、(2) python -m rlkit.diag.v8s_g0 をやり直す、(3) 全試験（実機を含む）、(4) 空きメモリ確認、(5) C-15 の方法で python -m rlkit.launch_detached --log runs/v8s_trial01/stdout.log -- python -m rlkit.v8.trainer --config rlkit/config_summer_v8.json（--resume は付けない）。モデルは ~/rl_models/v8s_trial01。
