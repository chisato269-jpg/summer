# C-5：執務者とゾーンの対応（R-5 の材料）

| 項目 | 内容 |
|---|---|
| 手順 | 指示書 V2-S 第 6-3 節 C-5 |
| 実施日 | 2026-09-16 |
| 位置づけ | 観測はテナント単位のまま維持する（R-5 の決定）。ゾーン別へ拡張できるかどうかの材料だけを報告する |

## 確認した事実

| 事項 | 確認結果 | 出所 |
|---|---|---|
| `setting.ini` の `rseed_oprm` | **1** | `Shizuku2/setting.ini` を診断スクリプトが読み取り（`C1/summary.json` の `C5_rseed_oprm`） |
| 執務者とゾーンの対応表の有無 | **ある**。`reference_ja.pdf` Appendix 2「Occupants」は、執務者 1 人ごとに No・Tenant（South／North）・Zone（S1〜S9／N1〜N9）・氏名・年齢・身長・体重・性別を並べた表である | `Shizuku2/reference_ja.pdf` Appendix 2（A2-1） |
| 表の前提 | 同表は乱数シード `rseed_oprm = 1` の場合の一覧である。`rseed_oprm` を変えると人数と各執務者の属性が変わる | `reference_ja.pdf` 1.2 節 3)・2.5 節 |
| テナント別の人数 | 南 **38** 人（No.1〜38）、北 **48** 人（No.39〜86）、合計 86 人。実装の定数 `TENANT_OCCUPANT_COUNT` と一致する | Appendix 2、Appendix 1 の BACnet オブジェクト（`Availability_OC_38` が tenant-1 の最大、`Availability_OC_48` が tenant-2 の最大） |
| 在室者数の実測との整合 | 規則ベース最良軌跡の 1 エピソードで観測した同時在室者数の最大は 南 **32** 人、北 **36** 人。総数（38・48）の内側であり矛盾しない（全員が同時に在室する時刻が無いだけである） | `C1/summary.json` の `C5_n_present_max_south`・`C5_n_present_max_north` |

## ゾーン別への拡張について（実行しない）

Appendix 2 の Zone 列を定数表として持てば、`get_thermal_sensation(tenant, occupant_index)` で読んだ個人の温冷感をゾーンへ割り当てられる。すなわち、v2 が API の制約（I-8・D-7）を理由にテナント単位へ後退させた観測 66–79 を、ゾーン別 7 段階（18 × 7 = 126 次元）へ戻す材料は揃っている。

ただし次の 2 点に注意が要る。第一に、対応表は `rseed_oprm = 1` に固有であり、この値を変える走行では作り直しになる。第二に、対応表は PDF 中の表であって機械可読な出力ファイルではないため、転記の正しさを別途確かめる必要がある（例えば、ゾーン別在室者数 `get_zone_occupant_number` の合計と、対応表に基づく在室者の集計が全ステップで一致することを 1 エピソードで照合する）。

本指示書の範囲では拡張しない（第 1-3 節・R-5）。
