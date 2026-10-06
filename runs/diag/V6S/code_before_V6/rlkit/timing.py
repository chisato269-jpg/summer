"""rlkit.timing — 二段の時間刻みの判定（現行 driver.zone_step_seconds・is_system_boundary と同じ式）。

エミュレータ通信に依存しない純粋関数だけを置き、観測・模擬環境・診断から共通に使う。
"""
from rlkit import constants


def zone_step_seconds(dt):
    """平日 06:00〜20:00 は 900 秒、それ以外は 3,600 秒。"""
    if dt.weekday() < 5 and constants.OCCUPIED_WEEKDAY_START_HOUR <= dt.hour < constants.OCCUPIED_WEEKDAY_END_HOUR:
        return constants.ZONE_STEP_OCCUPIED_SEC
    return constants.ZONE_STEP_UNOCCUPIED_SEC


def is_system_boundary(dt):
    """系統側の 3,600 秒境界（毎正時）か。エピソード開始が正時で刻みが 3600 の約数のため、分・秒が 0 と同値。"""
    return dt.minute == 0 and dt.second == 0 and dt.microsecond == 0
