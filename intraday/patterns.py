"""4 個即日形態，規則喺睇結果之前定死。全部只做好倉，每隻股每日每個形態最多一次。

訊號喺第 i 枝 K 線「收市」先確認，第 i+1 枝開市價入市（冇 look-ahead）。
"""
import numpy as np
import pandas as pd

OPENING_BARS = 6       # 開市頭 30 分鐘（6 枝 5 分鐘 K）
LAST_SIGNAL_BAR = 66   # 15:00 之後唔再開新倉，留時間畀佢走
VWAP_DIP = 0.005       # VWAP 重奪：當日最少試過跌穿 VWAP 0.5%

PATTERNS = {
    "ORB 開市區間突破": "收市升穿開市頭 30 分鐘嘅最高位",
    "突破前日高位": "開市喺前日高位之下，之後收市升穿前日高位",
    "VWAP 重奪": "當日曾經跌穿 VWAP 0.5% 以上，之後收市由下而上重返 VWAP",
    "好友吞沒": "連跌 3 枝之後，一枝陽燭完全包住前一枝陰燭",
}


def _first(mask: np.ndarray, lo: int, hi: int) -> int | None:
    idx = np.flatnonzero(mask[lo:hi + 1])
    return int(idx[0]) + lo if len(idx) else None


def find_signals(day: pd.DataFrame) -> dict[str, int]:
    """一日入面每個形態第一次出現嘅 K 線位置（第幾枝）。"""
    o, h, l, c = (day[k].to_numpy() for k in ("open", "high", "low", "close"))
    vwap = day.vwap.to_numpy()
    n = len(day)
    hi = min(LAST_SIGNAL_BAR, n - 3)
    out = {}
    if hi <= OPENING_BARS:
        return out

    # 1. ORB
    or_high = h[:OPENING_BARS].max()
    i = _first(c > or_high, OPENING_BARS, hi)
    if i is not None:
        out["ORB 開市區間突破"] = i

    # 2. 突破前日高位（開市已經高過就唔算「突破」）
    prev_high = day.prev_high.iloc[0]
    if not np.isnan(prev_high) and o[0] < prev_high:
        i = _first(c > prev_high, 0, hi)
        if i is not None:
            out["突破前日高位"] = i

    # 3. VWAP 重奪
    below_far = np.minimum.accumulate(l / vwap - 1) <= -VWAP_DIP
    cross_up = np.r_[False, (c[1:] > vwap[1:]) & (c[:-1] < vwap[:-1])]
    i = _first(cross_up & below_far, OPENING_BARS, hi)
    if i is not None:
        out["VWAP 重奪"] = i

    # 4. 好友吞沒
    down = c < o
    three_down = np.r_[[False] * 3, down[:-3] & down[1:-2] & down[2:-1]]
    engulf = np.r_[False, (c[1:] > o[1:]) & (o[1:] <= c[:-1]) & (c[1:] >= o[:-1])]
    i = _first(three_down & engulf, 4, hi)
    if i is not None:
        out["好友吞沒"] = i
    return out
