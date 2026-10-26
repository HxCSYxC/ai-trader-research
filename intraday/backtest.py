"""即日形態回測。

出場規則（所有形態一樣，預先定死）：
  止蝕 = 入市價 − 1 × ATR；止賺 = 入市價 + 2 × ATR；15:55 前未觸發就平倉。
  同一枝 K 線同時掂到止蝕同止賺 → 當止蝕（保守）。開市跳空穿咗 → 用開市價。
對照組：同一日、同一隻股、隨機一枝 K 線入市，用同一套出場規則。
"""
import zlib

import numpy as np
import pandas as pd

from intraday.data import SYMBOLS, add_session_columns, get_bars
from intraday.patterns import LAST_SIGNAL_BAR, PATTERNS, find_signals

STOP_ATR, TARGET_ATR = 1.0, 2.0
LOCKBOX_START = pd.Timestamp("2025-01-01")


def simulate(o, h, l, c, signal: int, atr: float) -> tuple[float, int, str]:
    """返回 (毛回報, 出場第幾枝, 出場原因)。"""
    e = signal + 1
    entry = o[e]
    stop, target = entry - STOP_ATR * atr, entry + TARGET_ATR * atr
    last = len(c) - 2  # 15:50 嗰枝收市（約 15:55）平倉
    for j in range(e, last + 1):
        if l[j] <= stop:
            px = o[j] if (j > e and o[j] < stop) else stop
            return px / entry - 1, j, "止蝕"
        if h[j] >= target:
            px = o[j] if (j > e and o[j] > target) else target
            return px / entry - 1, j, "止賺"
    return c[last] / entry - 1, last, "收市平倉"


def run_symbol(symbol: str, seed: int = 0) -> pd.DataFrame:
    df = add_session_columns(get_bars(symbol))
    rng = np.random.default_rng(zlib.crc32(symbol.encode()) + seed)  # 固定 seed，結果可以重現
    rows = []
    for day, g in df.groupby("day", sort=True):
        if len(g) < 70:  # 半日市或者數據唔齊就跳過
            continue
        o, h, l, c = (g[k].to_numpy() for k in ("open", "high", "low", "close"))
        atr = g.atr.to_numpy()
        for pattern, i in find_signals(g).items():
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue
            ret, j, why = simulate(o, h, l, c, i, atr[i])
            r = int(rng.integers(1, min(LAST_SIGNAL_BAR, len(g) - 3) + 1))
            rand_ret, _, _ = simulate(o, h, l, c, r, atr[r]) if atr[r] > 0 else (np.nan, 0, "")
            rows.append({
                "symbol": symbol, "day": day, "pattern": pattern,
                "entry_time": g.index[i + 1], "exit_time": g.index[j], "entry": o[i + 1],
                "gross": ret, "exit_reason": why, "random_gross": rand_ret,
            })
    return pd.DataFrame(rows)


def run_all(symbols=SYMBOLS) -> pd.DataFrame:
    return pd.concat([run_symbol(s) for s in symbols], ignore_index=True)


def summarize(trades: pd.DataFrame, cost_bps: float, start=None, end=None) -> pd.DataFrame:
    """每個形態嘅表現（扣除成本後）。成本係每邊 bps，一買一賣扣兩次。"""
    t = trades
    if start is not None:
        t = t[t.day >= start]
    if end is not None:
        t = t[t.day < end]
    cost = 2 * cost_bps / 1e4
    rows = {}
    for pattern in PATTERNS:
        p = t[t.pattern == pattern].dropna(subset=["random_gross"])
        if p.empty:
            continue
        net, rnd = p.gross - cost, p.random_gross - cost
        diff = net - rnd
        wins, losses = net[net > 0].sum(), -net[net < 0].sum()
        rows[pattern] = {
            "交易次數": len(p),
            "勝率 %": (net > 0).mean() * 100,
            "平均每筆 bps": net.mean() * 1e4,
            "t 值": net.mean() / net.std() * len(net) ** 0.5,
            "Profit factor": wins / losses if losses > 0 else np.nan,
            "隨機入市 bps": rnd.mean() * 1e4,
            "跑贏隨機 bps": diff.mean() * 1e4,
            "跑贏隨機 t 值": diff.mean() / diff.std() * len(diff) ** 0.5,
        }
    return pd.DataFrame(rows).T


def periods() -> dict[str, tuple]:
    return {
        f"開發期 2020 → {LOCKBOX_START.year - 1}": (None, LOCKBOX_START),
        f"鎖櫃期 {LOCKBOX_START.year} → 而家": (LOCKBOX_START, None),
    }
