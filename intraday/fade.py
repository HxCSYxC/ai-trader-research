"""假設 2：沽空突破（規則見 PREREGISTRATION.md）。只有鎖櫃期結果算數。

    python -m intraday.fade
"""
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from intraday.backtest import LOCKBOX_START
from intraday.data import SYMBOLS, add_session_columns, get_bars
from intraday.patterns import LAST_SIGNAL_BAR, OPENING_BARS, find_signals
from intraday.scan import COST_BPS, simulate_dir

FADE_PATTERNS = ["ORB 開市區間突破", "突破前日高位"]
T_THRESHOLD = 2.24  # Bonferroni：兩個假設


def fade_symbol(symbol: str) -> pd.DataFrame:
    df = add_session_columns(get_bars(symbol))
    rng = np.random.default_rng(zlib.crc32(symbol.encode()) + 2)
    rows = []
    for day, g in df.groupby("day", sort=True):
        if len(g) < 70:
            continue
        o, h, l, c = (g[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        atr = g.atr.to_numpy()
        hi = min(LAST_SIGNAL_BAR, len(g) - 3)
        for name, i in find_signals(g).items():
            if name not in FADE_PATTERNS or i < OPENING_BARS or not atr[i] > 0:
                continue
            ret = simulate_dir(o, h, l, c, i, atr[i], -1)
            r = int(rng.integers(OPENING_BARS, hi + 1))
            rnd = simulate_dir(o, h, l, c, r, atr[r], -1) if atr[r] > 0 else np.nan
            rows.append((symbol, day, name, g.index[i + 1], ret, rnd))
    return pd.DataFrame(rows, columns=["symbol", "day", "pattern", "entry_time", "gross", "random_gross"])


def evaluate(trades: pd.DataFrame) -> pd.DataFrame:
    cost = 2 * COST_BPS / 1e4
    out = {}
    for period, t in {"開發期（只作參考）": trades[trades.day < LOCKBOX_START],
                      "鎖櫃期（算數）": trades[trades.day >= LOCKBOX_START]}.items():
        for name in FADE_PATTERNS:
            p = t[t.pattern == name].dropna(subset=["random_gross"])
            net = p.gross - cost
            diff = net - (p.random_gross - cost)
            n = len(p)
            out[(period, name)] = {
                "交易次數": n, "勝率 %": (net > 0).mean() * 100,
                "平均每筆 bps": net.mean() * 1e4, "t 值": net.mean() / net.std() * n ** 0.5,
                "跑贏隨機 bps": diff.mean() * 1e4, "跑贏隨機 t 值": diff.mean() / diff.std() * n ** 0.5,
            }
    res = pd.DataFrame(out).T
    lock = res.index.get_level_values(0) == "鎖櫃期（算數）"
    res["通過"] = ""
    res.loc[lock, "通過"] = np.where(
        (res.loc[lock, "平均每筆 bps"] > 0) & (res.loc[lock, "t 值"] > T_THRESHOLD)
        & (res.loc[lock, "跑贏隨機 t 值"] > T_THRESHOLD), "✅", "❌")
    return res


def main():
    with ProcessPoolExecutor(4) as ex:
        trades = pd.concat(ex.map(fade_symbol, SYMBOLS), ignore_index=True)
    res = evaluate(trades)
    print(f"沽空突破：{len(trades):,} 筆交易，每邊成本 {COST_BPS} bps，門檻 t > {T_THRESHOLD}\n")
    print(res.round(2).to_string())


if __name__ == "__main__":
    main()
