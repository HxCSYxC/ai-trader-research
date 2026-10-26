"""全形態掃描，規則見 PREREGISTRATION.md（喺睇結果之前登記）。

    python -m intraday.scan            # 掃描 + 兩關判定，結果存喺 data/cache/intraday/scan_*.pkl
"""
import math
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import talib

from intraday.backtest import LOCKBOX_START, STOP_ATR, TARGET_ATR
from intraday.data import SYMBOLS, add_session_columns, get_bars
from intraday.patterns import LAST_SIGNAL_BAR, OPENING_BARS, find_signals

COST_BPS = 2.0
MIN_TRADES = 300
FDR = 0.05
LOCKBOX_T = 2.0
CDL_FUNCS = sorted(f for f in talib.get_functions() if f.startswith("CDL"))
OUT = Path(__file__).resolve().parents[1] / "data" / "cache" / "intraday"


def simulate_dir(o, h, l, c, signal: int, atr: float, direction: int) -> float:
    """direction 1＝好倉、-1＝沽空。返回毛回報（已按方向調整）。出場規則同 backtest.simulate 一樣。"""
    e = signal + 1
    entry = o[e]
    stop = entry - direction * STOP_ATR * atr
    target = entry + direction * TARGET_ATR * atr
    last = len(c) - 2
    for j in range(e, last + 1):
        hit_stop = l[j] <= stop if direction == 1 else h[j] >= stop
        if hit_stop:
            gap = (o[j] < stop) if direction == 1 else (o[j] > stop)
            px = o[j] if (j > e and gap) else stop
            return direction * (px / entry - 1)
        hit_target = h[j] >= target if direction == 1 else l[j] <= target
        if hit_target:
            gap = (o[j] > target) if direction == 1 else (o[j] < target)
            px = o[j] if (j > e and gap) else target
            return direction * (px / entry - 1)
    return direction * (c[last] / entry - 1)


def _hypotheses(df: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    """(形態, 方向) → 每枝 K 線有冇訊號。"""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    out = {}
    for f in CDL_FUNCS:
        v = getattr(talib, f)(o, h, l, c)
        out[(f, 1)] = v > 0
        out[(f, -1)] = v < 0
    return out


def scan_symbol(symbol: str) -> pd.DataFrame:
    df = add_session_columns(get_bars(symbol))
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr = df.atr.to_numpy()
    day_codes, day_values = pd.factorize(df.day)
    starts = np.r_[0, np.flatnonzero(np.diff(day_codes)) + 1]
    ends = np.r_[starts[1:], len(df)]
    day_len = (ends - starts)[day_codes]
    bar = df.bar.to_numpy()
    allowed = (bar >= OPENING_BARS) & (bar <= np.minimum(LAST_SIGNAL_BAR, day_len - 3)) & (day_len >= 70)

    signals = {}
    for key, mask in _hypotheses(df).items():
        idx = np.flatnonzero(mask & allowed)
        _, first = np.unique(day_codes[idx], return_index=True)  # 每日第一次
        signals[key] = idx[first]
    # 之前嗰 4 個即日形態（只做好倉）
    custom: dict[str, list[int]] = {}
    for d in np.flatnonzero((ends - starts) >= 70):
        g = df.iloc[starts[d]:ends[d]]
        for name, i in find_signals(g).items():
            if i >= OPENING_BARS:
                custom.setdefault(name, []).append(starts[d] + i)
    for name, idx in custom.items():
        signals[(name, 1)] = np.array(idx)

    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    rows = []
    for (pattern, direction), idx in signals.items():
        for g_i in idx:
            if not atr[g_i] > 0:
                continue
            d = day_codes[g_i]
            s, e = starts[d], ends[d]
            sl = slice(s, e)
            ret = simulate_dir(o[sl], h[sl], l[sl], c[sl], g_i - s, atr[g_i], direction)
            hi = min(LAST_SIGNAL_BAR, e - s - 3)
            r = int(rng.integers(OPENING_BARS, hi + 1))
            rnd = simulate_dir(o[sl], h[sl], l[sl], c[sl], r, atr[s + r], direction) if atr[s + r] > 0 else np.nan
            rows.append((symbol, day_values[d], pattern, direction, ret, rnd))
    return pd.DataFrame(rows, columns=["symbol", "day", "pattern", "direction", "gross", "random_gross"])


def run_scan(workers: int = 4) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(scan_symbol, SYMBOLS))
    trades = pd.concat(parts, ignore_index=True)
    trades["pattern"] = trades.pattern.astype("category")
    for col in ("gross", "random_gross"):
        trades[col] = trades[col].astype("float32")
    return trades


def _p_one_sided(t: float) -> float:
    return 0.5 * math.erfc(t / math.sqrt(2))


def evaluate(trades: pd.DataFrame) -> pd.DataFrame:
    """兩關判定。返回每個 (形態, 方向) 嘅開發期 / 鎖櫃期統計同結果。"""
    cost = 2 * COST_BPS / 1e4
    t = trades.dropna(subset=["random_gross"])
    dev = t[t.day < LOCKBOX_START]
    lock = t[t.day >= LOCKBOX_START]

    def stats(g):
        net = g.gross.astype(float) - cost
        diff = net - (g.random_gross.astype(float) - cost)
        n = len(g)
        return pd.Series({
            "n": n,
            "net_bps": net.mean() * 1e4,
            "net_t": net.mean() / net.std() * n ** 0.5 if n > 1 else np.nan,
            "vs_random_bps": diff.mean() * 1e4,
            "vs_random_t": diff.mean() / diff.std() * n ** 0.5 if n > 1 else np.nan,
        })

    res = dev.groupby(["pattern", "direction"], observed=True).apply(stats)
    res["p"] = res.vs_random_t.apply(lambda x: _p_one_sided(x) if pd.notna(x) else 1.0)

    # Benjamini–Hochberg（所有形態 × 方向一齊修正）
    m = len(res)
    order = res.p.sort_values()
    passed_rank = [k for k, p in enumerate(order, start=1) if p <= k / m * FDR]
    bh_cut = order.iloc[max(passed_rank) - 1] if passed_rank else -1
    res["bh_significant"] = res.p <= bh_cut
    res["stage1"] = res.bh_significant & (res.net_bps > 0) & (res.n >= MIN_TRADES)

    lock_stats = lock.groupby(["pattern", "direction"], observed=True).apply(stats)
    res = res.join(lock_stats.add_prefix("lock_"), how="left")
    # 鎖櫃期只對過咗第 1 關嘅形態有意義
    res["stage2"] = res.stage1 & (res.lock_net_bps > 0) & (res.lock_net_t > LOCKBOX_T)
    return res.sort_values("vs_random_t", ascending=False)


def main():
    import time

    t0 = time.time()
    trades = run_scan()
    trades.to_pickle(OUT / "scan_trades.pkl")
    res = evaluate(trades)
    res.to_pickle(OUT / "scan_results.pkl")

    print(f"掃描完成：{len(res)} 個（形態 × 方向），{len(trades):,} 筆交易，用咗 {time.time() - t0:.0f} 秒")
    print(f"第 1 關（開發期）通過：{int(res.stage1.sum())}｜第 2 關（鎖櫃期）通過：{int(res.stage2.sum())}")
    cols = ["n", "net_bps", "net_t", "vs_random_bps", "vs_random_t", "p", "bh_significant", "stage1"]
    print("\n開發期 vs_random_t 最高 15 個：")
    print(res[cols].head(15).round(3).to_string())
    if res.stage1.any():
        print("\n過咗第 1 關嘅形態，鎖櫃期結果：")
        print(res.loc[res.stage1, ["lock_n", "lock_net_bps", "lock_net_t", "stage2"]].round(3).to_string())


if __name__ == "__main__":
    main()
