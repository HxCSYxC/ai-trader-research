"""Features 同 label。

時間線：第 t 日收市後計 features、出訊號 → 第 t+1 日開市價成交。
所以 label 用「t+1 開市 → t+1+h 開市」嘅回報，同實際成交一致。
"""
import numpy as np
import pandas as pd

from strategies.indicators import ema, rsi


def make_features(df: pd.DataFrame, bench_close: pd.Series) -> pd.DataFrame:
    """全部只用第 t 日或之前嘅數據。"""
    c, v = df.Close, df.Volume
    f = pd.DataFrame(index=df.index)

    for n in (1, 5, 10, 20, 60):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (10, 20, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1

    f["rsi14"] = rsi(c, 14) / 100
    macd = ema(c, 12) - ema(c, 26)
    f["macd_hist"] = (macd - ema(macd, 9)) / c
    f["volatility_20"] = c.pct_change().rolling(20).std()
    f["range_14"] = ((df.High - df.Low) / c).rolling(14).mean()
    f["volume_ratio"] = np.log(v / v.rolling(20).mean())
    f["dist_high_252"] = c / c.rolling(252).max() - 1

    # 大市環境
    b = bench_close.reindex(df.index).ffill()
    f["spy_ret_20"] = b.pct_change(20)
    f["spy_dist_sma200"] = b / b.rolling(200).mean() - 1

    return f.replace([np.inf, -np.inf], np.nan)


def make_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    """1＝之後 horizon 日升，0＝跌。最後 horizon+1 日未知（NaN）。"""
    o = df.Open
    fwd = o.shift(-(horizon + 1)) / o.shift(-1) - 1
    return (fwd > 0).astype(float).where(fwd.notna())
