import hashlib
import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def get_daily_bars(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """拎日線 OHLCV（已調整拆股同派息）。"""
    # yfinance 內部用咗 pandas 3 deprecated 嘅寫法，啲 warning 唔關我哋事
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
    if df is None or df.empty:
        raise ValueError(f"搵唔到 {ticker} 嘅數據，檢查下 ticker 啱唔啱")

    df = df[OHLCV].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


CACHE_DIR = Path(__file__).parent / "cache"


def get_universe_bars(tickers: list[str], start: str, max_age_hours: float = 12) -> dict[str, pd.DataFrame]:
    """一次過拎多隻股。返回 {"Open": 日期×ticker, "Close": …, "Volume": …}，會 cache 喺硬碟。"""
    key = hashlib.md5(f"{sorted(tickers)}|{start}".encode()).hexdigest()[:12]
    path = CACHE_DIR / f"universe_{key}.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        return pd.read_pickle(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers, start=start, interval="1d", auto_adjust=True,
            progress=False, group_by="column", threads=True,
        )
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    bars = {}
    for field in ("Open", "Close", "Volume"):
        frame = raw[field].reindex(columns=tickers)
        # 停牌 / 數據缺口：用上一日價補；未上市之前保持 NaN
        bars[field] = frame.ffill() if field != "Volume" else frame
    # 下載失敗嘅 ticker 直接剔走（500 隻入面間中會有一兩隻）
    missing = [t for t in tickers if bars["Close"][t].isna().all()]
    if len(missing) > len(tickers) * 0.05:
        raise ValueError(f"太多 ticker 冇數據（{len(missing)} 隻），可能俾 Yahoo 限速，遲啲再試：{missing[:10]}")
    keep = [t for t in tickers if t not in missing]
    bars = {f: b[keep] for f, b in bars.items()}
    bars["missing"] = missing

    CACHE_DIR.mkdir(exist_ok=True)
    pd.to_pickle(bars, path)
    return bars
