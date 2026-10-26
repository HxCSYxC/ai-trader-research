"""即日 5 分鐘 K 線：由 Alpaca（SIP 完整市場數據）攞，只保留正常交易時段，cache 喺硬碟。"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import config  # noqa: F401  載入 .env

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "intraday"
START = "2020-01-01"

# 25 隻成交最活躍嘅 ETF 同大型股（用今日眼光揀，所以要同隨機入市比較）
SYMBOLS = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "NFLX", "CRM", "INTC", "MU",
    "JPM", "BAC", "XOM", "UNH", "LLY", "COST", "WMT", "BA", "DIS",
]


def _client():
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def _download(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    frames = []
    for year_start in pd.date_range(start.normalize(), end, freq="YS").union([start.normalize()]):
        year_end = min(year_start + pd.offsets.YearBegin(1), end)
        if year_start >= year_end:
            continue
        df = _client().get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=year_start.tz_localize("America/New_York"), end=year_end.tz_localize("America/New_York"),
            feed=DataFeed.SIP, adjustment=Adjustment.ALL,
        )).df
        if len(df):
            frames.append(df.droplevel("symbol"))
    df = pd.concat(frames)
    df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    df = df[~df.index.duplicated()].sort_index()
    # 只要正常時段 9:30–16:00（K 線開始時間 9:30 … 15:55）
    t = df.index.time
    df = df[(t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("15:55").time())]
    return df[["open", "high", "low", "close", "volume"]]


def get_bars(symbol: str, max_age_hours: float = 24) -> pd.DataFrame:
    """5 分鐘 K 線（index 係紐約時間）。最近 15 分鐘嘅數據免費帳戶攞唔到，所以只攞到尋日。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{symbol}_5m.pkl"
    end = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    if path.exists():
        cached = pd.read_pickle(path)
        fresh = (pd.Timestamp.now().timestamp() - path.stat().st_mtime) < max_age_hours * 3600
        if fresh or cached.index[-1].normalize() >= end - pd.Timedelta(days=1):
            return cached
    df = _download(symbol, pd.Timestamp(START), end)
    df.to_pickle(path)
    return df


def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    """加入每日 VWAP、ATR（5 分鐘，14 枝）、前日高位、當日第幾枝 K 線。"""
    df = df.copy()
    day = df.index.normalize()
    typical = (df.high + df.low + df.close) / 3
    pv = (typical * df.volume).groupby(day).cumsum()
    vol = df.volume.groupby(day).cumsum()
    df["vwap"] = pv / vol.replace(0, np.nan)

    prev_close = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev_close).abs(), (df.low - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    daily_high = df.high.groupby(day).max()
    df["prev_high"] = day.map(daily_high.shift(1))
    df["bar"] = df.groupby(day).cumcount()
    df["day"] = day
    return df
