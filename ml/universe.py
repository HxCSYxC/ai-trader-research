"""選股範圍。

⚠️ Survivorship bias：兩個範圍都係用「今日」嘅名單，過去十幾年都生存咗落嚟，
所以回測會偏樂觀。要公平比較，應該睇策略有冇贏到「平均買晒成個範圍」（同一個偏差）。
"""
import io
import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "sp500.csv"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# 50 隻人手揀嘅大型股（survivorship bias 最嚴重，用嚟做快速測試）
MEGA50 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CSCO", "ADBE", "CRM",
    "AMD", "INTC", "QCOM", "TXN", "IBM",
    "AMZN", "TSLA", "HD", "COST", "WMT", "PG", "KO", "PEP", "MCD", "NKE", "DIS", "NFLX",
    "BRK-B", "JPM", "V", "MA", "BAC", "GS", "MS", "AXP",
    "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE",
    "XOM", "CVX", "CAT", "GE", "HON", "BA", "MMM", "T", "VZ",
]
UNIVERSE = MEGA50  # 舊名，保留兼容


def sp500_constituents(max_age_days: float = 7) -> pd.DataFrame:
    """現時 S&P 500 成分股：ticker、公司名、GICS 行業、加入指數日期。由 Wikipedia 攞，cache 一星期。"""
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < max_age_days * 86400:
        cached = pd.read_csv(CACHE)
        if "added" in cached:  # 舊版 cache 冇呢欄，要重新攞
            return cached.assign(added=pd.to_datetime(cached["added"]))

    r = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0 (ai-trader research)"}, timeout=30)
    r.raise_for_status()
    t = pd.read_html(io.StringIO(r.text))[0]
    df = pd.DataFrame({
        "ticker": t["Symbol"].str.replace(".", "-", regex=False),  # BRK.B → BRK-B（yfinance 格式）
        "name": t["Security"],
        "sector": t["GICS Sector"],
        "added": pd.to_datetime(t["Date added"], errors="coerce"),
    })
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def get_universe(name: str) -> tuple[list[str], pd.Series, pd.Series | None]:
    """返回 (tickers, 行業, 可以開始揀嘅日期)。

    S&P 500：每隻股要等正式加入指數之後先可以揀。否則等於 2015 年就知道
    邊啲細股將來會升到入 S&P 500，回測會嚴重偏高（近年加入嘅股尤其嚴重）。
    """
    sp = sp500_constituents().set_index("ticker")
    if name == "50 隻大型股":
        return MEGA50, sp["sector"].reindex(MEGA50).fillna("Other"), None
    tickers = sp.index.tolist()
    return tickers, sp["sector"], sp["added"]


UNIVERSES = ["S&P 500（現時成分股）", "50 隻大型股"]
