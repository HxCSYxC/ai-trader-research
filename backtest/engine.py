import pandas as pd
from backtesting import Backtest

import config
from data.fetch import get_daily_bars


def run_backtest(df: pd.DataFrame, strategy_cls, **params) -> pd.Series:
    """喺下一枝 K 開市價成交（trade_on_close=False），避免用到未來價。"""
    bt = Backtest(
        df,
        strategy_cls,
        cash=config.DEFAULT_CASH,
        commission=config.COMMISSION,
        exclusive_orders=True,
        finalize_trades=True,
    )
    return bt.run(**params)


def benchmark_equity(index: pd.DatetimeIndex, ticker: str = config.BENCHMARK) -> pd.Series:
    """買入持有 benchmark 嘅資金曲線，對齊策略嘅日期。"""
    start = index[0].strftime("%Y-%m-%d")
    end = (index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    close = get_daily_bars(ticker, start, end)["Close"].reindex(index).ffill().bfill()
    return close / close.iloc[0] * config.DEFAULT_CASH


def summarize(stats: pd.Series, bench: pd.Series) -> dict:
    bench_return = (bench.iloc[-1] / bench.iloc[0] - 1) * 100
    bench_dd = ((bench / bench.cummax()) - 1).min() * 100
    return {
        "策略回報 %": stats["Return [%]"],
        "買入持有呢隻股 %": stats["Buy & Hold Return [%]"],
        f"買入持有 {config.BENCHMARK} %": bench_return,
        "年化回報 %": stats["Return (Ann.) [%]"],
        "Sharpe Ratio": stats["Sharpe Ratio"],
        "最大回撤 %": stats["Max. Drawdown [%]"],
        f"{config.BENCHMARK} 最大回撤 %": bench_dd,
        "交易次數": stats["# Trades"],
        "勝率 %": stats["Win Rate [%]"],
        "持倉時間 %": stats["Exposure Time [%]"],
    }
