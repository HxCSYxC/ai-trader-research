"""由數據到預測到回測，dashboard 同 CLI 共用。"""
import numpy as np
import pandas as pd

import config
from backtest.robustness import equity_return, max_drawdown, run_period
from data.fetch import get_daily_bars
from ml.features import make_features, make_labels
from ml.walkforward import walk_forward_proba
from strategies.baseline import SmaCross
from strategies.ml_signal import MLSignal


def load_data(ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    df = get_daily_bars(ticker, config.ML_DATA_START)
    bench = get_daily_bars(config.BENCHMARK, config.ML_DATA_START).Close
    return df, bench.reindex(df.index).ffill().bfill()


def predict(df: pd.DataFrame, bench_close: pd.Series, model_name: str, horizon: int):
    """返回 (加咗 Prob 欄嘅 df, labels)。Prob 全部係 out-of-sample。"""
    X = make_features(df, bench_close)
    y = make_labels(df, horizon)
    probs = walk_forward_proba(X, y, model_name, horizon)
    return df.assign(Prob=probs), y


def equity_curves(df_prob: pd.DataFrame, bench_close: pd.Series, ticker: str,
                  entry: float, exit: float) -> dict[str, pd.Series]:
    """全部由第一個預測日開始計，公平比較。"""
    start = df_prob.Prob.first_valid_index()

    def hold(close):
        c = close.loc[start:]
        return c / c.iloc[0] * config.DEFAULT_CASH

    return {
        "AI 訊號": run_period(df_prob, MLSignal, {"entry": entry, "exit": exit}, trade_from=start),
        "SMA Cross 20/50": run_period(df_prob, SmaCross, {}, trade_from=start),
        f"買入持有 {ticker}": hold(df_prob.Close),
        f"買入持有 {config.BENCHMARK}": hold(bench_close),
    }


def curve_stats(equity: pd.Series) -> dict:
    daily = equity.pct_change().dropna()
    years = len(equity) / 252
    return {
        "總回報 %": equity_return(equity),
        "年化回報 %": ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100,
        "最大回撤 %": max_drawdown(equity),
        "Sharpe": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0,
    }
