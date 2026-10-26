"""檢查 overfitting：參數 heatmap 同 in-sample / out-of-sample 測試。"""
import numpy as np
import pandas as pd

from backtest.engine import run_backtest


def grid_values(lo: int, hi: int, n: int = 12) -> list[int]:
    return sorted(set(np.linspace(lo, hi, n).round().astype(int).tolist()))


def equity_return(equity: pd.Series) -> float:
    return (equity.iloc[-1] / equity.iloc[0] - 1) * 100


def max_drawdown(equity: pd.Series) -> float:
    return ((equity / equity.cummax()) - 1).min() * 100


def run_period(df, strategy_cls, params: dict, trade_from=None) -> pd.Series:
    """跑回測，返回由 trade_from 開始嘅資金曲線。

    用全段數據跑但 trade_from 之前唔落單，咁指標有足夠歷史 warmup，
    又唔會偷睇到 trade_from 之後嘅價。
    """
    if trade_from is not None:
        params = {**params, "trade_from": trade_from}
    equity = run_backtest(df, strategy_cls, **params)["_equity_curve"]["Equity"]
    return equity.loc[trade_from:] if trade_from is not None else equity


def param_grid(
    df, strategy_cls, fixed: dict, x_name: str, x_vals, y_name: str, y_vals, trade_from=None
) -> pd.DataFrame:
    """每個 (x, y) 組合嘅回報 %。唔合理嘅組合（例如 fast >= slow）係 NaN。"""
    grid = pd.DataFrame(np.nan, index=pd.Index(y_vals, name=y_name), columns=pd.Index(x_vals, name=x_name))
    for y in y_vals:
        for x in x_vals:
            params = {**fixed, x_name: x, y_name: y}
            if strategy_cls.valid_params(params):
                grid.loc[y, x] = equity_return(run_period(df, strategy_cls, params, trade_from))
    return grid


def best_params(grid: pd.DataFrame) -> dict:
    y, x = grid.stack().idxmax()
    return {grid.columns.name: int(x), grid.index.name: int(y)}


def grid_rank(grid: pd.DataFrame, params: dict) -> tuple[int, int]:
    """params 喺 grid 入面排第幾（1 = 最好）。"""
    values = grid.stack().dropna().sort_values(ascending=False)
    key = (params[grid.index.name], params[grid.columns.name])
    return list(values.index).index(key) + 1, len(values)


def grid_correlation(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """In-sample 同 out-of-sample 排名嘅相關度（Spearman）。

    接近 1：前段好嘅參數後段都好，策略穩定。接近 0 或負數：前段嘅好成績係運氣。
    """
    pair = pd.concat([a.stack(), b.stack()], axis=1).dropna()
    ranks = pair.rank()  # Spearman＝排名嘅 Pearson，唔使裝 scipy
    return ranks.iloc[:, 0].corr(ranks.iloc[:, 1])
