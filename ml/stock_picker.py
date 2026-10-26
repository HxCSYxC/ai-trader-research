"""AI 選股 pipeline：數據 → features → walk-forward 分數 → 組合回測。dashboard 同 CLI 共用。

開發紀律：config.ML_LOCKBOX_START 之前係「開發期」，所有改良都只可以睇開發期嚟揀；
之後係「鎖櫃期」，揀定方法先打開睇一次。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from data.fetch import get_daily_bars, get_universe_bars
from ml.cross_section import (
    apply_regime, blend, excess_vs, forward_returns, information_coefficient, make_features,
    quintile_returns, rebalance_dates, simulate, top_k, top_k_buffered, walk_forward_scores,
)
from ml.pipeline import curve_stats
from ml.universe import get_universe

EW = "平均買晒"
SPY = f"買入持有 {config.BENCHMARK}"
VARIANTS = {
    "AI": "ML 分數最高嘅 K 隻",
    "動量（冇 ML）": "過去 12 個月（唔計最近 1 個月）升得最多嘅 K 隻",
    "AI + 動量混合": "AI 排名同動量排名各佔一半",
    "AI + 減少換手": "已經揸住嘅只要仲喺頭 2K 名就唔賣",
    "AI 行業內排名": "features 喺同一行業入面比較，避免全部揸晒熱門行業",
    "AI + 跌市持現金": "SPY 跌穿 200 日線嗰個月持現金",
}


@dataclass
class PickerData:
    universe: str
    bars: dict
    dates: pd.DatetimeIndex
    X: pd.DataFrame
    X_sector: pd.DataFrame
    fwd: pd.Series
    spy_close: pd.Series
    sectors: pd.Series


def eligible(X: pd.DataFrame, added: pd.Series | None) -> pd.DataFrame:
    """剔走當時仲未加入指數嘅股。"""
    if added is None:
        return X
    since = X.index.get_level_values("ticker").map(added)
    return X[np.asarray(X.index.get_level_values("date") >= since)]


def load(universe: str, max_age_hours: float = 12, until: pd.Timestamp | None = None) -> PickerData:
    """until：只用呢日之前嘅數據（落單嗰日要剔走今日未收市嘅 bar）。"""
    tickers, sectors, added = get_universe(universe)
    bars = get_universe_bars(tickers, config.ML_DATA_START, max_age_hours=max_age_hours)
    if until is not None:
        bars = {f: (b[b.index < until] if isinstance(b, pd.DataFrame) else b) for f, b in bars.items()}
    sectors = sectors.reindex(bars["Close"].columns)
    dates = rebalance_dates(bars["Close"].index)
    # 先篩走未入指數嘅股，再喺剩低嘅股入面重新排名
    X = eligible(make_features(bars, dates), added)
    return PickerData(
        universe=universe, bars=bars, dates=dates,
        X=_rerank(X), X_sector=_rerank(X, sectors),
        fwd=forward_returns(bars, dates),
        spy_close=_before(get_daily_bars(config.BENCHMARK, config.ML_DATA_START).Close, until),
        sectors=sectors,
    )


def _before(s: pd.Series, until):
    return s[s.index < until] if until is not None else s


def _rerank(X: pd.DataFrame, sectors: pd.Series | None = None) -> pd.DataFrame:
    """篩走未入指數嘅股之後重新排名（百分位嘅排名次序唔變，可以直接再排）。"""
    if sectors is None:
        return X.groupby(level="date").rank(pct=True)
    sector = X.index.get_level_values("ticker").map(sectors)
    return X.groupby([X.index.get_level_values("date"), sector]).rank(pct=True)


def score(data: PickerData, model_name: str, sector_neutral: bool = False) -> pd.Series:
    """慢嘅部分，dashboard 會 cache。"""
    return walk_forward_scores(data.X_sector if sector_neutral else data.X, data.fwd, model_name)


def momentum_scores(data: PickerData, like: pd.Series) -> pd.Series:
    return data.X["mom_12_1"].reindex(like.index).dropna()


def build_curves(data: PickerData, k: int, ai: pd.Series, ai_sector: pd.Series):
    """所有版本嘅資金曲線同換手率，全部由同一個 out-of-sample 起點開始。"""
    mom = momentum_scores(data, ai)
    everyone = {d: g.index.get_level_values("ticker").tolist() for d, g in ai.groupby(level="date")}
    holdings = {
        "AI": top_k(ai, k),
        "動量（冇 ML）": top_k(mom, k),
        "AI + 動量混合": top_k(blend(ai, mom), k),
        "AI + 減少換手": top_k_buffered(ai, k, 2 * k),
        "AI 行業內排名": top_k(ai_sector, k),
        "AI + 跌市持現金": apply_regime(top_k(ai, k), data.spy_close),
        EW: everyone,
    }
    curves, turnover = {}, {}
    for name, h in holdings.items():
        curves[name], turnover[name] = simulate(data.bars, data.dates, h)

    first = curves["AI"].index[0]
    spy = data.spy_close.loc[first:]
    curves[SPY], turnover[SPY] = spy / spy.iloc[0] * config.DEFAULT_CASH, 0.0
    end = min(c.index[-1] for c in curves.values())
    return {n: c.loc[:end] for n, c in curves.items()}, turnover


def period_table(curves: dict, turnover: dict, start=None, end=None) -> pd.DataFrame:
    """[start, end) 期間每個版本嘅表現，同埋相對「平均買晒」嘅超額回報。"""
    def cut(c):
        c = c.loc[start:] if start is not None else c
        return c[c.index < end] if end is not None else c

    base = cut(curves[EW])
    rows = {}
    for name, c in curves.items():
        c = cut(c)
        rows[name] = {**curve_stats(c), **(excess_vs(c, base) if name != EW else {}),
                      "每月換手 %": turnover[name] * 100}
    return pd.DataFrame(rows).T


def split_periods(curves: dict) -> dict[str, tuple]:
    first = curves["AI"].index[0]
    lock = pd.Timestamp(config.ML_LOCKBOX_START)
    return {
        f"開發期 {first:%Y-%m} → {lock - pd.Timedelta(days=1):%Y-%m}": (None, lock),
        f"鎖櫃期 {lock:%Y-%m} → 而家": (lock, None),
    }


def _between(s: pd.Series, start=None, end=None) -> pd.Series:
    d = s.index.get_level_values("date")
    mask = pd.Series(True, index=s.index)
    if start is not None:
        mask &= d >= start
    if end is not None:
        mask &= d < end
    return s[mask.to_numpy()]


def signal_quality(scores: dict[str, pd.Series], fwd: pd.Series, start=None, end=None) -> pd.DataFrame:
    rows = {}
    for name, s in scores.items():
        ic = information_coefficient(_between(s, start, end), fwd)
        rows[name] = {
            "平均 IC": ic.mean(),
            "IC > 0 嘅月份 %": (ic > 0).mean() * 100,
            "IC t 值": ic.mean() / ic.std() * len(ic) ** 0.5,
            "月份數": len(ic),
        }
    return pd.DataFrame(rows).T


def quintiles(scores: dict[str, pd.Series], fwd: pd.Series, start=None, end=None) -> pd.DataFrame:
    return pd.DataFrame({n: quintile_returns(_between(s, start, end), fwd) for n, s in scores.items()})


def latest_picks(scores: pd.Series, k: int) -> tuple[pd.Timestamp, pd.Series]:
    last = scores.index.get_level_values("date").max()
    return last, scores.xs(last, level="date").nlargest(k)
