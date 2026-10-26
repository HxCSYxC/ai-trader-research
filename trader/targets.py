"""計今個月要揸咩：同回測用同一套 features、model 同揀股方法。"""
from dataclasses import dataclass

import pandas as pd

import config
from ml.cross_section import blend, pick_buffered, predict_latest
from ml.stock_picker import load


@dataclass
class Targets:
    signal_date: pd.Timestamp   # 用邊日收市數據計
    weights: pd.Series          # ticker → 目標比重
    scores: pd.Series           # ticker → 排名分數（越高越好）
    prices: pd.Series           # ticker → 最近收市價


def new_york_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)


def compute_targets(strategy: str, model_name: str | None, k: int | None,
                    until: pd.Timestamp | None = None, held=()) -> Targets:
    """until 預設係紐約今日：只用尋日或之前已收市嘅數據，唔會偷睇今日。

    held：而家揸緊嘅 ticker。AI 策略會套用換手緩衝（config.AI_KEEP_MULT），避免 rank 邊緣嘅名因為
    重新下載數據嘅微小差異就換股。"""
    until = until if until is not None else new_york_today()
    data = load(config.TRADING_UNIVERSE, max_age_hours=1, until=until)
    last = data.X.index.get_level_values("date").max()
    mom = data.X["mom_12_1"].xs(last, level="date", drop_level=False).dropna()

    if strategy == "平均買晒":
        names = data.X.xs(last, level="date").dropna().index
        scores = pd.Series(1.0, index=names)
    elif strategy == "動量（冇 ML）":
        scores = mom.droplevel("date").nlargest(k)
    elif strategy in ("AI", "AI + 動量混合"):
        ai = predict_latest(data.X, data.fwd, model_name)
        s = ai if strategy == "AI" else blend(ai, mom)
        ranked = s.droplevel("date").sort_values(ascending=False)
        scores = ranked[pick_buffered(ranked.index.tolist(), set(held), k, config.AI_KEEP_MULT * k)]
    else:
        raise ValueError(f"模擬盤未支援策略：{strategy}")

    weight = min(1 / len(scores), config.MAX_POSITION_WEIGHT)
    prices = data.bars["Close"].iloc[-1].reindex(scores.index)
    return Targets(signal_date=last, weights=pd.Series(weight, index=scores.index),
                   scores=scores, prices=prices)
