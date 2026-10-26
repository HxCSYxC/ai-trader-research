"""Cross-sectional 選股：每月尾幫成籃股票排名，揸頭 K 隻，下個月再換。

時間線（同單股策略一樣冇 look-ahead）：
  月尾第 d 日收市 → 計 features、排名 → d+1 日開市買入 → 下個月尾 +1 日開市賣出 / 換股
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from strategies.indicators import rsi

MODELS = {
    "Ridge（線性）": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "Gradient Boosting": lambda: HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0,
        early_stopping=False, random_state=0,
    ),
}

FEATURE_NAMES = {
    "mom_12_1": "12 個月動量（唔計最近 1 個月）",
    "mom_6_1": "6 個月動量（唔計最近 1 個月）",
    "ret_1m": "最近 1 個月回報",
    "volatility_60": "60 日波幅",
    "dist_high_252": "距離 52 週高位",
    "dist_sma200": "距離 200 日線",
    "rsi14": "RSI 14",
    "volume_trend": "成交量趨勢（20 日 / 120 日）",
}


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每個月最後一個交易日。最後一個月未完都會包埋（用嚟出「今日」嘅排名）。"""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)


def _raw_features(close: pd.DataFrame, vol: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "mom_12_1": close.shift(21) / close.shift(252) - 1,
        "mom_6_1": close.shift(21) / close.shift(126) - 1,
        "ret_1m": close / close.shift(21) - 1,
        "volatility_60": close.pct_change(fill_method=None).rolling(60).std(),
        "dist_high_252": close / close.rolling(252).max() - 1,
        "dist_sma200": close / close.rolling(200).mean() - 1,
        "rsi14": close.apply(lambda s: pd.Series(rsi(s, 14), index=s.index)),
        "volume_trend": np.log(vol.rolling(20).mean() / vol.rolling(120).mean()),
    }


def make_features(bars: dict, dates: pd.DatetimeIndex, sectors: pd.Series | None = None) -> pd.DataFrame:
    """(date, ticker) × feature。每個日期內轉做百分位排名（0–1），咁唔同股票先可以比較。

    有 sectors 就喺同一行業入面排名（sector-neutral）：揀嘅係「行業入面最強」，
    唔會因為成個行業熱就全部揸晒同一行業。
    """
    # 成交量係 0 嗰幾日 log 會出 -inf，下面轉做 NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = _raw_features(bars["Close"], bars["Volume"])
    panel = pd.DataFrame({name: f.loc[dates].stack() for name, f in raw.items()})
    panel.index.names = ["date", "ticker"]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    if sectors is None:
        return panel.groupby(level="date").rank(pct=True)
    sector = panel.index.get_level_values("ticker").map(sectors)
    return panel.groupby([panel.index.get_level_values("date"), sector]).rank(pct=True)


def forward_returns(bars: dict, dates: pd.DatetimeIndex) -> pd.Series:
    """由 d+1 開市持有到下一個 rebalance 日 +1 開市嘅回報。最後一期未知（NaN）。"""
    opens = bars["Open"]
    pos = opens.index.get_indexer(dates)
    out = {}
    for k, d in enumerate(dates):
        e = pos[k] + 1
        x = pos[k + 1] + 1 if k + 1 < len(dates) else len(opens)
        if x < len(opens):
            out[d] = opens.iloc[x] / opens.iloc[e] - 1
    fwd = pd.DataFrame(out).T.reindex(dates).stack()
    fwd.index.names = ["date", "ticker"]
    return fwd


def walk_forward_scores(
    X: pd.DataFrame, fwd: pd.Series, model_name: str,
    min_train_months: int = 36, retrain_every: int = 3,
) -> pd.Series:
    """每個 rebalance 日嘅預測分數（越高越好），全部 out-of-sample。"""
    target = fwd.groupby(level="date").rank(pct=True)  # 預測「排第幾」，唔係預測幾多 %
    X = X.dropna()
    dates = X.index.get_level_values("date").unique().sort_values()
    scores, model = [], None

    for i in range(min_train_months, len(dates)):
        if model is None or (i - min_train_months) % retrain_every == 0:
            # 第 j 期嘅 label 要到第 j+1 期 +1 日開市先知，所以只可以用 j <= i-2
            train_dates = dates[: i - 1]
            X_tr = X[X.index.get_level_values("date").isin(train_dates)]
            y_tr = target.reindex(X_tr.index)
            ok = y_tr.notna()
            model = MODELS[model_name]()
            model.fit(X_tr[ok], y_tr[ok])
        X_te = X.xs(dates[i], level="date", drop_level=False)
        scores.append(pd.Series(model.predict(X_te), index=X_te.index))

    return pd.concat(scores).rename("score")


def predict_latest(X: pd.DataFrame, fwd: pd.Series, model_name: str) -> pd.Series:
    """只預測最新一期（落單用）。Train 法同 walk_forward_scores 一樣，只係唔使由頭行一次。"""
    target = fwd.groupby(level="date").rank(pct=True)
    X = X.dropna()
    dates = X.index.get_level_values("date").unique().sort_values()
    X_tr = X[X.index.get_level_values("date").isin(dates[: len(dates) - 2])]
    y_tr = target.reindex(X_tr.index)
    ok = y_tr.notna()
    model = MODELS[model_name]()
    model.fit(X_tr[ok], y_tr[ok])
    X_te = X.xs(dates[-1], level="date", drop_level=False)
    return pd.Series(model.predict(X_te), index=X_te.index, name="score")


def information_coefficient(scores: pd.Series, fwd: pd.Series) -> pd.Series:
    """每期分數同實際回報嘅排名相關度。業界 IC 平均有 0.03–0.05 已經算有用。"""
    df = pd.concat([scores.rename("s"), fwd.rename("r")], axis=1).dropna()
    ranks = df.groupby(level="date").rank()
    return ranks.groupby(level="date").apply(lambda g: g["s"].corr(g["r"])).dropna()


def quintile_returns(scores: pd.Series, fwd: pd.Series) -> pd.Series:
    """按分數分 5 組，每組平均每期回報 %。由 Q1（最差）到 Q5（最好）應該遞增。"""
    df = pd.concat([scores.rename("s"), fwd.rename("r")], axis=1).dropna()
    df["q"] = df.groupby(level="date")["s"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1
    )
    per_period = df.groupby([df.index.get_level_values("date"), "q"])["r"].mean().unstack()
    return per_period.mean() * 100


def top_k(scores: pd.Series, k: int) -> dict[pd.Timestamp, list[str]]:
    return {
        d: g.droplevel("date").nlargest(k).index.tolist()
        for d, g in scores.groupby(level="date")
    }


def pick_buffered(ranked: list[str], held, k: int, keep_within: int) -> list[str]:
    """ranked 係分數由高到低嘅 ticker。已經揸住嘅，只要仲排喺頭 keep_within 名就留低（最多 k 隻），空位先補排名最高嘅。"""
    keep = [t for t in ranked[:keep_within] if t in held][:k]
    return keep + [t for t in ranked if t not in keep][: k - len(keep)]


def top_k_buffered(scores: pd.Series, k: int, keep_within: int) -> dict[pd.Timestamp, list[str]]:
    """減少換手：已經揸住嘅股，只要仲排喺頭 keep_within 名就唔賣，空位先補排名最高嘅。"""
    out, held = {}, []
    for d, g in scores.groupby(level="date"):
        ranked = g.droplevel("date").sort_values(ascending=False).index.tolist()
        held = out[d] = pick_buffered(ranked, held, k, keep_within)
    return out


def blend(*scores: pd.Series) -> pd.Series:
    """幾個分數各自轉做排名再平均，例如 AI + 動量。"""
    ranks = [s.groupby(level="date").rank(pct=True) for s in scores]
    return pd.concat(ranks, axis=1).dropna().mean(axis=1)


def apply_regime(holdings: dict, spy_close: pd.Series, sma: int = 200) -> dict:
    """大市（SPY）跌穿 200 日線嗰個月就持現金（空 list）。"""
    above = spy_close > spy_close.rolling(sma).mean()
    return {d: (names if above.asof(d) else []) for d, names in holdings.items()}


def simulate(bars: dict, dates: pd.DatetimeIndex, holdings: dict,
             invested: float = 1.0, include_open: bool = False) -> tuple[pd.Series, float]:
    """等權揸 holdings，返回每日資金曲線同平均每月換手率。

    holdings[d] 係 [] ＝ 嗰個月持現金（冇利息）。冇 key ＝ 未開始。
    換股成本：每邊 config.COMMISSION × 交易咗嘅比例。
    invested：每期投資嘅比例（模擬盤留 2% 現金 → 0.98），其餘係冇利息現金。
    include_open：最後一期未有下一個換股日，都計到最新收市價（唔賣出）。
    """
    opens, close = bars["Open"], bars["Close"]
    pos = opens.index.get_indexer(dates)
    value, prev_w = float(config.DEFAULT_CASH), pd.Series(dtype=float)
    parts, turnovers = [], []

    for k in range(len(dates)):
        names = holdings.get(dates[k])
        e = pos[k] + 1
        x = pos[k + 1] + 1 if k + 1 < len(dates) else len(opens)
        if names is None or e >= len(opens):
            continue
        if x >= len(opens):
            if not include_open or not names:
                continue
            entry = opens.iloc[e][names]
            w = pd.Series(1 / len(names), index=names)
            value *= 1 - config.COMMISSION * w.sub(prev_w, fill_value=0).abs().sum()
            parts.append(((close.iloc[e:][names] / entry).mean(axis=1) * invested + 1 - invested) * value)
            break
        w = pd.Series(1 / len(names), index=names) if names else pd.Series(dtype=float)
        turnover = w.sub(prev_w, fill_value=0).abs().sum()
        value *= 1 - config.COMMISSION * turnover
        turnovers.append(turnover)

        if not names:
            parts.append(pd.Series(value, index=close.index[e:x]))
            prev_w = w
            continue

        entry = opens.iloc[e][names]
        parts.append(((close.iloc[e:x][names] / entry).mean(axis=1) * invested + 1 - invested) * value)
        growth = opens.iloc[x][names] / entry
        value *= growth.mean() * invested + 1 - invested
        prev_w = w * growth / (w * growth).sum()

    return pd.concat(parts), float(np.mean(turnovers)) if turnovers else 0.0


def excess_vs(curve: pd.Series, base: pd.Series) -> dict:
    """每月超額回報（策略 − 基準）：年化幾多、t 值。t > 2 先算唔似運氣。"""
    monthly = pd.concat([curve, base], axis=1).dropna().resample("ME").last().pct_change().dropna()
    diff = monthly.iloc[:, 0] - monthly.iloc[:, 1]
    return {
        "超額年化 %": diff.mean() * 12 * 100,
        "超額 t 值": diff.mean() / diff.std() * len(diff) ** 0.5 if diff.std() > 0 else 0.0,
    }
