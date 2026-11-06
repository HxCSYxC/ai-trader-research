"""Walk-forward purge：train 嘅 label 喺預測嗰陣一定已經知道。"""
import numpy as np
import pandas as pd
import pytest

from ml import cross_section, walkforward


class SpyModel:
    """記低每次 fit 用咗邊啲行、predict 預測邊啲行。"""

    def __init__(self, log):
        self.log = log

    def fit(self, X, y):
        self.log.append(("fit", X.index))
        return self

    def predict(self, X):
        self.log.append(("predict", X.index))
        return X.iloc[:, 0].to_numpy()

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.predict(X)))
        return np.column_stack([1 - p, p])


def spy_models(monkeypatch, module):
    log = []
    monkeypatch.setitem(module.MODELS, "spy", lambda: SpyModel(log))
    return log


def pairs(log):
    """每次 predict 配返對上一次 fit。"""
    out, fitted = [], None
    for kind, idx in log:
        if kind == "fit":
            fitted = idx
        else:
            out.append((fitted, idx))
    return out


# ---------- ml/walkforward.py（單股、每日）----------

def daily_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    X = pd.DataFrame(rng.normal(size=(n, 3)), index=idx, columns=list("abc"))
    X.iloc[:5] = np.nan                       # 頭幾日 features 未計到
    y = pd.Series((rng.random(n) > 0.5).astype(float), index=idx)
    return X, y


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_daily_purge_gap(monkeypatch, horizon):
    X, y = daily_data()
    log = spy_models(monkeypatch, walkforward)
    walkforward.walk_forward_proba(X, y, "spy", horizon, min_train=100, step=30)
    pos = X.index.get_indexer
    blocks = pairs(log)
    assert len(blocks) > 5
    for train, test in blocks:
        last_train, first_test = pos(train).max(), pos(test).min()
        # 第 i 行 label 要第 i+horizon+1 日先知，一定要早過預測嗰日
        assert last_train + horizon + 1 < first_test
        assert last_train + horizon + 2 == first_test   # 而且冇浪費：用晒可以用嘅數據


def test_daily_future_labels_cannot_change_predictions():
    """將預測嗰陣未知嘅 label 亂改，個 block 嘅預測完全唔變。"""
    X, y = daily_data()
    horizon, min_train, step = 5, 100, 30
    base = walkforward.walk_forward_proba(X, y, "Logistic Regression", horizon, min_train, step)
    start = 5 + min_train + 2 * step              # 第 3 個 block
    y2 = y.copy()
    y2.iloc[start - horizon - 1:] = 1 - y2.iloc[start - horizon - 1:]
    probs = walkforward.walk_forward_proba(X, y2, "Logistic Regression", horizon, min_train, step)
    assert base.notna().iloc[:start + step].sum() > 0
    pd.testing.assert_series_equal(base.iloc[:start + step], probs.iloc[:start + step])
    assert not base.iloc[start + step:].equals(probs.iloc[start + step:])   # 之後先用到


# ---------- ml/cross_section.py（每月一籃股）----------

def monthly_panel(n_dates=40, n_tickers=12, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-31", periods=n_dates, freq="ME")
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(n_tickers)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.random((len(idx), 3)), index=idx, columns=list("abc"))
    fwd = pd.Series(rng.normal(0, 0.05, len(idx)), index=idx)
    fwd.loc[dates[-1]] = np.nan                   # 最後一期未知
    return dates, X, fwd


@pytest.mark.parametrize("retrain_every", [1, 3])
def test_monthly_purge_gap(monkeypatch, retrain_every):
    dates, X, fwd = monthly_panel()
    log = spy_models(monkeypatch, cross_section)
    cross_section.walk_forward_scores(X, fwd, "spy", min_train_months=12, retrain_every=retrain_every)
    blocks = pairs(log)
    assert len(blocks) == len(dates) - 12
    for train, test in blocks:
        i = dates.get_loc(test.get_level_values("date").unique().item())
        j = dates.get_indexer(train.get_level_values("date").unique()).max()
        # 第 j 期 label 要到第 j+1 期之後一日開市先知 → 第 i 期只可以用 j <= i-2
        assert j <= i - 2


def test_monthly_future_labels_cannot_change_scores():
    dates, X, fwd = monthly_panel()
    base = cross_section.walk_forward_scores(X, fwd, "Ridge（線性）", min_train_months=12, retrain_every=1)
    i = 20
    fwd2 = fwd.copy()
    fwd2[fwd2.index.get_level_values("date") >= dates[i - 1]] = np.random.default_rng(9).normal(
        0, 0.05, int((fwd2.index.get_level_values("date") >= dates[i - 1]).sum()))
    scores = cross_section.walk_forward_scores(X, fwd2, "Ridge（線性）", min_train_months=12, retrain_every=1)
    upto = base.index.get_level_values("date") <= dates[i]
    pd.testing.assert_series_equal(base[upto], scores[upto])
    assert not base[~upto].equals(scores[~upto])


def test_predict_latest_purge(monkeypatch):
    dates, X, fwd = monthly_panel()
    log = spy_models(monkeypatch, cross_section)
    cross_section.predict_latest(X, fwd, "spy")
    [(train, test)] = pairs(log)
    assert set(test.get_level_values("date")) == {dates[-1]}
    assert train.get_level_values("date").max() <= dates[-3]


def test_forward_returns_start_next_open():
    """d 日收市先排名，d+1 開市買，下一個換股日 +1 開市賣。"""
    idx = pd.bdate_range("2020-01-01", periods=60)
    opens = pd.DataFrame({"A": np.arange(60, dtype=float) + 100}, index=idx)
    bars = {"Open": opens, "Close": opens + 0.5}
    dates = cross_section.rebalance_dates(idx)
    fwd = cross_section.forward_returns(bars, dates)
    d0, d1 = dates[0], dates[1]
    e, x = idx.get_loc(d0) + 1, idx.get_loc(d1) + 1
    assert fwd[(d0, "A")] == pytest.approx(opens.A.iloc[x] / opens.A.iloc[e] - 1)
    assert pd.isna(fwd.get((dates[-1], "A"), np.nan))   # 最後一期未知
