"""組合回測：換股成本 = config.COMMISSION × 換手。"""
import numpy as np
import pandas as pd
import pytest

import config
from ml.cross_section import rebalance_dates, simulate


def flat_market(n_days=90, tickers="ABCD"):
    """所有股價企死 100：資金只會因為成本而變。"""
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    px = pd.DataFrame(100.0, index=idx, columns=list(tickers))
    return {"Open": px, "Close": px.copy()}, rebalance_dates(idx)


def test_commission_charged_on_turnover():
    bars, dates = flat_market()
    # 第 1 期買 A、B（換手 1）；第 2 期唔變（0）；第 3 期全換 C、D（賣 1 + 買 1 = 2）
    holdings = {dates[0]: ["A", "B"], dates[1]: ["A", "B"], dates[2]: ["C", "D"]}
    curve, avg_turnover = simulate(bars, dates, holdings)
    c = config.COMMISSION
    assert avg_turnover == pytest.approx((1 + 0 + 2) / 3)
    assert curve.iloc[-1] == pytest.approx(config.DEFAULT_CASH * (1 - c * 1) * (1 - c * 2))


def test_going_to_cash_is_charged():
    bars, dates = flat_market()
    curve, avg_turnover = simulate(bars, dates, {dates[0]: ["A", "B"], dates[1]: [], dates[2]: []})
    c = config.COMMISSION
    assert avg_turnover == pytest.approx((1 + 1 + 0) / 3)
    assert curve.iloc[-1] == pytest.approx(config.DEFAULT_CASH * (1 - c) ** 2)


@pytest.mark.parametrize("commission", [0.0, 0.001, 0.01])
def test_cost_reads_config(monkeypatch, commission):
    monkeypatch.setattr(config, "COMMISSION", commission)
    bars, dates = flat_market()
    curve, _ = simulate(bars, dates, {dates[0]: ["A"], dates[1]: ["B"]})
    assert curve.iloc[-1] == pytest.approx(config.DEFAULT_CASH * (1 - commission) * (1 - 2 * commission))


def test_drift_counts_as_turnover():
    """揸住唔郁都會因為升跌令比重偏離等權，再平衡嗰部分都要計成本。"""
    bars, dates = flat_market()
    e1 = bars["Open"].index.get_loc(dates[1]) + 1
    bars["Open"].iloc[e1:, 0] = 300.0             # 第 1 期 A 由 100 升到 300，B 唔郁
    bars["Close"].iloc[e1:, 0] = 300.0
    curve, avg_turnover = simulate(bars, dates, {dates[0]: ["A", "B"], dates[1]: ["A", "B"]})
    # 期尾比重 A 0.75、B 0.25 → 調返 0.5 / 0.5，換手 0.25 + 0.25 = 0.5
    assert avg_turnover == pytest.approx((1 + 0.5) / 2)
    value_after_p1 = config.DEFAULT_CASH * (1 - config.COMMISSION) * 2.0
    e2 = bars["Open"].index.get_loc(dates[2]) + 1
    assert curve.loc[bars["Open"].index[e2 - 1]] == pytest.approx(value_after_p1 * (1 - 0.5 * config.COMMISSION))


def test_buys_at_next_open_not_signal_close():
    bars, dates = flat_market()
    d0 = bars["Close"].index.get_loc(dates[0])
    bars["Close"].iloc[d0, 0] = 1.0               # 排名嗰日收市價亂咗都唔影響入市價
    curve, _ = simulate(bars, dates, {dates[0]: ["A"]})
    assert curve.index[0] == bars["Open"].index[d0 + 1]
    assert np.isclose(curve.iloc[0], config.DEFAULT_CASH * (1 - config.COMMISSION))
