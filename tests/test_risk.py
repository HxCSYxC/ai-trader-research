"""風控：單股上限 10%、單張單上限 15%、現金緩衝、最少交易金額、先賣後買、kill switch。"""
import pandas as pd
import pytest

import config
from trader import risk, targets
from trader.broker import AccountInfo, Holding
from trader.risk import RiskError, check_account, plan_orders

EQUITY = 100_000.0
INVESTABLE = EQUITY * (1 - config.CASH_BUFFER)


def account(equity=EQUITY, last=EQUITY, blocked=False):
    return AccountInfo(equity=equity, last_equity=last, cash=equity, blocked=blocked)


def held(sym, value, price=10.0):
    return Holding(sym, value / price, value, price)


# ---------- kill switch / 帳戶檢查 ----------

def test_healthy_account_passes():
    check_account(account())


def test_kill_switch_file_blocks(tmp_path):
    risk.set_kill_switch(True, "test")
    assert risk.kill_switch_on()
    with pytest.raises(RiskError, match="Kill switch"):
        check_account(account())
    risk.set_kill_switch(False)
    assert not risk.kill_switch_on()
    check_account(account())


@pytest.mark.parametrize("value, on", [("1", True), ("true", True), (" YES ", True), ("on", True),
                                       ("0", False), ("", False), ("no", False)])
def test_kill_switch_env(monkeypatch, value, on):
    monkeypatch.setenv("TRADING_DISABLED", value)
    assert risk.kill_switch_on() is on
    if on:
        with pytest.raises(RiskError, match="TRADING_DISABLED"):
            check_account(account())


@pytest.mark.parametrize("info, msg", [
    (account(blocked=True), "被鎖"),
    (account(equity=0), "資產係 0"),
    (account(equity=94_000, last=100_000), "今日跌咗"),
])
def test_bad_account_blocks(info, msg):
    with pytest.raises(RiskError, match=msg):
        check_account(info)


def test_daily_loss_just_under_limit_passes():
    check_account(account(equity=EQUITY * (1 - config.MAX_DAILY_LOSS) + 1, last=EQUITY))


# ---------- 落單計劃 ----------

def test_cash_buffer_kept():
    w = pd.Series(0.10, index=list("ABCDEFGHIJ"))
    orders = plan_orders(w, pd.Series(10.0, index=w.index), {}, EQUITY)
    assert sum(o.dollars for o in orders) == pytest.approx(INVESTABLE)
    assert sum(o.dollars for o in orders) <= EQUITY * (1 - config.CASH_BUFFER) + 1e-6
    assert all(o.dollars == pytest.approx(0.10 * INVESTABLE) for o in orders)


@pytest.mark.parametrize("weight, gap, trades", [
    (0.04, config.MIN_TRADE_DOLLARS - 5, False),    # 目標 $3,920：差少過 $50 → 唔做
    (0.04, config.MIN_TRADE_DOLLARS + 5, True),
    (0.10, 0.009 * 0.10 * INVESTABLE, False),       # 目標 $9,800：差少過目標 1%（$98）→ 唔做
    (0.10, 0.011 * 0.10 * INVESTABLE, True),
])
def test_min_trade_size(weight, gap, trades):
    current = weight * INVESTABLE - gap
    orders = plan_orders(pd.Series({"A": weight}), pd.Series({"A": 10.0}), {"A": held("A", current)}, EQUITY)
    assert bool(orders) is trades


def test_sells_before_buys():
    positions = {"OLD": held("OLD", 5_000), "A": held("A", 20_000), "B": held("B", 1_000)}
    w = pd.Series({"B": 0.10, "A": 0.10, "NEW": 0.10})
    orders = plan_orders(w, pd.Series(10.0, index=w.index), positions, EQUITY)
    sides = [o.side for o in orders]
    assert sides == sorted(sides, key=lambda s: s != "sell")   # 所有 sell 喺 buy 前面
    assert {(o.symbol, o.side) for o in orders} == {("OLD", "sell"), ("A", "sell"), ("B", "buy"), ("NEW", "buy")}
    assert next(o for o in orders if o.symbol == "OLD").sell_all


def test_max_order_size():
    w = pd.Series({"A": 0.20})                      # 0.2 × 98,000 = 19,600 > 15%
    with pytest.raises(RiskError, match="超過資產"):
        plan_orders(w, pd.Series({"A": 10.0}), {}, EQUITY)


def test_sell_all_exempt_from_max_order():
    orders = plan_orders(pd.Series(dtype=float), pd.Series(dtype=float), {"X": held("X", 60_000)}, EQUITY)
    assert [(o.symbol, o.sell_all) for o in orders] == [("X", True)]


def test_no_leverage():
    w = pd.Series(0.10, index=[f"S{i}" for i in range(11)])
    with pytest.raises(RiskError, match="100%"):
        plan_orders(w, pd.Series(10.0, index=w.index), {}, EQUITY)


def test_missing_price():
    with pytest.raises(RiskError, match="冇有效價格"):
        plan_orders(pd.Series({"A": 0.05}), pd.Series(dtype=float), {}, EQUITY)


def test_max_orders_per_run(monkeypatch):
    monkeypatch.setattr(config, "MAX_ORDERS_PER_RUN", 3)
    w = pd.Series(0.05, index=list("ABCD"))
    with pytest.raises(RiskError, match="上限"):
        plan_orders(w, pd.Series(10.0, index=w.index), {}, EQUITY)


# ---------- 單股上限 10%（trader/targets.py）----------

class FakeData:
    def __init__(self, n):
        names = [f"S{i}" for i in range(n)]
        last = pd.Timestamp("2024-05-31")
        idx = pd.MultiIndex.from_product([[last], names], names=["date", "ticker"])
        self.X = pd.DataFrame({"mom_12_1": range(n)}, index=idx, dtype=float)
        self.bars = {"Close": pd.DataFrame([[10.0] * n], index=[last], columns=names)}


@pytest.mark.parametrize("n, strategy, k, weight", [
    (5, "平均買晒", None, config.MAX_POSITION_WEIGHT),   # 1/5 = 20% → 封頂 10%
    (40, "平均買晒", None, 1 / 40),
    (30, "動量（冇 ML）", 20, 1 / 20),
    (30, "動量（冇 ML）", 4, config.MAX_POSITION_WEIGHT),
])
def test_position_cap(monkeypatch, n, strategy, k, weight):
    monkeypatch.setattr(targets, "load", lambda *a, **kw: FakeData(n))
    t = targets.compute_targets(strategy, None, k, until=pd.Timestamp("2024-06-03"))
    assert (t.weights <= config.MAX_POSITION_WEIGHT + 1e-12).all()
    assert t.weights.sum() <= 1 + 1e-12
    assert t.weights.iloc[0] == pytest.approx(weight)
