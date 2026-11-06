"""每月換股時間表：頭 REBALANCE_WINDOW_DAYS 個交易日先落單；過咗窗口都未換到就報 missed。"""
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import config
from trader import rebalance as rb
from trader import run
from trader.broker import AccountInfo
from trader.targets import Targets

# 2026 年 6 月：1 號係星期一 → 第 1–5 個交易日係 1–5 號，第 6 個係 8 號，第 7 個係 9 號
JUNE = pd.bdate_range("2026-06-01", "2026-06-30")


class FakeBroker:
    def __init__(self, today, orders=0, is_open=True):
        self.today, self.orders, self.is_open = today, orders, is_open
        self.orders_since_args = []

    def info(self):
        return AccountInfo(equity=100_000.0, last_equity=100_000.0, cash=100_000.0, blocked=False)

    def trading_days(self, start, end):
        return [d.date() for d in JUNE if start <= d.date() <= end]

    def market_open(self):
        return self.is_open

    def orders_since(self, start):
        self.orders_since_args.append(start)
        return self.orders

    def positions(self):
        return {}

    def tradable(self, symbol):
        return True

    def wait(self, ids):
        return {}

    def submit(self, *args, **kwargs):
        raise AssertionError("測試唔准落單")


@pytest.fixture
def setup(monkeypatch):
    """假 broker、假 journal、假目標；返回一個 function 行一次 rebalance。"""
    planned = []
    monkeypatch.setattr(rb, "journal", SimpleNamespace(
        start_run=lambda *a: 1, finish_run=lambda *a, **kw: None, log_targets=lambda *a: None,
        log_order=lambda *a, **kw: None, update_order=lambda *a: None,
        executed_this_month=lambda *a: False,
    ))
    monkeypatch.setattr(rb, "compute_targets", lambda *a, **kw: Targets(
        signal_date=pd.Timestamp("2026-05-29"), weights=pd.Series(dtype=float),
        scores=pd.Series(dtype=float), prices=pd.Series(dtype=float)))

    def fake_plan(*args):
        planned.append(args)
        return []
    monkeypatch.setattr(rb, "plan_orders", fake_plan)

    def go(day: int, orders=0, is_open=True, force=False):
        today = JUNE[day - 1]
        broker = FakeBroker(today, orders, is_open)
        monkeypatch.setattr(rb, "new_york_today", lambda: today)
        monkeypatch.setattr(rb, "AlpacaBroker", lambda account: broker)
        report = rb.rebalance("ALPACA", execute=True, force=force)
        return report, bool(planned), broker
    return go


def test_window_is_five_days():
    assert config.REBALANCE_WINDOW_DAYS == 5


@pytest.mark.parametrize("day", [1, 5])
def test_inside_window_reaches_planning(setup, day):
    report, planned, _ = setup(day)
    assert planned and report.status == "ok"


def test_inside_window_with_orders_skips(setup):
    report, planned, _ = setup(5, orders=3)
    assert report.status == "skipped" and not planned


@pytest.mark.parametrize("is_open", [True, False])
def test_day_after_window_without_orders_is_missed(setup, is_open):
    """未開市都要報 missed：呢個檢查喺開市檢查之前。"""
    report, planned, broker = setup(6, orders=0, is_open=is_open)
    assert report.status == "missed" and not planned
    assert broker.orders_since_args == [pd.Timestamp("2026-06-01", tz="America/New_York")]


def test_day_after_window_with_orders_skips(setup):
    report, planned, _ = setup(6, orders=1)
    assert report.status == "skipped" and not planned


def test_later_days_skip(setup):
    report, planned, _ = setup(7, orders=0)
    assert report.status == "skipped" and not planned


def test_inside_window_market_closed_skips(setup):
    report, planned, _ = setup(3, is_open=False)
    assert report.status == "skipped" and not planned


def test_force_ignores_window(setup):
    report, planned, _ = setup(9, force=True)
    assert planned and report.status == "ok"


def test_trading_day_of_month():
    b = FakeBroker(None)
    assert rb.trading_day_of_month(b, pd.Timestamp("2026-06-01")) == 1
    assert rb.trading_day_of_month(b, pd.Timestamp("2026-06-08")) == 6
    assert rb.trading_day_of_month(b, pd.Timestamp("2026-06-06")) == 0   # 星期六唔開市


# ---------- trader/run.py：有 missed 就 exit 1，GitHub Actions 會變紅 ----------

@pytest.mark.parametrize("statuses, code", [(["ok", "skipped"], None), (["skipped", "missed"], 1)])
def test_cli_exit_code(monkeypatch, statuses, code):
    reports = iter(rb.RunReport(account=f"A{i}", strategy="s", mode="execute", status=s)
                   for i, s in enumerate(statuses))
    monkeypatch.setattr(run, "accounts", lambda selected: [f"A{i}" for i in range(len(statuses))])
    monkeypatch.setattr(run, "rebalance", lambda *a, **kw: next(reports))
    monkeypatch.setattr(sys, "argv", ["trader.run", "rebalance", "--execute"])
    if code is None:
        run.main()
    else:
        with pytest.raises(SystemExit) as e:
            run.main()
        assert e.value.code == code
