"""每月換股流程：檢查 → 計目標 → 計單 → （execute 先）落單 → 記錄。"""
from dataclasses import dataclass, field

import pandas as pd

import config
from trader import journal
from trader.broker import AlpacaBroker, OfflineBroker
from trader.risk import PlannedOrder, RiskError, check_account, plan_orders
from trader.targets import Targets, compute_targets, new_york_today


@dataclass
class RunReport:
    account: str
    strategy: str
    mode: str
    status: str = "ok"
    message: str = ""
    equity: float = 0.0
    targets: Targets | None = None
    orders: list[PlannedOrder] = field(default_factory=list)
    fills: dict = field(default_factory=dict)


def signal_cutoff() -> pd.Timestamp:
    """用上個月最後一個交易日收市嘅數據，同回測一致（月尾出訊號，下個月頭落單）。"""
    return new_york_today().replace(day=1)


def month_start_new_york(today: pd.Timestamp) -> pd.Timestamp:
    return today.replace(day=1).tz_localize("America/New_York")


def is_rebalance_window(broker: AlpacaBroker, today: pd.Timestamp) -> bool:
    days = broker.trading_days(today.replace(day=1).date(), today.date())
    return 0 < len(days) <= config.REBALANCE_WINDOW_DAYS and days[-1] == today.date()


def rebalance(account: str, execute: bool = False, offline: bool = False, force: bool = False) -> RunReport:
    settings = config.ACCOUNTS.get(account, config.ACCOUNTS["ALPACA"])
    strategy = settings["strategy"]
    mode = "execute" if execute else "preview"
    report = RunReport(account=account if not offline else "OFFLINE", strategy=strategy, mode=mode)
    run_id = journal.start_run(report.account, strategy, mode)

    try:
        if execute and config.DASHBOARD_READONLY:
            raise RiskError("唯讀模式唔可以落單")
        broker = OfflineBroker() if offline else AlpacaBroker(account)
        info = broker.info()
        report.equity = info.equity
        check_account(info)

        today = new_york_today()
        if execute and not force:
            if not broker.market_open():
                return _finish(run_id, report, "skipped", "未開市")
            if not is_rebalance_window(broker, today):
                return _finish(run_id, report, "skipped", f"唔係每月頭 {config.REBALANCE_WINDOW_DAYS} 個交易日")
            if journal.executed_this_month(account, today):
                return _finish(run_id, report, "skipped", "今個月已經換過股")
            # GitHub Actions 每次都係新機，SQLite 冇記錄；以 Alpaca 嘅成交記錄為準，避免同一個月換兩次
            if n := broker.orders_since(month_start_new_york(today)):
                return _finish(run_id, report, "skipped", f"今個月已經換過股（Alpaca 有 {n} 張單）")
        if execute and not broker.market_open():
            return _finish(run_id, report, "skipped", "未開市，唔可以落單")

        positions = broker.positions()
        targets = compute_targets(strategy, settings["model"], settings["k"], until=signal_cutoff(),
                                  held=positions.keys())
        if not offline:
            ok = [s for s in targets.weights.index if broker.tradable(s)]
            targets.weights = targets.weights.reindex(ok)
        report.targets = targets
        journal.log_targets(run_id, targets.weights, targets.scores)

        report.orders = plan_orders(targets.weights, targets.prices, positions, info.equity)
        if not execute:
            for o in report.orders:
                journal.log_order(run_id, o)
            return _finish(run_id, report, "ok", "預覽，冇落單", signal_date=str(targets.signal_date.date()))

        _execute(broker, run_id, report)
        unfilled = [oid for oid, f in report.fills.items() if f["status"] != "filled"]
        status, msg = ("ok", "全部成交") if not unfilled else ("ok", f"{len(unfilled)} 張單未完全成交")
        return _finish(run_id, report, status, msg, signal_date=str(targets.signal_date.date()))

    except RiskError as e:
        return _finish(run_id, report, "blocked", str(e))
    except Exception as e:  # 記低再拋出，等 log 見到成個 traceback
        _finish(run_id, report, "error", f"{type(e).__name__}: {e}")
        raise


def _execute(broker: AlpacaBroker, run_id: int, report: RunReport) -> None:
    sells = [o for o in report.orders if o.side == "sell"]
    buys = [o for o in report.orders if o.side == "buy"]

    ids = [_submit(broker, run_id, o) for o in sells]
    report.fills.update(_wait(broker, ids))

    # 賣完先買，買單總額唔可以超過手頭現金
    cash = broker.info().cash
    total = sum(o.dollars for o in buys)
    scale = min(1.0, cash * 0.995 / total) if total > 0 else 1.0
    for o in buys:
        o.dollars *= scale
    ids = [_submit(broker, run_id, o) for o in buys]
    report.fills.update(_wait(broker, ids))


def _submit(broker, run_id: int, o: PlannedOrder) -> str | None:
    try:
        oid, detail = broker.submit(o.symbol, o.side, o.dollars, o.price, o.sell_all)
        journal.log_order(run_id, o, oid, detail, status="submitted" if oid else "skipped")
        return oid
    except Exception as e:  # 一張單失敗唔好拖累其他單
        journal.log_order(run_id, o, None, f"{type(e).__name__}: {e}", status="failed")
        return None


def _wait(broker, ids: list) -> dict:
    fills = broker.wait([i for i in ids if i])
    for oid, f in fills.items():
        journal.update_order(oid, f["status"], f["filled_qty"], f["filled_avg_price"])
    return fills


def _finish(run_id: int, report: RunReport, status: str, message: str, **fields) -> RunReport:
    report.status, report.message = status, message
    journal.finish_run(run_id, status, message, equity=report.equity, n_orders=len(report.orders), **fields)
    return report
