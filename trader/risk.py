"""風控：落單之前逐項檢查，任何一項唔過就成個 run 停。"""
import os
from dataclasses import dataclass

import pandas as pd

import config
from trader.broker import AccountInfo, Holding


class RiskError(Exception):
    pass


@dataclass
class PlannedOrder:
    symbol: str
    side: str          # "buy" / "sell"
    dollars: float
    price: float
    sell_all: bool = False
    reason: str = ""


def kill_switch_reason() -> str | None:
    """本機用 KILL_SWITCH 檔；GitHub Actions 冇呢個檔，改用 repository variable TRADING_DISABLED=1。"""
    if os.path.exists(config.KILL_SWITCH_FILE):
        return f"KILL_SWITCH 檔（{config.KILL_SWITCH_FILE}）"
    if os.getenv("TRADING_DISABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        return "環境變數 TRADING_DISABLED"
    return None


def kill_switch_on() -> bool:
    return kill_switch_reason() is not None


def set_kill_switch(on: bool, note: str = "") -> None:
    if on:
        with open(config.KILL_SWITCH_FILE, "w") as f:
            f.write(f"{pd.Timestamp.now()} {note}\n")
    elif os.path.exists(config.KILL_SWITCH_FILE):
        os.remove(config.KILL_SWITCH_FILE)


def check_account(info: AccountInfo) -> None:
    if reason := kill_switch_reason():
        raise RiskError(f"Kill switch 開咗（{reason}），唔會落任何單")
    if info.blocked:
        raise RiskError("Alpaca 帳戶被鎖，唔可以交易")
    if info.equity <= 0:
        raise RiskError("帳戶資產係 0")
    if info.last_equity > 0:
        change = info.equity / info.last_equity - 1
        if change < -config.MAX_DAILY_LOSS:
            raise RiskError(f"帳戶今日跌咗 {change:.1%}，超過上限 {config.MAX_DAILY_LOSS:.0%}，今日唔落單")


def plan_orders(weights: pd.Series, prices: pd.Series, positions: dict[str, Holding],
                equity: float) -> list[PlannedOrder]:
    """由目前持倉去到目標比重要落嘅單。先賣後買。"""
    investable = equity * (1 - config.CASH_BUFFER)
    sells, buys = [], []

    for sym, h in positions.items():
        if sym not in weights.index:
            sells.append(PlannedOrder(sym, "sell", h.market_value, h.price, sell_all=True, reason="唔再喺目標"))

    for sym, w in weights.items():
        target = w * investable
        held = positions.get(sym)
        current = held.market_value if held else 0.0
        price = held.price if held else float(prices.get(sym, float("nan")))
        diff = target - current
        if abs(diff) < max(config.MIN_TRADE_DOLLARS, 0.01 * target):
            continue
        if pd.isna(price) or price <= 0:
            raise RiskError(f"{sym} 冇有效價格")
        side = "buy" if diff > 0 else "sell"
        order = PlannedOrder(sym, side, abs(diff), price, reason="新買入" if not held else "調整比重")
        (buys if side == "buy" else sells).append(order)

    orders = sells + buys
    if len(orders) > config.MAX_ORDERS_PER_RUN:
        raise RiskError(f"要落 {len(orders)} 張單，超過上限 {config.MAX_ORDERS_PER_RUN}")
    for o in orders:
        if not o.sell_all and o.dollars > config.MAX_ORDER_PCT * equity:
            raise RiskError(f"{o.symbol} 單張單 ${o.dollars:,.0f} 超過資產 {config.MAX_ORDER_PCT:.0%}")
    if sum(weights) > 1:
        raise RiskError("目標比重加埋超過 100%（唔用槓桿）")
    return orders
