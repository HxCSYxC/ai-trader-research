"""連接 Alpaca 模擬盤。ticker 喺項目入面用 yfinance 格式（BRK-B），落單先轉做 Alpaca 格式（BRK.B）。"""
import math
import os
import time
from dataclasses import dataclass

import pandas as pd

import config


@dataclass
class Holding:
    symbol: str
    qty: float
    market_value: float
    price: float


@dataclass
class AccountInfo:
    equity: float
    last_equity: float  # 上一個交易日收市嘅資產
    cash: float
    blocked: bool


def to_alpaca(t: str) -> str:
    return t.replace("-", ".")


def from_alpaca(s: str) -> str:
    return s.replace(".", "-")


def has_keys(account: str) -> bool:
    return bool(os.getenv(f"{account}_API_KEY")) and bool(os.getenv(f"{account}_SECRET_KEY"))


class AlpacaBroker:
    """永遠係模擬盤：paper=True 寫死，冇任何參數可以改。"""

    def __init__(self, account: str = "ALPACA"):
        from alpaca.trading.client import TradingClient

        if not has_keys(account):
            raise RuntimeError(f"未設定 {account}_API_KEY / {account}_SECRET_KEY，請喺 .env 填入 Paper Trading 嘅 key")
        self.account = account
        self.client = TradingClient(
            os.environ[f"{account}_API_KEY"], os.environ[f"{account}_SECRET_KEY"], paper=True
        )
        self._asset_cache: dict[str, tuple[bool, bool]] = {}

    def info(self) -> AccountInfo:
        a = self.client.get_account()
        return AccountInfo(
            equity=float(a.equity), last_equity=float(a.last_equity), cash=float(a.cash),
            blocked=bool(a.trading_blocked or a.account_blocked),
        )

    def positions(self) -> dict[str, Holding]:
        return {
            from_alpaca(p.symbol): Holding(
                symbol=from_alpaca(p.symbol), qty=float(p.qty),
                market_value=float(p.market_value), price=float(p.current_price),
            )
            for p in self.client.get_all_positions()
        }

    def market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def trading_days(self, start, end) -> list:
        from alpaca.trading.requests import GetCalendarRequest

        return [c.date for c in self.client.get_calendar(GetCalendarRequest(start=start, end=end))]

    def _assets(self) -> dict[str, tuple[bool, bool]]:
        """一次過攞晒所有美股嘅 (tradable, fractionable)，避免逐隻問撞 API 限速。"""
        if not self._asset_cache:
            # 用原始 JSON：alpaca-py 0.44 未識新交易所（例如 TXSE），用 get_all_assets 會解析失敗
            assets = self.client.get("/assets", {"status": "active", "asset_class": "us_equity"})
            self._asset_cache = {
                from_alpaca(a["symbol"]): (bool(a.get("tradable")), bool(a.get("fractionable")))
                for a in assets
            }
        return self._asset_cache

    def tradable(self, symbol: str) -> bool:
        return self._assets().get(symbol, (False, False))[0]

    def fractionable(self, symbol: str) -> bool:
        return all(self._assets().get(symbol, (False, False)))

    def submit(self, symbol: str, side: str, dollars: float, price: float, sell_all: bool = False):
        """返回 (order_id, 描述)。可以碎股就用金額落單，唔得就用整數股。"""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        if sell_all:
            o = self.client.close_position(to_alpaca(symbol))
            return str(o.id), "全部賣出"
        if self.fractionable(symbol):
            req = MarketOrderRequest(symbol=to_alpaca(symbol), notional=round(dollars, 2),
                                     side=OrderSide(side), time_in_force=TimeInForce.DAY)
            desc = f"${dollars:,.2f}"
        else:
            qty = math.floor(dollars / price)
            if qty < 1:
                return None, "唔可以碎股，金額唔夠 1 股，跳過"
            req = MarketOrderRequest(symbol=to_alpaca(symbol), qty=qty,
                                     side=OrderSide(side), time_in_force=TimeInForce.DAY)
            desc = f"{qty} 股"
        o = self.client.submit_order(req)
        time.sleep(0.35)  # Alpaca 限每分鐘 200 個請求，平均買晒要落幾百張單
        return str(o.id), desc

    def wait(self, order_ids: list[str], timeout: float = 180) -> dict[str, dict]:
        """等到全部成交 / 取消 / 超時，返回每張單嘅狀態。每次一個請求攞晒所有單，唔會撞限速。"""
        done = {"filled", "canceled", "expired", "rejected", "done_for_day"}
        result, deadline = {}, time.time() + timeout
        pending = {i for i in order_ids if i}
        while pending and time.time() < deadline:
            # 用原始 JSON，同 _assets 一樣避開 alpaca-py 解析新欄位出錯
            for o in self.client.get("/orders", {"status": "all", "limit": 500, "direction": "desc"}):
                if o["id"] in pending:
                    result[o["id"]] = {
                        "status": o["status"],
                        "filled_qty": float(o.get("filled_qty") or 0),
                        "filled_avg_price": float(o.get("filled_avg_price") or 0),
                    }
            pending = {i for i in pending if result.get(i, {}).get("status") not in done}
            if pending:
                time.sleep(3)
        for i in pending:  # 超時仲未完成嘅單，標記出嚟
            if i in result:
                result[i]["status"] = f"timeout（{result[i]['status']}）"
            else:
                result[i] = {"status": "timeout", "filled_qty": 0.0, "filled_avg_price": 0.0}
        return result

    def orders_since(self, start) -> int:
        """start 之後落、已經有成交或者仲未完成嘅單數（取消 / 拒絕而冇成交嘅唔計）。
        由 Alpaca 自己嘅記錄計，唔靠本機 SQLite，所以 GitHub Actions 每次新機都知今個月換咗未。"""
        after = pd.Timestamp(start).tz_convert("UTC").isoformat()
        orders = self.client.get("/orders", {"status": "all", "after": after, "limit": 500, "direction": "desc"})
        dead = {"canceled", "expired", "rejected"}
        return sum(float(o.get("filled_qty") or 0) > 0 or o["status"] not in dead for o in orders)

    def portfolio_history(self, period: str = "1M"):
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        h = self.client.get_portfolio_history(GetPortfolioHistoryRequest(period=period, timeframe="1D"))
        return h.timestamp, h.equity


class OfflineBroker:
    """冇 key 都可以預覽：一個 $100,000、冇持倉、永遠開市嘅假帳戶。唔會落任何單。"""

    account = "OFFLINE"

    def info(self) -> AccountInfo:
        return AccountInfo(equity=config.DEFAULT_CASH, last_equity=config.DEFAULT_CASH,
                           cash=config.DEFAULT_CASH, blocked=False)

    def positions(self) -> dict[str, Holding]:
        return {}

    def market_open(self) -> bool:
        return True

    def tradable(self, symbol: str) -> bool:
        return True

    def submit(self, *args, **kwargs):
        raise RuntimeError("離線模式唔可以落單")
