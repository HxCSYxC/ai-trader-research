"""Baseline 策略：之後 AI 策略要贏到呢啲先有意義。全部只做 long。"""
from backtesting.lib import crossover

from strategies.base import BaseStrategy
from strategies.indicators import rsi, sma


class SmaCross(BaseStrategy):
    """短期 MA 上穿長期 MA 買入，下穿就賣。"""

    fast = 20
    slow = 50

    @classmethod
    def valid_params(cls, params: dict) -> bool:
        return params.get("fast", cls.fast) < params.get("slow", cls.slow)

    def init(self):
        self.ma_fast = self.I(sma, self.data.Close, self.fast, name=f"SMA{self.fast}")
        self.ma_slow = self.I(sma, self.data.Close, self.slow, name=f"SMA{self.slow}")

    def next(self):
        if not self.can_trade():
            return
        if crossover(self.ma_fast, self.ma_slow) and not self.position:
            self.buy()
        elif crossover(self.ma_slow, self.ma_fast) and self.position:
            self.position.close()


class RsiReversion(BaseStrategy):
    """升市（價喺 200 日線之上）入面，RSI 超賣就買，超買就賣。"""

    period = 14
    lower = 30
    upper = 70
    trend = 200

    @classmethod
    def valid_params(cls, params: dict) -> bool:
        return params.get("lower", cls.lower) < params.get("upper", cls.upper)

    def init(self):
        self.rsi = self.I(rsi, self.data.Close, self.period, name="RSI")
        self.ma_trend = self.I(sma, self.data.Close, self.trend, name=f"SMA{self.trend}")

    def next(self):
        if not self.can_trade():
            return
        price = self.data.Close[-1]
        if not self.position:
            if self.rsi[-1] < self.lower and price > self.ma_trend[-1]:
                self.buy()
        elif self.rsi[-1] > self.upper:
            self.position.close()
