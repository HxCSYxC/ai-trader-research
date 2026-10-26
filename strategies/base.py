from backtesting import Strategy


class BaseStrategy(Strategy):
    # 呢個日期之前只計指標、唔落單（用嚟做 out-of-sample 測試，指標有足夠 warmup）
    trade_from = None

    def can_trade(self) -> bool:
        return self.trade_from is None or self.data.index[-1] >= self.trade_from

    @classmethod
    def valid_params(cls, params: dict) -> bool:
        return True
