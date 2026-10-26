import numpy as np

from strategies.base import BaseStrategy


class MLSignal(BaseStrategy):
    """data 要有 Prob 欄（walk-forward 預測嘅 P(升)）。高過 entry 買，低過 exit 賣。"""

    entry = 0.5
    exit = 0.5

    @classmethod
    def valid_params(cls, params: dict) -> bool:
        return params.get("exit", cls.exit) <= params.get("entry", cls.entry)

    def init(self):
        self.prob = self.I(lambda p: p, self.data.Prob, name="P(up)", overlay=False)

    def next(self):
        p = self.prob[-1]
        if not self.can_trade() or np.isnan(p):
            return
        if not self.position and p > self.entry:
            self.buy()
        elif self.position and p < self.exit:
            self.position.close()
