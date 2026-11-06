"""半年報告（trader/recent.py）同模擬盤（trader/targets.py）要用同一套揀股方法，包括換手緩衝。"""
from types import SimpleNamespace

import pandas as pd

import config
from trader import recent

TICKERS = [f"S{i:02d}" for i in range(50)]
DATES = pd.to_datetime(["2026-01-30", "2026-02-27"])


def _fake_data():
    idx = pd.MultiIndex.from_product([DATES, TICKERS], names=["date", "ticker"])
    X = pd.DataFrame({"mom_12_1": 0.5}, index=idx)
    return SimpleNamespace(X=X, fwd=pd.Series(dtype=float))


def test_ai_account_keeps_holdings_inside_buffer(monkeypatch):
    k = config.ACCOUNTS["ALPACA"]["k"]
    month1 = TICKERS                                  # S00 最高分
    month2 = TICKERS[k:] + TICKERS[:k]                # 上個月頭 k 名而家排第 31–50

    def fake_predict(X, fwd, model):
        d = X.index.get_level_values("date").max()
        order = month1 if d == DATES[0] else month2
        idx = pd.MultiIndex.from_product([[d], order], names=["date", "ticker"])
        return pd.Series(range(len(order), 0, -1), index=idx, dtype=float)

    monkeypatch.setattr(recent, "predict_latest", fake_predict)
    hold = recent.holdings_for(_fake_data(), list(DATES))["ALPACA"]

    assert hold[DATES[0]] == TICKERS[:k]
    # 動量全部一樣，所以混合排名跟 AI。上個月嘅 S00–S19 而家排第 31–50：
    # 排喺頭 AI_KEEP_MULT × k = 40 名入面嘅 S00–S09 留低，空位補排名最高嘅 S20–S29
    keep_within = config.AI_KEEP_MULT * k
    kept = [t for t in month2[:keep_within] if t in TICKERS[:k]]
    assert set(hold[DATES[1]]) == set(kept) | set(month2[: k - len(kept)])
    assert set(hold[DATES[1]]) != set(month2[:k])  # 冇緩衝就會全部換晒
