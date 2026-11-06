"""intraday.scan.evaluate 嘅 Benjamini–Hochberg，同手計結果對數（要 TA-Lib）。"""
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib")
from intraday import scan  # noqa: E402

N = 400  # > MIN_TRADES


def trades_with_p(p_values: dict, base: dict | None = None) -> pd.DataFrame:
    """每個形態整 N 筆開發期交易，令「跑贏隨機」嘅單邊 p 值啱啱等於指定數值。

    diff = μ + z，z 係 ±c 交替（樣本標準差啱啱 1），所以 t = μ√N → μ = t / √N。
    """
    base = base or {}
    c = np.sqrt((N - 1) / N)
    z = np.where(np.arange(N) % 2 == 0, c, -c)
    dev_days = pd.bdate_range("2021-01-04", periods=N)
    lock_days = pd.bdate_range("2025-01-02", periods=20)
    rows = []
    for name, p in p_values.items():
        t = NormalDist().inv_cdf(1 - p)
        rnd = base.get(name, 0.001)
        diff = 1e-3 * (t / np.sqrt(N) + z)         # 乘個比例做回報，t 值唔變
        for d, x in zip(dev_days, diff):
            rows.append(("SPY", d, name, 1, rnd + x, rnd))
        for i, d in enumerate(lock_days):          # 鎖櫃期：穩定賺錢
            rows.append(("SPY", d, name, 1, 0.01 + (0.001 if i % 2 else -0.001), 0.0))
    return pd.DataFrame(rows, columns=["symbol", "day", "pattern", "direction", "gross", "random_gross"])


# 手計（m = 5，FDR 5%）：
#   排名 k        1      2      3      4      5
#   p           0.004  0.025  0.028  0.060  0.500
#   k/m × 5%    0.010  0.020  0.030  0.040  0.050
#   p ≤ 門檻？    ✓      ✗      ✓      ✗      ✗
# 最大通過排名 = 3 → 頭 3 個都顯著（P2 自己唔過門檻，但 step-up 會一齊收）。
# Bonferroni（0.05 / 5 = 0.01）只會收 P1。
P = {"P1": 0.004, "P2": 0.025, "P3": 0.028, "P4": 0.060, "P5": 0.500}


def test_p_values_reproduced():
    res = scan.evaluate(trades_with_p(P)).droplevel("direction")
    for name, p in P.items():
        assert res.loc[name, "p"] == pytest.approx(p, rel=1e-6)
        assert res.loc[name, "n"] == N


def test_bh_hand_computed():
    res = scan.evaluate(trades_with_p(P)).droplevel("direction")
    assert set(res.index[res.bh_significant]) == {"P1", "P2", "P3"}


def test_more_hypotheses_raise_the_bar():
    """多 5 個冇料嘅形態（m = 10）：門檻變 0.005, 0.010, 0.015 …，淨係 P1 過到。"""
    nulls = {f"N{i}": 0.5 + 0.08 * i for i in range(5)}
    res = scan.evaluate(trades_with_p(P | nulls)).droplevel("direction")
    assert set(res.index[res.bh_significant]) == {"P1"}


def test_stage1_also_needs_positive_net():
    """P3 跑贏隨機，但隨機入市本身蝕好多 → 扣成本後仍然蝕錢，唔過第 1 關。"""
    res = scan.evaluate(trades_with_p(P, base={"P3": -0.01})).droplevel("direction")
    assert res.loc["P3", "bh_significant"] and res.loc["P3", "net_bps"] < 0
    assert set(res.index[res.stage1]) == {"P1", "P2"}
    assert set(res.index[res.stage2]) == {"P1", "P2"}


def test_one_sided_p():
    assert scan._p_one_sided(0.0) == pytest.approx(0.5)
    assert scan._p_one_sided(1.6448536269514722) == pytest.approx(0.05)
