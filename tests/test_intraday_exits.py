"""即日出場規則：下一枝開市價入市、止蝕 / 止賺、跳空、同一枝兩樣都掂當止蝕。"""
import numpy as np
import pytest

from intraday.backtest import simulate

ATR = 1.0  # 入市價 100 → 好倉止蝕 99、止賺 102；淡倉止蝕 101、止賺 98


def flat_bars(n=10, px=100.0):
    """全日企喺 100 附近（±0.5），唔會掂到止蝕止賺。"""
    o = np.full(n, px)
    return o.copy(), o + 0.5, o - 0.5, o.copy()


def test_entry_is_next_bar_open():
    o, h, l, c = flat_bars()
    o[1] = 50.0                     # 訊號枝收市 100，下一枝開 50
    h[1], l[1], c[1] = 50.5, 49.5, 50.0
    o[2:], h[2:], l[2:], c[2:] = 50.0, 50.5, 49.5, 50.0
    ret, j, why = simulate(o, h, l, c, signal=0, atr=ATR)
    assert why == "收市平倉" and ret == pytest.approx(0.0)   # 以 50 計，唔係 100


def test_signal_bar_and_earlier_do_not_matter():
    """改訊號枝或之前嘅 K 線，結果唔變 → 冇用到入市前未知嘅價。"""
    o, h, l, c = flat_bars()
    h[5] = 103.0                   # 第 5 枝之後先掂止賺
    base = simulate(o, h, l, c, signal=2, atr=ATR)
    o2, h2, l2, c2 = (a.copy() for a in (o, h, l, c))
    h2[:3], l2[:3] = 500.0, 1.0    # 入市之前（0–2 枝）亂改
    assert simulate(o2, h2, l2, c2, signal=2, atr=ATR) == base


def test_bars_after_exit_do_not_matter():
    o, h, l, c = flat_bars()
    l[4] = 98.0
    base = simulate(o, h, l, c, signal=0, atr=ATR)
    h[5:], l[5:] = 500.0, 1.0
    assert simulate(o, h, l, c, signal=0, atr=ATR) == base


def test_long_stop():
    o, h, l, c = flat_bars()
    l[3] = 98.5
    ret, j, why = simulate(o, h, l, c, signal=0, atr=ATR)
    assert (why, j) == ("止蝕", 3) and ret == pytest.approx(-0.01)


def test_long_target():
    o, h, l, c = flat_bars()
    h[4] = 102.5
    ret, j, why = simulate(o, h, l, c, signal=0, atr=ATR)
    assert (why, j) == ("止賺", 4) and ret == pytest.approx(0.02)


def test_same_bar_hits_both_counts_as_stop():
    o, h, l, c = flat_bars()
    h[3], l[3] = 103.0, 98.0
    ret, _, why = simulate(o, h, l, c, signal=0, atr=ATR)
    assert why == "止蝕" and ret == pytest.approx(-0.01)


def test_gap_through_stop_fills_at_open():
    o, h, l, c = flat_bars()
    o[3], h[3], l[3], c[3] = 97.0, 97.5, 96.5, 97.0
    ret, _, why = simulate(o, h, l, c, signal=0, atr=ATR)
    assert why == "止蝕" and ret == pytest.approx(-0.03)   # 用開市價 97，唔係 99


def test_gap_through_target_fills_at_open():
    o, h, l, c = flat_bars()
    o[3], h[3], l[3], c[3] = 105.0, 105.5, 104.5, 105.0
    ret, _, why = simulate(o, h, l, c, signal=0, atr=ATR)
    assert why == "止賺" and ret == pytest.approx(0.05)


def test_entry_bar_cannot_gap():
    """入市嗰枝開市價就係入市價，掂到止蝕只可以係止蝕價。"""
    o, h, l, c = flat_bars()
    l[1] = 90.0
    ret, j, _ = simulate(o, h, l, c, signal=0, atr=ATR)
    assert j == 1 and ret == pytest.approx(-0.01)


def test_time_exit_uses_second_last_bar_close():
    o, h, l, c = flat_bars()
    c[-2], c[-1] = 101.0, 150.0
    l[-1] = 1.0                    # 最後一枝唔再睇
    ret, j, why = simulate(o, h, l, c, signal=0, atr=ATR)
    assert (why, j) == ("收市平倉", len(c) - 2) and ret == pytest.approx(0.01)


# ---------- scan.simulate_dir（要 TA-Lib）----------

@pytest.fixture
def simulate_dir():
    pytest.importorskip("talib")
    from intraday.scan import simulate_dir
    return simulate_dir


def test_long_matches_backtest(simulate_dir):
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 40)))
        o = np.r_[100, c[:-1]] * np.exp(rng.normal(0, 0.003, 40))
        h = np.maximum(o, c) * (1 + rng.uniform(0, 0.004, 40))
        l = np.minimum(o, c) * (1 - rng.uniform(0, 0.004, 40))
        s = int(rng.integers(0, 30))
        assert simulate_dir(o, h, l, c, s, 0.5, 1) == pytest.approx(simulate(o, h, l, c, s, 0.5)[0])


@pytest.mark.parametrize("lo, hi, o_bar, expected", [
    (99.5, 101.5, 100.0, -0.01),   # 升到 101 → 止蝕
    (97.5, 100.5, 100.0, 0.02),    # 跌到 98 → 止賺
    (97.0, 102.0, 100.0, -0.01),   # 同一枝兩樣都掂 → 止蝕
    (102.5, 103.5, 103.0, -0.03),  # 跳空高開穿止蝕 → 用開市價 103
    (95.5, 96.5, 96.0, 0.04),      # 跳空低開穿止賺 → 用開市價 96
])
def test_short_exits(simulate_dir, lo, hi, o_bar, expected):
    o, h, l, c = flat_bars()
    o[3], h[3], l[3], c[3] = o_bar, hi, lo, o_bar
    assert simulate_dir(o, h, l, c, 0, ATR, -1) == pytest.approx(expected)


def test_short_time_exit(simulate_dir):
    o, h, l, c = flat_bars()
    c[-2] = 99.0
    assert simulate_dir(o, h, l, c, 0, ATR, -1) == pytest.approx(0.01)
