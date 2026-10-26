"""用模擬盤同一套策略，回測最近 N 個月（例如睇下如果早幾個月開始會點）。

    python -m trader.recent --months 6

同模擬盤一致：S&P 500（只揀已入指數）、月尾訊號、下一日開市換股、Top 20、留 2% 現金、每邊成本 0.05%。
AI 每個月只用嗰個月尾或之前嘅數據重新 train（同 trader/targets.py 一樣用 predict_latest）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from ml.cross_section import blend, predict_latest, simulate, top_k
from ml.pipeline import curve_stats
from ml.stock_picker import load

OUT = Path(config.PROJECT_DIR) / "data" / "cache" / "recent_report.json"
INVESTED = 1 - config.CASH_BUFFER


def holdings_for(data, signal_dates: list[pd.Timestamp]) -> dict[str, dict]:
    """每個策略喺每個訊號日要揸咩。"""
    out = {name: {} for name in config.ACCOUNTS}
    for d in signal_dates:
        X_d = data.X[data.X.index.get_level_values("date") <= d]
        mom = data.X["mom_12_1"].xs(d, level="date", drop_level=False).dropna()
        for account, s in config.ACCOUNTS.items():
            if s["strategy"] == "平均買晒":
                out[account][d] = data.X.xs(d, level="date").dropna().index.tolist()
            elif s["strategy"] == "動量（冇 ML）":
                out[account][d] = top_k(mom, s["k"])[d]
            elif s["strategy"] == "AI + 動量混合":
                ai = predict_latest(X_d, data.fwd, s["model"])
                out[account][d] = top_k(blend(ai, mom), s["k"])[d]
    return out


def period_returns(data, d: pd.Timestamp, next_d: pd.Timestamp | None, names: list[str]) -> pd.Series:
    """每隻股喺呢期嘅回報（開市買、下期開市賣；最後一期計到最新收市）。"""
    opens, close = data.bars["Open"], data.bars["Close"]
    e = opens.index.get_loc(d) + 1
    entry = opens.iloc[e][names]
    if next_d is not None and opens.index.get_loc(next_d) + 1 < len(opens):
        exit_ = opens.iloc[opens.index.get_loc(next_d) + 1][names]
    else:
        exit_ = close.iloc[-1][names]
    return (exit_ / entry - 1).sort_values(ascending=False)


def build(months: int) -> dict:
    data = load(config.TRADING_UNIVERSE, max_age_hours=1)
    last = data.bars["Close"].index[-1]
    start = last - pd.DateOffset(months=months)
    month_ends = [d for d in data.dates if d < last]
    # 由 start 嗰陣揸緊嘅組合開始（start 之前最近一個月尾訊號）
    first = max(d for d in month_ends if d < start)
    signal_dates = [d for d in month_ends if d >= first]

    hold = holdings_for(data, signal_dates)
    dates = pd.DatetimeIndex(signal_dates)
    curves, turnover = {}, {}
    for account, h in hold.items():
        curve, turnover[account] = simulate(data.bars, dates, h, invested=INVESTED, include_open=True)
        curve = curve[curve.index > start]
        curves[account] = curve / curve.iloc[0] * config.DEFAULT_CASH
    spy = data.spy_close[data.spy_close.index > start]
    curves["SPY"] = spy / spy.iloc[0] * config.DEFAULT_CASH
    common_end = min(c.index[-1] for c in curves.values())  # 所有曲線截到同一日
    curves = {a: c[c.index <= common_end] for a, c in curves.items()}

    names = {a: config.ACCOUNTS[a]["strategy"] for a in config.ACCOUNTS} | {"SPY": f"買入持有 {config.BENCHMARK}"}
    stats = {a: {k: round(float(v), 2) for k, v in curve_stats(c).items()} for a, c in curves.items()}
    monthly = {a: (c.resample("ME").last().pct_change().fillna(c.resample("ME").last().iloc[0] / c.iloc[0] - 1) * 100)
               .round(2).rename(lambda t: t.strftime("%Y-%m")).to_dict() for a, c in curves.items()}

    by_month = []
    for i, d in enumerate(signal_dates):
        nxt = signal_dates[i + 1] if i + 1 < len(signal_dates) else None
        row = {"signal": d.strftime("%Y-%m-%d"), "holding_month": (d + pd.offsets.MonthBegin(1)).strftime("%Y-%m")}
        for account in ("ALPACA", "ALPACA_MOM"):
            r = period_returns(data, d, nxt, hold[account][d])
            row[account] = {
                "tickers": hold[account][d],
                "best": [[t, round(float(v) * 100, 1)] for t, v in r.head(3).items()],
                "worst": [[t, round(float(v) * 100, 1)] for t, v in r.tail(3)[::-1].items()],
            }
        by_month.append(row)

    return {
        "start": curves["SPY"].index[0].strftime("%Y-%m-%d"),
        "end": common_end.strftime("%Y-%m-%d"),
        "names": names,
        "stats": stats,
        "turnover_pct": {a: round(v * 100, 1) for a, v in turnover.items()},
        "monthly": monthly,
        "curves": {a: {"dates": [t.strftime("%Y-%m-%d") for t in c.index], "values": [round(float(v), 2) for v in c]}
                   for a, c in curves.items()},
        "by_month": by_month,
        "overlap": {
            a: len(set(hold["ALPACA"][signal_dates[-1]]) & set(hold[a][signal_dates[-1]]))
            for a in ("ALPACA_MOM",)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=6)
    args = parser.parse_args()
    report = build(args.months)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"{report['start']} → {report['end']}")
    for a, s in report["stats"].items():
        print(f"  {report['names'][a]:<14} {s}  換手 {report['turnover_pct'].get(a, 0)}%")
    print("\n每月回報 %：")
    print(pd.DataFrame(report["monthly"]).rename(columns=report["names"]).to_string())
    print(f"\n已寫入 {OUT}")


if __name__ == "__main__":
    main()
