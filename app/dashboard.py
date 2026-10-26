"""Streamlit dashboard：streamlit run app/dashboard.py"""
import hmac
import os
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from backtest.engine import benchmark_equity, run_backtest, summarize
from data.fetch import get_daily_bars
from strategies import STRATEGIES
from strategies.indicators import rsi, sma
from ml_view import ml_page
from picker_view import picker_page
from trading_view import trading_page
from intraday_view import intraday_page
from robustness_view import heatmap_page, oos_page

st.set_page_config(page_title="AI Trader", layout="wide")


def require_password() -> None:
    """唯讀（公開）模式一定要密碼；冇設定密碼就唔開放（fail closed）。"""
    if not config.DASHBOARD_READONLY:
        return
    password = os.getenv("DASHBOARD_PASSWORD", "")
    if len(password) < 12:
        st.error("公開模式要喺 .env 設定 DASHBOARD_PASSWORD（最少 12 個字）。")
        st.stop()
    if st.session_state.get("authed"):
        return
    st.title("AI Trader")
    entered = st.text_input("密碼", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), password.encode()):
            st.session_state.authed = True
            st.rerun()
        time.sleep(1.5)  # 拖慢估密碼
        st.error("密碼唔啱")
    st.stop()


require_password()


@st.cache_data(ttl=3600, show_spinner="拎緊數據…")
def load_bars(ticker: str, start: str) -> pd.DataFrame:
    return get_daily_bars(ticker, start)


@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark(ticker: str, start: str, _index: pd.DatetimeIndex) -> pd.Series:
    # DatetimeIndex hash 唔到，用 ticker + start 做 cache key（同一組數據 index 一樣）
    return benchmark_equity(_index)


# ---------- Sidebar ----------
st.sidebar.title("AI Trader")
ticker = st.sidebar.text_input("Ticker", "AAPL").strip().upper()
start = st.sidebar.date_input("開始日期", date(2018, 1, 1), max_value=date.today())

strategy_name = st.sidebar.selectbox("策略", list(STRATEGIES))
strategy_cls, param_specs = STRATEGIES[strategy_name]
params = {
    name: st.sidebar.slider(name, lo, hi, default)
    for name, (lo, hi, default) in param_specs.items()
}
st.sidebar.caption(
    f"成本：每邊 {config.COMMISSION:.2%}｜本金：${config.DEFAULT_CASH:,}｜"
    "訊號喺下一日開市價成交"
)

if not ticker:
    st.stop()

try:
    df = load_bars(ticker, start.isoformat())
except ValueError as e:
    st.error(str(e))
    st.stop()

PAGES = ["回測", "參數 Heatmap", "In / Out-of-sample", "AI 訊號", "AI 選股", "即日形態", "模擬盤"]
page = st.segmented_control("頁面", PAGES, default=PAGES[0], label_visibility="collapsed", key="page") or PAGES[0]

if page == "AI 訊號":
    ml_page(ticker)
    st.stop()

if page == "AI 選股":
    picker_page()
    st.stop()

if page == "模擬盤":
    trading_page()
    st.stop()

if page == "即日形態":
    intraday_page()
    st.stop()

if page != PAGES[0]:
    bench_close = load_bars(config.BENCHMARK, start.isoformat()).Close.reindex(df.index).ffill().bfill()
    view = heatmap_page if page == PAGES[1] else oos_page
    view(ticker, start.isoformat(), strategy_name, param_specs, params, df, bench_close)
    st.stop()

if not strategy_cls.valid_params(params):
    st.warning("參數唔合理（例如 fast 要細過 slow、lower 要細過 upper）")
    st.stop()

stats = run_backtest(df, strategy_cls, **params)
equity = stats["_equity_curve"]["Equity"]
trades = stats["_trades"]
bench = load_benchmark(ticker, start.isoformat(), equity.index)
summary = summarize(stats, bench)

# ---------- 重點數字 ----------
st.header(f"{ticker} · {strategy_name}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("策略回報", f"{summary['策略回報 %']:.0f}%")
c2.metric("買入持有", f"{summary['買入持有呢隻股 %']:.0f}%")
c3.metric(f"{config.BENCHMARK}", f"{summary[f'買入持有 {config.BENCHMARK} %']:.0f}%")
c4.metric("最大回撤", f"{summary['最大回撤 %']:.0f}%")
c5.metric("交易次數", int(summary["交易次數"]))

beat_bench = summary["策略回報 %"] > summary[f"買入持有 {config.BENCHMARK} %"]
if beat_bench:
    st.success(f"呢段時間贏 {config.BENCHMARK}。記住：歷史表現唔代表將來，小心 overfitting。")
else:
    st.info(f"呢段時間輸 {config.BENCHMARK}，直接買 {config.BENCHMARK} 會更好。")

# ---------- K 線圖 ----------
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.15, 0.25], vertical_spacing=0.03
)
fig.add_trace(
    go.Candlestick(
        x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name=ticker
    ),
    row=1, col=1,
)

ma_periods = [params[k] for k in ("fast", "slow", "trend") if k in params]
for n in ma_periods:
    fig.add_trace(
        go.Scatter(x=df.index, y=sma(df.Close, n), name=f"SMA{n}", line=dict(width=1.2)),
        row=1, col=1,
    )

if len(trades):
    fig.add_trace(
        go.Scatter(
            x=trades.EntryTime, y=trades.EntryPrice, mode="markers", name="買入",
            marker=dict(symbol="triangle-up", size=11, color="#16a34a"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trades.ExitTime, y=trades.ExitPrice, mode="markers", name="賣出",
            marker=dict(symbol="triangle-down", size=11, color="#dc2626"),
        ),
        row=1, col=1,
    )

fig.add_trace(
    go.Bar(x=df.index, y=df.Volume, name="成交量", marker_color="#94a3b8"), row=2, col=1
)

period = params.get("period", 14)
fig.add_trace(
    go.Scatter(x=df.index, y=rsi(df.Close, period), name=f"RSI{period}", line=dict(width=1.2)),
    row=3, col=1,
)
for level in (params.get("lower", 30), params.get("upper", 70)):
    fig.add_hline(y=level, line_dash="dot", line_color="gray", row=3, col=1)

fig.update_layout(
    height=750, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.02),
)
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
st.plotly_chart(fig, width="stretch")

# ---------- 資金曲線 ----------
st.subheader("資金曲線")
hold = df.Close.reindex(equity.index) / df.Close.iloc[0] * config.DEFAULT_CASH
eq_fig = go.Figure()
eq_fig.add_trace(go.Scatter(x=equity.index, y=equity, name=strategy_name))
eq_fig.add_trace(go.Scatter(x=hold.index, y=hold, name=f"買入持有 {ticker}"))
eq_fig.add_trace(go.Scatter(x=bench.index, y=bench, name=f"買入持有 {config.BENCHMARK}"))
eq_fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="$")
st.plotly_chart(eq_fig, width="stretch")

# ---------- 詳細 ----------
left, right = st.columns([1, 2])
with left:
    st.subheader("統計")
    st.dataframe(
        pd.Series(summary, name="數值").round(2).to_frame(), width="stretch"
    )
with right:
    st.subheader("交易記錄")
    if len(trades):
        st.dataframe(
            trades[["EntryTime", "ExitTime", "EntryPrice", "ExitPrice", "Size", "ReturnPct", "PnL"]]
            .assign(ReturnPct=lambda t: (t.ReturnPct * 100).round(2))
            .rename(columns={"ReturnPct": "Return %"}),
            width="stretch", hide_index=True,
        )
    else:
        st.write("呢段時間冇交易訊號。")
