"""AI 訊號頁：walk-forward ML 預測 + 回測。"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from backtest.engine import run_backtest
from ml.pipeline import curve_stats, equity_curves, load_data, predict
from ml.walkforward import MODELS, prediction_metrics
from strategies.ml_signal import MLSignal

# 類別色（固定次序）：AI、SMA、持有股票、SPY
SERIES_COLORS = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
                 "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"]}


@st.cache_data(ttl=3600, show_spinner="Walk-forward train 緊 model（大約 5–10 秒）…")
def cached_predict(ticker: str, model_name: str, horizon: int):
    df, bench = load_data(ticker)
    df_prob, y = predict(df, bench, model_name, horizon)
    return df_prob, y, bench


def ml_page(ticker: str):
    st.header(f"{ticker} · AI 訊號")
    st.caption(
        f"用 {config.ML_DATA_START[:4]} 年至今嘅數據：頭 3 年淨係用嚟 train，之後每季 retrain 一次，"
        "每次只用當時已知嘅數據。下面所有結果都係 out-of-sample。呢頁唔用 sidebar 嘅日期同策略。"
    )

    c1, c2, c3, c4 = st.columns(4)
    model_name = c1.selectbox("Model", list(MODELS))
    horizon = c2.selectbox("預測幾多日後", [1, 5, 10, 20], index=1)
    entry = c3.slider("買入門檻 P(升) >", 0.40, 0.70, 0.50, 0.01)
    exit_ = c4.slider("賣出門檻 P(升) <", 0.30, 0.70, 0.50, 0.01)
    st.caption("⚠️ 門檻應該事先定好。睇住結果調門檻，又會變返 overfitting。")
    if exit_ > entry:
        st.warning("賣出門檻唔可以高過買入門檻")
        st.stop()

    df_prob, y, bench = cached_predict(ticker, model_name, horizon)
    start = df_prob.Prob.first_valid_index()
    if start is None:
        st.error("數據唔夠長，train 唔到 model（要大約 4 年以上歷史）。")
        st.stop()

    # ---------- 預測質素 ----------
    m = prediction_metrics(df_prob.Prob, y)
    st.subheader("① 預測準唔準？")
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC", f"{m['AUC']:.3f}", help="0.5＝擲毫。0.55 以上先算有少少預測能力。")
    c2.metric("Model 準確率", f"{m['準確率 %']:.1f}%")
    c3.metric("永遠估「升」嘅準確率", f"{m['永遠估升嘅準確率 %']:.1f}%",
              help="最笨嘅方法：每日都估升。Model 要贏到呢個先有意義。")

    if m["AUC"] < 0.53:
        st.error("AUC 接近 0.5，model 基本上冇預測能力，同擲毫差唔多。")
    elif m["準確率 %"] <= m["永遠估升嘅準確率 %"]:
        st.warning("AUC 有少少，但準確率仲輸「永遠估升」。")
    else:
        st.success("Model 有啲預測能力。下一步要睇扣除成本之後賺唔賺到錢。")

    # ---------- 回測 ----------
    st.subheader("② 用嚟交易賺唔賺錢？")
    curves = equity_curves(df_prob, bench, ticker, entry, exit_)
    table = pd.DataFrame({k: curve_stats(v) for k, v in curves.items()}).T
    st.dataframe(table.round(2), width="stretch")

    colors = SERIES_COLORS["dark" if st.context.theme.type == "dark" else "light"]
    eq_fig = go.Figure()
    for (name, eq), color in zip(curves.items(), colors):
        eq_fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name, line=dict(width=2, color=color)))
    eq_fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="$ (log)", yaxis_type="log",
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"), hovermode="x unified",
    )
    st.plotly_chart(eq_fig, width="stretch")

    # ---------- 訊號 ----------
    st.subheader("③ 訊號")
    stats = run_backtest(df_prob, MLSignal, entry=entry, exit=exit_, trade_from=start)
    trades = stats["_trades"]
    d = df_prob.loc[start:]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=d.index, y=d.Close, name=ticker, line=dict(width=1.5, color=colors[2])), row=1, col=1)
    if len(trades):
        fig.add_trace(go.Scatter(
            x=trades.EntryTime, y=trades.EntryPrice, mode="markers", name="買入",
            marker=dict(symbol="triangle-up", size=9, color="#0ca30c")), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trades.ExitTime, y=trades.ExitPrice, mode="markers", name="賣出",
            marker=dict(symbol="triangle-down", size=9, color="#e34948")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.Prob, name="P(升)", line=dict(width=1, color=colors[0])), row=2, col=1)
    fig.add_hline(y=entry, line_dash="dot", line_color="#0ca30c", row=2, col=1)
    fig.add_hline(y=exit_, line_dash="dot", line_color="#e34948", row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"交易次數：{stats['# Trades']}｜持倉時間：{stats['Exposure Time [%]']:.0f}%")
