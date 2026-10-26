"""即日形態頁：4 個形態嘅回測（同隨機入市比較），同埋逐日 5 分鐘 K 線睇入市 / 出市位。"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from intraday.backtest import STOP_ATR, TARGET_ATR, periods, summarize
from intraday.data import SYMBOLS, add_session_columns, get_bars
from intraday.patterns import OPENING_BARS, PATTERNS
from intraday.run import TRADES_CACHE, load_trades

PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#9085e9"],
}


@st.cache_data(ttl=24 * 3600, show_spinner="計緊所有交易（第一次大約 1–2 分鐘）…")
def cached_trades(mtime: float) -> pd.DataFrame:
    return load_trades()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_day(symbol: str) -> pd.DataFrame:
    return add_session_columns(get_bars(symbol))


def intraday_page() -> None:
    st.header("即日形態（短線）")
    st.caption(
        f"{len(SYMBOLS)} 隻成交最活躍嘅股票 / ETF，5 分鐘 K 線，只做好倉。訊號收市確認，下一枝開市入市；"
        f"止蝕 {STOP_ATR:g}×ATR、止賺 {TARGET_ATR:g}×ATR、15:55 前一定平倉。"
        "對照組係同一日同一隻股、隨機時間入市，出場規則一樣。"
    )
    if not TRADES_CACHE.exists():
        st.info("未有回測結果，先喺 Terminal 行：`python -m intraday.run`")
        st.stop()

    trades = cached_trades(TRADES_CACHE.stat().st_mtime)
    c1, c2 = st.columns(2)
    cost = c1.slider("每邊成本（bps）", 0.0, 10.0, 2.0, 0.5,
                     help="買賣價差 + 滑價。大型股大約 1–3 bps，細啲嘅股會更貴。")
    unlocked = c2.toggle("🔓 打開鎖櫃期（2025 →）", value=False)
    labels = list(periods())
    label = st.radio("期間", labels, horizontal=True, label_visibility="collapsed") if unlocked else labels[0]
    start, end = periods()[label]
    st.subheader(label)

    table = summarize(trades, cost, start, end)
    st.dataframe(table.round(2), width="stretch")
    st.caption("bps＝萬分之一（1 bps＝0.01%）。「跑贏隨機 t 值」大過 2 先算形態本身有料，唔係靠股票本身向上。")

    edge = table[table["跑贏隨機 t 值"] >= 2]
    profitable = edge[edge["平均每筆 bps"] > 0]
    if len(profitable):
        st.success(f"{', '.join(profitable.index)}：跑贏隨機入市，而且扣成本後仲賺錢。要喺鎖櫃期再驗證。")
    elif len(edge):
        st.warning(f"{', '.join(edge.index)}：跑贏隨機入市（t ≥ 2），但扣咗成本之後平均每筆仍然蝕錢，用唔到。"
                   f"另外試咗 {len(table)} 個形態，t 啱啱過 2 有機會只係運氣。")
    else:
        st.info(f"冇一個形態顯著跑贏隨機入市（最高 t = {table['跑贏隨機 t 值'].max():.2f}）。")

    scan_section()

    with st.expander("每個形態嘅定義"):
        for name, desc in PATTERNS.items():
            st.markdown(f"- **{name}**：{desc}")

    # ---------- 累積回報 ----------
    t = trades if start is None else trades[trades.day >= start]
    t = t if end is None else t[t.day < end]
    colors = PALETTE["dark" if st.context.theme.type == "dark" else "light"]
    fig = go.Figure()
    for name, color in zip(PATTERNS, colors):
        p = t[t.pattern == name]
        if p.empty:
            continue
        net = (p.gross - 2 * cost / 1e4).groupby(p.day).mean().cumsum() * 100
        rnd = (p.random_gross - 2 * cost / 1e4).groupby(p.day).mean().cumsum() * 100
        fig.add_trace(go.Scatter(x=net.index, y=net, name=name, line=dict(width=2, color=color)))
        fig.add_trace(go.Scatter(x=rnd.index, y=rnd, name=f"{name}（隨機）", showlegend=False,
                                 line=dict(width=1, color=color, dash="dot")))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="累積 %（每日平均）",
                      legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("實線＝形態，虛線＝同色形態嘅隨機入市對照。每日將嗰日所有交易平均，再累加。")

    # ---------- 逐日睇圖 ----------
    st.subheader("逐日睇圖")
    c1, c2 = st.columns(2)
    symbol = c1.selectbox("股票", SYMBOLS)
    days = sorted(t[t.symbol == symbol].day.unique(), reverse=True)
    if not days:
        st.write("呢段時間冇交易。")
        return
    day = c2.selectbox("日期", days, format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
    g = cached_day(symbol)
    g = g[g.day == day]
    day_trades = t[(t.symbol == symbol) & (t.day == day)]

    fig = go.Figure(go.Candlestick(x=g.index, open=g.open, high=g.high, low=g.low, close=g.close, name=symbol))
    fig.add_trace(go.Scatter(x=g.index, y=g.vwap, name="VWAP", line=dict(width=1.2, color="#9085e9")))
    or_high = g.high.iloc[:OPENING_BARS].max()
    fig.add_hline(y=or_high, line_dash="dot", line_color="#d95926", annotation_text="開市 30 分鐘高位")
    fig.add_hline(y=g.prev_high.iloc[0], line_dash="dot", line_color="#c98500", annotation_text="前日高位")
    for (name, color) in zip(PATTERNS, colors):
        tr = day_trades[day_trades.pattern == name]
        for _, r in tr.iterrows():
            exit_px = r.entry * (1 + r.gross)
            fig.add_trace(go.Scatter(
                x=[r.entry_time, r.exit_time], y=[r.entry, exit_px], mode="markers+lines", name=name,
                line=dict(color=color, width=1.5, dash="dash"),
                marker=dict(size=11, symbol=["triangle-up", "x"], color=color),
                hovertemplate=f"{name}<br>%{{x|%H:%M}} $%{{y:.2f}}<br>{r.exit_reason}：{r.gross * 100:+.2f}%<extra></extra>",
            ))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"))
    st.plotly_chart(fig, width="stretch")
    if len(day_trades):
        st.dataframe(day_trades[["pattern", "entry_time", "exit_time", "entry", "gross", "exit_reason"]]
                     .assign(gross=lambda d: (d.gross * 100).round(2)).rename(columns={"gross": "回報 %"}),
                     width="stretch", hide_index=True)


def scan_section() -> None:
    """全形態掃描結果（規則見 intraday/PREREGISTRATION.md）。"""
    from intraday.scan import OUT

    path = OUT / "scan_results.pkl"
    st.subheader("全形態掃描（TA-Lib 61 個 + 4 個自訂）")
    if not path.exists():
        st.info("未掃描，喺 Terminal 行：`python -m intraday.scan`")
        return
    r = pd.read_pickle(path).reset_index()
    r["方向"] = r.direction.map({1: "好倉", -1: "沽空"})
    passed1, passed2 = int(r.stage1.sum()), int(r.stage2.sum())
    msg = f"{len(r)} 個（形態 × 方向）｜第 1 關（開發期）通過 {passed1}｜第 2 關（鎖櫃期）通過 {passed2}"
    (st.success if passed2 else st.info)(("🎯 搵到！" if passed2 else "冇搵到。") + msg)
    st.caption("第 1 關：≥ 300 筆、扣成本後平均 > 0、跑贏隨機（Benjamini–Hochberg 修正，FDR 5%）。"
               "第 2 關：鎖櫃期扣成本後平均 > 0 而且 t > 2。規則喺睇結果之前登記。")
    cols = {"pattern": "形態", "方向": "方向", "n": "交易次數", "net_bps": "平均每筆 bps", "net_t": "t 值",
            "vs_random_bps": "跑贏隨機 bps", "vs_random_t": "跑贏隨機 t 值", "bh_significant": "BH 顯著",
            "stage1": "第 1 關", "stage2": "第 2 關"}
    st.dataframe(r[list(cols)].rename(columns=cols).round(2), width="stretch", hide_index=True, height=400)
