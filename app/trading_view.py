"""模擬盤頁：帳戶狀態、持倉、預覽 / 執行換股、kill switch、交易日誌。"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import config
from trader import journal
from trader.broker import AlpacaBroker, has_keys
from trader.rebalance import rebalance
from trader.risk import kill_switch_on, set_kill_switch


def show_report(r) -> None:
    msg = f"{r.strategy} · {r.mode} → **{r.status}**：{r.message}"
    {"ok": st.success, "skipped": st.info, "blocked": st.error}.get(r.status, st.warning)(msg)
    if r.targets is not None:
        st.caption(f"訊號日期 {r.targets.signal_date:%Y-%m-%d}｜目標 {len(r.targets.weights)} 隻，每隻 {r.targets.weights.iloc[0]:.1%}")
    if r.orders:
        df = pd.DataFrame([vars(o) for o in r.orders])[["symbol", "side", "dollars", "reason"]]
        st.dataframe(df.round(2), width="stretch", hide_index=True, height=min(400, 38 + 35 * len(df)))


def account_panel(account: str) -> None:
    settings = config.ACCOUNTS[account]
    st.subheader(f"{settings['strategy']}（{account}）")
    try:
        b = AlpacaBroker(account)
        info, pos, is_open = b.info(), b.positions(), b.market_open()
    except Exception as e:
        st.error(f"連唔到 Alpaca：{e}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("資產", f"${info.equity / 1000:,.1f}k", f"{info.equity / info.last_equity - 1:+.2%} 今日")
    c2.metric("現金", f"${info.cash / 1000:,.1f}k")
    c3.metric("持倉", f"{len(pos)} 隻")
    c4.metric("市場", "開市" if is_open else "休市")

    try:
        ts, eq = b.portfolio_history("3M")
        hist = pd.Series(eq, index=pd.to_datetime(ts, unit="s")).dropna()
        hist = hist[hist > 0]
        if len(hist) > 1:
            fig = go.Figure(go.Scatter(x=hist.index, y=hist, line=dict(width=2, color="#2a78d6"),
                                       hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="$")
            st.plotly_chart(fig, width="stretch", key=f"hist_{account}")
    except Exception:
        pass

    if pos:
        df = pd.DataFrame([vars(h) for h in pos.values()]).sort_values("market_value", ascending=False)
        df["比重 %"] = df.market_value / info.equity * 100
        st.dataframe(df.round(2), width="stretch", hide_index=True, height=min(400, 38 + 35 * len(df)))

    if config.DASHBOARD_READONLY:
        return
    c1, c2 = st.columns(2)
    if c1.button("預覽今個月嘅單", key=f"preview_{account}"):
        with st.spinner("計緊目標持倉…"):
            show_report(rebalance(account, execute=False))
    with c2:
        confirm = st.checkbox("我明白會喺模擬盤落市價單", key=f"confirm_{account}")
        if st.button("執行換股", key=f"exec_{account}", disabled=not (confirm and is_open),
                     help="要開市先撳得到。會即刻用市價單換到目標持倉。"):
            with st.spinner("落緊單，等成交…"):
                show_report(rebalance(account, execute=True, force=True))


def trading_page() -> None:
    load_dotenv(override=True)  # 填完 .env 唔使重啟 dashboard
    st.header("模擬盤")
    st.caption("只會連 Alpaca Paper Trading（code 寫死），唔涉及真錢。每月頭 3 個交易日自動換股，亦可以喺度手動預覽 / 執行。")

    on = kill_switch_on()
    if config.DASHBOARD_READONLY:
        st.caption(f"👀 唯讀模式｜Kill switch：{'🔴 開咗' if on else '🟢 關咗'}")
    else:
        new = st.toggle("🔴 Kill switch（開咗就唔會落任何單）", value=on)
        if new != on:
            set_kill_switch(new, "dashboard")
            st.rerun()

    ready = [a for a in config.ACCOUNTS if has_keys(a)]
    if not ready:
        st.info(
            "未有 Alpaca key。步驟：\n"
            "1. 喺 Alpaca dashboard 揀 **Paper Trading**，generate API key\n"
            "2. 將項目入面嘅 `.env.example` 複製做 `.env`，填入 `ALPACA_API_KEY` 同 `ALPACA_SECRET_KEY`\n"
            "3. 返嚟呢頁撳 R 重新整理\n\n"
            "未有 key 都可以用假帳戶（$100,000、冇持倉）預覽："
        )
        if not config.DASHBOARD_READONLY and st.button("離線預覽"):
            with st.spinner("計緊目標持倉…"):
                show_report(rebalance("ALPACA", offline=True))
    for a in ready:
        account_panel(a)
        st.divider()

    st.subheader("交易日誌")
    runs = journal.recent_runs()
    if runs.empty:
        st.write("未有記錄。")
        return
    st.dataframe(runs.drop(columns=["mode"]).assign(模式=runs["mode"]), width="stretch", hide_index=True)
    run_id = st.selectbox("睇邊次嘅單", runs["id"].tolist())
    st.dataframe(journal.run_orders(run_id), width="stretch", hide_index=True)
