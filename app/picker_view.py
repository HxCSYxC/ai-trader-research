"""AI 選股頁：cross-sectional 排名 + 每月換股，開發期 / 鎖櫃期分開睇。"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from ml.cross_section import FEATURE_NAMES, MODELS
from ml.stock_picker import (
    EW, SPY, VARIANTS, build_curves, latest_picks, load, momentum_scores, period_table,
    quintiles, score, signal_quality, split_periods,
)
from ml.universe import UNIVERSES

# 類別色固定跟住策略（唔會因為揀少咗而轉色）
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
CURVE_ORDER = [*VARIANTS, EW, SPY]


@st.cache_resource(ttl=12 * 3600, show_spinner="下載緊數據、計緊 features（S&P 500 第一次大約 1–2 分鐘）…")
def cached_load(universe: str):
    return load(universe)


@st.cache_data(ttl=12 * 3600, show_spinner="Walk-forward train 緊 model（Gradient Boosting 會慢好多）…")
def cached_scores(universe: str, model_name: str, sector_neutral: bool, _data) -> pd.Series:
    return score(_data, model_name, sector_neutral)


def picker_page():
    st.header("AI 選股")
    st.caption(
        "每個月尾幫成個範圍排名，第二日開市等權買入頭 K 隻，下個月再換。"
        "Model 只用當時已知嘅數據 train（walk-forward）。呢頁唔用 sidebar 嘅設定。"
    )

    c1, c2, c3 = st.columns(3)
    universe = c1.selectbox("範圍", UNIVERSES)
    model_name = c2.selectbox("Model", list(MODELS))
    k = c3.selectbox("每次揸幾多隻", [10, 20, 30, 50], index=1)

    st.warning(
        "⚠️ **Survivorship bias**：範圍係用今日嘅名單，已經剔走咗過去退市、被收購嘅公司，"
        "所以絕對回報會偏高。S&P 500 已經做咗部分修正：每隻股要等正式加入指數之後先可以揀。"
        "要睇嘅係有冇贏到「平均買晒」（同一個偏差），同埋超額回報嘅 t 值。"
    )

    data = cached_load(universe)
    ai = cached_scores(universe, model_name, False, data)
    ai_sector = cached_scores(universe, model_name, True, data)
    curves, turnover = build_curves(data, k, ai, ai_sector)
    signals = {"AI": ai, "AI 行業內排名": ai_sector, "動量": momentum_scores(data, ai)}

    periods = split_periods(curves)
    unlocked = st.toggle(
        "🔓 打開鎖櫃期", value=False,
        help=f"{config.ML_LOCKBOX_START[:4]} 年之後嘅數據係「鎖櫃」：應該淨係用開發期揀定方法，最後先打開睇一次。"
             "打開之後再返去改方法，鎖櫃期就唔再係真正嘅 out-of-sample。",
    )
    labels = list(periods) if unlocked else list(periods)[:1]
    label = st.radio("期間", labels, horizontal=True, label_visibility="collapsed") if unlocked else labels[0]
    start, end = periods[label]
    st.subheader(label)

    # ---------- ① 訊號 ----------
    st.markdown("#### ① 排名準唔準？")
    sq = signal_quality(signals, data.fwd, start, end)
    st.dataframe(sq.round(3), width="stretch")
    st.caption("IC＝每月排名同實際回報排名嘅相關度，0＝冇用，0.03–0.05 已經算有用。t 值大過 2 先算唔似運氣。")

    q = quintiles(signals, data.fwd, start, end)
    theme = "dark" if st.context.theme.type == "dark" else "light"
    colors = dict(zip(CURVE_ORDER, PALETTE[theme]))
    signal_colors = {"AI": colors["AI"], "AI 行業內排名": colors["AI 行業內排名"], "動量": colors["動量（冇 ML）"]}
    q_fig = go.Figure()
    for name, col in q.items():
        q_fig.add_trace(go.Bar(
            x=[f"Q{i}" for i in q.index], y=col, name=name, marker_color=signal_colors[name],
            hovertemplate=f"{name}<br>%{{x}}：平均每月 %{{y:.2f}}%<extra></extra>",
        ))
    q_fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10), barmode="group", bargap=0.3, bargroupgap=0.08,
        yaxis_title="平均每月回報 %", legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"),
    )
    st.plotly_chart(q_fig, width="stretch")
    st.caption("Q1 係排名最差嘅一組、Q5 最好。如果排名有用，柱應該由左至右越嚟越高。")

    # ---------- ② 組合 ----------
    st.markdown("#### ② 組合回測（超額＝相對平均買晒）")
    table = period_table(curves, turnover, start, end)
    st.dataframe(table.round(2), width="stretch")
    with st.expander("每個版本係乜"):
        for name, desc in VARIANTS.items():
            st.markdown(f"- **{name}**：{desc}")

    shown = st.multiselect("圖表顯示", CURVE_ORDER, default=["AI", "動量（冇 ML）", EW, SPY])
    eq_fig = go.Figure()
    for name in shown:
        c = curves[name]
        c = c.loc[start:] if start is not None else c
        c = c[c.index < end] if end is not None else c
        eq_fig.add_trace(go.Scatter(x=c.index, y=c / c.iloc[0] * config.DEFAULT_CASH, name=name,
                                    line=dict(width=2, color=colors[name])))
    eq_fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="$ (log)", yaxis_type="log",
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"), hovermode="x unified",
    )
    st.plotly_chart(eq_fig, width="stretch")

    best_t = table.drop(index=[EW, SPY])["超額 t 值"].max()
    if best_t >= 2:
        st.success(f"有版本嘅超額回報 t 值 ≥ 2（最高 {best_t:.2f}），統計上唔太似運氣。")
    else:
        st.info(f"冇一個版本嘅超額回報 t 值過到 2（最高 {best_t:.2f}）。即係話贏「平均買晒」嘅部分，分唔清係實力定運氣。")

    # ---------- ③ 最新排名 ----------
    if unlocked:
        date, picks = latest_picks(ai, k)
        st.markdown(f"#### ③ 最新 AI 排名（{date:%Y-%m-%d} 收市）")
        feats = data.X.xs(date, level="date").reindex(picks.index)
        shown_feats = (feats * 100).round(0).rename(columns=FEATURE_NAMES)
        shown_feats.insert(0, "行業", data.sectors.reindex(picks.index))
        shown_feats.insert(1, "AI 分數", picks.round(3))
        st.dataframe(shown_feats, width="stretch")
        st.caption("Feature 係喺成個範圍入面嘅百分位（100＝最高）。模擬盤研究用，唔係投資建議。")
