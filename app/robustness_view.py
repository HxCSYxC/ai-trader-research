"""參數 Heatmap 同 In / Out-of-sample 兩頁。"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from backtest.robustness import (
    best_params,
    equity_return,
    grid_correlation,
    grid_rank,
    grid_values,
    max_drawdown,
    param_grid,
    run_period,
)
from strategies import STRATEGIES

# 分歧色：輸 benchmark 紅、贏 benchmark 藍、打和灰
NEG, POS = "#e34948", "#2a78d6"


@st.cache_data(show_spinner="計緊每個參數組合…")
def cached_grid(
    ticker, start, strategy_name, fixed: tuple, x_name, x_vals: tuple, y_name, y_vals: tuple,
    segment: str, split: str | None, _df: pd.DataFrame,
) -> pd.DataFrame:
    """segment: full＝全段 / in＝split 之前 / out＝split 之後。ticker 同 start 係 cache key。"""
    cls, _ = STRATEGIES[strategy_name]
    args = (dict(fixed), x_name, list(x_vals), y_name, list(y_vals))
    if segment == "in":
        return param_grid(_df[_df.index < split], cls, *args)
    if segment == "out":
        return param_grid(_df, cls, *args, trade_from=pd.Timestamp(split))
    return param_grid(_df, cls, *args)


def period_return(close: pd.Series, start=None, end=None) -> float:
    """[start, end) 期間嘅買入持有回報 %。"""
    if start is not None:
        close = close[close.index >= start]
    if end is not None:
        close = close[close.index < end]
    return equity_return(close)


def heatmap(grid: pd.DataFrame, bench_ret: float, marks: dict[str, dict]) -> go.Figure:
    dark = st.context.theme.type == "dark"
    mid = "#383835" if dark else "#f0efec"
    diff = grid - bench_ret
    limit = max(abs(diff.min().min()), abs(diff.max().max()), 1)

    fig = go.Figure(
        go.Heatmap(
            z=diff.values, x=grid.columns, y=grid.index,
            customdata=grid.values, zmid=0, zmin=-limit, zmax=limit,
            colorscale=[[0, NEG], [0.5, mid], [1, POS]],
            text=grid.map(lambda v: "" if pd.isna(v) else f"{v:.0f}").values,
            texttemplate="%{text}", textfont=dict(size=10),
            colorbar=dict(title=f"vs {config.BENCHMARK}<br>(%)"),
            hovertemplate=(
                f"{grid.columns.name}=%{{x}}<br>{grid.index.name}=%{{y}}<br>"
                f"回報 %{{customdata:.1f}}%<br>vs {config.BENCHMARK} %{{z:+.1f}}%<extra></extra>"
            ),
            xgap=2, ygap=2,
        )
    )
    for label, (p, symbol) in marks.items():
        fig.add_trace(
            go.Scatter(
                x=[p[grid.columns.name]], y=[p[grid.index.name]], mode="markers", name=label,
                # open symbol 嘅外框用 marker.color
                marker=dict(symbol=symbol, size=18, color="#ffffff" if dark else "#111111",
                            line=dict(width=2.5)),
                hovertemplate=f"{label}<extra></extra>",
            )
        )
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title=grid.columns.name, yaxis_title=grid.index.name,
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"),
    )
    return fig


def axis_controls(param_specs: dict) -> tuple[str, list[int], str, list[int]]:
    names = list(param_specs)
    c1, c2 = st.columns(2)
    x_name = c1.selectbox("X 軸參數", names, index=0)
    y_name = c2.selectbox("Y 軸參數", [n for n in names if n != x_name], index=0)
    x_vals = grid_values(*param_specs[x_name][:2])
    y_vals = grid_values(*param_specs[y_name][:2])
    return x_name, x_vals, y_name, y_vals


def heatmap_page(ticker, start, strategy_name, param_specs, params, df, bench_close):
    st.header(f"{ticker} · {strategy_name} · 參數 Heatmap")
    st.caption(
        f"每格係一組參數喺全段時間嘅回報 %。藍＝贏買入持有 {config.BENCHMARK}，紅＝輸。"
        "其他參數用 sidebar 嘅數值。"
    )
    x_name, x_vals, y_name, y_vals = axis_controls(param_specs)
    fixed = tuple((k, v) for k, v in params.items() if k not in (x_name, y_name))

    grid = cached_grid(ticker, start, strategy_name, fixed, x_name, tuple(x_vals),
                       y_name, tuple(y_vals), "full", None, df)
    bench_ret = equity_return(bench_close)
    values = grid.stack().dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"贏 {config.BENCHMARK} 嘅組合", f"{(values > bench_ret).mean():.0%}")
    c2.metric("回報中位數", f"{values.median():.0f}%")
    c3.metric("最好", f"{values.max():.0f}%")
    c4.metric(f"{config.BENCHMARK}", f"{bench_ret:.0f}%")

    st.plotly_chart(heatmap(grid, bench_ret, {"你而家嘅參數": (params, "square-open")}), width="stretch")

    share = (values > bench_ret).mean()
    if share >= 0.7:
        st.success("大部分組合都贏，策略對參數唔敏感，比較可信。不過仍然要做 out-of-sample 測試。")
    elif share >= 0.3:
        st.warning("一半半：結果好睇靠揀啱參數，有 overfitting 風險。")
    else:
        st.error(f"大部分組合都輸 {config.BENCHMARK}，就算搵到一格好嘅，都好可能係運氣。")


def oos_page(ticker, start, strategy_name, param_specs, params, df, bench_close):
    st.header(f"{ticker} · {strategy_name} · In / Out-of-sample")
    st.caption(
        "模擬真實情況：只用前段數據（in-sample）揀最好嘅參數，然後喺後段（out-of-sample）"
        "睇下佢係咪仲好。後段嘅結果先係你將來大概會見到嘅表現。"
    )
    x_name, x_vals, y_name, y_vals = axis_controls(param_specs)

    dates = df.index
    split = st.select_slider(
        "分割日期", options=list(dates.date), value=dates[int(len(dates) * 0.6)].date(),
        format_func=lambda d: d.strftime("%Y-%m"),
    )
    split_str = split.isoformat()
    split_ts = pd.Timestamp(split)
    if (dates < split_ts).sum() < 250 or (dates >= split_ts).sum() < 120:
        st.warning("前段最少要大約一年、後段最少半年數據，郁下分割日期。")
        st.stop()

    fixed = tuple((k, v) for k, v in params.items() if k not in (x_name, y_name))
    key = (ticker, start, strategy_name, fixed, x_name, tuple(x_vals), y_name, tuple(y_vals))
    is_grid = cached_grid(*key, "in", split_str, df)
    oos_grid = cached_grid(*key, "out", split_str, df)

    cls, _ = STRATEGIES[strategy_name]
    best = {**dict(fixed), **best_params(is_grid)}
    rank, total = grid_rank(oos_grid, best)
    corr = grid_correlation(is_grid, oos_grid)

    is_bench = period_return(bench_close, end=split_ts)
    oos_bench = period_return(bench_close, start=split_ts)
    close = df.Close

    def row(label, p):
        if not cls.valid_params(p):
            return {"": label, "In-sample 回報 %": None, "Out-of-sample 回報 %": None, "Out-of-sample 最大回撤 %": None}
        eq_in = run_period(df[df.index < split_ts], cls, p)
        eq_out = run_period(df, cls, p, trade_from=split_ts)
        return {"": label, "In-sample 回報 %": equity_return(eq_in),
                "Out-of-sample 回報 %": equity_return(eq_out),
                "Out-of-sample 最大回撤 %": max_drawdown(eq_out)}

    oos_close = close[close.index >= split_ts]
    table = pd.DataFrame([
        row(f"前段最好參數 {best_params(is_grid)}", best),
        row("你而家嘅參數", params),
        {"": f"買入持有 {ticker}", "In-sample 回報 %": period_return(close, end=split_ts),
         "Out-of-sample 回報 %": equity_return(oos_close), "Out-of-sample 最大回撤 %": max_drawdown(oos_close)},
        {"": f"買入持有 {config.BENCHMARK}", "In-sample 回報 %": is_bench,
         "Out-of-sample 回報 %": oos_bench,
         "Out-of-sample 最大回撤 %": max_drawdown(bench_close[bench_close.index >= split_ts])},
    ]).set_index("")

    c1, c2, c3 = st.columns(3)
    c1.metric("前段最好參數，喺後段排第幾", f"{rank} / {total}")
    c2.metric("前後段排名相關度", f"{corr:.2f}", help="1＝前段好嘅後段都好；0＝冇關係；負數＝前段好嘅後段反而差")
    c3.metric(f"後段 {config.BENCHMARK}", f"{oos_bench:.0f}%")

    best_oos = table.iloc[0]["Out-of-sample 回報 %"]
    if corr < 0.3 or best_oos < oos_bench:
        st.error(
            f"前段揀出嚟嘅參數喺後段{'輸咗' if best_oos < oos_bench else '冇明顯優勢'}"
            f"（後段 {best_oos:.0f}% vs {config.BENCHMARK} {oos_bench:.0f}%），"
            f"相關度 {corr:.2f}。前段嘅好成績大部分係運氣 / overfitting。"
        )
    else:
        st.success(
            f"前段揀出嚟嘅參數喺後段仲贏 {config.BENCHMARK}，而且排名相關度 {corr:.2f}。"
            "係好兆頭，不過最好再試多幾隻股同唔同分割日期。"
        )

    st.dataframe(table.round(1), width="stretch")

    marks = {"前段最好": (best, "star-open"), "你而家嘅參數": (params, "square-open")}
    st.subheader(f"In-sample（～{split:%Y-%m}）")
    st.plotly_chart(heatmap(is_grid, is_bench, marks), width="stretch", key="is_heatmap")
    st.subheader(f"Out-of-sample（{split:%Y-%m}～）")
    st.plotly_chart(heatmap(oos_grid, oos_bench, marks), width="stretch", key="oos_heatmap")
    st.caption("兩幅圖嘅形狀越似，代表策略越穩定；如果左邊最藍嗰格喺右邊變紅，就係 overfitting。")
