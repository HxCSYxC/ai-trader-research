"""一次過測多隻股：python -m ml.run --model "Gradient Boosting" --horizon 5"""
import argparse

import pandas as pd

import config
from ml.pipeline import curve_stats, equity_curves, load_data, predict
from ml.walkforward import MODELS, prediction_metrics

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "KO", "XOM", "SPY"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--model", default="Logistic Regression", choices=list(MODELS))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--entry", type=float, default=0.5)
    parser.add_argument("--exit", type=float, default=0.5)
    args = parser.parse_args()

    rows = []
    for t in args.tickers:
        df, bench = load_data(t)
        df_prob, y = predict(df, bench, args.model, args.horizon)
        m = prediction_metrics(df_prob.Prob, y)
        curves = equity_curves(df_prob, bench, t, args.entry, args.exit)
        s = {k: curve_stats(v) for k, v in curves.items()}
        rows.append({
            "Ticker": t,
            "AUC": m["AUC"],
            "準確率": m["準確率 %"],
            "估升準確率": m["永遠估升嘅準確率 %"],
            "AI 年化": s["AI 訊號"]["年化回報 %"],
            "持有年化": s[f"買入持有 {t}"]["年化回報 %"],
            "SPY 年化": s[f"買入持有 {config.BENCHMARK}"]["年化回報 %"],
            "AI 回撤": s["AI 訊號"]["最大回撤 %"],
            "持有回撤": s[f"買入持有 {t}"]["最大回撤 %"],
            "AI Sharpe": s["AI 訊號"]["Sharpe"],
            "持有 Sharpe": s[f"買入持有 {t}"]["Sharpe"],
        })
        print(f"done {t}", flush=True)

    table = pd.DataFrame(rows).set_index("Ticker")
    start = df_prob.Prob.first_valid_index().date()
    print(f"\n{args.model}, horizon={args.horizon}, out-of-sample 由 {start} 開始\n")
    print(table.round(2).to_string())
    print("\n平均：")
    print(table.mean().round(2).to_string())


if __name__ == "__main__":
    main()
