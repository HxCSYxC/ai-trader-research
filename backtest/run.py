"""命令行回測：python -m backtest.run --ticker AAPL --strategy "SMA Cross" """
import argparse

from backtest.engine import benchmark_equity, run_backtest, summarize
from data.fetch import get_daily_bars
from strategies import STRATEGIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--strategy", default="SMA Cross", choices=list(STRATEGIES))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    strategy_cls, _ = STRATEGIES[args.strategy]
    df = get_daily_bars(args.ticker, args.start, args.end)
    stats = run_backtest(df, strategy_cls)
    bench = benchmark_equity(stats["_equity_curve"].index)

    print(f"\n{args.strategy} on {args.ticker}  ({df.index[0].date()} → {df.index[-1].date()})\n")
    for k, v in summarize(stats, bench).items():
        print(f"  {k:<20} {v:>10.2f}")


if __name__ == "__main__":
    main()
