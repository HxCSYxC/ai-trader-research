"""即日形態回測命令行。

    python -m intraday.run                 # 只睇開發期（2020–2024）
    python -m intraday.run --unlock        # 揀定之後先打開鎖櫃期（2025 →）
    python -m intraday.run --cost 5        # 每邊成本（bps），預設 2
"""
import argparse
from pathlib import Path

import pandas as pd

from intraday.backtest import periods, run_all, summarize

TRADES_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "intraday" / "trades.pkl"


def load_trades(refresh: bool = False) -> pd.DataFrame:
    if TRADES_CACHE.exists() and not refresh:
        return pd.read_pickle(TRADES_CACHE)
    trades = run_all()
    trades.to_pickle(TRADES_CACHE)
    return trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", type=float, default=2.0, help="每邊成本 bps（買賣價差 + 滑價）")
    parser.add_argument("--unlock", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="重新計所有交易")
    args = parser.parse_args()

    trades = load_trades(args.refresh)
    print(f"總共 {len(trades):,} 筆交易，{trades.symbol.nunique()} 隻股，每邊成本 {args.cost} bps")
    items = list(periods().items())
    for label, (start, end) in items if args.unlock else items[:1]:
        print(f"\n######## {label} ########")
        print(summarize(trades, args.cost, start, end).round(2).to_string())
    if not args.unlock:
        print("\n🔒 鎖櫃期未打開。")


if __name__ == "__main__":
    main()
