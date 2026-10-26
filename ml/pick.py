"""命令行 AI 選股。

    python -m ml.pick                       # 只睇開發期
    python -m ml.pick --unlock              # 揀定方法之後，先打開鎖櫃期
"""
import argparse

from ml.cross_section import MODELS
from ml.stock_picker import (
    build_curves, latest_picks, load, momentum_scores, period_table, quintiles, score,
    signal_quality, split_periods,
)
from ml.universe import UNIVERSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default=UNIVERSES[0], choices=UNIVERSES)
    parser.add_argument("--model", default=list(MODELS)[0], choices=list(MODELS))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--unlock", action="store_true", help="打開鎖櫃期")
    args = parser.parse_args()

    data = load(args.universe)
    ai = score(data, args.model)
    ai_sector = score(data, args.model, sector_neutral=True)
    curves, turnover = build_curves(data, args.k, ai, ai_sector)
    signals = {"AI": ai, "AI 行業內排名": ai_sector, "動量": momentum_scores(data, ai)}

    print(f"\n== {args.universe}（{data.bars['Close'].shape[1]} 隻）, {args.model}, Top {args.k} ==")
    periods = list(split_periods(curves).items())
    for label, (start, end) in periods if args.unlock else periods[:1]:
        print(f"\n######## {label} ########")
        print("\n① 訊號質素")
        print(signal_quality(signals, data.fwd, start, end).round(3).to_string())
        print("\n② 分 5 組平均每月回報 %（Q1 最差 → Q5 最好）")
        print(quintiles(signals, data.fwd, start, end).T.round(2).to_string())
        print("\n③ 組合（超額＝相對平均買晒）")
        print(period_table(curves, turnover, start, end).round(2).to_string())

    if not args.unlock:
        print("\n🔒 鎖櫃期未打開。揀定方法之後用 --unlock 睇。")
    else:
        date, picks = latest_picks(ai, args.k)
        print(f"\n④ 最新 AI 排名（{date.date()}）：{', '.join(picks.index)}")


if __name__ == "__main__":
    main()
