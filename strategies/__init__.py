from strategies.baseline import RsiReversion, SmaCross

# 名 -> (策略 class, {參數: (最細, 最大, 預設)})，dashboard 用嚟整 slider
STRATEGIES = {
    "SMA Cross": (SmaCross, {"fast": (5, 100, 20), "slow": (20, 250, 50)}),
    "RSI Reversion": (
        RsiReversion,
        {"period": (5, 30, 14), "lower": (10, 45, 30), "upper": (55, 90, 70), "trend": (50, 250, 200)},
    ),
}
