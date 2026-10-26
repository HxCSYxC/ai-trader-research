"""模擬盤命令行。

    python -m trader.run status                    # 帳戶、持倉、kill switch
    python -m trader.run preview --offline         # 冇 key 都得：假帳戶預覽今個月嘅單
    python -m trader.run preview                   # 用真 paper account 預覽，唔落單
    python -m trader.run rebalance --execute       # 定時任務用：只會喺每月頭 3 個交易日、開市時先落單
    python -m trader.run rebalance --execute --force   # 手動強制換股（仍然要開市）
    python -m trader.run kill / unkill             # 開 / 關 kill switch
"""
import argparse

import pandas as pd

import config
from trader.broker import AlpacaBroker, has_keys
from trader.rebalance import rebalance
from trader.risk import kill_switch_reason, set_kill_switch


def print_report(r) -> None:
    print(f"\n[{r.account}] {r.strategy} · {r.mode} → {r.status}：{r.message}")
    if r.targets is not None:
        print(f"訊號日期：{r.targets.signal_date.date()}｜目標 {len(r.targets.weights)} 隻，"
              f"每隻 {r.targets.weights.iloc[0]:.1%}｜資產 ${r.equity:,.0f}")
        if len(r.targets.weights) <= 30:
            print("目標：" + ", ".join(r.targets.weights.index))
    if r.orders:
        df = pd.DataFrame([vars(o) for o in r.orders])
        print(f"\n{len(df)} 張單（賣 {sum(df.side == 'sell')}、買 {sum(df.side == 'buy')}），"
              f"買入總額 ${df.loc[df.side == 'buy', 'dollars'].sum():,.0f}")
        print(df[["symbol", "side", "dollars", "reason"]].round(2).head(40).to_string(index=False))
    if r.fills:
        statuses = pd.Series([f["status"] for f in r.fills.values()]).value_counts()
        print("\n成交狀態：" + ", ".join(f"{k} {v}" for k, v in statuses.items()))


def accounts(selected: str | None) -> list[str]:
    names = [selected] if selected else [a for a in config.ACCOUNTS if has_keys(a)]
    if not names:
        print("冇帳戶有 key。可以先用 --offline 預覽，或者喺 .env 填入 Paper Trading key。")
    return names


def status(account: str | None) -> None:
    reason = kill_switch_reason()
    print(f"Kill switch：{f'🔴 開咗（{reason}）' if reason else '🟢 關咗'}")
    for a in accounts(account):
        b = AlpacaBroker(a)
        info, pos = b.info(), b.positions()
        print(f"\n[{a}] {config.ACCOUNTS.get(a, {}).get('strategy', '')}")
        print(f"資產 ${info.equity:,.2f}｜現金 ${info.cash:,.2f}｜今日 {info.equity / info.last_equity - 1:+.2%}"
              f"｜{'開市' if b.market_open() else '休市'}｜持倉 {len(pos)} 隻")
        for h in sorted(pos.values(), key=lambda h: -h.market_value)[:20]:
            print(f"  {h.symbol:<6} ${h.market_value:>10,.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["status", "preview", "rebalance", "kill", "unkill"])
    parser.add_argument("--account", help="帳戶前綴，例如 ALPACA_MOM；唔填就做晒有 key 嘅帳戶")
    parser.add_argument("--offline", action="store_true", help="用假帳戶，唔使 key")
    parser.add_argument("--execute", action="store_true", help="真係落單（模擬盤）")
    parser.add_argument("--force", action="store_true", help="唔理係咪每月頭幾日")
    args = parser.parse_args()

    if args.command == "kill":
        set_kill_switch(True, "manual")
        print("🔴 Kill switch 已開，所有落單會停。")
    elif args.command == "unkill":
        set_kill_switch(False)
        reason = kill_switch_reason()
        print(f"🔴 KILL_SWITCH 檔已刪，但{reason}仍然開住。" if reason else "🟢 Kill switch 已關。")
    elif args.command == "status":
        status(args.account)
    elif args.offline:
        print_report(rebalance(args.account or "ALPACA", offline=True))
    else:
        execute = args.command == "rebalance" and args.execute
        for a in accounts(args.account):
            print_report(rebalance(a, execute=execute, force=args.force))


if __name__ == "__main__":
    main()
