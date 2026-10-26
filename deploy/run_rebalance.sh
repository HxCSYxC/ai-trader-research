#!/bin/bash
# launchd 每個工作日香港時間 23:00 行一次。唔係每月頭 3 個交易日 / 未開市 / 今個月換過，程式會自己跳過。
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  .venv/bin/python -m trader.run rebalance --execute
  echo
} >> logs/rebalance.log 2>&1
