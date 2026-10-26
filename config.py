import os

from dotenv import load_dotenv

load_dotenv()

# 回測設定
DEFAULT_CASH = 100_000
# Alpaca 免佣，但真實落單有 slippage，每邊當 0.05% 成本計
COMMISSION = 0.0005
BENCHMARK = "SPY"

# ML：用長啲嘅歷史，頭 3 年只用嚟 train，之後每季 retrain 一次
ML_DATA_START = "2010-01-01"
ML_MIN_TRAIN_DAYS = 756
ML_RETRAIN_EVERY = 63
# 選股：呢日之前係開發期（可以睇住改），之後係鎖櫃期（揀定方法先睇一次）
ML_LOCKBOX_START = "2022-01-01"

# ---------- 模擬盤（第 4 階段）----------
# 本項目只支援 Alpaca 模擬盤（paper trading），code 入面寫死，唔會連真錢帳戶。
# 每個帳戶喺 .env 用 <前綴>_API_KEY / <前綴>_SECRET_KEY；冇填 key 嘅帳戶會自動跳過。
ACCOUNTS = {
    "ALPACA": {"strategy": "AI + 動量混合", "model": "Gradient Boosting", "k": 20},
    "ALPACA_MOM": {"strategy": "動量（冇 ML）", "model": None, "k": 20},
    "ALPACA_EW": {"strategy": "平均買晒", "model": None, "k": None},
}
TRADING_UNIVERSE = "S&P 500（現時成分股）"
# AI 策略嘅換手緩衝：已經揸住嘅股只要仲排喺頭 AI_KEEP_MULT × k 名就唔賣。
# 原因：GBM 對訓練數據 1e-6 級嘅浮點差都敏感，rank ~20 附近嘅名會跳。
AI_KEEP_MULT = 2

# 風控
MAX_POSITION_WEIGHT = 0.10   # 單一股票最多佔資產 10%
CASH_BUFFER = 0.02           # 留 2% 現金，避免價格郁動令買單唔夠錢
MAX_DAILY_LOSS = 0.05        # 帳戶今日跌超過 5% 就唔落單
MAX_ORDER_PCT = 0.15         # 單一張單最多係資產 15%（全部賣出除外）
MIN_TRADE_DOLLARS = 50       # 少過呢個金額嘅調整唔做，慳換手
MAX_ORDERS_PER_RUN = 600
REBALANCE_WINDOW_DAYS = 3    # 每月頭 3 個交易日內都可以補做（例如部機瞓咗）

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
KILL_SWITCH_FILE = os.path.join(PROJECT_DIR, "KILL_SWITCH")
DB_PATH = os.path.join(PROJECT_DIR, "db", "trading.sqlite")

# ---------- 公開 dashboard ----------
# DASHBOARD_READONLY=1 時：收起所有落單 / kill switch 掣，並且一定要輸入 .env 嘅 DASHBOARD_PASSWORD 先睇到。
DASHBOARD_READONLY = os.getenv("DASHBOARD_READONLY", "0") == "1"
