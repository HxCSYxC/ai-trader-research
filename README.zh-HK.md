# AI Trader

[English](README.md) | **廣東話**

自己用嘅美股 AI 交易研究系統，用 Alpaca **模擬盤**（paper trading）驗證，唔涉及真錢。

> **呢個係展示版。** 每月自動換股喺另一個私人 repo 運行；呢度嘅 GitHub Actions 冇開定時，亦冇任何 key。
> 研究結論：用公開價量數據，喺大型美股搵唔到扣成本後穩定賺錢嘅 edge（日線 ML、S&P 500 AI 選股、即日形態全部測試過），詳情睇下面「進度」同 `intraday/PREREGISTRATION.md`。
>
> 唔係投資建議。

## 開始

```bash
source .venv/bin/activate
streamlit run app/dashboard.py          # 網站：http://localhost:8501
python -m backtest.run --ticker AAPL --strategy "SMA Cross"   # 命令行回測
python -m ml.run --model "Logistic Regression" --horizon 5    # ML 單股訊號，一次過測多隻
python -m ml.pick --k 20                  # AI 選股（S&P 500 每月排名，只睇開發期）
python -m ml.pick --k 20 --unlock         # 揀定方法之後先打開鎖櫃期（2022 年之後）
```

重新安裝：`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
研究用（即日形態全形態掃描要 TA-Lib）：`.venv/bin/pip install -r requirements-research.txt`

## 模擬盤（Alpaca Paper Trading）

```bash
python -m trader.run preview --offline        # 冇 key 都得：假帳戶預覽今個月嘅單
python -m trader.run status                   # 帳戶、持倉、kill switch
python -m trader.run preview                  # 用 paper account 預覽，唔落單
python -m trader.run rebalance --execute      # 定時任務用：每月頭 3 個交易日、開市時先落單
python -m trader.run kill / unkill            # 開 / 關 kill switch
```

- 只會連模擬盤（`paper=True` 寫死）。key 放 `.env`，格式睇 `.env.example`
- 帳戶同策略喺 `config.ACCOUNTS` 設定，風控參數都喺 `config.py`
- 定時（主力）：GitHub Actions，部 Mac 熄咗都照行，睇下面「雲端定時換股」
- 定時（舊，已停用）：`deploy/com.aitrader.rebalance.plist`（星期一至五香港時間 23:00）。已經 unload，唔好同 GitHub Actions 兩邊一齊行；真係要用返先 `cp deploy/com.aitrader.rebalance.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.aitrader.rebalance.plist`，同時喺 GitHub 停用個 workflow
- 本機 run 同單記錄喺 `db/trading.sqlite`，dashboard「模擬盤」頁睇得到
- 「今個月換咗未」以 Alpaca 嘅落單記錄為準（今個月有成交或者未完成嘅單就跳過），所以雲端 / 本機都唔會一個月換兩次

## 雲端定時換股（GitHub Actions）

`.github/workflows/rebalance.yml`：星期一至五 15:17 UTC（香港 23:17，紐約開市後）行一次，17:17 UTC 再行一次做後備。
未開市、唔係每月頭 3 個交易日、或者 Alpaca 今個月已經有單，程式會自己跳過。

設定（repo 一定要係 **Private**）：
1. GitHub repo → Settings → Secrets and variables → Actions → **Secrets** → New repository secret，逐個加：
   `ALPACA_API_KEY`、`ALPACA_SECRET_KEY`、`ALPACA_MOM_API_KEY`、`ALPACA_MOM_SECRET_KEY`、`ALPACA_EW_API_KEY`、`ALPACA_EW_SECRET_KEY`
   （值同 `.env` 一樣，全部係 Paper Trading key）
2. Actions → Rebalance (paper) → Run workflow → mode 揀 `preview`：只預覽，唔落單，用嚟測試
3. 停用本機 launchd（已經做咗）：`launchctl unload ~/Library/LaunchAgents/com.aitrader.rebalance.plist`

- **Kill switch**：雲端冇 `KILL_SWITCH` 檔，改用 Settings → Secrets and variables → Actions → **Variables** 加 `TRADING_DISABLED` = `1`（刪咗或者改做 `0` 就恢復）。本機 export `TRADING_DISABLED=1` 效果一樣
- **日誌**：每次 run 嘅輸出喺 Actions 頁嘅 Summary；SQLite 日誌同 log 上載做 artifact（`journal-<run id>`，留 90 日）
- 手動 `scheduled` 模式同定時任務一樣（唔會 `--force`）；要強制換股請喺本機行 `python -m trader.run rebalance --execute --force`

## 畀其他人睇（唯讀 + 密碼）

```bash
.venv/bin/streamlit run app/dashboard.py --server.address 127.0.0.1   # 自己用：全功能，只限本機
bash deploy/public_dashboard.sh                                         # 公開版：唯讀 + 密碼，127.0.0.1:8502
cloudflared tunnel --url http://127.0.0.1:8502                          # 生成公開網址（https://xxx.trycloudflare.com）
```

- 密碼喺 `.env` 嘅 `DASHBOARD_PASSWORD`（最少 12 個字，冇設定就唔開放）
- 唯讀模式冇落單、預覽、kill switch 掣；落單程式本身亦會拒絕（`DASHBOARD_READONLY=1`）
- Quick tunnel 每次重開網址都會變；關咗 cloudflared 或者部機瞓咗，人哋就睇唔到

## 即日形態（短線研究）

```bash
python -m intraday.run              # 4 個形態，開發期 2020–2024（第一次要下載 25 隻股嘅 5 分鐘 K，約 15 分鐘）
python -m intraday.run --cost 5     # 每邊成本 bps（預設 2）
python -m intraday.run --unlock     # 鎖櫃期 2025 →，揀定假設先開
```

- 形態：ORB 開市區間突破、突破前日高位、VWAP 重奪、好友吞沒；全部只做好倉
- 出場：止蝕 1×ATR、止賺 2×ATR、15:55 前平倉；同「同日同股隨機入市」比較
- 開發期結論：扣 2 bps 成本後 4 個形態全部蝕錢；兩個突破形態仲顯著輸過隨機入市
- 全形態掃描（`python -m intraday.scan`，TA-Lib 61 個 × 好 / 淡）：91 個組合冇一個過關（規則見 `intraday/PREREGISTRATION.md`）
- 假設 2 沽空突破（`python -m intraday.fade`）：鎖櫃期跑贏隨機沽空（t 5.8 / 4.1），但扣成本後每筆仍蝕 4–5 bps，不合格

## 結構

| 路徑 | 用途 |
|---|---|
| `data/fetch.py` | 用 yfinance 拎日線數據 |
| `strategies/indicators.py` | SMA、EMA、RSI |
| `strategies/baseline.py` | Baseline 策略（SMA Cross、RSI Reversion） |
| `strategies/__init__.py` | 策略列表，加新策略喺度註冊 |
| `backtest/engine.py` | 回測、同 SPY 比較 |
| `backtest/robustness.py` | 參數 grid、in/out-of-sample 測試 |
| `app/dashboard.py` | Streamlit 網站（回測頁） |
| `app/robustness_view.py` | 參數 Heatmap、In / Out-of-sample 頁 |
| `ml/features.py` | ML features 同 label |
| `ml/walkforward.py` | Walk-forward train / 預測、AUC |
| `ml/pipeline.py` | 預測 → 回測 → 同 benchmark 比較 |
| `strategies/ml_signal.py` | 用 P(升) 落單嘅策略 |
| `app/ml_view.py` | AI 訊號頁 |
| `ml/universe.py` | 選股範圍：S&P 500（Wikipedia）同 50 隻大型股 |
| `ml/cross_section.py` | 選股 features、walk-forward 排名、IC、組合模擬 |
| `ml/stock_picker.py` | 選股 pipeline |
| `app/picker_view.py` | AI 選股頁 |
| `trader/broker.py` | Alpaca 模擬盤連線（同離線假帳戶） |
| `trader/targets.py` | 用最新月尾數據計目標持倉 |
| `trader/risk.py` | 風控同計單 |
| `trader/journal.py` | SQLite 交易日誌 |
| `trader/rebalance.py` | 每月換股流程 |
| `trader/run.py` | 模擬盤命令行 |
| `trader/recent.py` | 用模擬盤策略回測最近 N 個月（`python -m trader.recent --months 6`） |
| `.github/workflows/rebalance.yml` | 雲端定時換股（GitHub Actions） |
| `app/trading_view.py` | 模擬盤頁 |
| `intraday/data.py` | Alpaca 5 分鐘 K 線（SIP）、VWAP、ATR |
| `intraday/patterns.py` | 4 個即日形態嘅規則 |
| `intraday/backtest.py` | 即日回測同隨機入市對照 |
| `app/intraday_view.py` | 即日形態頁 |

## 回測規則
- 本金 $100,000，每邊成本 0.05%（模擬 slippage）
- 訊號喺收市出，**下一日開市價**成交，避免 look-ahead
- 只做 long，唔用槓桿

## 進度
- [x] 第 1 階段：數據 + K 線圖
- [x] 第 2 階段：Baseline 策略 + 回測
- [x] Overfitting 檢查：參數 Heatmap、In / Out-of-sample
- [x] 第 3 階段：ML 訊號（walk-forward validation）— 10 隻股平均 AUC ≈ 0.50，未有預測能力
- [x] 第 3b 階段：AI 選股（cross-sectional）— IC 0.02（t=1.15），贏平均買晒 5%/年，但輸淨係動量
- [x] 第 3c 階段：S&P 500 範圍、4 個改良版本、開發期 / 鎖櫃期分開、只揀已入指數嘅股
  - 結論：AI IC ≈ 0，冇版本嘅超額回報 t 值過 2；價量 features 喺大型股搵唔到穩定 edge
- [x] 第 4 階段：Alpaca Paper Trading + 風控
- [x] 定時換股搬上 GitHub Actions（Mac 熄機都照行）
