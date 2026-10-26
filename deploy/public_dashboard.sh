#!/bin/bash
# 公開版 dashboard：唯讀 + 密碼，只綁 127.0.0.1:8502，經 Cloudflare Tunnel 對外。
cd "$(dirname "$0")/.." || exit 1
DASHBOARD_READONLY=1 exec .venv/bin/streamlit run app/dashboard.py \
  --server.port 8502 --server.address 127.0.0.1 --server.headless true \
  --client.toolbarMode viewer --browser.gatherUsageStats false
