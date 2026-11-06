"""所有測試只用合成數據：冇網絡、冇 Alpaca key、唔會落單。"""
import os
import socket

import pytest

import config


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    # config 會 load .env；測試入面清走所有 key 同 kill switch 設定，免得用到真帳戶
    for k in list(os.environ):
        if k.startswith("ALPACA") or k in ("TRADING_DISABLED", "DASHBOARD_PASSWORD"):
            monkeypatch.delenv(k)
    monkeypatch.setattr(config, "KILL_SWITCH_FILE", str(tmp_path / "KILL_SWITCH"))

    def _no_network(*args, **kwargs):
        raise RuntimeError("測試唔准連網絡")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
