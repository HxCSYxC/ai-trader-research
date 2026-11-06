"""只會連 Alpaca 模擬盤：TradingClient 一定係 paper=True。用假 client，唔會連網絡。"""
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import config
from trader import broker

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_client(monkeypatch):
    import alpaca.trading.client

    mock = MagicMock(name="TradingClient")
    monkeypatch.setattr(alpaca.trading.client, "TradingClient", mock)
    return mock


@pytest.mark.parametrize("account", list(config.ACCOUNTS))
def test_always_paper(monkeypatch, fake_client, account):
    monkeypatch.setenv(f"{account}_API_KEY", "test-key")
    monkeypatch.setenv(f"{account}_SECRET_KEY", "test-secret")
    b = broker.AlpacaBroker(account)
    fake_client.assert_called_once_with("test-key", "test-secret", paper=True)
    assert b.client is fake_client.return_value


def test_no_keys_no_client(fake_client):
    with pytest.raises(RuntimeError, match="未設定"):
        broker.AlpacaBroker("ALPACA")
    fake_client.assert_not_called()


def test_no_way_to_turn_off_paper():
    assert list(inspect.signature(broker.AlpacaBroker.__init__).parameters) == ["self", "account"]


def test_no_live_trading_anywhere():
    """成個項目都唔可以出現 paper=False，亦唔可以用 live endpoint。"""
    for f in ROOT.rglob("*.py"):
        if ".venv" in f.parts or "tests" in f.parts:
            continue
        src = f.read_text()
        assert "paper=False" not in src, f
        assert "api.alpaca.markets" not in src.replace("paper-api.alpaca.markets", ""), f


def test_offline_broker_never_submits():
    with pytest.raises(RuntimeError):
        broker.OfflineBroker().submit("AAPL", "buy", 100.0, 10.0)
