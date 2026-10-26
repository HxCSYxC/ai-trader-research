"""SQLite 交易日誌：每次 run、每個目標持倉、每張單都記低。"""
import os
import sqlite3
from contextlib import contextmanager

import pandas as pd

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    account TEXT NOT NULL,
    strategy TEXT NOT NULL,
    mode TEXT NOT NULL,           -- preview / execute
    signal_date TEXT,
    equity REAL,
    n_orders INTEGER,
    status TEXT NOT NULL,         -- ok / skipped / blocked / error
    message TEXT
);
CREATE TABLE IF NOT EXISTS targets (
    run_id INTEGER, symbol TEXT, weight REAL, score REAL
);
CREATE TABLE IF NOT EXISTS orders (
    run_id INTEGER, symbol TEXT, side TEXT, dollars REAL, sell_all INTEGER, reason TEXT,
    order_id TEXT, detail TEXT, status TEXT, filled_qty REAL, filled_avg_price REAL
);
"""


@contextmanager
def connect():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def start_run(account: str, strategy: str, mode: str) -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO runs (ts, account, strategy, mode, status) VALUES (?, ?, ?, ?, 'running')",
            (pd.Timestamp.now().isoformat(timespec="seconds"), account, strategy, mode),
        )
        return cur.lastrowid


def finish_run(run_id: int, status: str, message: str = "", **fields) -> None:
    cols = ", ".join(f"{k} = ?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE runs SET status = ?, message = ?{', ' + cols if cols else ''} WHERE id = ?",
                    (status, message, *fields.values(), run_id))


def log_targets(run_id: int, weights: pd.Series, scores: pd.Series) -> None:
    with connect() as con:
        con.executemany("INSERT INTO targets VALUES (?, ?, ?, ?)",
                        [(run_id, s, float(w), float(scores.get(s, 0))) for s, w in weights.items()])


def log_order(run_id: int, o, order_id=None, detail="", status="planned", filled_qty=0.0, filled_avg_price=0.0):
    with connect() as con:
        con.execute("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, o.symbol, o.side, o.dollars, int(o.sell_all), o.reason,
                     order_id, detail, status, filled_qty, filled_avg_price))


def update_order(order_id: str, status: str, filled_qty: float, filled_avg_price: float) -> None:
    with connect() as con:
        con.execute("UPDATE orders SET status = ?, filled_qty = ?, filled_avg_price = ? WHERE order_id = ?",
                    (status, filled_qty, filled_avg_price, order_id))


def executed_this_month(account: str, today: pd.Timestamp) -> bool:
    with connect() as con:
        row = con.execute(
            "SELECT 1 FROM runs WHERE account = ? AND mode = 'execute' AND status = 'ok' AND substr(ts, 1, 7) = ?",
            (account, today.strftime("%Y-%m")),
        ).fetchone()
    return row is not None


def recent_runs(limit: int = 30) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(f"SELECT * FROM runs ORDER BY id DESC LIMIT {int(limit)}", con)


def run_orders(run_id: int) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query("SELECT * FROM orders WHERE run_id = ?", con, params=(run_id,))
