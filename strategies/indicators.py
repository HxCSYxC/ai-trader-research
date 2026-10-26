"""技術指標。全部只用當日同之前嘅數據，冇 look-ahead。"""
import numpy as np
import pandas as pd


def sma(values, n: int) -> np.ndarray:
    return pd.Series(values).rolling(n).mean().to_numpy()


def ema(values, n: int) -> np.ndarray:
    return pd.Series(values).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(values, n: int = 14) -> np.ndarray:
    """Wilder's RSI。"""
    delta = pd.Series(values).diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss
    return (100 - 100 / (1 + rs)).to_numpy()
