"""Walk-forward：每次只用「當時已知」嘅數據 train，預測之後一段，再向前推。"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

MODELS = {
    "Logistic Regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)
    ),
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
        early_stopping=False, random_state=0,
    ),
}


def walk_forward_proba(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    horizon: int,
    min_train: int = config.ML_MIN_TRAIN_DAYS,
    step: int = config.ML_RETRAIN_EVERY,
) -> pd.Series:
    """返回每日嘅 out-of-sample P(升)。頭 min_train 日冇預測（NaN）。"""
    probs = pd.Series(np.nan, index=X.index, name="Prob")
    has_x = X.notna().all(axis=1).to_numpy()
    first = int(np.argmax(has_x))

    for start in range(first + min_train, len(X), step):
        end = min(start + step, len(X))
        # Purge：第 i 行嘅 label 要等到第 i+horizon+1 日先知，所以只用 i < start-horizon-1
        train = slice(first, start - horizon - 1)
        X_tr, y_tr = X.iloc[train], y.iloc[train]
        ok = X_tr.notna().all(axis=1) & y_tr.notna()
        if y_tr[ok].nunique() < 2:
            continue

        model = MODELS[model_name]()
        model.fit(X_tr[ok], y_tr[ok])

        X_te = X.iloc[start:end]
        te_ok = X_te.notna().all(axis=1)
        if te_ok.any():
            probs.loc[X_te.index[te_ok]] = model.predict_proba(X_te[te_ok])[:, 1]

    return probs


def prediction_metrics(probs: pd.Series, y: pd.Series) -> dict:
    """預測質素。AUC 0.5＝擲毫，>0.55 先算有少少料。"""
    ok = probs.notna() & y.notna()
    p, t = probs[ok], y[ok]
    return {
        "預測日數": int(ok.sum()),
        "準確率 %": ((p > 0.5) == (t == 1)).mean() * 100,
        "永遠估升嘅準確率 %": t.mean() * 100,
        "AUC": roc_auc_score(t, p) if t.nunique() == 2 else float("nan"),
    }
