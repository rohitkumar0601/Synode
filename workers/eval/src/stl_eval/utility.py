"""Utility: train on synthetic, test on real (TSTR) and importance agreement."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from stl_core.metadata import ColumnType, TableMetadata
from stl_core.schemas import UtilityMetrics
from xgboost import XGBClassifier

from stl_eval.preprocessing import numeric_view

MODELS = ("logreg", "random_forest", "xgboost")


def _model(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                         random_state=seed, n_jobs=-1, eval_metric="logloss")


def _fit_predict(name, seed, X, y, X_test) -> tuple[np.ndarray, object]:
    present = np.unique(y)  # remap to contiguous labels (XGBoost requires it)
    model = _model(name, seed).fit(X, np.searchsorted(present, y))
    return present[model.predict(X_test)], model


def tstr(real_train: pd.DataFrame, synth: pd.DataFrame, test: pd.DataFrame,
         metadata: TableMetadata, seed: int) -> UtilityMetrics:
    if metadata.task != "classification":
        raise NotImplementedError("utility metrics support classification targets only")
    target, feats = metadata.target, metadata.feature_columns
    labels = {c: i for i, c in enumerate(sorted(real_train[target].astype(str).unique()))}

    def xy(df: pd.DataFrame):
        df = numeric_view(df, metadata)
        keep = df[target].astype(str).isin(labels)
        return df.loc[keep, feats], df.loc[keep, target].astype(str).map(labels).to_numpy()

    cats = [c for c in feats if metadata.columns[c] == ColumnType.categorical]
    nums = [c for c in feats if c not in cats]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=50), cats),
        ("num", StandardScaler(), nums),
    ])
    Xr_df, yr = xy(real_train)
    Xs_df, ys = xy(synth)
    Xt_df, yt = xy(test)
    pre.fit(Xr_df)
    Xr, Xs, Xt = pre.transform(Xr_df), pre.transform(Xs_df), pre.transform(Xt_df)

    real_scores: dict[str, float | None] = {}
    ratios: dict[str, float | None] = {}
    importances = {}
    single_class = len(np.unique(ys)) < 2
    for name in MODELS:
        pred_r, model_r = _fit_predict(name, seed, Xr, yr, Xt)
        real_f1 = float(f1_score(yt, pred_r, average="macro"))
        real_scores[name] = real_f1
        if single_class:
            ratios[name] = 0.0
            continue
        pred_s, model_s = _fit_predict(name, seed, Xs, ys, Xt)
        synth_f1 = float(f1_score(yt, pred_s, average="macro"))
        ratios[name] = synth_f1 / real_f1 if real_f1 > 0 else None
        if name == "random_forest":
            importances = {"real": model_r.feature_importances_,
                           "synth": model_s.feature_importances_}
    rho = None
    if importances:
        rho = float(spearmanr(importances["real"], importances["synth"]).statistic)
        rho = None if np.isnan(rho) else rho
    return UtilityMetrics(tstr=ratios, real_scores=real_scores, importance_spearman=rho)
