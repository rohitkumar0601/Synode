"""Meta-dataset assembly, LODO evaluation and predictor artifacts for RQ3."""

from __future__ import annotations

import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from stl_core.aggregate import aggregate
from stl_core.decision import recommend
from stl_core.schemas import MetricReport
from stl_runner import keys
from stl_runner.pipeline import load_jobs
from stl_runner.storage import ArtifactStore

META_COLUMNS = {"dataset_id", "openml_id", "name", "label", "n_reports", "excluded_reason"}
SCORE_PREFIX = "score_"


def dataset_id(row: dict[str, Any]) -> str:
    return str(row.get("dataset_id") or f"cc18_{row['id']}")


def run_id(row: dict[str, Any]) -> str:
    return str(row.get("run_id") or f"meta-{dataset_id(row)}")


def _load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _reports(store: ArtifactStore, rid: str) -> list[MetricReport]:
    if not store.exists(keys.run_manifest(rid)):
        return []
    out: list[MetricReport] = []
    for job in load_jobs(store, rid):
        key = keys.report(job.run_id, job.job_id)
        if store.exists(key):
            out.append(MetricReport.model_validate(store.read_json(key)))
    return out


def _feature_record(store: ArtifactStore, did: str) -> tuple[dict[str, float], str | None]:
    key = keys.dataset_metafeatures(did)
    if not store.exists(key):
        return {}, "missing metafeatures"
    rec = store.read_json(key)
    if rec.get("error"):
        return {}, str(rec["error"])
    feats = rec.get("features")
    if not isinstance(feats, dict):
        return {}, "missing metafeatures"
    return {str(k): (math.nan if v is None else float(v)) for k, v in feats.items()}, None


def build_meta_dataset(store: ArtifactStore, datasets_yaml: str | Path) -> pd.DataFrame:
    """One row per frozen dataset with meta-features, winner label and generator scores."""
    rows: list[dict[str, Any]] = []
    for item in _load_yaml(datasets_yaml).get("datasets", []):
        did = dataset_id(item)
        feats, reason = _feature_record(store, did)
        reports = _reports(store, run_id(item))
        row: dict[str, Any] = {
            "dataset_id": did,
            "openml_id": item.get("id"),
            "name": item.get("name", did),
            "n_reports": len(reports),
            "excluded_reason": reason,
        }
        row.update(feats)
        if reports:
            rec = recommend(aggregate(reports), preset="balanced")
            row["label"] = rec.winner or "none"
            for cand in rec.ranking:
                row[f"{SCORE_PREFIX}{cand.generator}"] = cand.score
        else:
            row["label"] = "none"
            row["excluded_reason"] = reason or "missing reports"
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        c for c in df.columns
        if c not in META_COLUMNS and not c.startswith(SCORE_PREFIX)
        and pd.api.types.is_numeric_dtype(df[c])
    )


def _median_impute(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    med = train.median(numeric_only=True).fillna(0.0)
    return train.fillna(med), test.fillna(med), {str(k): float(v) for k, v in med.items()}


def _mode(labels: pd.Series) -> str:
    counts = Counter(str(x) for x in labels)
    return sorted(counts, key=lambda k: (-counts[k], k))[0]


def _regret(row: pd.Series, pred: str) -> float | None:
    true = row.get(f"{SCORE_PREFIX}{row['label']}")
    got = row.get(f"{SCORE_PREFIX}{pred}")
    if true is None or got is None or pd.isna(true) or pd.isna(got):
        return None
    return float(true) - float(got)


def lodo_evaluate(meta_df: pd.DataFrame) -> dict[str, Any]:
    """Leave-one-dataset-out evaluation with train-fold imputation and baseline."""
    eligible = meta_df[meta_df["label"].notna() & (meta_df["label"] != "none")].reset_index(drop=True)
    features = _feature_columns(eligible)
    predictions: list[dict[str, Any]] = []
    if eligible.empty or not features:
        return {"features": features, "predictions": [], "accuracy": None,
                "balanced_accuracy": None, "mean_regret": None,
                "baseline_accuracy": None, "baseline_balanced_accuracy": None,
                "baseline_mean_regret": None, "n_none": int((meta_df["label"] == "none").sum())}
    for i, row in eligible.iterrows():
        train = eligible.drop(index=i)
        if train.empty:
            pred = baseline = str(row["label"])
        else:
            baseline = _mode(train["label"])
            x_train, x_test, _ = _median_impute(train[features], row[features].to_frame().T)
            y_train = train["label"].astype(str)
            if y_train.nunique() < 2:
                pred = baseline
            else:
                model = RandomForestClassifier(
                    n_estimators=300, random_state=0, class_weight="balanced")
                model.fit(x_train, y_train)
                pred = str(model.predict(x_test)[0])
        predictions.append({
            "dataset_id": row["dataset_id"],
            "true": str(row["label"]),
            "predicted": pred,
            "baseline": baseline,
            "regret": _regret(row, pred),
            "baseline_regret": _regret(row, baseline),
        })
    y = [p["true"] for p in predictions]
    pred = [p["predicted"] for p in predictions]
    base = [p["baseline"] for p in predictions]
    regrets = [p["regret"] for p in predictions if p["regret"] is not None]
    base_regrets = [p["baseline_regret"] for p in predictions if p["baseline_regret"] is not None]
    return {
        "features": features,
        "predictions": predictions,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mean_regret": float(np.mean(regrets)) if regrets else None,
        "baseline_accuracy": float(accuracy_score(y, base)),
        "baseline_balanced_accuracy": float(balanced_accuracy_score(y, base)),
        "baseline_mean_regret": float(np.mean(base_regrets)) if base_regrets else None,
        "n_none": int((meta_df["label"] == "none").sum()),
    }


def train_final(meta_df: pd.DataFrame) -> dict[str, Any]:
    train = meta_df[meta_df["label"].notna() & (meta_df["label"] != "none")].reset_index(drop=True)
    features = _feature_columns(train)
    if train.empty or not features:
        raise ValueError("no labelled meta-dataset rows with numeric features")
    x, _, medians = _median_impute(train[features], train[features].iloc[:0])
    y = train["label"].astype(str)
    model = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
    model.fit(x, y)
    return {
        "model": model,
        "features": features,
        "medians": medians,
        "labels": sorted(y.unique()),
        "metadata": {
            "n_datasets": int(len(train)),
            "trained_at": _now(),
            "sklearn_version": sklearn.__version__,
        },
    }


def save_predictor(store: ArtifactStore, artifact: dict[str, Any], summary: dict[str, Any] | None = None) -> None:
    buf = io.BytesIO()
    joblib.dump(artifact, buf)
    store.write_bytes(keys.meta_predictor(), buf.getvalue())
    meta = dict(artifact["metadata"])
    meta.update({"features": artifact["features"], "labels": artifact["labels"]})
    if summary is not None:
        meta["lodo"] = summary
    store.write_json(keys.meta_predictor_summary(), meta)


def load_predictor(store: ArtifactStore) -> dict[str, Any] | None:
    if not store.exists(keys.meta_predictor()):
        return None
    return joblib.load(io.BytesIO(store.read_bytes(keys.meta_predictor())))


def predict_features(artifact: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    cols = artifact["features"]
    med = artifact["medians"]
    row = {c: features.get(c, med.get(c, 0.0)) for c in cols}
    x = pd.DataFrame([row], columns=cols).apply(pd.to_numeric, errors="coerce")
    x = x.fillna(pd.Series(med))
    model = artifact["model"]
    probs = model.predict_proba(x)[0]
    labels = [str(c) for c in model.classes_]
    probabilities = {label: float(prob) for label, prob in zip(labels, probs, strict=True)}
    generator = max(probabilities, key=probabilities.get)
    return {
        "available": True,
        "generator": generator,
        "probabilities": probabilities,
        "model": artifact["metadata"],
    }


def write_json(store: ArtifactStore, key: str, obj: dict[str, Any]) -> None:
    store.write_bytes(key, json.dumps(obj, indent=2, allow_nan=False).encode())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
