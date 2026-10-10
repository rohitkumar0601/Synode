"""RQ3 meta-dataset, leave-one-dataset-out (LODO) evaluation and the predictor artifact.

Meta-dataset (``build_meta_dataset``): one row per dataset of the frozen YAML
(``experiments/conf/meta/datasets.yaml``), dataset ``cc18_<id>``, run ``<run_prefix>cc18_<id>``.

* meta-features: ``datasets/<id>/metafeatures.json`` ``features`` (null -> NaN);
* ``label``: the winner of the existing decision engine, ``recommend(aggregate(reports),
  "balanced")`` over the run's reports (seeds aggregated), with ``train_copy`` reports dropped
  so it is never a candidate; ``"none"`` when no candidate passes the privacy gate with a
  complete score (kept and reported, excluded from training and accuracy);
* ``score__<gen>``: the candidate's balanced score (gate-failing candidates keep their score);
  ``pass__<gen>``: whether it passed the gate.

A dataset whose meta-features are missing or failed, that has no run / no reports, or whose
run is incomplete (a planned job of the manifest, train_copy included, without a report; a
report is terminal, "done" or "failed") is not a row: it is listed in ``df.attrs["excluded"]``
as ``{"dataset_id", "reason"}`` -- an incomplete run is never labelled ``"none"``.

Regret of predicting generator ``g`` on a labelled dataset (``regret``): with P the
generators that passed the gate with a finite score, ``best = max(P)``;
``best - score(g)`` if ``g`` is in P, else ``best - min(P)`` -- picking a generator that
failed the gate (or was not scored) costs as much as the worst acceptable pick. The number
of such picks is reported separately (``n_gate_failing``) because the penalty is 0 when only
one generator passed. The same text is written to lodo.json as ``regret_definition``.
"""

from __future__ import annotations

import io
import json
import math
import warnings
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
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

NONE_LABEL = "none"
EXCLUDED_GENERATORS = frozenset({"train_copy"})
META_COLUMNS = frozenset({"dataset_id", "openml_id", "name", "label", "n_reports"})
SCORE_PREFIX = "score__"
PASS_PREFIX = "pass__"
PRESET = "balanced"
MIN_LABELLED = 3
MODEL_PARAMS = {"n_estimators": 300, "class_weight": "balanced"}
REGRET_DEFINITION = (
    "P = generators that passed the privacy gate with a finite balanced score on the dataset; "
    "regret(g) = max(P) - score(g) if g is in P, else max(P) - min(P) (a gate-failing or "
    "unscored pick costs as much as the worst acceptable pick; 0 when |P| = 1, so "
    "n_gate_failing counts such picks separately)"
)


# ---------------------------------------------------------------- meta-dataset


def _features(store: ArtifactStore, did: str) -> tuple[dict[str, float] | None, str | None]:
    key = keys.dataset_metafeatures(did)
    if not store.exists(key):
        return None, "missing metafeatures.json"
    rec = store.read_json(key)
    feats = rec.get("features")
    if not isinstance(feats, dict) or not feats:
        return None, f"metafeature extraction failed: {rec.get('error') or 'no features'}"
    return {str(k): (math.nan if v is None else float(v)) for k, v in feats.items()}, None


def _reports(store: ArtifactStore, rid: str) -> tuple[list[MetricReport] | None, str | None]:
    if not store.exists(keys.run_manifest(rid)):
        return None, f"no run {rid}"
    jobs = load_jobs(store, rid)
    reports = {}
    for job in jobs:
        key = keys.report(job.run_id, job.job_id)
        if store.exists(key):  # a report is terminal: status "done" or "failed"
            reports[job.job_id] = MetricReport.model_validate(store.read_json(key))
    if len(reports) < len(jobs):
        return None, f"incomplete run ({len(reports)}/{len(jobs)} jobs reported)"
    out = [r for r in reports.values() if r.generator not in EXCLUDED_GENERATORS]
    if not out:
        return None, f"no reports in run {rid}"
    return out, None


def build_meta_dataset(store: ArtifactStore, datasets_yaml: str | Path,
                       run_prefix: str = "meta-") -> pd.DataFrame:
    with open(datasets_yaml, encoding="utf-8") as f:
        items = (yaml.safe_load(f) or {}).get("datasets") or []
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for item in items:
        did = f"cc18_{item['id']}"
        feats, f_reason = _features(store, did)
        reports, r_reason = _reports(store, f"{run_prefix}{did}")
        reasons = [r for r in (f_reason, r_reason) if r]
        if reasons:
            excluded.append({"dataset_id": did, "reason": "; ".join(reasons)})
            continue
        rec = recommend(aggregate(reports), preset=PRESET)
        row: dict[str, Any] = {"dataset_id": did, "openml_id": item["id"],
                               "name": item.get("name", did),
                               "label": rec.winner or NONE_LABEL, "n_reports": len(reports)}
        for cand in rec.ranking:
            row[f"{SCORE_PREFIX}{cand.generator}"] = (math.nan if cand.score is None
                                                      else float(cand.score))
            row[f"{PASS_PREFIX}{cand.generator}"] = bool(cand.gate_passed)
        row.update(feats)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["excluded"] = excluded
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        c for c in df.columns
        if c not in META_COLUMNS and not str(c).startswith((SCORE_PREFIX, PASS_PREFIX))
        and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    )


def _is_none(labels: pd.Series) -> pd.Series:
    return labels.isna() | (labels.astype(str) == NONE_LABEL)


# ---------------------------------------------------------------- imputation / baseline


def fit_imputer(train_x: pd.DataFrame) -> pd.Series:
    """Per-column medians of the training rows (non-finite ignored; all-missing -> 0.0)."""
    x = train_x.astype(float).replace([np.inf, -np.inf], np.nan)
    return x.median().fillna(0.0)


def apply_imputer(x: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return x.astype(float).replace([np.inf, -np.inf], np.nan).fillna(medians)


def _mode(labels: pd.Series) -> str:
    counts = Counter(str(x) for x in labels)
    return sorted(counts, key=lambda k: (-counts[k], k))[0]


def _model(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=random_state, **MODEL_PARAMS)


# ---------------------------------------------------------------- regret


def _passing_scores(row: pd.Series) -> dict[str, float]:
    out = {}
    for col in row.index:
        if not str(col).startswith(SCORE_PREFIX):
            continue
        gen = col[len(SCORE_PREFIX):]
        score, passed = row[col], row.get(f"{PASS_PREFIX}{gen}")
        if (passed is not None and not pd.isna(passed) and bool(passed)
                and score is not None and not pd.isna(score)):
            out[gen] = float(score)
    return out


def regret(row: pd.Series, predicted: str) -> float | None:
    """See the module docstring; None if no generator passed the gate with a score."""
    passing = _passing_scores(row)
    if not passing:
        return None
    best = max(passing.values())
    return best - passing.get(predicted, min(passing.values()))


# ---------------------------------------------------------------- LODO


def _ba(y_true: list[str], y_pred: list[str]) -> float:
    with warnings.catch_warnings():  # predicted classes absent from y_true are expected here
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return float(balanced_accuracy_score(y_true, y_pred))


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def lodo_evaluate(meta_df: pd.DataFrame, random_state: int = 0) -> dict[str, Any]:
    """Leave-one-dataset-out: for each labelled dataset, fit the imputer (medians), the
    meta-model and the majority baseline on the other labelled datasets only, predict it."""
    none_mask = _is_none(meta_df["label"]) if "label" in meta_df else pd.Series(dtype=bool)
    labelled = meta_df[~none_mask]
    features = feature_columns(labelled)
    n_classes = int(labelled["label"].nunique()) if len(labelled) else 0
    out: dict[str, Any] = {
        "insufficient_data": False, "reason": None,
        "n_datasets": int(len(meta_df)), "n_labelled": int(len(labelled)),
        "n_none": int(none_mask.sum()),
        "none_datasets": [str(d) for d in meta_df.loc[none_mask, "dataset_id"]]
        if len(meta_df) else [],
        "excluded": list(meta_df.attrs.get("excluded", [])),
        "labels": sorted(str(x) for x in labelled["label"].unique()) if len(labelled) else [],
        "features": features,
        "model_params": {"type": "RandomForestClassifier", "random_state": random_state,
                         **MODEL_PARAMS},
        "regret_definition": REGRET_DEFINITION,
        "predictions": [],
        "accuracy": None, "balanced_accuracy": None, "mean_regret": None,
        "n_gate_failing": None,
        "baseline_accuracy": None, "baseline_balanced_accuracy": None,
        "baseline_mean_regret": None, "baseline_n_gate_failing": None,
    }
    if len(labelled) < MIN_LABELLED:
        out.update(insufficient_data=True,
                   reason=f"insufficient data: {len(labelled)} labelled dataset(s), "
                          f"LODO needs at least {MIN_LABELLED}")
        return out
    if n_classes < 2:
        out.update(insufficient_data=True,
                   reason="insufficient data: every labelled dataset has the same winner "
                          "class")
        return out
    if not features:
        out.update(insufficient_data=True, reason="insufficient data: no numeric meta-features")
        return out

    preds = []
    for idx in labelled.index:
        train = labelled.drop(index=idx)
        test = labelled.loc[[idx]]
        y_train = train["label"].astype(str)
        baseline = _mode(y_train)
        medians = fit_imputer(train[features])
        if y_train.nunique() < 2:
            predicted = y_train.iloc[0]  # single class in the training folds
        else:
            model = _model(random_state).fit(apply_imputer(train[features], medians), y_train)
            predicted = str(model.predict(apply_imputer(test[features], medians))[0])
        row = labelled.loc[idx]
        passing = _passing_scores(row)
        preds.append({
            "dataset_id": str(row["dataset_id"]), "true": str(row["label"]),
            "predicted": predicted, "baseline": baseline,
            "regret": regret(row, predicted), "baseline_regret": regret(row, baseline),
            "predicted_passed_gate": predicted in passing,
            "baseline_passed_gate": baseline in passing,
        })
    y = [p["true"] for p in preds]
    yp = [p["predicted"] for p in preds]
    yb = [p["baseline"] for p in preds]
    out.update(
        predictions=preds,
        accuracy=float(accuracy_score(y, yp)), balanced_accuracy=_ba(y, yp),
        mean_regret=_mean([p["regret"] for p in preds]),
        n_gate_failing=sum(not p["predicted_passed_gate"] for p in preds),
        baseline_accuracy=float(accuracy_score(y, yb)), baseline_balanced_accuracy=_ba(y, yb),
        baseline_mean_regret=_mean([p["baseline_regret"] for p in preds]),
        baseline_n_gate_failing=sum(not p["baseline_passed_gate"] for p in preds),
    )
    return out


# ---------------------------------------------------------------- final predictor


def _version(dist: str) -> str | None:
    try:
        return importlib_metadata.version(dist)
    except importlib_metadata.PackageNotFoundError:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def train_final(meta_df: pd.DataFrame, random_state: int = 0) -> dict[str, Any]:
    """Fit the meta-model on every labelled dataset. Raises ValueError if there is none."""
    none_mask = _is_none(meta_df["label"])
    train = meta_df[~none_mask]
    features = feature_columns(train)
    if train.empty or not features:
        raise ValueError("no labelled meta-dataset rows with numeric meta-features to train on")
    medians = fit_imputer(train[features])
    y = train["label"].astype(str)
    model = _model(random_state).fit(apply_imputer(train[features], medians), y)
    return {
        "model": model,
        "features": features,
        "medians": {k: float(v) for k, v in medians.items()},
        "labels": sorted(y.unique()),
        "metadata": {
            "n_datasets": int(len(train)), "n_none": int(none_mask.sum()),
            "trained_at": _now(), "preset": PRESET,
            "model_params": {"type": "RandomForestClassifier", "random_state": random_state,
                      **MODEL_PARAMS},
            "sklearn_version": sklearn.__version__, "pymfe_version": _version("pymfe"),
        },
    }


def write_json(store: ArtifactStore, key: str, obj: dict[str, Any]) -> None:
    store.write_bytes(key, json.dumps(obj, indent=2, allow_nan=False).encode())


def save_artifact(store: ArtifactStore, artifact: dict[str, Any],
                  lodo: dict[str, Any] | None = None) -> None:
    """meta/predictor.joblib (the artifact) + meta/predictor.json (summary, no model)."""
    buf = io.BytesIO()
    joblib.dump(artifact, buf)
    store.write_bytes(keys.meta_predictor(), buf.getvalue())
    summary = {**artifact["metadata"], "features": artifact["features"],
               "labels": artifact["labels"], "medians": artifact["medians"]}
    if lodo is not None:
        summary["lodo"] = {k: v for k, v in lodo.items() if k not in ("predictions", "features")}
    write_json(store, keys.meta_predictor_summary(), summary)


def load_artifact(store: ArtifactStore) -> dict[str, Any] | None:
    if not store.exists(keys.meta_predictor()):
        return None
    return joblib.load(io.BytesIO(store.read_bytes(keys.meta_predictor())))


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def predict(artifact: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Best-generator prediction for one dataset's meta-features (missing / non-finite /
    unknown values -> the training medians; extra keys ignored)."""
    cols = artifact["features"]
    x = pd.DataFrame([[_as_float(features.get(c)) for c in cols]], columns=cols)
    x = apply_imputer(x, pd.Series(artifact["medians"], dtype=float))
    model = artifact["model"]
    probs = model.predict_proba(x)[0]
    probabilities = {str(c): float(p) for c, p in zip(model.classes_, probs, strict=True)}
    return {"generator": max(probabilities, key=probabilities.get),
            "probabilities": probabilities}


# Names used by stl_meta.tasks (Task 5).
save_predictor = save_artifact
load_predictor = load_artifact


def predict_features(artifact: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    return {"available": True, **predict(artifact, features), "model": artifact["metadata"]}
