"""Dataset meta-features (RQ3) via pymfe.

``extract`` describes one table with pymfe's ``general``, ``statistical`` and
``info-theory`` groups (summaries ``mean`` and ``sd``), computed on a deterministic
stratified sample of at most ``max_rows`` rows.

Input preparation, done on the sample only (pymfe needs no NaN in numeric arrays):

* ``id`` columns are dropped; rows with a missing target are dropped;
* numerical columns are cast to float and missing values imputed with the sample median
  (0.0 if the column has no value at all);
* datetime columns become numeric (nanoseconds since the epoch), imputed the same way;
* categorical columns become strings, missing values imputed with the sample mode, and are
  passed to pymfe as ``cat_cols`` (indices), so they are treated as nominal attributes.

Registered datasets are already imputed by ``stl_core.metadata.prepare``; the imputation
here only matters for raw tables.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pymfe
from pymfe.mfe import MFE
from stl_core.metadata import ColumnType, TableMetadata
from stl_core.split import stratified_subsample
from stl_runner import keys
from stl_runner.pipeline import ensure_splits, load_metadata
from stl_runner.storage import ArtifactStore

GROUPS = ("general", "statistical", "info-theory")
SUMMARY = ("mean", "sd")


class MetaFeatureError(RuntimeError):
    """pymfe produced no meta-features at all (fit/extract failed, timed out or was empty)."""


def sample_rows(df: pd.DataFrame, metadata: TableMetadata, max_rows: int,
                seed: int) -> pd.DataFrame:
    """At most ``max_rows`` rows, stratified by the target (every class keeps at least one
    row) and deterministic for ``seed``; a plain random sample if stratification fails."""
    df = df[df[metadata.target].notna()].reset_index(drop=True)
    if len(df) <= max_rows:
        return df
    try:
        return stratified_subsample(df, metadata, max_rows, seed)
    except ValueError:  # e.g. too few rows to bin a regression target
        return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _prepare(df: pd.DataFrame, metadata: TableMetadata) -> tuple[np.ndarray, np.ndarray,
                                                                  list[int]]:
    cols = [c for c in metadata.feature_columns if c in df.columns]
    X = pd.DataFrame(index=df.index)
    cat_cols: list[int] = []
    for i, c in enumerate(cols):
        kind = metadata.columns[c]
        if kind == ColumnType.categorical:
            s = df[c].astype(object)
            mode = s.dropna().astype(str).mode()
            X[c] = s.where(s.notna(), mode.iloc[0] if len(mode) else "missing").astype(str)
            cat_cols.append(i)
        else:
            if kind == ColumnType.datetime:
                dt = pd.to_datetime(df[c], errors="coerce")
                s = pd.Series(dt.astype("int64").astype(float), index=df.index).where(dt.notna())
            else:
                s = pd.to_numeric(df[c], errors="coerce").astype(float)
            s = s.where(np.isfinite(s))
            median = s.median()
            X[c] = s.fillna(0.0 if pd.isna(median) else median)
    y = df[metadata.target]
    y = y.astype(str) if metadata.task == "classification" else y.astype(float)
    return X.to_numpy(dtype=object), y.to_numpy(), cat_cols


def _to_float(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def _run_pymfe(X: np.ndarray, y: np.ndarray, cat_cols: list[int], seed: int):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        mfe = MFE(groups=list(GROUPS), summary=list(SUMMARY), random_state=seed,
                  suppress_warnings=True)
        mfe.fit(X, y, cat_cols=cat_cols, suppress_warnings=True)
        return mfe.extract(suppress_warnings=True)


def extract(df: pd.DataFrame, target: str, metadata: TableMetadata, seed: int,
            max_rows: int = 5000, timeout_s: float | None = None) -> dict[str, float]:
    """Meta-features of ``df`` as ``{name: float}``; a feature pymfe could not compute (or
    that is not finite) is NaN. Raises MetaFeatureError only if pymfe fails entirely.

    ``timeout_s`` runs pymfe in a worker thread and gives up after that many seconds. The
    thread cannot be killed: it keeps running in the background until pymfe returns, and
    its result is discarded.
    """
    if target != metadata.target:
        raise ValueError(f"target {target!r} does not match metadata target "
                         f"{metadata.target!r}")
    X, y, cat_cols = _prepare(sample_rows(df, metadata, max_rows, seed), metadata)
    try:
        if timeout_s is None:
            names, values = _run_pymfe(X, y, cat_cols, seed)
        else:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(_run_pymfe, X, y, cat_cols, seed)
                names, values = future.result(timeout=timeout_s)
            finally:
                pool.shutdown(wait=False)
    except concurrent.futures.TimeoutError as exc:
        raise MetaFeatureError(f"meta-feature extraction timed out after {timeout_s}s") from exc
    except Exception as exc:
        raise MetaFeatureError(f"pymfe failed: {type(exc).__name__}: {exc}") from exc
    out = {str(n): _to_float(v) for n, v in zip(names, values, strict=True)}
    if not out:
        raise MetaFeatureError("pymfe returned no meta-features")
    return out


def metafeatures_key(dataset_id: str) -> str:
    return keys.dataset_metafeatures(keys.validate_component(dataset_id, "dataset id"))


def compute_for_dataset(store: ArtifactStore, dataset_id: str, seed: int = 0,
                        max_rows: int = 5000, timeout_s: float | None = None) -> dict:
    """Meta-features of a registered dataset's TRAIN split for ``seed``, written to
    ``datasets/<id>/metafeatures.json`` and returned. NaN features are stored as null.

    If extraction fails entirely, ``{"error": ..., "seed", "pymfe_version", "groups",
    "computed_at"}`` (no ``features`` key) is written instead, so the dataset can be
    excluded with a recorded reason, and MetaFeatureError is re-raised."""
    metadata = load_metadata(store, dataset_id)
    train = ensure_splits(store, dataset_id, seed).train
    record = {"seed": seed, "pymfe_version": pymfe.__version__, "groups": list(GROUPS),
              "summary": list(SUMMARY)}
    try:
        feats = extract(train, metadata.target, metadata, seed, max_rows=max_rows,
                        timeout_s=timeout_s)
    except MetaFeatureError as exc:
        record.update(error=str(exc), computed_at=_now())
        _write(store, metafeatures_key(dataset_id), record)
        raise
    record = {"features": {k: None if math.isnan(v) else v for k, v in feats.items()},
              "n_rows_used": int(min(len(train), max_rows)), **record, "computed_at": _now()}
    _write(store, metafeatures_key(dataset_id), record)
    return record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(store: ArtifactStore, key: str, obj: dict) -> None:
    store.write_bytes(key, json.dumps(obj, indent=2, allow_nan=False).encode())
