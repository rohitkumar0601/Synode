"""Align synthetic tables to real ones and encode them for distances and models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from stl_core.metadata import MISSING_CATEGORY, ColumnType, TableMetadata


def _as_number(value: str) -> float | None:
    try:
        x = float(value)
    except ValueError:
        return None
    return x if np.isfinite(x) else None


def _match_numeric_categories(values: pd.Series, real: pd.Series) -> pd.Series:
    """Map synthetic categories unseen in real data but numerically equal to a real
    category (e.g. "1.0" vs "1") onto that real category's string."""
    real_cats = set(real.astype(str).unique())
    by_number: dict[float, str] = {}
    for cat in sorted(real_cats):
        x = _as_number(cat)
        if x is not None:
            by_number.setdefault(x, cat)
    if not by_number:
        return values
    mapping = {}
    for v in values.unique():
        if v not in real_cats and (x := _as_number(v)) is not None and x in by_number:
            mapping[v] = by_number[x]
    return values.replace(mapping) if mapping else values


def unseen_category_shares(real: pd.DataFrame, synth: pd.DataFrame,
                           metadata: TableMetadata) -> dict[str, float]:
    """Per categorical column, the share of (aligned) synthetic values absent from real."""
    return {
        col: float((~synth[col].astype(str).isin(set(real[col].astype(str)))).mean())
        for col in metadata.model_columns
        if metadata.columns[col] == ColumnType.categorical and len(synth)
    }


def align(real: pd.DataFrame, synth: pd.DataFrame, metadata: TableMetadata) -> pd.DataFrame:
    cols = metadata.model_columns
    missing = [c for c in cols if c not in synth.columns]
    if missing:
        raise ValueError(f"synthetic data is missing columns: {missing}")
    out = synth[cols].copy()
    for col in cols:
        ct = metadata.columns[col]
        if ct == ColumnType.numerical:
            values = pd.to_numeric(out[col], errors="coerce")
            if values.isna().mean() > 0.5:
                raise ValueError(f"synthetic column {col!r} is not numeric")
            out[col] = values.fillna(real[col].median()).astype(float)
        elif ct == ColumnType.categorical:
            values = out[col].where(out[col].notna(), MISSING_CATEGORY).astype(str)
            out[col] = _match_numeric_categories(values, real[col]).astype(object)
        elif ct == ColumnType.datetime:
            values = pd.to_datetime(out[col], errors="coerce")
            out[col] = values.fillna(real[col].median())
    return out.reset_index(drop=True)


def numeric_view(df: pd.DataFrame, metadata: TableMetadata) -> pd.DataFrame:
    out = df.copy()
    for col in metadata.of_type(ColumnType.datetime):
        if col in out:
            out[col] = pd.to_datetime(out[col]).astype("datetime64[ns]").astype("int64") / 1e9
    return out


class DistanceEncoder:
    """Min-max numericals and one-hot categoricals, both fitted on a reference table.

    Numericals are scaled by the reference range only (not clipped), so values outside it
    fall outside [0, 1]; categories unseen in the reference encode as all zeros.
    """

    def __init__(self, metadata: TableMetadata) -> None:
        self.metadata = metadata
        cols = metadata.model_columns
        self.num_cols = [c for c in cols if metadata.columns[c] != ColumnType.categorical]
        self.cat_cols = [c for c in cols if metadata.columns[c] == ColumnType.categorical]

    def fit(self, ref: pd.DataFrame) -> DistanceEncoder:
        ref = numeric_view(ref, self.metadata)
        self.lo = {c: float(ref[c].min()) for c in self.num_cols}
        self.span = {c: (float(ref[c].max()) - self.lo[c]) or 1.0 for c in self.num_cols}
        self.categories = {c: sorted(ref[c].astype(str).unique()) for c in self.cat_cols}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = numeric_view(df, self.metadata)
        blocks = [
            ((df[c].to_numpy(float) - self.lo[c]) / self.span[c])[:, None]
            for c in self.num_cols
        ]
        for c in self.cat_cols:
            codes = pd.Categorical(df[c].astype(str), categories=self.categories[c]).codes
            onehot = np.zeros((len(df), len(self.categories[c])))
            hit = codes >= 0
            onehot[np.flatnonzero(hit), codes[hit]] = 1.0
            blocks.append(onehot)
        return np.hstack(blocks).astype(np.float32)
