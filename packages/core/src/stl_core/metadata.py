"""Table metadata: column types, target column, inference and preparation."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import pandas as pd
from pydantic import BaseModel, model_validator

MISSING_CATEGORY = "__missing__"


class ColumnType(str, Enum):
    numerical = "numerical"
    categorical = "categorical"
    datetime = "datetime"
    id = "id"


class TableMetadata(BaseModel):
    columns: dict[str, ColumnType]
    target: str
    task: Literal["classification", "regression"] = "classification"

    @model_validator(mode="after")
    def _check_target(self) -> TableMetadata:
        if self.target not in self.columns:
            raise ValueError(f"target column {self.target!r} is not in columns")
        if self.columns[self.target] == ColumnType.id:
            raise ValueError(f"target column {self.target!r} cannot be an id column")
        return self

    def of_type(self, t: ColumnType) -> list[str]:
        return [c for c, ct in self.columns.items() if ct == t]

    @property
    def model_columns(self) -> list[str]:
        return [c for c, ct in self.columns.items() if ct != ColumnType.id]

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.model_columns if c != self.target]

    def to_sdv_dict(self) -> dict:
        return {"columns": {c: {"sdtype": ct.value} for c, ct in self.columns.items()}}


def _infer_column(s: pd.Series, max_categories: int) -> ColumnType:
    if pd.api.types.is_datetime64_any_dtype(s):
        return ColumnType.datetime
    if pd.api.types.is_bool_dtype(s):
        return ColumnType.categorical
    nunique = s.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and nunique <= 10:
            return ColumnType.categorical
        return ColumnType.numerical
    if nunique == len(s) and len(s) > max_categories:
        return ColumnType.id
    return ColumnType.categorical


def infer_metadata(df: pd.DataFrame, target: str, max_categories: int = 50) -> TableMetadata:
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found; columns are {list(df.columns)}")
    columns = {str(c): _infer_column(df[c], max_categories) for c in df.columns}
    if columns[target] == ColumnType.id:
        columns[target] = ColumnType.categorical
    task = "classification" if columns[target] == ColumnType.categorical else "regression"
    return TableMetadata(columns=columns, target=target, task=task)


def prepare(df: pd.DataFrame, metadata: TableMetadata) -> pd.DataFrame:
    """Select metadata columns, drop rows with missing target, impute the rest.

    Raises ValueError if a numerical or datetime column has no usable value left (nothing
    to impute from).
    """
    out = df[list(metadata.columns)].copy()
    out = out[out[metadata.target].notna()]
    if len(out):
        empty = [
            c for c, ct in metadata.columns.items()
            if (ct == ColumnType.numerical and pd.to_numeric(out[c], errors="coerce").isna().all())
            or (ct == ColumnType.datetime and pd.to_datetime(out[c], errors="coerce").isna().all())
        ]
        if empty:
            raise ValueError(f"columns entirely missing (after dropping rows with missing "
                             f"target): {empty}")
    for col, ct in metadata.columns.items():
        if ct == ColumnType.numerical:
            values = pd.to_numeric(out[col], errors="coerce")
            out[col] = values.fillna(values.median()).astype(float)
        elif ct == ColumnType.categorical:
            out[col] = out[col].astype(object).where(out[col].notna(), MISSING_CATEGORY).astype(str)
            out[col] = out[col].astype(object)
        elif ct == ColumnType.datetime:
            values = pd.to_datetime(out[col], errors="coerce")
            out[col] = values.fillna(values.median())
    return out.reset_index(drop=True)
