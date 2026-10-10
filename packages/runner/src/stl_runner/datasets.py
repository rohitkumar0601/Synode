"""Load benchmark datasets from OpenML or local files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def load_dataset(cfg: Mapping) -> tuple[pd.DataFrame, str]:
    source = cfg.get("source")
    if source == "openml":
        from sklearn.datasets import fetch_openml

        if cfg.get("openml_id") is not None:
            kwargs = {"data_id": int(cfg["openml_id"])}
        else:
            kwargs = {"name": cfg["openml_name"], "version": cfg.get("version", "active")}
        df = fetch_openml(as_frame=True, parser="auto", **kwargs).frame
    elif source == "file":
        path = Path(cfg["path"])
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    else:
        raise ValueError(f"unknown dataset source {source!r}; use 'openml' or 'file'")
    df = df.drop(columns=list(cfg.get("drop") or []))
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(object)
    return df.reset_index(drop=True), str(cfg["target"])
