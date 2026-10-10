"""Deterministic stratified TRAIN/CONTROL/TEST split."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stl_core.metadata import TableMetadata

MIN_ROWS_PER_STRATUM = 5


@dataclass
class Splits:
    train: pd.DataFrame
    control: pd.DataFrame
    test: pd.DataFrame

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for part in (self.train, self.control, self.test):
            h.update(pd.util.hash_pandas_object(part, index=False).values.tobytes())
        return h.hexdigest()


def _strata(df: pd.DataFrame, metadata: TableMetadata) -> pd.Series:
    y = df[metadata.target]
    if metadata.task == "classification":
        return y.astype(str)
    bins = min(10, max(1, len(df) // (MIN_ROWS_PER_STRATUM * 4)))
    return pd.qcut(y.rank(method="first"), q=bins, labels=False).astype(str)


def split(
    df: pd.DataFrame,
    metadata: TableMetadata,
    seed: int,
    shares: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Splits:
    strata = _strata(df, metadata)
    counts = strata.value_counts()
    too_small = counts[counts < MIN_ROWS_PER_STRATUM]
    if len(too_small):
        name, n = too_small.index[0], int(too_small.iloc[0])
        raise ValueError(
            f"target class {name!r} has {n} rows; a stratified 60/20/20 split needs at least "
            f"{MIN_ROWS_PER_STRATUM}. Merge or drop rare classes."
        )
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[], [], []]
    for _, idx in sorted(strata.groupby(strata).groups.items()):
        idx = rng.permutation(np.asarray(idx))
        n_train = int(round(len(idx) * shares[0]))
        n_control = int(round(len(idx) * shares[1]))
        parts[0].extend(idx[:n_train])
        parts[1].extend(idx[n_train : n_train + n_control])
        parts[2].extend(idx[n_train + n_control :])
    train, control, test = (
        df.loc[sorted(p)].reset_index(drop=True) for p in parts
    )
    return Splits(train=train, control=control, test=test)


def stratified_subsample(
    df: pd.DataFrame, metadata: TableMetadata, n: int, seed: int
) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    strata = _strata(df, metadata)
    rng = np.random.default_rng(seed)
    frac = n / len(df)
    keep: list[int] = []
    for _, idx in sorted(strata.groupby(strata).groups.items()):
        idx = np.asarray(idx)
        k = max(1, int(round(len(idx) * frac)))
        keep.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
    if len(keep) > n:
        keep = list(rng.choice(keep, size=n, replace=False))
    keep = sorted(keep)
    if len(keep) < n:
        rest = np.setdiff1d(df.index.to_numpy(), keep)
        keep = sorted(keep + list(rng.choice(rest, size=n - len(keep), replace=False)))
    return df.loc[keep].reset_index(drop=True)
