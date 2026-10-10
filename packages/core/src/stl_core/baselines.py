"""Sanity baselines: train-copy (privacy ceiling breaker) and independent marginals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stl_core.generator import Generator
from stl_core.metadata import ColumnType, TableMetadata


class TrainCopy(Generator):
    """Returns real training rows. Perfect fidelity, worst privacy."""

    name = "train_copy"

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        self._train = train_df.reset_index(drop=True)

    def sample(self, n: int) -> pd.DataFrame:
        replace = n > len(self._train)
        return self._train.sample(n=n, replace=replace, random_state=self.seed).reset_index(
            drop=True
        )


class IndependentMarginals(Generator):
    """Samples each column independently from its marginal distribution.

    Numerical columns use inverse-CDF interpolation between order statistics, so sampled
    values are new numbers rather than copies of training values (copied unique values
    would be flagged by singling-out attacks and make this baseline fail privacy for the
    wrong reason). Other columns are sampled by observed frequency.
    """

    name = "independent_marginals"

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        self._columns = list(train_df.columns)
        self._numeric = {c for c in self._columns
                         if metadata.columns.get(c) == ColumnType.numerical}
        self._values = {c: train_df[c].to_numpy() for c in self._columns}
        self._dtypes = train_df.dtypes

    def sample(self, n: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        data = {}
        for c in self._columns:
            if c in self._numeric:
                data[c] = np.quantile(self._values[c].astype(float), rng.random(n))
            else:
                data[c] = rng.choice(self._values[c], size=n, replace=True)
        return pd.DataFrame(data).astype(self._dtypes.to_dict())


_BUILDERS = {TrainCopy.name: TrainCopy, IndependentMarginals.name: IndependentMarginals}


def build(name: str, seed: int = 0, **params) -> Generator:
    return _BUILDERS[name](seed=seed, **params)
