"""SDV-backed generators."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from stl_core.generator import Generator
from stl_core.metadata import TableMetadata


class GaussianCopula(Generator):
    name = "gaussian_copula"

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)
        columns = {c: {"sdtype": metadata.columns[c].value} for c in train_df.columns}
        sdv_metadata = Metadata.load_from_dict({"tables": {"table": {"columns": columns}}})
        kwargs = {}
        if "default_distribution" in self.params:
            kwargs["default_distribution"] = self.params["default_distribution"]
        self._model = GaussianCopulaSynthesizer(sdv_metadata, **kwargs)
        self._model.fit(train_df)

    def sample(self, n: int) -> pd.DataFrame:
        np.random.seed(self.seed)
        return self._model.sample(num_rows=n)


def build(name: str, seed: int = 0, **params) -> Generator:
    if name != GaussianCopula.name:
        raise KeyError(f"stl_sdv does not provide {name!r}")
    return GaussianCopula(seed=seed, **params)
