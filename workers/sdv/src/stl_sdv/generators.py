"""SDV-backed generators."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from stl_core.generator import Generator, library_version, resolve_params
from stl_core.metadata import TableMetadata


class GaussianCopula(Generator):
    name = "gaussian_copula"

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)
        columns = {c: {"sdtype": metadata.columns[c].value} for c in train_df.columns}
        sdv_metadata = Metadata.load_from_dict({"tables": {"table": {"columns": columns}}})
        self._model = GaussianCopulaSynthesizer(sdv_metadata, **self._synth_kwargs())
        self._model.fit(train_df)

    def _synth_kwargs(self) -> dict:
        """Only default_distribution is forwarded (when set); SDV defaults otherwise."""
        value = self.params.get("default_distribution")
        return {} if value is None else {"default_distribution": value}

    def resolved_params(self) -> dict:
        # Library defaults: GaussianCopulaSynthesizer.__init__
        # (third_party/SDV/sdv/single_table/copulas.py:97-105).
        return resolve_params("sdv", library_version("sdv"), GaussianCopulaSynthesizer,
                              self._synth_kwargs(), skip=("metadata",))

    def sample(self, n: int) -> pd.DataFrame:
        np.random.seed(self.seed)
        return self._model.sample(num_rows=n)


def build(name: str, seed: int = 0, **params) -> Generator:
    if name != GaussianCopula.name:
        raise KeyError(f"stl_sdv does not provide {name!r}")
    return GaussianCopula(seed=seed, **params)
