"""The interface every synthetic-data generator implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from stl_core.metadata import TableMetadata


class Generator(ABC):
    name: str = "generator"

    def __init__(self, seed: int = 0, **params) -> None:
        self.seed = seed
        self.params = params

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None: ...

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame: ...
