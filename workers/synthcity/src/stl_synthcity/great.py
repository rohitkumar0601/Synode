"""GReaT (LLM-based tabular generator) via be-great, called directly.

synthcity's own `great` plugin passes a torch.device where be-great 0.0.14 expects a string.
"""

from __future__ import annotations

import random
import tempfile

import numpy as np
import pandas as pd
import torch
from stl_core.generator import Generator
from stl_core.metadata import ColumnType, TableMetadata

from stl_synthcity.generators import capture_train_stats, restore_types


class GReaTGenerator(Generator):
    """GReaT fine-tuned LLM generator.

    `guided_sampling` defaults to True: be-great's default ("legacy") sampler discards every
    row the LLM does not emit in valid "col is value" form and loops until n valid rows exist.
    At low epochs that yields ~1 valid row per 100 draws (measured on CPU, 1 epoch, 120 rows:
    the test did not finish in 34 min; 3 rows took ~19 min). Guided sampling prompts
    feature-by-feature and returns exactly n rows (20 rows in ~88 s).
    """

    name = "great"

    def __init__(self, seed: int = 0, epochs: int = 50, llm: str = "distilgpt2",
                 batch_size: int = 8, guided_sampling: bool = True, **params) -> None:
        super().__init__(seed=seed, epochs=epochs, llm=llm, batch_size=batch_size,
                         guided_sampling=guided_sampling, **params)
        self.epochs, self.llm, self.batch_size = epochs, llm, batch_size
        self.guided_sampling = guided_sampling
        self._tmp: tempfile.TemporaryDirectory | None = None

    def _seed_all(self) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        from be_great import GReaT
        self._seed_all()
        self._columns = list(train_df.columns)
        self._dtypes = train_df.dtypes.to_dict()
        self._ids = [c for c in self._columns if metadata.columns.get(c) == ColumnType.id]
        self._cats = {c for c in self._columns if metadata.columns.get(c) == ColumnType.categorical}
        self._medians, self._cat_values = capture_train_stats(train_df, self._cats)
        # Checkpoints are never reloaded (we sample from the in-memory model): save_strategy="no"
        # (be-great forwards train kwargs to TrainingArguments) keeps the temp dir empty.
        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory(prefix="stl-great-")
        self._model = GReaT(llm=self.llm, epochs=self.epochs, batch_size=self.batch_size,
                            experiment_dir=self._tmp.name, save_strategy="no")
        self._model.fit(train_df.drop(columns=self._ids))

    def sample(self, n: int) -> pd.DataFrame:
        self._seed_all()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        out = self._model.sample(n_samples=n, device=device,
                                 guided_sampling=self.guided_sampling)
        if len(out) < n:
            raise RuntimeError(f"GReaT returned {len(out)} rows, expected {n}")
        out = out.iloc[:n]
        return restore_types(out, self._columns, self._dtypes, self._cats, self._ids,
                             self._medians, self._cat_values)
