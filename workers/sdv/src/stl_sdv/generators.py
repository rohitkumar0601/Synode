"""SDV-backed generators."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
from stl_core.generator import Generator, library_version, resolve_params
from stl_core.metadata import TableMetadata


def _sdv_metadata(train_df: pd.DataFrame, metadata: TableMetadata) -> Metadata:
    columns = {c: {"sdtype": metadata.columns[c].value} for c in train_df.columns}
    return Metadata.load_from_dict({"tables": {"table": {"columns": columns}}})


class GaussianCopula(Generator):
    name = "gaussian_copula"

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)
        self._model = GaussianCopulaSynthesizer(_sdv_metadata(train_df, metadata),
                                                **self._synth_kwargs())
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


class SdvDeep(Generator):
    """SDV's CTGANSynthesizer / TVAESynthesizer (ctgan package) with SDV's shipped defaults.

    Defaults come from the synthesizer signatures (SDV 1.38.5,
    third_party/SDV/sdv/single_table/ctgan.py): CTGAN :169-189 (epochs=300 :186,
    batch_size=500 :182, pac=10, enable_gpu=True :188); TVAE :372-386 (epochs=300 :383,
    batch_size=500 :381, loss_factor=2, enable_gpu=True :385). Only ``epochs`` is ever
    passed, and only when the budget profile sets it (library profile: None -> SDV's 300).
    ``enable_gpu`` keeps SDV's default True; ctgan uses CUDA when available and falls back
    to CPU otherwise (the sdv env ships CPU-only torch).

    Seeding: python/numpy/torch global RNGs are seeded before fit (CTGAN/TVAE training
    draws from them), and SDV's own sampling RNG is set to ``seed`` before every sample()
    call (SDV otherwise uses its FIXED_RNG_SEED, base.py:65). On CPU this makes fit+sample
    reproducible for a fixed seed; CUDA kernels may still be nondeterministic.
    """

    _SYNTHESIZERS = {"sdv_ctgan": CTGANSynthesizer, "sdv_tvae": TVAESynthesizer}

    def __init__(self, name: str, seed: int = 0, **params) -> None:
        if name not in self._SYNTHESIZERS:
            raise KeyError(f"stl_sdv does not provide {name!r}")
        super().__init__(seed=seed, **params)
        self.name = name
        self._cls = self._SYNTHESIZERS[name]

    def _synth_kwargs(self) -> dict:
        epochs = self.params.get("epochs")
        return {} if epochs is None else {"epochs": int(epochs)}

    def _seed_all(self) -> None:
        import torch

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        self._seed_all()
        self._model = self._cls(_sdv_metadata(train_df, metadata), **self._synth_kwargs())
        self._model.fit(train_df)

    def resolved_params(self) -> dict:
        # ``cuda`` is SDV's deprecated alias of enable_gpu (None = unused); not a parameter.
        return resolve_params("sdv", library_version("sdv"), self._cls, self._synth_kwargs(),
                              skip=("metadata", "cuda"))

    def sample(self, n: int) -> pd.DataFrame:
        self._seed_all()
        # SDV's documented reproducibility hook (BaseSingleTableSynthesizer, base.py:882).
        self._model._set_random_state(self.seed)
        return self._model.sample(num_rows=n)


def build(name: str, seed: int = 0, **params) -> Generator:
    if name == GaussianCopula.name:
        return GaussianCopula(seed=seed, **params)
    if name in SdvDeep._SYNTHESIZERS:
        return SdvDeep(name, seed=seed, **params)
    raise KeyError(f"stl_sdv does not provide {name!r}")
