"""GReaT (LLM-based tabular generator) via be-great, called directly.

synthcity's own `great` plugin passes a torch.device where be-great 0.0.14 expects a string.
"""

from __future__ import annotations

import random
import tempfile

import numpy as np
import pandas as pd
import torch
from stl_core.generator import (
    Generator,
    json_safe,
    library_version,
    resolve_params,
    signature_defaults,
)
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

    def __init__(self, seed: int = 0, epochs: int | None = None, llm: str = "distilgpt2",
                 batch_size: int | None = None, guided_sampling: bool = True,
                 **params) -> None:
        """``epochs`` / ``batch_size`` None (budget profile `library`) are not passed, so
        be-great's own defaults apply (epochs=100, batch_size=8: be_great/great.py:218-221).
        ``llm`` is required by be-great; distilgpt2 is synthcity's default for its great
        plugin (synthcity/plugins/generic/plugin_great.py:51-54)."""
        given = {"epochs": epochs, "batch_size": batch_size}
        super().__init__(seed=seed, llm=llm, guided_sampling=guided_sampling,
                         **{k: v for k, v in given.items() if v is not None}, **params)
        self.epochs, self.llm, self.batch_size = epochs, llm, batch_size
        self.guided_sampling = guided_sampling
        self._tmp: tempfile.TemporaryDirectory | None = None

    def _great_kwargs(self, experiment_dir: str = "<temporary directory>") -> dict:
        """kwargs for be_great.GReaT(...). Checkpoints are never reloaded (we sample from the
        in-memory model): save_strategy="no" (forwarded to TrainingArguments) keeps the
        temp dir empty."""
        kw = {"llm": self.llm, "experiment_dir": experiment_dir, "save_strategy": "no"}
        kw.update({k: v for k, v in (("epochs", self.epochs), ("batch_size", self.batch_size))
                   if v is not None})
        return kw

    def _llm_source(self) -> str:
        """What be-great's loader receives as ``llm`` (a hub id, or a pinned snapshot path)."""
        return self.llm

    def resolved_params(self) -> dict:
        from be_great import GReaT

        rp = resolve_params("be-great", library_version("be-great"), GReaT,
                            self._great_kwargs())
        sample = signature_defaults(GReaT.sample, skip=("n_samples", "device"))
        rp["params"]["sample"] = json_safe({**sample, "guided_sampling": self.guided_sampling})
        rp["overrides"] = sorted([*rp["overrides"], "sample.guided_sampling"])
        return rp

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
        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory(prefix="stl-great-")
        self._model = GReaT(**{**self._great_kwargs(self._tmp.name), "llm": self._llm_source()})
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


class GReaTQwenGenerator(GReaTGenerator):
    """GReaT on the tabular-pretrained checkpoint ``tabularisai/Qwen3-0.3B-distil``.

    be-great's own docstring calls it "pre-trained on tabular GReaT-format rows"
    (be_great/great.py:570) and every README example for it uses ``batch_size=32, epochs=5,
    fp16=True`` (be-great 0.0.14 METADATA lines 99, 123, 247-250, 260-270). For this
    checkpoint those README values are the "library default": be-great's signature defaults
    (epochs=100, batch_size=8) are generic, not tuned for it. They are still passed
    explicitly, so they are listed in ``overrides`` with the reason in ``override_notes``.
    A budget profile's epochs (equal50, smoke) win over the README's 5. ``fp16`` is only
    passed when CUDA is available (the README examples are GPU; fp16 fails on CPU).

    Hub check 2026-10-12 (HfApi().model_info): not gated, licence apache-2.0 (card
    ``license``), Qwen3ForCausalLM fine-tuned from Qwen/Qwen3-0.6B, 375.8M BF16 params,
    model.safetensors 751,651,024 B, repo 767,538,061 B. The revision is pinned:
    ``snapshot_download(revision=...)`` and the local snapshot path go to be-great's own
    ``GReaT(llm=...)`` loader (from_pretrained), so a later push to the repo cannot change
    a benchmark run.
    """

    name = "great_qwen"
    CHECKPOINT = {"id": "tabularisai/Qwen3-0.3B-distil",
                  "revision": "d778cacea5028a5fae52b981e9aed4699489a0d2",
                  "license": "apache-2.0", "size_bytes": 767538061,
                  "source": "huggingface-hub"}
    README_EPOCHS, README_BATCH_SIZE = 5, 32
    _README = "per be-great README for this checkpoint (batch_size=32, epochs=5, fp16=True)"

    def __init__(self, seed: int = 0, epochs: int | None = None,
                 batch_size: int | None = None, **params) -> None:
        self._notes = {
            "epochs": f"budget profile ({epochs})" if epochs is not None else self._README,
            "batch_size": (self._README if batch_size is None
                           else f"explicit parameter ({batch_size})"),
            "llm": "tabular-pretrained checkpoint (be_great/great.py:570)",
        }
        self.fp16 = bool(torch.cuda.is_available())
        if self.fp16:
            self._notes["fp16"] = self._README + "; passed only when CUDA is available"
        params.setdefault("llm", self.CHECKPOINT["id"])
        super().__init__(
            seed=seed,
            epochs=self.README_EPOCHS if epochs is None else epochs,
            batch_size=self.README_BATCH_SIZE if batch_size is None else batch_size,
            **params)

    def _great_kwargs(self, experiment_dir: str = "<temporary directory>") -> dict:
        kw = super()._great_kwargs(experiment_dir)
        if self.fp16:
            kw["fp16"] = True  # forwarded by be-great to TrainingArguments
        return kw

    def _llm_source(self) -> str:
        if self.llm != self.CHECKPOINT["id"]:
            return self.llm
        from huggingface_hub import snapshot_download
        return snapshot_download(self.CHECKPOINT["id"], revision=self.CHECKPOINT["revision"])

    def resolved_params(self) -> dict:
        rp = super().resolved_params()
        rp["override_notes"] = {k: v for k, v in self._notes.items() if k in rp["overrides"]}
        if self.llm == self.CHECKPOINT["id"]:
            rp["checkpoint"] = dict(self.CHECKPOINT)
        return rp


def build(name: str, seed: int = 0, **params) -> Generator:
    """Catalog entry point for the be-great generators that live in this module."""
    if name == "great_qwen":
        return GReaTQwenGenerator(seed=seed, **params)
    if name == "great":
        return GReaTGenerator(seed=seed, **params)
    raise KeyError(f"stl_synthcity.great does not provide {name!r}")
