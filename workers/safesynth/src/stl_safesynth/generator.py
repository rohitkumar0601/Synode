"""NVIDIA Safe-Synthesizer generator wrapper.

Uses SmolLM3-3B (HuggingFaceTB/SmolLM3-3B, Apache-2.0, ~3.08B params, 6.18 GB) with QLoRA
fine-tuning. DP variant uses DP-SGD (Opacus) at epsilon=8.0 (library default).
Runs on Linux with an NVIDIA A100 GPU (or larger); CPU execution is for config validation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stl_core.generator import Generator
from stl_core.metadata import TableMetadata

CHECKPOINT = {
    "id": "HuggingFaceTB/SmolLM3-3B",
    "license": "apache-2.0",
    "size_bytes": 6178163042,
    "parameters": 3075098624,
    "source": "huggingface-hub",
}


class SafeSynthesizerGenerator(Generator):
    """NVIDIA Safe-Synthesizer generator for standard and DP-SGD fine-tuning."""

    def __init__(
        self,
        name: str = "nss_smollm3",
        seed: int = 0,
        pretrained_model: str = "HuggingFaceTB/SmolLM3-3B",
        dp_enabled: bool = False,
        epsilon: float = 8.0,
        epochs: int | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            seed=seed,
            pretrained_model=pretrained_model,
            dp_enabled=dp_enabled,
            epsilon=epsilon,
            epochs=epochs,
            **params,
        )
        self.name = name
        self.pretrained_model = pretrained_model
        self.dp_enabled = dp_enabled or (name == "nss_smollm3_dp_eps8")
        self.epsilon = epsilon
        self.epochs = epochs
        self._builder = None
        self._results = None

    def resolved_params(self) -> dict[str, Any]:
        """Record resolved parameters according to library signature defaults."""
        params: dict[str, Any] = {
            "pretrained_model": self.pretrained_model,
            "peft_implementation": "QLORA",
            "lora_r": 32,
            "lora_alpha_over_r": 1.0,
            "dp_enabled": self.dp_enabled,
        }
        if self.dp_enabled:
            params["epsilon"] = self.epsilon
            params["delta"] = "auto"
            params["per_sample_max_grad_norm"] = 1.0
        if self.epochs is not None:
            params["epochs"] = self.epochs

        return {
            "library": "nemo-safe-synthesizer",
            "library_version": "0.1.0",
            "params": params,
            "overrides": ["dp_enabled", "epsilon"] if self.dp_enabled else [],
            "checkpoint": CHECKPOINT,
        }

    def build_config(self) -> Any:
        """Construct the SafeSynthesizerParameters config without importing heavy backends."""
        from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters

        training_kwargs: dict[str, Any] = {
            "pretrained_model": self.pretrained_model,
        }

        dp_kwargs: dict[str, Any] = {}
        if self.dp_enabled:
            dp_kwargs["dp_enabled"] = True
            dp_kwargs["epsilon"] = self.epsilon
            dp_kwargs["delta"] = "auto"
        else:
            dp_kwargs["dp_enabled"] = False

        return SafeSynthesizerParameters(
            training=training_kwargs,
            privacy=dp_kwargs,
        )

    def build_builder(self, df: pd.DataFrame, save_path: str | Path | None = None) -> Any:
        """Construct the fluent SafeSynthesizer builder and configure data and models."""
        from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

        builder = SafeSynthesizer(save_path=save_path).with_data_source(df)
        train_kwargs: dict[str, Any] = {
            "pretrained_model": self.pretrained_model,
        }
        if self.epochs is not None:
            train_kwargs["num_train_epochs"] = self.epochs

        builder = builder.with_train(**train_kwargs)
        if self.dp_enabled:
            builder = builder.with_differential_privacy(
                dp_enabled=True,
                epsilon=self.epsilon,
                delta="auto",
            )
        return builder

    def _seed_all(self) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)
        try:
            import torch

            torch.manual_seed(self.seed)
        except ImportError:
            pass

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        if not cuda_available:
            raise RuntimeError(
                f"Generator {self.name!r} requires an NVIDIA GPU (CUDA). "
                "Run on a GPU instance (e.g. via notebooks/gpu_offline_run.ipynb) "
                "and import results."
            )
        self._seed_all()
        self._builder = self.build_builder(train_df)
        self._builder.process_data().train()

    def sample(self, n: int) -> pd.DataFrame:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        if not cuda_available or self._builder is None:
            raise RuntimeError(
                f"Generator {self.name!r} requires an NVIDIA GPU (CUDA) and a fitted model."
            )
        self._seed_all()
        self._builder.with_generate(num_records=n)
        self._builder.generate()
        self._builder.save_results()
        self._results = self._builder.results
        if hasattr(self._results, "synthetic_data") and self._results.synthetic_data is not None:
            return self._results.synthetic_data
        raise RuntimeError("Safe-Synthesizer did not produce synthetic data output.")


def build(name: str, seed: int = 0, **params: Any) -> Generator:
    """Catalog entry point for Safe-Synthesizer generators."""
    if name == "nss_smollm3":
        return SafeSynthesizerGenerator(name=name, seed=seed, dp_enabled=False, **params)
    if name == "nss_smollm3_dp_eps8":
        return SafeSynthesizerGenerator(name=name, seed=seed, dp_enabled=True, **params)
    raise KeyError(f"stl_safesynth does not provide generator {name!r}")
