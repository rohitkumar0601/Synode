"""The interface every synthetic-data generator implements."""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from stl_core.metadata import TableMetadata


def json_safe(value: Any) -> Any:
    """``value`` as JSON-ready data: tuples become lists, dict keys strings, non-finite
    floats and anything else JSON cannot hold (paths, devices, classes) their str()."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def signature_defaults(fn: Callable, skip: Iterable[str] = ()) -> dict[str, Any]:
    """Keyword defaults of ``fn`` (a function or class) from ``inspect.signature``; required
    parameters, *args/**kwargs, ``self`` and the names in ``skip`` are left out."""
    skip = {"self", *skip}
    return {name: p.default for name, p in inspect.signature(fn).parameters.items()
            if name not in skip and p.default is not inspect.Parameter.empty
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}


def resolve_params(library: str, version: str | None, fn: Callable | None,
                   passed: dict[str, Any], skip: Iterable[str] = ()) -> dict[str, Any]:
    """The generator_params record: ``fn``'s signature defaults overlaid with the kwargs we
    actually pass (``passed``; None values must already be dropped), the library name and
    version, and ``overrides`` = the sorted keys we pass explicitly."""
    defaults = signature_defaults(fn, skip) if fn is not None else {}
    return {"library": library, "library_version": version,
            "params": json_safe({**defaults, **passed}), "overrides": sorted(passed)}


def library_version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


class Generator(ABC):
    name: str = "generator"

    def __init__(self, seed: int = 0, **params) -> None:
        self.seed = seed
        self.params = params

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None: ...

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame: ...

    def resolved_params(self) -> dict[str, Any]:
        """The hyperparameters the underlying library is (or will be) called with: library
        signature defaults merged with the kwargs this wrapper passes, plus library name and
        version and the explicitly passed keys (``overrides``). Wrappers override this; the
        default (unknown library, no params) keeps other generators working."""
        return {"library": None, "library_version": None, "params": {}, "overrides": []}
