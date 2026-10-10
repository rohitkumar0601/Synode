"""Synthcity-backed generators (and GReaT via be-great)."""

from __future__ import annotations

import functools
import random

import numpy as np
import pandas as pd
import torch
from loguru import logger
from stl_core.generator import Generator, library_version, resolve_params
from stl_core.metadata import MISSING_CATEGORY, ColumnType, TableMetadata

# Single place for logging noise: this disables ALL loguru output from the "synthcity"
# package (including ERROR). We do so because synthcity emits CRITICAL lines for optional
# plugins that fail to load (goggle, great), which are not errors for us. Exceptions
# still propagate normally; only log records are dropped.
logger.disable("synthcity")

PLUGINS = {"marginals": "marginal_distributions", "bayesian_network": "bayesian_network",
           "ctgan": "ctgan", "tvae": "tvae", "tabddpm": "ddpm",
           "privbayes_eps0.1": "privbayes", "privbayes_eps1": "privbayes",
           "privbayes_eps10": "privbayes"}


# synthcity's automatic PrivBayes degree (K=0, fsolve on usefulness) explodes at small epsilon
# and OOMs in pd.crosstab on Adult; fixing K=2 is what avoids the OOM. n_bins is a no-op in
# synthcity 0.2.12: its _encode never bins numeric columns (`or` instead of `and`), so their
# domain is the exact training values. PrivBayes here is NOT epsilon-DP and is kept only as
# a nominal-DP audit (catalog family "nominal_dp_audit"); the RQ2 sweep uses smartnoise MST.
PRIVBAYES_DEFAULTS = {"K": 2, "n_bins": 20}


def _numeric_key(v) -> str | None:
    try:
        return str(float(v))
    except (TypeError, ValueError):
        return None


def _restore_categorical(values: list[str], s: pd.Series) -> pd.Series:
    # Generators may hand numeric-like categories back as floats ("1" -> 1.0);
    # map them to the matching TRAIN category when every TRAIN value is numeric-like.
    # Nulls become MISSING_CATEGORY (as in stl_core.metadata.prepare), not "None"/"nan".
    s = s.astype(object).where(s.notna(), MISSING_CATEGORY)
    keys = {_numeric_key(v): v for v in values}
    if values and None not in keys:
        return s.map(lambda v: keys.get(_numeric_key(v), str(v))).astype(object)
    return s.astype(str).astype(object)


def capture_train_stats(train_df: pd.DataFrame, cats: set[str]) -> tuple[dict, dict]:
    """TRAIN medians (numeric columns) and categorical value lists, for restore_types."""
    medians = {c: train_df[c].median() for c in train_df.columns
               if pd.api.types.is_numeric_dtype(train_df[c])
               and not pd.api.types.is_bool_dtype(train_df[c])}
    cat_values = {c: sorted(train_df[c].dropna().astype(str).unique()) for c in cats}
    return medians, cat_values


def restore_types(out: pd.DataFrame, columns: list[str], dtypes: dict, cats: set[str],
                  ids: list[str], medians: dict, cat_values: dict) -> pd.DataFrame:
    """Re-add id columns, restore column order and TRAIN dtypes on generated rows.

    Unparseable / missing numerics fall back to the TRAIN median.
    """
    out = out.copy().reset_index(drop=True)
    for c in ids:
        out[c] = [f"syn-{i}" for i in range(len(out))]
    out = out[columns]
    for c, dt in dtypes.items():
        if c in cats or dt is object or dt == "object":
            out[c] = _restore_categorical(cat_values.get(c, []), out[c])
        elif pd.api.types.is_integer_dtype(dt):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(medians[c]).round().astype(dt)
        elif pd.api.types.is_float_dtype(dt):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(medians[c]).astype(float)
    return out


@functools.cache
def _plugin_class(plugin: str) -> type:
    from synthcity.plugins import Plugins
    return Plugins().get_type(plugin)


class SynthcityGenerator(Generator):
    def __init__(self, name: str, seed: int = 0, **params) -> None:
        super().__init__(seed=seed, **params)
        self.name = name

    def _plugin_kwargs(self) -> dict:
        """The kwargs passed to the synthcity plugin. None values are dropped so that the
        plugin's own default applies (budget profile `library` sets epochs to None)."""
        kw = {k: v for k, v in self.params.items() if v is not None}
        if PLUGINS[self.name] == "privbayes":
            kw = {**PRIVBAYES_DEFAULTS, **kw}
        if "epochs" in kw:
            kw["n_iter"] = kw.pop("epochs")
        return {"random_state": self.seed, **kw}

    def resolved_params(self) -> dict:
        # Library defaults = the plugin class's __init__ signature, e.g. CTGANPlugin
        # n_iter=2000 (third_party/synthcity/src/synthcity/plugins/generic/plugin_ctgan.py:108),
        # TVAEPlugin n_iter=1000 (plugin_tvae.py:96), TabDDPMPlugin n_iter=1000
        # (plugin_ddpm.py:97). synthcity exposes no other defaults table (hyperparameter_space
        # gives search ranges, not defaults).
        return resolve_params("synthcity", library_version("synthcity"),
                              _plugin_class(PLUGINS[self.name]), self._plugin_kwargs())

    def _seed_all(self) -> None:
        # Some plugins (e.g. marginal_distributions) draw from global RNGs.
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        from synthcity.plugins import Plugins
        self._seed_all()
        self._columns = list(train_df.columns)
        self._dtypes = train_df.dtypes.to_dict()
        self._ids = [c for c in self._columns if metadata.columns.get(c) == ColumnType.id]
        self._cats = {c for c in self._columns if metadata.columns.get(c) == ColumnType.categorical}
        self._medians, self._cat_values = capture_train_stats(train_df, self._cats)
        self._plugin = Plugins().get(PLUGINS[self.name], **self._plugin_kwargs())
        self._plugin.fit(train_df.drop(columns=self._ids))

    def sample(self, n: int) -> pd.DataFrame:
        self._seed_all()
        frames, have = [], 0
        for _ in range(3):
            df = self._plugin.generate(count=n - have).dataframe()
            frames.append(df)
            have += len(df)
            if have >= n:
                break
        if have < n:
            raise RuntimeError("synthcity returned fewer rows than requested")
        out = pd.concat(frames, ignore_index=True).iloc[:n]
        return self._restore_types(out)

    def _restore_types(self, out: pd.DataFrame) -> pd.DataFrame:
        return restore_types(out, self._columns, self._dtypes, self._cats, self._ids,
                             self._medians, self._cat_values)


def build(name: str, seed: int = 0, **params) -> Generator:
    if name == "great":
        from stl_synthcity.great import GReaTGenerator
        return GReaTGenerator(seed=seed, **params)
    if name not in PLUGINS:
        raise KeyError(f"stl_synthcity does not provide {name!r}")
    return SynthcityGenerator(name, seed=seed, **params)
