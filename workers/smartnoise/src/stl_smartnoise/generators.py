"""SmartNoise-backed generators: MST, the vetted epsilon-DP synthesizer for the RQ2 sweep."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import random
import warnings

import numpy as np
import pandas as pd
from stl_core.bounds import bound_value_problems
from stl_core.generator import Generator
from stl_core.metadata import MISSING_CATEGORY, ColumnType, TableMetadata

PREPROCESSOR_SHARE = 0.2  # epsilon share spent privately on bounds of non-public-bound columns
MST_BINS = 10  # bins per continuous column (smartnoise's default for the 'cube' style)
MST_NAMES = ("mst_eps0.1", "mst_eps1", "mst_eps10")


# --- dtype restoration: a minimal copy of stl_synthcity's helper (separate environments) ---

def _numeric_key(v) -> str | None:
    try:
        return str(float(v))
    except (TypeError, ValueError):
        return None


def _restore_categorical(values: list[str], s: pd.Series) -> pd.Series:
    # Numeric-like categories may come back as floats ("1" -> 1.0): map them to the TRAIN
    # category when every TRAIN value is numeric-like. Nulls become MISSING_CATEGORY.
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


@contextlib.contextmanager
def _quiet():
    """Silence smartnoise's budget-split print and mbi's warning about the DataFrame that
    smartnoise itself passes in (library noise, not ours to fix)."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pandas dataframe inputs are deprecated",
                                category=UserWarning, module="mbi.dataset")
        yield


def _log1p_bin_transformer(lower: float, upper: float, column: str):
    """BinTransformer over log1p(value) with public bounds [lower, upper] (raw scale):
    values are clipped to the bounds, binned on the log1p scale, and decoded as expm1 of
    the log-scale bin midpoint -- except that the lowest bin decodes to the public lower
    bound when that bound is an integer (a mass point such as 0 for capital-gain), so such
    values are reproducible. The decode rule uses only public information: 0 epsilon, as
    is the whole transformer (no data-dependent fit). (A ChainTransformer of smartnoise's
    LogTransformer + BinTransformer is not usable: LogTransformer is log, not log1p, and
    MST rejects the chain's continuous intermediate cardinality.)"""
    from snsynth.transform import BinTransformer

    lower_is_mass_point = float(lower).is_integer()

    class Log1pBinTransformer(BinTransformer):
        def _transform(self, val):
            val = float(val)
            if np.isnan(val):
                raise ValueError(f"NaN in log-scale column {column!r}: MST needs complete "
                                 "numeric values (impute before generating)")
            return super()._transform(float(np.log1p(min(max(val, lower), upper))))

        def _inverse_transform(self, val):
            if lower_is_mass_point and val == 0:
                return float(lower)
            return float(np.expm1(super()._inverse_transform(val)))

    return Log1pBinTransformer(bins=MST_BINS, lower=float(np.log1p(lower)),
                               upper=float(np.log1p(upper)))


def bounds_sha256(public_bounds: dict, log_scale: list) -> str:
    """sha256 of the sorted JSON of the resolved public bounds and log-scale columns."""
    resolved = {"public_bounds": {c: [float(lo), float(hi)]
                                  for c, (lo, hi) in public_bounds.items()},
                "public_log_scale": sorted(log_scale)}
    return hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()


def _public_constraints(public_bounds: dict, log_scale: list, continuous: list) -> dict:
    """smartnoise transformer constraints for the publicly bounded continuous columns."""
    from snsynth.transform import BinTransformer

    unknown = sorted(set(public_bounds) - set(continuous))
    if unknown:
        raise ValueError(f"public_bounds names {unknown}, which is not a continuous column "
                         f"(continuous columns: {continuous})")
    bad_log = sorted(set(log_scale) - set(public_bounds))
    if bad_log:
        raise ValueError(f"public_log_scale names {bad_log}, which have no public_bounds")
    constraints = {}
    problems = bound_value_problems(public_bounds, log_scale)
    if problems:
        raise ValueError(f"invalid public_bounds: {'; '.join(problems)}")
    for col, bound in public_bounds.items():
        lower, upper = (float(v) for v in bound)
        if col in log_scale:
            constraints[col] = _log1p_bin_transformer(lower, upper, col)
        else:  # BinTransformer clips values outside [lower, upper] into the edge bins
            constraints[col] = BinTransformer(bins=MST_BINS, lower=lower, upper=upper)
    return constraints


class MSTGenerator(Generator):
    """smartnoise-synth MST (McKenna et al.), (epsilon, delta)-DP with delta = 1e-9.

    Privacy budget: the total is ``params["epsilon"]`` (the catalog epsilon). smartnoise's
    ``Synthesizer.create("mst", epsilon=E)`` sets the synthesizer's budget to E, and
    ``fit(..., preprocessor_eps=P)`` spends P (Laplace-based approximate bounds, split over
    the continuous columns that need them) and *subtracts what it spent from E* before MST
    runs (see ``snsynth.base.Synthesizer._get_train_data``).

    Bounds (``privacy_spent["bounds"]`` records the path):
      * ``"public"`` -- every continuous column has a public codebook bound in
        ``params["public_bounds"]`` ({column: [lower, upper]}, never derived from the data):
        each is binned by a ``BinTransformer(lower, upper)``, P = 0 and MST gets all of
        epsilon. Values outside the bounds are clipped into the edge bins. Columns listed in
        ``params["public_log_scale"]`` (nonnegative, heavy-tailed) are binned on the log1p
        scale over the same public bounds -- still 0 epsilon.
      * ``"private"`` / ``"mixed"`` -- some (or all) continuous columns lack a public bound:
        P = 0.2 * epsilon is spent on those columns only, MST gets 0.8 * epsilon.
      * ``"none"`` -- no continuous columns; MST gets all of epsilon.

    Columns: categorical -> ``categorical_columns``; numerical and datetime (as epoch
    seconds) -> continuous (binned into 10 bins; synthetic values are bin midpoints); ids
    are dropped and recreated as ``syn-{i}``. Caveats inherited from smartnoise: the
    categorical domain (set of categories) and the row count are taken as public.

    Reproducibility: numpy/random are seeded before fit and sample, but the Gaussian
    measurement noise comes from OpenDP's CSPRNG, so runs are not bit-reproducible.
    """

    name = "mst"

    def __init__(self, name: str = "mst", seed: int = 0, **params) -> None:
        super().__init__(seed=seed, **params)
        self.name = name

    def _seed_all(self) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        from snsynth import Synthesizer

        self._seed_all()
        epsilon = float(self.params["epsilon"])
        self._columns = list(train_df.columns)
        self._dtypes = train_df.dtypes.to_dict()
        kind = {c: metadata.columns.get(c) for c in self._columns}
        self._ids = [c for c in self._columns if kind[c] == ColumnType.id]
        self._cats = {c for c in self._columns if kind[c] == ColumnType.categorical}
        self._datetimes = [c for c in self._columns
                           if kind[c] == ColumnType.datetime
                           or pd.api.types.is_datetime64_any_dtype(train_df[c])]
        self._medians, self._cat_values = capture_train_stats(train_df, self._cats)
        data = train_df.drop(columns=self._ids).copy()
        for c in self._datetimes:
            data[c] = pd.to_datetime(data[c]).astype("int64") / 1e9
        cats = [c for c in data.columns if c in self._cats]
        for c in cats:
            data[c] = data[c].astype(object).where(data[c].notna(), MISSING_CATEGORY).astype(str)
        continuous = [c for c in data.columns if c not in self._cats]
        for c in continuous:
            data[c] = pd.to_numeric(data[c], errors="coerce").astype(float)
        public_bounds = dict(self.params.get("public_bounds") or {})
        log_scale = list(self.params.get("public_log_scale") or [])
        constraints = _public_constraints(public_bounds, log_scale, continuous)
        private = [c for c in continuous if c not in constraints]
        if not continuous:
            bounds_path = "none"
        elif not private:
            bounds_path = "public"
        else:
            bounds_path = "mixed" if constraints else "private"
        self._model = Synthesizer.create("mst", epsilon=epsilon)
        preprocessor_eps = PREPROCESSOR_SHARE * epsilon if private else 0.0
        try:
            with _quiet():
                self._model.fit(data, transformer=constraints, preprocessor_eps=preprocessor_eps,
                                categorical_columns=cats, continuous_columns=private)
        except ValueError as exc:
            if "could not find bounds" not in str(exc):
                raise
            raise ValueError(
                f"MST could not privately infer bounds for the continuous columns {private} "
                f"from {len(data)} rows: preprocessor epsilon {preprocessor_eps:g} "
                f"({preprocessor_eps / len(private):g} per column) is too small for "
                "smartnoise's approx_bounds; give public_bounds for these columns, or use "
                "a larger epsilon or more rows") from exc
        preprocessor = float(self._model._transformer.odometer.spent[0])
        self.privacy_spent = {"preprocessor": preprocessor,
                              "synthesizer": float(self._model.epsilon),
                              "bounds": bounds_path,
                              "bounds_sha256": bounds_sha256(public_bounds, log_scale)}

    def sample(self, n: int) -> pd.DataFrame:
        self._seed_all()
        with _quiet():
            out = self._model.sample(n)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out, columns=[c for c in self._columns if c not in self._ids])
        for c in self._datetimes:
            seconds = pd.to_numeric(out[c], errors="coerce")
            seconds = seconds.fillna(seconds.median() if seconds.notna().any() else 0.0)
            out[c] = pd.to_datetime(seconds, unit="s").astype(self._dtypes[c])
        return restore_types(out, self._columns, self._dtypes, self._cats, self._ids,
                             self._medians, self._cat_values)


def build(name: str, seed: int = 0, **params) -> Generator:
    if name not in MST_NAMES:
        raise KeyError(f"stl_smartnoise does not provide {name!r}")
    return MSTGenerator(name, seed=seed, **params)
