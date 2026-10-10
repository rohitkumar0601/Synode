# S1 — Core Contract, Eval Suite, CLI and Sanity Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `packages/core`, `packages/runner`, the eval worker's metric suite, the SDV worker's Gaussian Copula, and the Hydra-configured experiment CLI, ending with an automated sanity gate on UCI Adult (train-copy worst on privacy, independent marginals worst on column-pair trends).

**Architecture:** `stl_core` (pandas/numpy/pydantic only) defines metadata, splitting, the `Generator` ABC, a lazy generator catalog, result schemas and the decision engine. `stl_runner` adds storage (local FS or S3 via fsspec), dataset loading, cost measurement and the generate pipeline. Each worker (`workers/eval`, `workers/sdv`) is its own uv project with its own lockfile and Python pin, depending on core and runner by path. The CLI (`experiments/run.py`) runs a stage (`prepare`, `generate`, `evaluate`, `report`) inside whichever environment has the needed libraries; stages hand off through the artifact store.

**Tech Stack:** Python 3.11 (eval) / 3.12 (sdv) / ≥3.10 (core, runner), uv, pandas, numpy 1.26 (eval), pydantic v2, fsspec, scikit-learn, xgboost, SDMetrics 0.32, Anonymeter 1.0.0, SDV 1.38, Hydra 1.3, MLflow (skinny), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-05-synth-trust-lab-design.md`

## Global Constraints

- **No git commits or pushes.** The owner commits. Every task ends with a "Checkpoint" step (run tests + `git status`), never `git commit`.
- `packages/core` runtime dependencies: pandas, numpy, pydantic **only**.
- Python pins: eval `>=3.11,<3.12`; sdv `>=3.12,<3.13`; core and runner `>=3.10`.
- Anonymeter pinned `==1.0.0`.
- Split: stratified on target, 60/20/20 TRAIN/CONTROL/TEST, disjoint, deterministic per seed.
- Row cap 20,000 (stratified subsample) on registration.
- Gate defaults: max Anonymeter risk 0.10; DCR share 0.6; a null privacy metric fails the gate.
- Presets: balanced (⅓,⅓,⅓), privacy_first (0.2,0.2,0.6), utility_first (0.2,0.6,0.2) as (fidelity, utility, privacy).
- **Windows MAX_PATH:** the repo lives under a long OneDrive path and `LongPathsEnabled=0`. Virtual environments MUST live outside the repo at `${STL_VENV_ROOT:-$HOME/.stl}/<env>`; always run Python through `scripts/envrun.sh <env> …`, which sets `UV_PROJECT_ENVIRONMENT`. Never create `.venv` inside the repo (SciPy imports fail with missing-module errors when paths exceed 260 chars — verified 2026-10-05).
- Known resolver trap: in the eval env SciPy resolves to 1.17.1 with numpy 1.26.4; this imports correctly at short paths (verified). Do not add a SciPy pin unless an import test fails.
- SDMetrics: use the unified `sdmetrics.reports.QualityReport` with `{"tables": {"t": {...}}}` metadata and `{"t": df}` data (the single-table report is deprecated).
- Missing values: `prepare()` imputes numerical NaN with the column median and categorical NaN with the literal `"__missing__"`, and drops rows whose target is NaN. All downstream code may assume NaN-free frames.

## Review Focus

1. **A target class with fewer than 5 rows** — a stratified split cannot place it in all three parts; expect a clear `ValueError` naming the class, not a crash deep in pandas. (Test in Task 2.)
2. **Synthetic data with categories never seen in real data, missing columns, or wrong dtypes** — metrics must align/coerce and either score it or fail with a clear message naming the column. (Tests in Task 8 and Task 11.)
3. **Small datasets where `n_attacks` exceeds CONTROL size** — Anonymeter raises; expect the attack count to be clamped. (Test in Task 10.)
4. **A synthetic table whose target has a single class** — classifiers cannot fit; expect TSTR ratio 0 for that model, not an exception. (Test in Task 9.)
5. **All candidates failing the gate** — expect `winner=None` and a reason that lists each candidate's first failure. (Test in Task 5.)

## File Structure

```
scripts/envrun.sh                         run a command in a named uv env stored outside the repo
ruff.toml                                 lint config for all Python
packages/core/
  pyproject.toml
  src/stl_core/__init__.py
  src/stl_core/metadata.py                ColumnType, TableMetadata, infer_metadata, prepare
  src/stl_core/split.py                   Splits, split, stratified_subsample
  src/stl_core/generator.py               Generator ABC
  src/stl_core/baselines.py               TrainCopy, IndependentMarginals, build()
  src/stl_core/catalog.py                 GeneratorInfo, CATALOG, load_generator, is_available
  src/stl_core/schemas.py                 JobSpec, *Metrics, MetricReport, AggregatedReport, Recommendation …
  src/stl_core/aggregate.py               flatten, aggregate
  src/stl_core/decision.py                axis_scores, check_gate, pareto_front, weighted_score, recommend
  tests/…                                 one test file per module
packages/runner/
  pyproject.toml
  src/stl_runner/__init__.py
  src/stl_runner/keys.py                  artifact key layout
  src/stl_runner/storage.py               ArtifactStore (fsspec)
  src/stl_runner/cost.py                  measure() context manager (time + peak RSS)
  src/stl_runner/datasets.py              load_dataset(cfg) from openml or csv
  src/stl_runner/pipeline.py              register_dataset, ensure_splits, plan_run, generate_job, run_generate_with_cap
  src/stl_runner/tracking.py              log_report to MLflow (lazy import)
  tests/…
workers/eval/
  pyproject.toml                          py3.11, sdmetrics, anonymeter==1.0.0, sklearn, xgboost, mlflow-skinny, hydra-core
  src/stl_eval/__init__.py
  src/stl_eval/preprocessing.py           align, DistanceEncoder, ml_matrices
  src/stl_eval/fidelity.py                quality, detection_auc, precision_recall
  src/stl_eval/utility.py                 tstr
  src/stl_eval/privacy.py                 dcr_share, anonymeter_risks, mia_advantage
  src/stl_eval/evaluate.py                EvalConfig, evaluate, evaluate_stored_job
  tests/…                                 incl. test_sanity_gate.py
workers/sdv/
  pyproject.toml                          py3.12, sdv
  src/stl_sdv/__init__.py
  src/stl_sdv/generators.py               GaussianCopula, build()
  tests/test_copula.py
experiments/
  conf/config.yaml
  conf/datasets/adult.yaml
  conf/generators/default.yaml
  run.py                                  argparse subcommands + Hydra compose
.github/workflows/ci.yml
```

Package distribution names: `stl-core`, `stl-runner`, `stl-eval`, `stl-sdv`. Import names: `stl_core`, `stl_runner`, `stl_eval`, `stl_sdv`.

---

### Task 1: Repo tooling and core package skeleton with metadata

**Files:**
- Create: `scripts/envrun.sh`, `ruff.toml`, `packages/core/pyproject.toml`, `packages/core/src/stl_core/__init__.py`, `packages/core/src/stl_core/metadata.py`
- Test: `packages/core/tests/test_metadata.py`

**Interfaces:**
- Produces:
  - `ColumnType(str, Enum)` with members `numerical`, `categorical`, `datetime`, `id`
  - `TableMetadata(BaseModel)`: `columns: dict[str, ColumnType]`, `target: str`, `task: Literal["classification","regression"] = "classification"`; properties `feature_columns -> list[str]` (non-id, non-target), `model_columns -> list[str]` (non-id, includes target); methods `of_type(t: ColumnType) -> list[str]`, `to_sdv_dict() -> dict`
  - `infer_metadata(df: pd.DataFrame, target: str, max_categories: int = 50) -> TableMetadata`
  - `prepare(df: pd.DataFrame, metadata: TableMetadata) -> pd.DataFrame`

- [ ] **Step 1: Create tooling files**

`scripts/envrun.sh`:
```bash
#!/usr/bin/env bash
# Run a command inside one of the project's uv environments.
# Environments live outside the repo (Windows MAX_PATH); see plan Global Constraints.
# Usage: scripts/envrun.sh <core|runner|eval|sdv|synthcity|datadesigner|api> <command...>
set -euo pipefail
ENV_NAME="$1"; shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$ENV_NAME" in
  core|runner) PROJECT="$ROOT/packages/$ENV_NAME" ;;
  api)         PROJECT="$ROOT/services/api" ;;
  *)           PROJECT="$ROOT/workers/$ENV_NAME" ;;
esac
export UV_PROJECT_ENVIRONMENT="${STL_VENV_ROOT:-$HOME/.stl}/$ENV_NAME"
exec uv run --project "$PROJECT" "$@"
```

`ruff.toml`:
```toml
line-length = 100
target-version = "py310"

[lint]
select = ["E", "F", "I", "B", "UP"]
```

`packages/core/pyproject.toml`:
```toml
[project]
name = "stl-core"
version = "0.1.0"
description = "Synthetic Data Trust Lab contract: metadata, splits, generator interface, schemas, decision engine"
requires-python = ">=3.10"
dependencies = ["pandas>=1.5", "numpy>=1.24", "pydantic>=2.5"]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/stl_core"]
```

`packages/core/src/stl_core/__init__.py`:
```python
"""Synthetic Data Trust Lab core contract."""
```

- [ ] **Step 2: Write the failing test**

`packages/core/tests/test_metadata.py`:
```python
import numpy as np
import pandas as pd
import pytest

from stl_core.metadata import ColumnType, TableMetadata, infer_metadata, prepare


def _frame(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "age": rng.integers(18, 90, n),
            "income": rng.normal(50_000, 10_000, n),
            "city": rng.choice(["a", "b", "c"], n),
            "children": rng.integers(0, 4, n),
            "joined": pd.date_range("2020-01-01", periods=n, freq="D"),
            "label": rng.choice(["yes", "no"], n),
        }
    )


def test_infer_metadata_types():
    md = infer_metadata(_frame(), target="label")
    assert md.columns["customer_id"] == ColumnType.id
    assert md.columns["age"] == ColumnType.numerical
    assert md.columns["income"] == ColumnType.numerical
    assert md.columns["city"] == ColumnType.categorical
    assert md.columns["children"] == ColumnType.categorical  # <= 10 distinct integers
    assert md.columns["joined"] == ColumnType.datetime
    assert md.columns["label"] == ColumnType.categorical
    assert md.task == "classification"


def test_feature_and_model_columns():
    md = infer_metadata(_frame(), target="label")
    assert "customer_id" not in md.model_columns
    assert "label" in md.model_columns
    assert "label" not in md.feature_columns


def test_target_must_exist_and_not_be_id():
    with pytest.raises(ValueError, match="target"):
        TableMetadata(columns={"a": ColumnType.numerical}, target="b")
    with pytest.raises(ValueError, match="target"):
        TableMetadata(columns={"a": ColumnType.id}, target="a")


def test_infer_missing_target_raises():
    with pytest.raises(ValueError, match="nope"):
        infer_metadata(_frame(), target="nope")


def test_to_sdv_dict():
    md = infer_metadata(_frame(), target="label")
    d = md.to_sdv_dict()
    assert d["columns"]["age"] == {"sdtype": "numerical"}
    assert d["columns"]["customer_id"] == {"sdtype": "id"}


def test_prepare_imputes_and_drops_missing_target():
    df = _frame(20)
    df.loc[0, "income"] = np.nan
    df.loc[1, "city"] = None
    df.loc[2, "label"] = None
    md = infer_metadata(_frame(), target="label")
    out = prepare(df, md)
    assert len(out) == 19
    assert out.isna().sum().sum() == 0
    assert (out["city"] == "__missing__").sum() == 1
    assert out["city"].dtype == object
    assert out.index.tolist() == list(range(19))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bash scripts/envrun.sh core pytest packages/core/tests/test_metadata.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'stl_core.metadata'`

- [ ] **Step 4: Write implementation**

`packages/core/src/stl_core/metadata.py`:
```python
"""Table metadata: column types, target column, inference and preparation."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import pandas as pd
from pydantic import BaseModel, model_validator

MISSING_CATEGORY = "__missing__"


class ColumnType(str, Enum):
    numerical = "numerical"
    categorical = "categorical"
    datetime = "datetime"
    id = "id"


class TableMetadata(BaseModel):
    columns: dict[str, ColumnType]
    target: str
    task: Literal["classification", "regression"] = "classification"

    @model_validator(mode="after")
    def _check_target(self) -> TableMetadata:
        if self.target not in self.columns:
            raise ValueError(f"target column {self.target!r} is not in columns")
        if self.columns[self.target] == ColumnType.id:
            raise ValueError(f"target column {self.target!r} cannot be an id column")
        return self

    def of_type(self, t: ColumnType) -> list[str]:
        return [c for c, ct in self.columns.items() if ct == t]

    @property
    def model_columns(self) -> list[str]:
        return [c for c, ct in self.columns.items() if ct != ColumnType.id]

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.model_columns if c != self.target]

    def to_sdv_dict(self) -> dict:
        return {"columns": {c: {"sdtype": ct.value} for c, ct in self.columns.items()}}


def _infer_column(s: pd.Series, max_categories: int) -> ColumnType:
    if pd.api.types.is_datetime64_any_dtype(s):
        return ColumnType.datetime
    if pd.api.types.is_bool_dtype(s):
        return ColumnType.categorical
    nunique = s.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and nunique <= 10:
            return ColumnType.categorical
        return ColumnType.numerical
    if nunique == len(s) and len(s) > max_categories:
        return ColumnType.id
    return ColumnType.categorical


def infer_metadata(df: pd.DataFrame, target: str, max_categories: int = 50) -> TableMetadata:
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found; columns are {list(df.columns)}")
    columns = {str(c): _infer_column(df[c], max_categories) for c in df.columns}
    if columns[target] == ColumnType.id:
        columns[target] = ColumnType.categorical
    task = "classification" if columns[target] == ColumnType.categorical else "regression"
    return TableMetadata(columns=columns, target=target, task=task)


def prepare(df: pd.DataFrame, metadata: TableMetadata) -> pd.DataFrame:
    """Select metadata columns, drop rows with missing target, impute the rest."""
    out = df[list(metadata.columns)].copy()
    out = out[out[metadata.target].notna()]
    for col, ct in metadata.columns.items():
        if ct == ColumnType.numerical:
            values = pd.to_numeric(out[col], errors="coerce")
            out[col] = values.fillna(values.median()).astype(float)
        elif ct == ColumnType.categorical:
            out[col] = out[col].astype(object).where(out[col].notna(), MISSING_CATEGORY).astype(str)
            out[col] = out[col].astype(object)
        elif ct == ColumnType.datetime:
            values = pd.to_datetime(out[col], errors="coerce")
            out[col] = values.fillna(values.median())
    return out.reset_index(drop=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `bash scripts/envrun.sh core pytest packages/core/tests/test_metadata.py -q`
Expected: 6 passed

- [ ] **Step 6: Lint and checkpoint**

Run: `bash scripts/envrun.sh core ruff check packages/core` → `All checks passed!`
Run: `git status --short` (inspect only; do not commit).

---

### Task 2: Stratified split and subsample

**Files:**
- Create: `packages/core/src/stl_core/split.py`
- Test: `packages/core/tests/test_split.py`

**Interfaces:**
- Consumes: `TableMetadata` (Task 1)
- Produces:
  - `@dataclass Splits(train: pd.DataFrame, control: pd.DataFrame, test: pd.DataFrame)` with `fingerprint() -> str` (sha256 hex)
  - `split(df, metadata, seed: int, shares: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> Splits`
  - `stratified_subsample(df, metadata, n: int, seed: int) -> pd.DataFrame`
  - `MIN_ROWS_PER_STRATUM = 5`

- [ ] **Step 1: Write the failing test**

`packages/core/tests/test_split.py`:
```python
import numpy as np
import pandas as pd
import pytest

from stl_core.metadata import ColumnType, TableMetadata
from stl_core.split import split, stratified_subsample

MD = TableMetadata(
    columns={"x": ColumnType.numerical, "y": ColumnType.categorical}, target="y"
)


def _frame(n=1000, pos_rate=0.2):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"x": rng.normal(size=n), "y": np.where(rng.random(n) < pos_rate, "pos", "neg")}
    )


def test_split_sizes_and_disjoint():
    df = _frame().reset_index().rename(columns={"index": "rid"})
    md = TableMetadata(columns={**MD.columns, "rid": ColumnType.id}, target="y")
    s = split(df, md, seed=0)
    assert abs(len(s.train) - 600) <= 2
    assert abs(len(s.control) - 200) <= 2
    assert len(s.train) + len(s.control) + len(s.test) == 1000
    ids = [set(p["rid"]) for p in (s.train, s.control, s.test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_split_is_stratified():
    df = _frame()
    s = split(df, MD, seed=0)
    overall = (df["y"] == "pos").mean()
    for part in (s.train, s.control, s.test):
        assert abs((part["y"] == "pos").mean() - overall) < 0.02


def test_split_deterministic_per_seed():
    df = _frame()
    a, b, c = split(df, MD, seed=3), split(df, MD, seed=3), split(df, MD, seed=4)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_rare_class_raises_clear_error():
    df = _frame(100)
    df.loc[:2, "y"] = "rare"
    with pytest.raises(ValueError, match="'rare'"):
        split(df, MD, seed=0)


def test_regression_target_uses_quantile_bins():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=500), "t": rng.normal(size=500)})
    md = TableMetadata(
        columns={"x": ColumnType.numerical, "t": ColumnType.numerical},
        target="t",
        task="regression",
    )
    s = split(df, md, seed=0)
    assert len(s.train) + len(s.control) + len(s.test) == 500


def test_stratified_subsample_keeps_ratio():
    df = _frame(5000)
    sub = stratified_subsample(df, MD, n=1000, seed=0)
    assert len(sub) == 1000
    assert abs((sub["y"] == "pos").mean() - (df["y"] == "pos").mean()) < 0.01
    assert stratified_subsample(df, MD, n=10_000, seed=0).shape == df.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash scripts/envrun.sh core pytest packages/core/tests/test_split.py -q`
Expected: FAIL — `No module named 'stl_core.split'`

- [ ] **Step 3: Write implementation**

`packages/core/src/stl_core/split.py`:
```python
"""Deterministic stratified TRAIN/CONTROL/TEST split."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stl_core.metadata import TableMetadata

MIN_ROWS_PER_STRATUM = 5


@dataclass
class Splits:
    train: pd.DataFrame
    control: pd.DataFrame
    test: pd.DataFrame

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for part in (self.train, self.control, self.test):
            h.update(pd.util.hash_pandas_object(part, index=False).values.tobytes())
        return h.hexdigest()


def _strata(df: pd.DataFrame, metadata: TableMetadata) -> pd.Series:
    y = df[metadata.target]
    if metadata.task == "classification":
        return y.astype(str)
    bins = min(10, max(1, len(df) // (MIN_ROWS_PER_STRATUM * 4)))
    return pd.qcut(y.rank(method="first"), q=bins, labels=False).astype(str)


def split(
    df: pd.DataFrame,
    metadata: TableMetadata,
    seed: int,
    shares: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Splits:
    strata = _strata(df, metadata)
    counts = strata.value_counts()
    too_small = counts[counts < MIN_ROWS_PER_STRATUM]
    if len(too_small):
        name, n = too_small.index[0], int(too_small.iloc[0])
        raise ValueError(
            f"target class {name!r} has {n} rows; a stratified 60/20/20 split needs at least "
            f"{MIN_ROWS_PER_STRATUM}. Merge or drop rare classes."
        )
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[], [], []]
    for _, idx in sorted(strata.groupby(strata).groups.items()):
        idx = rng.permutation(np.asarray(idx))
        n_train = int(round(len(idx) * shares[0]))
        n_control = int(round(len(idx) * shares[1]))
        parts[0].extend(idx[:n_train])
        parts[1].extend(idx[n_train : n_train + n_control])
        parts[2].extend(idx[n_train + n_control :])
    train, control, test = (
        df.loc[sorted(p)].reset_index(drop=True) for p in parts
    )
    return Splits(train=train, control=control, test=test)


def stratified_subsample(
    df: pd.DataFrame, metadata: TableMetadata, n: int, seed: int
) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    strata = _strata(df, metadata)
    rng = np.random.default_rng(seed)
    frac = n / len(df)
    keep: list[int] = []
    for _, idx in sorted(strata.groupby(strata).groups.items()):
        idx = np.asarray(idx)
        k = max(1, int(round(len(idx) * frac)))
        keep.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
    if len(keep) > n:
        keep = list(rng.choice(keep, size=n, replace=False))
    keep = sorted(keep)
    if len(keep) < n:
        rest = np.setdiff1d(df.index.to_numpy(), keep)
        keep = sorted(keep + list(rng.choice(rest, size=n - len(keep), replace=False)))
    return df.loc[keep].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bash scripts/envrun.sh core pytest packages/core/tests/test_split.py -q`
Expected: 6 passed

- [ ] **Step 5: Checkpoint**

Run: `bash scripts/envrun.sh core ruff check packages/core` and `git status --short`.

---

### Task 3: Generator interface, sanity baselines and catalog

**Files:**
- Create: `packages/core/src/stl_core/generator.py`, `packages/core/src/stl_core/baselines.py`, `packages/core/src/stl_core/catalog.py`
- Test: `packages/core/tests/test_baselines.py`, `packages/core/tests/test_catalog.py`

**Interfaces:**
- Consumes: `TableMetadata` (Task 1)
- Produces:
  - `class Generator(ABC)`: attribute `name: str`; `__init__(self, seed: int = 0, **params)` stores `self.seed`, `self.params`; abstract `fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None`; abstract `sample(self, n: int) -> pd.DataFrame`
  - `stl_core.baselines.TrainCopy`, `stl_core.baselines.IndependentMarginals`, `stl_core.baselines.build(name: str, seed: int, **params) -> Generator`
  - `@dataclass(frozen=True) GeneratorInfo(name, family, queue, module, cpu_ok=True, params: dict = {})`
  - `CATALOG: dict[str, GeneratorInfo]`
  - `load_generator(name: str, seed: int = 0, **overrides) -> Generator` — imports `info.module` and calls its `build(name, seed=seed, **{**info.params, **overrides})`; raises `KeyError` for unknown names and `ImportError` with message `"generator 'x' needs module 'm' (queue q.y); run it in that worker's environment"` when the module is missing
  - `is_available(name: str) -> bool`
  - Every generator module (core baselines, `stl_sdv.generators`, later `stl_synthcity.generators`) exposes `build(name, seed, **params) -> Generator`.

- [ ] **Step 1: Write the failing tests**

`packages/core/tests/test_baselines.py`:
```python
import numpy as np
import pandas as pd

from stl_core.baselines import IndependentMarginals, TrainCopy, build
from stl_core.metadata import ColumnType, TableMetadata

MD = TableMetadata(
    columns={"a": ColumnType.numerical, "b": ColumnType.numerical, "y": ColumnType.categorical},
    target="y",
)


def _train(n=500):
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": 3 * a + rng.normal(scale=0.1, size=n),
                         "y": np.where(a > 0, "p", "n")})


def test_train_copy_returns_real_rows():
    train = _train()
    g = TrainCopy(seed=0)
    g.fit(train, MD)
    out = g.sample(200)
    assert len(out) == 200
    merged = out.merge(train.drop_duplicates(), how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
    assert len(g.sample(1000)) == 1000  # oversampling with replacement


def test_independent_marginals_breaks_correlation_keeps_shapes():
    train = _train(2000)
    g = IndependentMarginals(seed=0)
    g.fit(train, MD)
    out = g.sample(2000)
    assert list(out.columns) == list(train.columns)
    assert abs(np.corrcoef(out["a"], out["b"])[0, 1]) < 0.1
    assert abs(out["a"].mean() - train["a"].mean()) < 0.1
    assert set(out["y"]) <= set(train["y"])
    assert out["a"].isin(train["a"]).mean() < 0.05  # numerics are not copied values


def test_build_by_name_and_seeded():
    train = _train()
    g1, g2 = build("independent_marginals", seed=5), build("independent_marginals", seed=5)
    g1.fit(train, MD)
    g2.fit(train, MD)
    pd.testing.assert_frame_equal(g1.sample(50), g2.sample(50))
    assert build("train_copy", seed=0).name == "train_copy"
```

`packages/core/tests/test_catalog.py`:
```python
import pytest

from stl_core.catalog import CATALOG, is_available, load_generator


def test_catalog_has_full_lineup():
    expected = {
        "train_copy", "independent_marginals", "marginals", "gaussian_copula",
        "bayesian_network", "ctgan", "tvae", "tabddpm", "great",
        "privbayes_eps0.1", "privbayes_eps1", "privbayes_eps10",
    }
    assert expected <= set(CATALOG)
    assert CATALOG["gaussian_copula"].queue == "q.sdv"
    assert CATALOG["ctgan"].queue == "q.synthcity"
    assert CATALOG["privbayes_eps1"].params == {"epsilon": 1.0}
    assert CATALOG["tabddpm"].cpu_ok is False


def test_load_core_generator():
    assert load_generator("train_copy", seed=1).seed == 1
    assert is_available("train_copy")


def test_unknown_generator():
    with pytest.raises(KeyError, match="nope"):
        load_generator("nope")


def test_missing_module_message():
    if is_available("ctgan"):
        pytest.skip("synthcity installed in this environment")
    with pytest.raises(ImportError, match="q.synthcity"):
        load_generator("ctgan")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `bash scripts/envrun.sh core pytest packages/core/tests/test_baselines.py packages/core/tests/test_catalog.py -q`
Expected: FAIL — `No module named 'stl_core.baselines'`

- [ ] **Step 3: Write implementation**

`packages/core/src/stl_core/generator.py`:
```python
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
```

`packages/core/src/stl_core/baselines.py`:
```python
"""Sanity baselines: train-copy (privacy ceiling breaker) and independent marginals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stl_core.generator import Generator
from stl_core.metadata import ColumnType, TableMetadata


class TrainCopy(Generator):
    """Returns real training rows. Perfect fidelity, worst privacy."""

    name = "train_copy"

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        self._train = train_df.reset_index(drop=True)

    def sample(self, n: int) -> pd.DataFrame:
        replace = n > len(self._train)
        return self._train.sample(n=n, replace=replace, random_state=self.seed).reset_index(
            drop=True
        )


class IndependentMarginals(Generator):
    """Samples each column independently from its marginal distribution.

    Numerical columns use inverse-CDF interpolation between order statistics, so sampled
    values are new numbers rather than copies of training values (copied unique values
    would be flagged by singling-out attacks and make this baseline fail privacy for the
    wrong reason). Other columns are sampled by observed frequency.
    """

    name = "independent_marginals"

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        self._columns = list(train_df.columns)
        self._numeric = {c for c in self._columns
                         if metadata.columns.get(c) == ColumnType.numerical}
        self._values = {c: train_df[c].to_numpy() for c in self._columns}
        self._dtypes = train_df.dtypes

    def sample(self, n: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        data = {}
        for c in self._columns:
            if c in self._numeric:
                data[c] = np.quantile(self._values[c].astype(float), rng.random(n))
            else:
                data[c] = rng.choice(self._values[c], size=n, replace=True)
        return pd.DataFrame(data).astype(self._dtypes.to_dict())


_BUILDERS = {TrainCopy.name: TrainCopy, IndependentMarginals.name: IndependentMarginals}


def build(name: str, seed: int = 0, **params) -> Generator:
    return _BUILDERS[name](seed=seed, **params)
```

`packages/core/src/stl_core/catalog.py`:
```python
"""Static catalog of generators: which queue owns each and where its code lives.

Modules are imported lazily so the catalog can be read in any environment.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field

from stl_core.generator import Generator


@dataclass(frozen=True)
class GeneratorInfo:
    name: str
    family: str
    queue: str
    module: str
    cpu_ok: bool = True
    params: dict = field(default_factory=dict)


_SC = "stl_synthcity.generators"
_ENTRIES = [
    GeneratorInfo("train_copy", "sanity_ceiling", "q.sdv", "stl_core.baselines"),
    GeneratorInfo("independent_marginals", "sanity_floor", "q.sdv", "stl_core.baselines"),
    GeneratorInfo("marginals", "sanity_floor", "q.synthcity", _SC),
    GeneratorInfo("gaussian_copula", "statistical", "q.sdv", "stl_sdv.generators"),
    GeneratorInfo("bayesian_network", "graphical", "q.synthcity", _SC),
    GeneratorInfo("ctgan", "deep", "q.synthcity", _SC),
    GeneratorInfo("tvae", "deep", "q.synthcity", _SC),
    GeneratorInfo("tabddpm", "diffusion", "q.synthcity", _SC, cpu_ok=False),
    GeneratorInfo("great", "llm", "q.synthcity", _SC, cpu_ok=False),
    GeneratorInfo("privbayes_eps0.1", "differential_privacy", "q.synthcity", _SC,
                  params={"epsilon": 0.1}),
    GeneratorInfo("privbayes_eps1", "differential_privacy", "q.synthcity", _SC,
                  params={"epsilon": 1.0}),
    GeneratorInfo("privbayes_eps10", "differential_privacy", "q.synthcity", _SC,
                  params={"epsilon": 10.0}),
]
CATALOG: dict[str, GeneratorInfo] = {e.name: e for e in _ENTRIES}


def is_available(name: str) -> bool:
    return importlib.util.find_spec(CATALOG[name].module.split(".")[0]) is not None


def load_generator(name: str, seed: int = 0, **overrides) -> Generator:
    if name not in CATALOG:
        raise KeyError(f"unknown generator {name!r}; known: {sorted(CATALOG)}")
    info = CATALOG[name]
    if not is_available(name):
        raise ImportError(
            f"generator {name!r} needs module {info.module!r} (queue {info.queue}); "
            "run it in that worker's environment"
        )
    module = importlib.import_module(info.module)
    return module.build(name, seed=seed, **{**info.params, **overrides})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bash scripts/envrun.sh core pytest packages/core/tests -q`
Expected: all passed (`test_missing_module_message` passes, not skipped, in the core env)

- [ ] **Step 5: Checkpoint**

Run: `bash scripts/envrun.sh core ruff check packages/core` and `git status --short`.

---

### Task 4: Result schemas and seed aggregation

**Files:**
- Create: `packages/core/src/stl_core/schemas.py`, `packages/core/src/stl_core/aggregate.py`
- Test: `packages/core/tests/test_schemas.py`, `packages/core/tests/test_aggregate.py`

**Interfaces:**
- Produces (pydantic v2 models in `stl_core.schemas`):
  - `JobSpec(job_id: str, run_id: str, dataset_id: str, generator: str, seed: int, params: dict[str, Any] = {}, time_cap_seconds: float | None = None)`
  - `CostMetrics(fit_seconds: float, sample_seconds: float, peak_memory_mb: float | None = None)`
  - `FidelityMetrics(quality_score, column_shapes, column_pair_trends, detection_auc, alpha_precision, beta_recall)` — all `float | None = None`
  - `UtilityMetrics(tstr: dict[str, float | None] = {}, real_scores: dict[str, float | None] = {}, importance_spearman: float | None = None)`
  - `PrivacyMetrics(dcr_share, singling_out_risk, linkability_risk, inference_risk, mia_advantage)` — all `float | None = None`
  - `MetricReport(job_id, run_id, dataset_id, generator, seed, status: Literal["done","failed"], error: str | None = None, warnings: list[str] = [], cost: CostMetrics | None = None, fidelity: FidelityMetrics, utility: UtilityMetrics, privacy: PrivacyMetrics)` with `flat() -> dict[str, float | None]` (keys `fidelity.<f>`, `utility.tstr.<model>`, `utility.importance_spearman`, `privacy.<f>`, `cost.<f>`; `utility.real_scores` excluded)
  - `AxisScores(fidelity: float | None = None, utility: float | None = None, privacy: float | None = None)`
  - `AggregatedReport(generator: str, n_seeds: int, n_failed: int, mean: dict[str, float | None], std: dict[str, float | None], missing: list[str])`
  - `RankedCandidate(generator: str, scores: AxisScores, score: float | None, gate_passed: bool, gate_failures: list[str], on_pareto_front: bool)`
  - `Recommendation(preset: str, weights: tuple[float, float, float], winner: str | None, reason: str, ranking: list[RankedCandidate])`
  - constant `PRIVACY_KEYS: tuple[str, ...]` = the five `privacy.*` flat keys
- Produces (`stl_core.aggregate`): `aggregate(reports: list[MetricReport]) -> list[AggregatedReport]` sorted by generator name

- [ ] **Step 1: Write the failing tests**

`packages/core/tests/test_schemas.py`:
```python
from stl_core.schemas import (
    PRIVACY_KEYS, CostMetrics, FidelityMetrics, MetricReport, PrivacyMetrics, UtilityMetrics,
)


def _report(**kw):
    base = dict(job_id="j", run_id="r", dataset_id="d", generator="g", seed=0, status="done",
                fidelity=FidelityMetrics(quality_score=0.8),
                utility=UtilityMetrics(tstr={"logreg": 0.9}, real_scores={"logreg": 0.7}),
                privacy=PrivacyMetrics(dcr_share=0.5),
                cost=CostMetrics(fit_seconds=1.0, sample_seconds=0.5))
    base.update(kw)
    return MetricReport(**base)


def test_flat_keys():
    flat = _report().flat()
    assert flat["fidelity.quality_score"] == 0.8
    assert flat["utility.tstr.logreg"] == 0.9
    assert flat["privacy.dcr_share"] == 0.5
    assert flat["privacy.mia_advantage"] is None
    assert flat["cost.fit_seconds"] == 1.0
    assert not any(k.startswith("utility.real_scores") for k in flat)


def test_privacy_keys():
    assert len(PRIVACY_KEYS) == 5 and all(k.startswith("privacy.") for k in PRIVACY_KEYS)


def test_round_trip_json():
    r = _report()
    assert MetricReport.model_validate_json(r.model_dump_json()) == r
```

`packages/core/tests/test_aggregate.py`:
```python
import math

from stl_core.aggregate import aggregate
from stl_core.schemas import FidelityMetrics, MetricReport, PrivacyMetrics, UtilityMetrics


def _r(gen, seed, q, dcr=0.5, status="done"):
    return MetricReport(job_id=f"{gen}{seed}", run_id="r", dataset_id="d", generator=gen,
                        seed=seed, status=status, fidelity=FidelityMetrics(quality_score=q),
                        utility=UtilityMetrics(), privacy=PrivacyMetrics(dcr_share=dcr))


def test_mean_std_over_seeds():
    aggs = aggregate([_r("a", 0, 0.6), _r("a", 1, 0.8), _r("b", 0, 0.5)])
    assert [a.generator for a in aggs] == ["a", "b"]
    a = aggs[0]
    assert a.n_seeds == 2 and a.n_failed == 0
    assert math.isclose(a.mean["fidelity.quality_score"], 0.7)
    assert math.isclose(a.std["fidelity.quality_score"], 0.1414, rel_tol=1e-3)
    assert aggs[1].std["fidelity.quality_score"] == 0.0


def test_missing_metric_recorded():
    aggs = aggregate([_r("a", 0, 0.6, dcr=0.5), _r("a", 1, 0.6, dcr=None)])
    assert "privacy.dcr_share" in aggs[0].missing
    assert aggs[0].mean["privacy.dcr_share"] == 0.5


def test_failed_seeds_excluded():
    aggs = aggregate([_r("a", 0, 0.6), _r("a", 1, 0.0, status="failed")])
    assert aggs[0].n_failed == 1 and aggs[0].mean["fidelity.quality_score"] == 0.6


def test_all_failed():
    agg = aggregate([_r("a", 0, 0.6, status="failed")])[0]
    assert agg.n_failed == 1 and agg.mean == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `bash scripts/envrun.sh core pytest packages/core/tests/test_schemas.py packages/core/tests/test_aggregate.py -q`
Expected: FAIL — `No module named 'stl_core.schemas'`

- [ ] **Step 3: Write implementation**

`packages/core/src/stl_core/schemas.py`:
```python
"""Pydantic schemas shared by workers, CLI, API and frontend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class JobSpec(BaseModel):
    job_id: str
    run_id: str
    dataset_id: str
    generator: str
    seed: int
    params: dict[str, Any] = Field(default_factory=dict)
    time_cap_seconds: float | None = None


class CostMetrics(BaseModel):
    fit_seconds: float
    sample_seconds: float
    peak_memory_mb: float | None = None


class FidelityMetrics(BaseModel):
    quality_score: float | None = None
    column_shapes: float | None = None
    column_pair_trends: float | None = None
    detection_auc: float | None = None
    alpha_precision: float | None = None
    beta_recall: float | None = None


class UtilityMetrics(BaseModel):
    tstr: dict[str, float | None] = Field(default_factory=dict)
    real_scores: dict[str, float | None] = Field(default_factory=dict)
    importance_spearman: float | None = None


class PrivacyMetrics(BaseModel):
    dcr_share: float | None = None
    singling_out_risk: float | None = None
    linkability_risk: float | None = None
    inference_risk: float | None = None
    mia_advantage: float | None = None


PRIVACY_KEYS: tuple[str, ...] = tuple(f"privacy.{k}" for k in PrivacyMetrics.model_fields)


class MetricReport(BaseModel):
    job_id: str
    run_id: str
    dataset_id: str
    generator: str
    seed: int
    status: Literal["done", "failed"]
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    cost: CostMetrics | None = None
    fidelity: FidelityMetrics = Field(default_factory=FidelityMetrics)
    utility: UtilityMetrics = Field(default_factory=UtilityMetrics)
    privacy: PrivacyMetrics = Field(default_factory=PrivacyMetrics)

    def flat(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        out.update({f"fidelity.{k}": v for k, v in self.fidelity.model_dump().items()})
        out.update({f"utility.tstr.{k}": v for k, v in self.utility.tstr.items()})
        out["utility.importance_spearman"] = self.utility.importance_spearman
        out.update({f"privacy.{k}": v for k, v in self.privacy.model_dump().items()})
        if self.cost is not None:
            out.update({f"cost.{k}": v for k, v in self.cost.model_dump().items()})
        return out


class AxisScores(BaseModel):
    fidelity: float | None = None
    utility: float | None = None
    privacy: float | None = None


class AggregatedReport(BaseModel):
    generator: str
    n_seeds: int
    n_failed: int
    mean: dict[str, float | None]
    std: dict[str, float | None]
    missing: list[str] = Field(default_factory=list)


class RankedCandidate(BaseModel):
    generator: str
    scores: AxisScores
    score: float | None
    gate_passed: bool
    gate_failures: list[str]
    on_pareto_front: bool


class Recommendation(BaseModel):
    preset: str
    weights: tuple[float, float, float]
    winner: str | None
    reason: str
    ranking: list[RankedCandidate]
```

`packages/core/src/stl_core/aggregate.py`:
```python
"""Aggregate per-seed reports into mean ± std per generator."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from stl_core.schemas import AggregatedReport, MetricReport


def aggregate(reports: list[MetricReport]) -> list[AggregatedReport]:
    by_gen: dict[str, list[MetricReport]] = defaultdict(list)
    for r in reports:
        by_gen[r.generator].append(r)
    out = []
    for gen in sorted(by_gen):
        rs = by_gen[gen]
        done = [r.flat() for r in rs if r.status == "done"]
        keys = sorted({k for f in done for k in f})
        mean: dict[str, float | None] = {}
        std: dict[str, float | None] = {}
        missing: list[str] = []
        for k in keys:
            vals = [f.get(k) for f in done]
            present = [float(v) for v in vals if v is not None]
            if len(present) < len(vals):
                missing.append(k)
            if present:
                mean[k] = float(np.mean(present))
                std[k] = float(np.std(present, ddof=1)) if len(present) > 1 else 0.0
            else:
                mean[k], std[k] = None, None
        out.append(AggregatedReport(generator=gen, n_seeds=len(rs),
                                    n_failed=len(rs) - len(done), mean=mean, std=std,
                                    missing=missing))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bash scripts/envrun.sh core pytest packages/core/tests -q`
Expected: all passed

- [ ] **Step 5: Checkpoint**

Run: `bash scripts/envrun.sh core ruff check packages/core` and `git status --short`.

---

### Task 5: Decision engine

**Files:**
- Create: `packages/core/src/stl_core/decision.py`
- Test: `packages/core/tests/test_decision.py`

**Interfaces:**
- Consumes: `AggregatedReport`, `AxisScores`, `RankedCandidate`, `Recommendation`, `PRIVACY_KEYS` (Task 4)
- Produces:
  - `@dataclass(frozen=True) GateConfig(max_anonymeter_risk: float = 0.10, max_dcr_share: float = 0.6)`
  - `PRESETS: dict[str, tuple[float, float, float]]` keys `balanced`, `privacy_first`, `utility_first`
  - `axis_scores(flat: dict[str, float | None]) -> AxisScores`
  - `check_gate(agg: AggregatedReport, gate: GateConfig) -> list[str]` — empty list means pass; each failure reads naturally after "but", e.g. `"failed the singling-out check (risk 0.34 > 0.10)"`, `"failed the DCR check (share 0.71 > 0.60)"`, `"is missing the DCR share metric"`, `"had every seed fail"`
  - `pareto_front(points: dict[str, tuple[float, ...]]) -> set[str]` (maximise every coordinate)
  - `weighted_score(scores: AxisScores, weights: tuple[float, float, float]) -> float | None`
  - `resolve_weights(preset: str = "balanced", weights: tuple[float, float, float] | None = None) -> tuple[str, tuple[float, float, float]]` — custom weights normalised to sum 1 and labelled `"custom"`
  - `recommend(aggs: list[AggregatedReport], preset: str = "balanced", weights: tuple | None = None, gate: GateConfig = GateConfig()) -> Recommendation`

- [ ] **Step 1: Write the failing test**

`packages/core/tests/test_decision.py`:
```python
import math

import pytest

from stl_core.decision import (
    GateConfig, PRESETS, axis_scores, check_gate, pareto_front, recommend, resolve_weights,
    weighted_score,
)
from stl_core.schemas import AggregatedReport, AxisScores

GOOD_PRIVACY = {
    "privacy.dcr_share": 0.5, "privacy.singling_out_risk": 0.0,
    "privacy.linkability_risk": 0.0, "privacy.inference_risk": 0.0,
    "privacy.mia_advantage": 0.0,
}


def _agg(gen, fid=0.8, tstr=0.9, imp=0.9, privacy=None, n_failed=0, missing=()):
    mean = {"fidelity.quality_score": fid, "fidelity.detection_auc": 0.5,
            "fidelity.alpha_precision": fid, "fidelity.beta_recall": fid,
            "utility.tstr.logreg": tstr, "utility.importance_spearman": imp,
            **GOOD_PRIVACY, **(privacy or {})}
    return AggregatedReport(generator=gen, n_seeds=3, n_failed=n_failed, mean=mean,
                            std={k: 0.0 for k in mean}, missing=list(missing))


def test_axis_scores_formulas():
    s = axis_scores(_agg("g", fid=0.8, tstr=1.4, imp=-0.2,
                         privacy={"privacy.dcr_share": 0.7,
                                  "privacy.singling_out_risk": 0.2,
                                  "privacy.mia_advantage": 0.4}).mean)
    assert math.isclose(s.fidelity, (0.8 + 1.0 + 0.8 + 0.8) / 4)
    assert math.isclose(s.utility, 0.5 * 1.0 + 0.5 * 0.0)  # both clipped
    assert math.isclose(s.privacy, ((1 - 0.2) + (1 - 0.4) + (1 - 0.4)) / 3)


def test_privacy_none_when_any_privacy_metric_missing():
    flat = dict(_agg("g").mean)
    flat["privacy.mia_advantage"] = None
    assert axis_scores(flat).privacy is None


def test_gate_failures_read_naturally():
    agg = _agg("g", privacy={"privacy.singling_out_risk": 0.34, "privacy.dcr_share": 0.71})
    fails = check_gate(agg, GateConfig())
    assert "failed the singling-out check (risk 0.34 > 0.10)" in fails
    assert "failed the DCR check (share 0.71 > 0.60)" in fails
    assert check_gate(_agg("ok"), GateConfig()) == []


def test_gate_missing_and_all_failed():
    assert check_gate(_agg("g", missing=["privacy.dcr_share"]), GateConfig()) == [
        "is missing the DCR share metric"]
    assert check_gate(_agg("g", n_failed=3), GateConfig()) == ["had every seed fail"]


def test_pareto_front():
    pts = {"a": (1, 1, 0), "b": (0, 0, 1), "c": (0.5, 0.5, 0), "d": (1, 1, 0)}
    assert pareto_front(pts) == {"a", "b", "d"}


def test_weights():
    assert resolve_weights("privacy_first") == ("privacy_first", PRESETS["privacy_first"])
    label, w = resolve_weights(weights=(2, 1, 1))
    assert label == "custom" and math.isclose(w[0], 0.5)
    with pytest.raises(KeyError, match="unknown preset"):
        resolve_weights("nope")
    assert weighted_score(AxisScores(fidelity=1, utility=0, privacy=None), (1, 1, 1)) is None


def test_recommend_gates_then_ranks_with_reason():
    aggs = [
        _agg("tvae", fid=0.7, tstr=0.95, imp=0.9),
        _agg("ctgan", fid=0.95, tstr=0.9, imp=0.9,
             privacy={"privacy.singling_out_risk": 0.3}),
        _agg("copula", fid=0.6, tstr=0.7, imp=0.6),
    ]
    rec = recommend(aggs, preset="balanced")
    assert rec.winner == "tvae"
    assert [c.generator for c in rec.ranking if c.gate_passed] == ["tvae", "copula"]
    assert rec.reason.startswith("tvae: highest balanced score")
    assert "Runner-up: copula" in rec.reason
    assert "ctgan had higher fidelity but failed the singling-out check" in rec.reason
    front = {c.generator for c in rec.ranking if c.on_pareto_front}
    assert "tvae" in front and "ctgan" not in front


def test_recommend_no_winner():
    aggs = [_agg("train_copy", privacy={"privacy.dcr_share": 0.9}),
            _agg("x", missing=["privacy.inference_risk"])]
    rec = recommend(aggs)
    assert rec.winner is None
    assert rec.reason.startswith("No generator passed the privacy gate.")
    assert "train_copy failed the DCR check" in rec.reason
    assert "x is missing the inference risk metric" in rec.reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash scripts/envrun.sh core pytest packages/core/tests/test_decision.py -q`
Expected: FAIL — `No module named 'stl_core.decision'`

- [ ] **Step 3: Write implementation**

`packages/core/src/stl_core/decision.py`:
```python
"""Privacy-gated decision engine: gate, Pareto front, weighted score, reason."""

from __future__ import annotations

from dataclasses import dataclass

from stl_core.schemas import (
    PRIVACY_KEYS, AggregatedReport, AxisScores, RankedCandidate, Recommendation,
)

AXES = ("fidelity", "utility", "privacy")
PRESETS: dict[str, tuple[float, float, float]] = {
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "privacy_first": (0.2, 0.2, 0.6),
    "utility_first": (0.2, 0.6, 0.2),
}
ANONYMETER_CHECKS = {
    "privacy.singling_out_risk": "singling-out",
    "privacy.linkability_risk": "linkability",
    "privacy.inference_risk": "inference",
}
METRIC_LABELS = {
    "privacy.dcr_share": "DCR share",
    "privacy.singling_out_risk": "singling-out risk",
    "privacy.linkability_risk": "linkability risk",
    "privacy.inference_risk": "inference risk",
    "privacy.mia_advantage": "membership-inference advantage",
}


@dataclass(frozen=True)
class GateConfig:
    max_anonymeter_risk: float = 0.10
    max_dcr_share: float = 0.6


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def axis_scores(flat: dict[str, float | None]) -> AxisScores:
    get = flat.get
    det = get("fidelity.detection_auc")
    fidelity = _mean([
        get("fidelity.quality_score"),
        None if det is None else 1 - 2 * abs(det - 0.5),
        get("fidelity.alpha_precision"),
        get("fidelity.beta_recall"),
    ])
    tstr = [_clip01(v) for k, v in flat.items() if k.startswith("utility.tstr.") and v is not None]
    t = sum(tstr) / len(tstr) if tstr else None
    imp = get("utility.importance_spearman")
    imp = None if imp is None else _clip01(imp)
    if t is not None and imp is not None:
        utility = 0.5 * t + 0.5 * imp
    else:
        utility = t if t is not None else imp
    privacy = None
    if all(get(k) is not None for k in PRIVACY_KEYS):
        anon = max(get(k) for k in ANONYMETER_CHECKS)
        privacy = (
            (1 - _clip01(anon))
            + (1 - 2 * max(0.0, get("privacy.dcr_share") - 0.5))
            + (1 - _clip01(get("privacy.mia_advantage")))
        ) / 3
    return AxisScores(
        fidelity=None if fidelity is None else _clip01(fidelity),
        utility=None if utility is None else _clip01(utility),
        privacy=None if privacy is None else _clip01(privacy),
    )


def check_gate(agg: AggregatedReport, gate: GateConfig) -> list[str]:
    if agg.n_failed >= agg.n_seeds:
        return ["had every seed fail"]
    failures = [
        f"is missing the {METRIC_LABELS[k]} metric"
        for k in PRIVACY_KEYS
        if agg.mean.get(k) is None or k in agg.missing
    ]
    for key, label in ANONYMETER_CHECKS.items():
        v = agg.mean.get(key)
        if v is not None and v > gate.max_anonymeter_risk:
            failures.append(
                f"failed the {label} check (risk {v:.2f} > {gate.max_anonymeter_risk:.2f})"
            )
    dcr = agg.mean.get("privacy.dcr_share")
    if dcr is not None and dcr > gate.max_dcr_share:
        failures.append(f"failed the DCR check (share {dcr:.2f} > {gate.max_dcr_share:.2f})")
    return failures


def pareto_front(points: dict[str, tuple[float, ...]]) -> set[str]:
    front = set()
    for a, pa in points.items():
        dominated = any(
            all(qb >= qa for qa, qb in zip(pa, pb)) and any(qb > qa for qa, qb in zip(pa, pb))
            for b, pb in points.items()
            if b != a
        )
        if not dominated:
            front.add(a)
    return front


def weighted_score(scores: AxisScores, weights: tuple[float, float, float]) -> float | None:
    values = [getattr(scores, a) for a in AXES]
    if any(v is None for v in values):
        return None
    total = sum(weights)
    return sum(w * v for w, v in zip(weights, values)) / total


def resolve_weights(
    preset: str = "balanced", weights: tuple[float, float, float] | None = None
) -> tuple[str, tuple[float, float, float]]:
    if weights is not None:
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("custom weights must sum to a positive number")
        return "custom", tuple(float(w) / total for w in weights)  # type: ignore[return-value]
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return preset, PRESETS[preset]


def _reason(winner, passing, ranking, label) -> str:
    if winner is None:
        parts = [
            f"{c.generator} {c.gate_failures[0] if c.gate_failures else 'has incomplete scores'}"
            for c in ranking
        ]
        return "No generator passed the privacy gate. " + "; ".join(parts) + "."
    best_axis = max(AXES, key=lambda a: getattr(winner.scores, a))
    text = (
        f"{winner.generator}: highest {label} score ({winner.score:.2f}) among models passing "
        f"the privacy gate; strongest on {best_axis} ({getattr(winner.scores, best_axis):.2f})."
    )
    if len(passing) > 1:
        text += f" Runner-up: {passing[1].generator} ({passing[1].score:.2f})."
    for c in ranking:
        if c.gate_passed:
            continue
        better = [
            a for a in ("fidelity", "utility")
            if getattr(c.scores, a) is not None
            and getattr(c.scores, a) > getattr(winner.scores, a)
        ]
        if better:
            text += f" {c.generator} had higher {better[0]} but {c.gate_failures[0]}."
    return text


def recommend(
    aggs: list[AggregatedReport],
    preset: str = "balanced",
    weights: tuple[float, float, float] | None = None,
    gate: GateConfig = GateConfig(),
) -> Recommendation:
    label, w = resolve_weights(preset, weights)
    candidates = []
    for agg in aggs:
        scores = axis_scores(agg.mean)
        failures = check_gate(agg, gate)
        candidates.append(dict(generator=agg.generator, scores=scores,
                               score=weighted_score(scores, w), gate_failures=failures,
                               gate_passed=not failures))
    eligible = {
        c["generator"]: tuple(getattr(c["scores"], a) for a in AXES)
        for c in candidates
        if c["gate_passed"] and c["score"] is not None
    }
    front = pareto_front(eligible)
    ranking = sorted(
        (RankedCandidate(**c, on_pareto_front=c["generator"] in front) for c in candidates),
        key=lambda c: (c.score is None, -(c.score or 0.0), c.generator),
    )
    passing = [c for c in ranking if c.gate_passed and c.score is not None]
    winner = passing[0] if passing else None
    return Recommendation(preset=label, weights=w, winner=winner.generator if winner else None,
                          reason=_reason(winner, passing, ranking, label), ranking=ranking)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bash scripts/envrun.sh core pytest packages/core/tests -q`
Expected: all passed

- [ ] **Step 5: Checkpoint**

Run: `bash scripts/envrun.sh core ruff check packages/core` and `git status --short`.

---

### Task 6: Runner package — keys, storage, cost measurement, dataset loading

**Files:**
- Create: `packages/runner/pyproject.toml`, `packages/runner/src/stl_runner/__init__.py`, `packages/runner/src/stl_runner/keys.py`, `packages/runner/src/stl_runner/storage.py`, `packages/runner/src/stl_runner/cost.py`, `packages/runner/src/stl_runner/datasets.py`
- Modify: `.gitignore` (append `mlartifacts/`, `*.db`)
- Test: `packages/runner/tests/test_storage.py`, `packages/runner/tests/test_cost.py`, `packages/runner/tests/test_datasets.py`

**Interfaces:**
- Consumes: pydantic `BaseModel` (any schema from Task 4)
- Produces:
  - `stl_runner.keys`: `dataset_raw(d)`, `dataset_metadata(d)`, `split_part(d, seed, part)`, `run_manifest(r)`, `synthetic(r, j)`, `cost(r, j)`, `status(r, j)`, `report(r, j)` → `str` keys (POSIX, relative)
  - `ArtifactStore(root: str, storage_options: dict | None = None)` with `.root`, `.storage_options`, `exists(key) -> bool`, `write_bytes/read_bytes`, `write_df(key, df)`, `read_df(key) -> pd.DataFrame`, `write_json(key, obj: dict | BaseModel)`, `read_json(key) -> dict`, `list(prefix: str) -> list[str]` (keys relative to root, sorted)
  - `store_from_env() -> ArtifactStore` — `STL_STORAGE_ROOT` (default `outputs/store`); for `s3://` roots reads `S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
  - `stl_runner.cost.measure()` context manager yielding `Measurement(seconds: float, peak_mb: float)` (filled on exit)
  - `stl_runner.datasets.load_dataset(cfg: Mapping) -> tuple[pd.DataFrame, str]` — `cfg["source"]` is `openml` (`openml_id` or `openml_name` + `version`) or `file` (`path`, `.csv` or `.parquet`); optional `drop: list[str]`; converts pandas `category` columns to `object`

- [ ] **Step 1: Create the package**

`packages/runner/pyproject.toml`:
```toml
[project]
name = "stl-runner"
version = "0.1.0"
description = "Synthetic Data Trust Lab runtime: storage, datasets, pipeline, tracking"
requires-python = ">=3.10"
dependencies = [
  "stl-core",
  "pyarrow>=14",
  "fsspec>=2024.2",
  "psutil>=5.9",
  "scikit-learn>=1.3",
  "hydra-core>=1.3",
]

[project.optional-dependencies]
s3 = ["s3fs>=2024.2"]
mlflow = ["mlflow-skinny>=3", "sqlalchemy>=2", "alembic>=1.13"]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[tool.uv.sources]
stl-core = { path = "../core", editable = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/stl_runner"]

[tool.pytest.ini_options]
markers = ["network: needs internet access (OpenML)"]
```

`packages/runner/src/stl_runner/__init__.py`:
```python
"""Synthetic Data Trust Lab runtime shared by the CLI and the workers."""
```

`packages/runner/src/stl_runner/keys.py`:
```python
"""Artifact key layout. Every stage reads and writes through these keys."""


def dataset_raw(dataset_id: str) -> str:
    return f"datasets/{dataset_id}/raw.parquet"


def dataset_metadata(dataset_id: str) -> str:
    return f"datasets/{dataset_id}/metadata.json"


def split_part(dataset_id: str, seed: int, part: str) -> str:
    return f"datasets/{dataset_id}/splits/{seed}/{part}.parquet"


def run_manifest(run_id: str) -> str:
    return f"runs/{run_id}/manifest.json"


def synthetic(run_id: str, job_id: str) -> str:
    return f"runs/{run_id}/{job_id}/synthetic.parquet"


def cost(run_id: str, job_id: str) -> str:
    return f"runs/{run_id}/{job_id}/cost.json"


def status(run_id: str, job_id: str) -> str:
    return f"runs/{run_id}/{job_id}/status.json"


def report(run_id: str, job_id: str) -> str:
    return f"runs/{run_id}/{job_id}/report.json"
```

Append to `.gitignore`:
```
mlartifacts/
*.db
```

- [ ] **Step 2: Write the failing tests**

`packages/runner/tests/test_storage.py`:
```python
import pandas as pd
from pydantic import BaseModel

from stl_runner.storage import ArtifactStore, store_from_env


class M(BaseModel):
    a: int


def test_df_round_trip(tmp_path):
    store = ArtifactStore(str(tmp_path))
    df = pd.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]})
    store.write_df("datasets/d/raw.parquet", df)
    assert store.exists("datasets/d/raw.parquet")
    pd.testing.assert_frame_equal(store.read_df("datasets/d/raw.parquet"), df)


def test_json_and_models(tmp_path):
    store = ArtifactStore(str(tmp_path))
    store.write_json("a/b.json", {"k": 1})
    store.write_json("a/m.json", M(a=3))
    assert store.read_json("a/b.json") == {"k": 1}
    assert M.model_validate(store.read_json("a/m.json")).a == 3
    assert store.list("a") == ["a/b.json", "a/m.json"]
    assert not store.exists("a/zzz.json")
    assert store.list("missing") == []


def test_store_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STL_STORAGE_ROOT", str(tmp_path / "s"))
    store = store_from_env()
    store.write_json("x.json", {"ok": True})
    assert (tmp_path / "s" / "x.json").exists()
```

`packages/runner/tests/test_cost.py`:
```python
import time

import numpy as np

from stl_runner.cost import measure


def test_measure_time_and_memory():
    with measure() as m:
        a = np.ones((2000, 2000))
        time.sleep(0.1)
        del a
    assert m.seconds >= 0.1
    assert m.peak_mb > 10
```

`packages/runner/tests/test_datasets.py`:
```python
import pandas as pd
import pytest

from stl_runner.datasets import load_dataset


def test_load_file_csv(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "junk": [0, 0]}).to_csv(p, index=False)
    df, target = load_dataset({"source": "file", "path": str(p), "target": "b", "drop": ["junk"]})
    assert target == "b" and list(df.columns) == ["a", "b"]


def test_category_columns_become_object(tmp_path):
    p = tmp_path / "d.parquet"
    pd.DataFrame({"c": pd.Categorical(["a", "b"]), "y": [0, 1]}).to_parquet(p)
    df, _ = load_dataset({"source": "file", "path": str(p), "target": "y"})
    assert df["c"].dtype == object


def test_unknown_source():
    with pytest.raises(ValueError, match="source"):
        load_dataset({"source": "ftp", "target": "y"})


@pytest.mark.network
def test_openml_adult():
    df, target = load_dataset({"source": "openml", "openml_name": "adult", "version": 2,
                               "target": "class"})
    assert target == "class" and len(df) > 40_000
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `bash scripts/envrun.sh runner pytest packages/runner/tests -q`
Expected: FAIL — `No module named 'stl_runner.storage'`

- [ ] **Step 4: Write implementation**

`packages/runner/src/stl_runner/storage.py`:
```python
"""Artifact store over fsspec: local directory or S3/MinIO."""

from __future__ import annotations

import io
import json
import os

import fsspec
import pandas as pd
from pydantic import BaseModel


class ArtifactStore:
    def __init__(self, root: str, storage_options: dict | None = None) -> None:
        self.root = root
        self.storage_options = storage_options or {}
        self.fs, base = fsspec.core.url_to_fs(root, **self.storage_options)
        self._base = base.rstrip("/")

    def _path(self, key: str) -> str:
        return f"{self._base}/{key}"

    def exists(self, key: str) -> bool:
        return self.fs.exists(self._path(key))

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        with self.fs.open(path, "wb") as f:
            f.write(data)

    def read_bytes(self, key: str) -> bytes:
        with self.fs.open(self._path(key), "rb") as f:
            return f.read()

    def write_df(self, key: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.write_bytes(key, buf.getvalue())

    def read_df(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.read_bytes(key)))

    def write_json(self, key: str, obj: dict | BaseModel) -> None:
        if isinstance(obj, BaseModel):
            text = obj.model_dump_json(indent=2)
        else:
            text = json.dumps(obj, indent=2, default=str)
        self.write_bytes(key, text.encode())

    def read_json(self, key: str) -> dict:
        return json.loads(self.read_bytes(key))

    def list(self, prefix: str) -> list[str]:
        path = self._path(prefix)
        if not self.fs.exists(path):
            return []
        base = self.fs._strip_protocol(self._base).rstrip("/") + "/"
        found = [self.fs._strip_protocol(p) for p in self.fs.find(path)]
        return sorted(p[len(base):] if p.startswith(base) else p for p in found)


def store_from_env() -> ArtifactStore:
    root = os.environ.get("STL_STORAGE_ROOT", "outputs/store")
    options: dict = {}
    if root.startswith("s3://"):
        options = {
            "key": os.environ.get("AWS_ACCESS_KEY_ID"),
            "secret": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "client_kwargs": {"endpoint_url": os.environ.get("S3_ENDPOINT_URL")},
        }
    return ArtifactStore(root, options)
```

`packages/runner/src/stl_runner/cost.py`:
```python
"""Wall-clock time and peak resident memory for a block of code."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psutil


@dataclass
class Measurement:
    seconds: float = 0.0
    peak_mb: float = 0.0


@contextmanager
def measure(poll_seconds: float = 0.05) -> Iterator[Measurement]:
    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(poll_seconds):
            peak[0] = max(peak[0], proc.memory_info().rss)

    thread = threading.Thread(target=poll, daemon=True)
    m = Measurement()
    thread.start()
    start = time.perf_counter()
    try:
        yield m
    finally:
        m.seconds = time.perf_counter() - start
        stop.set()
        thread.join()
        m.peak_mb = max(peak[0], proc.memory_info().rss) / 2**20
```

`packages/runner/src/stl_runner/datasets.py`:
```python
"""Load benchmark datasets from OpenML or local files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def load_dataset(cfg: Mapping) -> tuple[pd.DataFrame, str]:
    source = cfg.get("source")
    if source == "openml":
        from sklearn.datasets import fetch_openml

        if cfg.get("openml_id") is not None:
            kwargs = {"data_id": int(cfg["openml_id"])}
        else:
            kwargs = {"name": cfg["openml_name"], "version": cfg.get("version", "active")}
        df = fetch_openml(as_frame=True, parser="auto", **kwargs).frame
    elif source == "file":
        path = Path(cfg["path"])
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    else:
        raise ValueError(f"unknown dataset source {source!r}; use 'openml' or 'file'")
    df = df.drop(columns=list(cfg.get("drop") or []))
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(object)
    return df.reset_index(drop=True), str(cfg["target"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `bash scripts/envrun.sh runner pytest packages/runner/tests -q`
Expected: all passed (the `network` test downloads Adult once, ~2 MB, cached under `~/scikit_learn_data`)

- [ ] **Step 6: Checkpoint**

Run: `bash scripts/envrun.sh runner ruff check packages/runner` and `git status --short`.

---

### Task 7: Runner pipeline — register, split, plan, generate with time cap

**Files:**
- Create: `packages/runner/src/stl_runner/pipeline.py`
- Test: `packages/runner/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `prepare`, `TableMetadata` (Task 1); `split`, `stratified_subsample`, `Splits` (Task 2); `load_generator` (Task 3); `JobSpec`, `CostMetrics` (Task 4); `ArtifactStore`, `keys`, `measure` (Task 6)
- Produces:
  - `register_dataset(store, dataset_id: str, df: pd.DataFrame, metadata: TableMetadata, max_rows: int = 20_000, seed: int = 0) -> pd.DataFrame`
  - `load_metadata(store, dataset_id) -> TableMetadata`
  - `ensure_splits(store, dataset_id, seed: int) -> Splits` (creates once, then reads)
  - `plan_run(store, run_id, dataset_id, generators: list[str], seeds: list[int], params: dict[str, dict] | None = None, time_cap_seconds: float | None = None) -> list[JobSpec]` — `job_id = f"{generator}__s{seed}"`; writes manifest and a `queued` status for jobs that have none
  - `load_jobs(store, run_id) -> list[JobSpec]`
  - `job_state(store, job) -> dict` — `{"state": ..., "error": ...}`; states `queued | fitting | sampling | generated | evaluating | done | failed`
  - `set_state(store, job, state: str, error: str | None = None) -> None`
  - `generate_job(store, job) -> CostMetrics` — samples `len(train)` rows; writes synthetic, cost and `generated` state
  - `run_generate_with_cap(store, job) -> str` — returns final state; on exception or time-cap breach writes `failed` with the error text

- [ ] **Step 1: Write the failing test**

`packages/runner/tests/test_pipeline.py`:
```python
import numpy as np
import pandas as pd

from stl_core.metadata import ColumnType, TableMetadata
from stl_core.schemas import JobSpec
from stl_runner import keys
from stl_runner.pipeline import (
    ensure_splits, generate_job, job_state, load_jobs, load_metadata, plan_run, register_dataset,
    run_generate_with_cap,
)
from stl_runner.storage import ArtifactStore

MD = TableMetadata(columns={"x": ColumnType.numerical, "c": ColumnType.categorical,
                            "y": ColumnType.categorical}, target="y")


def _df(n=500):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=n), "c": rng.choice(["a", "b"], n),
                       "y": rng.choice(["p", "n"], n)})
    df.loc[0, "x"] = np.nan
    return df


def _store(tmp_path):
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", _df(), MD, max_rows=400)
    return store


def test_register_caps_and_prepares(tmp_path):
    store = _store(tmp_path)
    raw = store.read_df(keys.dataset_raw("toy"))
    assert len(raw) == 400 and raw.isna().sum().sum() == 0
    assert load_metadata(store, "toy") == MD


def test_ensure_splits_is_stable(tmp_path):
    store = _store(tmp_path)
    a = ensure_splits(store, "toy", 0)
    b = ensure_splits(store, "toy", 0)
    assert a.fingerprint() == b.fingerprint()
    assert store.exists(keys.split_part("toy", 0, "control"))


def test_plan_and_generate(tmp_path):
    store = _store(tmp_path)
    jobs = plan_run(store, "r1", "toy", ["train_copy", "independent_marginals"], [0, 1])
    assert [j.job_id for j in jobs][:2] == ["train_copy__s0", "train_copy__s1"]
    assert load_jobs(store, "r1") == jobs
    assert job_state(store, jobs[0])["state"] == "queued"
    cost = generate_job(store, jobs[0])
    assert cost.fit_seconds >= 0 and cost.peak_memory_mb > 0
    synth = store.read_df(keys.synthetic("r1", "train_copy__s0"))
    assert len(synth) == len(ensure_splits(store, "toy", 0).train)
    assert job_state(store, jobs[0])["state"] == "generated"


def test_generate_failure_is_recorded(tmp_path):
    store = _store(tmp_path)
    job = JobSpec(job_id="ctgan__s0", run_id="r1", dataset_id="toy", generator="ctgan", seed=0)
    assert run_generate_with_cap(store, job) == "failed"
    assert "q.synthcity" in job_state(store, job)["error"]


def test_time_cap_breach_fails_job(tmp_path):
    store = _store(tmp_path)
    job = JobSpec(job_id="train_copy__s0", run_id="r1", dataset_id="toy",
                  generator="train_copy", seed=0, time_cap_seconds=0.05)
    assert run_generate_with_cap(store, job) == "failed"
    assert "time cap" in job_state(store, job)["error"]


def test_capped_run_succeeds_with_room(tmp_path):
    store = _store(tmp_path)
    job = JobSpec(job_id="train_copy__s0", run_id="r1", dataset_id="toy",
                  generator="train_copy", seed=0, time_cap_seconds=120)
    assert run_generate_with_cap(store, job) == "generated"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash scripts/envrun.sh runner pytest packages/runner/tests/test_pipeline.py -q`
Expected: FAIL — `No module named 'stl_runner.pipeline'`

- [ ] **Step 3: Write implementation**

`packages/runner/src/stl_runner/pipeline.py`:
```python
"""Pipeline stages shared by the CLI (local backend) and the Celery workers."""

from __future__ import annotations

import multiprocessing as mp
import traceback

import pandas as pd

from stl_core.catalog import load_generator
from stl_core.metadata import TableMetadata, prepare
from stl_core.schemas import CostMetrics, JobSpec
from stl_core.split import Splits, split, stratified_subsample
from stl_runner import keys
from stl_runner.cost import measure
from stl_runner.storage import ArtifactStore

PARTS = ("train", "control", "test")


def register_dataset(
    store: ArtifactStore,
    dataset_id: str,
    df: pd.DataFrame,
    metadata: TableMetadata,
    max_rows: int = 20_000,
    seed: int = 0,
) -> pd.DataFrame:
    prepared = stratified_subsample(prepare(df, metadata), metadata, max_rows, seed)
    store.write_df(keys.dataset_raw(dataset_id), prepared)
    store.write_json(keys.dataset_metadata(dataset_id), metadata)
    return prepared


def load_metadata(store: ArtifactStore, dataset_id: str) -> TableMetadata:
    return TableMetadata.model_validate(store.read_json(keys.dataset_metadata(dataset_id)))


def ensure_splits(store: ArtifactStore, dataset_id: str, seed: int) -> Splits:
    part_keys = [keys.split_part(dataset_id, seed, p) for p in PARTS]
    if all(store.exists(k) for k in part_keys):
        return Splits(*(store.read_df(k) for k in part_keys))
    df = store.read_df(keys.dataset_raw(dataset_id))
    splits = split(df, load_metadata(store, dataset_id), seed)
    for k, part in zip(part_keys, (splits.train, splits.control, splits.test)):
        store.write_df(k, part)
    return Splits(*(store.read_df(k) for k in part_keys))


def set_state(store: ArtifactStore, job: JobSpec, state: str, error: str | None = None) -> None:
    store.write_json(keys.status(job.run_id, job.job_id), {"state": state, "error": error})


def job_state(store: ArtifactStore, job: JobSpec) -> dict:
    key = keys.status(job.run_id, job.job_id)
    return store.read_json(key) if store.exists(key) else {"state": "queued", "error": None}


def plan_run(
    store: ArtifactStore,
    run_id: str,
    dataset_id: str,
    generators: list[str],
    seeds: list[int],
    params: dict[str, dict] | None = None,
    time_cap_seconds: float | None = None,
) -> list[JobSpec]:
    params = params or {}
    jobs = [
        JobSpec(job_id=f"{g}__s{s}", run_id=run_id, dataset_id=dataset_id, generator=g, seed=s,
                params=dict(params.get(g, {})), time_cap_seconds=time_cap_seconds)
        for g in generators
        for s in seeds
    ]
    store.write_json(keys.run_manifest(run_id), {
        "run_id": run_id, "dataset_id": dataset_id,
        "jobs": [j.model_dump(mode="json") for j in jobs],
    })
    for job in jobs:
        if not store.exists(keys.status(run_id, job.job_id)):
            set_state(store, job, "queued")
    return jobs


def load_jobs(store: ArtifactStore, run_id: str) -> list[JobSpec]:
    manifest = store.read_json(keys.run_manifest(run_id))
    return [JobSpec.model_validate(j) for j in manifest["jobs"]]


def generate_job(store: ArtifactStore, job: JobSpec) -> CostMetrics:
    metadata = load_metadata(store, job.dataset_id)
    splits = ensure_splits(store, job.dataset_id, job.seed)
    generator = load_generator(job.generator, seed=job.seed, **job.params)
    set_state(store, job, "fitting")
    with measure() as fit_m:
        generator.fit(splits.train, metadata)
    set_state(store, job, "sampling")
    with measure() as sample_m:
        synthetic = generator.sample(len(splits.train))
    cost = CostMetrics(fit_seconds=fit_m.seconds, sample_seconds=sample_m.seconds,
                       peak_memory_mb=max(fit_m.peak_mb, sample_m.peak_mb))
    store.write_df(keys.synthetic(job.run_id, job.job_id), synthetic)
    store.write_json(keys.cost(job.run_id, job.job_id), cost)
    set_state(store, job, "generated")
    return cost


def _generate_entry(root: str, storage_options: dict, job_json: str) -> None:
    store = ArtifactStore(root, storage_options)
    job = JobSpec.model_validate_json(job_json)
    try:
        generate_job(store, job)
    except BaseException:
        set_state(store, job, "failed", traceback.format_exc())
        raise SystemExit(1) from None


def run_generate_with_cap(store: ArtifactStore, job: JobSpec) -> str:
    if job.time_cap_seconds is None:
        try:
            generate_job(store, job)
        except Exception:
            set_state(store, job, "failed", traceback.format_exc())
        return job_state(store, job)["state"]
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_generate_entry,
                       args=(store.root, store.storage_options, job.model_dump_json()))
    proc.start()
    proc.join(job.time_cap_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        set_state(store, job, "failed", f"time cap of {job.time_cap_seconds}s exceeded")
    return job_state(store, job)["state"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bash scripts/envrun.sh runner pytest packages/runner/tests -q -m "not network"`
Expected: all passed

- [ ] **Step 5: Checkpoint**

Run: `bash scripts/envrun.sh runner ruff check packages/runner` and `git status --short`.

---

### Task 8: Eval worker environment, preprocessing and fidelity metrics

**Files:**
- Create: `workers/eval/pyproject.toml`, `workers/eval/src/stl_eval/__init__.py`, `workers/eval/src/stl_eval/preprocessing.py`, `workers/eval/src/stl_eval/fidelity.py`, `workers/eval/tests/conftest.py`
- Test: `workers/eval/tests/test_preprocessing.py`, `workers/eval/tests/test_fidelity.py`

**Interfaces:**
- Consumes: `TableMetadata`, `ColumnType`, `MISSING_CATEGORY` (Task 1)
- Produces:
  - `align(real: pd.DataFrame, synth: pd.DataFrame, metadata) -> pd.DataFrame` — returns `synth[metadata.model_columns]` coerced to real's types; raises `ValueError` naming missing columns or a numeric column whose values are mostly non-numeric
  - `numeric_view(df, metadata) -> pd.DataFrame` — datetimes become float seconds
  - `DistanceEncoder(metadata)` with `fit(ref) -> self`, `transform(df) -> np.ndarray[float32]` (min-max numerics by ref range; one-hot categoricals by ref categories; unseen → all-zero)
  - `quality(real, synth, metadata) -> tuple[float, float, float]` (overall, column shapes, column pair trends)
  - `detection_auc(real, synth, metadata, seed: int, max_rows: int = 5000) -> float`
  - `precision_recall(real, synth, metadata, seed: int, k: int = 5, max_rows: int = 5000) -> tuple[float, float]` — kNN precision/recall (Kynkäänniemi et al. 2019), reported as α-precision / β-recall
  - Test fixture `toy` in `conftest.py`: returns `(train, control, test, metadata)` of a 1,500-row correlated mixed-type table; fixture `fresh` returns another independent 900-row draw from the same distribution

- [ ] **Step 1: Create the environment**

`workers/eval/pyproject.toml`:
```toml
[project]
name = "stl-eval"
version = "0.1.0"
description = "Evaluation worker: fidelity, utility, privacy metrics"
requires-python = ">=3.11,<3.12"
dependencies = [
  "stl-core",
  "stl-runner[mlflow,s3]",
  "sdmetrics>=0.32",
  "anonymeter==1.0.0",
  "scikit-learn>=1.3",
  "xgboost>=2",
  "scipy>=1.10",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[tool.uv.sources]
stl-core = { path = "../../packages/core", editable = true }
stl-runner = { path = "../../packages/runner", editable = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/stl_eval"]

[tool.pytest.ini_options]
markers = ["network: needs internet access (OpenML)", "slow: takes more than a minute"]
filterwarnings = ["ignore::FutureWarning", "ignore::DeprecationWarning"]
```

`workers/eval/src/stl_eval/__init__.py`:
```python
"""Evaluation worker for the Synthetic Data Trust Lab."""
```

Run: `bash scripts/envrun.sh eval python -c "import anonymeter, sdmetrics, xgboost, scipy.stats; print('ok')"`
Expected: `ok` (uv creates the env at `~/.stl/eval`, downloading Python 3.11 if needed, and writes `workers/eval/uv.lock`).

`workers/eval/tests/conftest.py`:
```python
import numpy as np
import pandas as pd
import pytest

from stl_core.metadata import ColumnType, TableMetadata
from stl_core.split import split

MD = TableMetadata(
    columns={"x1": ColumnType.numerical, "x2": ColumnType.numerical,
             "c": ColumnType.categorical, "y": ColumnType.categorical},
    target="y",
)


def draw(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 2 * x1 + rng.normal(scale=0.3, size=n)
    c = np.where(x1 + rng.normal(scale=0.5, size=n) > 0, "hi", "lo")
    c = np.where(rng.random(n) < 0.1, "mid", c)
    y = np.where(x1 + 0.5 * (c == "hi") + rng.normal(scale=0.5, size=n) > 0.3, "pos", "neg")
    return pd.DataFrame({"x1": x1, "x2": x2, "c": c.astype(object), "y": y.astype(object)})


@pytest.fixture(scope="session")
def toy():
    s = split(draw(1500, 0), MD, seed=0)
    return s.train, s.control, s.test, MD


@pytest.fixture(scope="session")
def fresh():
    return draw(900, 99)
```

- [ ] **Step 2: Write the failing tests**

`workers/eval/tests/test_preprocessing.py`:
```python
import numpy as np
import pytest

from stl_eval.preprocessing import DistanceEncoder, align


def test_align_coerces_and_orders(toy):
    train, _, _, md = toy
    synth = train[["y", "c", "x2", "x1"]].copy()
    synth["x1"] = synth["x1"].astype(str)
    synth.loc[0, "c"] = "never-seen"
    out = align(train, synth, md)
    assert list(out.columns) == md.model_columns
    assert out["x1"].dtype == float
    assert out.loc[0, "c"] == "never-seen"


def test_align_missing_column_named(toy):
    train, _, _, md = toy
    with pytest.raises(ValueError, match="x2"):
        align(train, train.drop(columns=["x2"]), md)


def test_align_non_numeric_column_named(toy):
    train, _, _, md = toy
    bad = train.copy()
    bad["x1"] = "abc"
    with pytest.raises(ValueError, match="x1"):
        align(train, bad, md)


def test_distance_encoder_ranges_and_unseen(toy):
    train, _, _, md = toy
    enc = DistanceEncoder(md).fit(train)
    X = enc.transform(train)
    assert X.dtype == np.float32
    assert X.min() >= 0 and X.max() <= 1
    other = train.head(3).copy()
    other["c"] = "zzz"
    Xo = enc.transform(other)
    assert np.isfinite(Xo).all()
    # layout: x1, x2, then one-hot c (hi, lo, mid), then one-hot y (neg, pos)
    assert X.shape[1] == 7
    assert Xo[:, 2:5].sum() == 0  # unseen category encodes as all zeros
```

`workers/eval/tests/test_fidelity.py`:
```python
import numpy as np

from stl_eval.fidelity import detection_auc, precision_recall, quality


def test_quality_high_for_same_distribution(toy, fresh):
    train, _, _, md = toy
    overall, shapes, pairs = quality(train, fresh, md)
    assert overall > 0.85 and shapes > 0.85 and pairs > 0.85


def test_quality_pairs_drop_when_columns_shuffled(toy, fresh):
    train, _, _, md = toy
    shuffled = fresh.copy()
    rng = np.random.default_rng(0)
    for col in shuffled:
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
    _, _, pairs_good = quality(train, fresh, md)
    _, _, pairs_bad = quality(train, shuffled, md)
    assert pairs_bad < pairs_good - 0.1


def test_detection_auc(toy, fresh):
    train, _, _, md = toy
    assert detection_auc(train, fresh, md, seed=0) < 0.65
    shifted = fresh.copy()
    shifted["x1"] = shifted["x1"] + 3
    assert detection_auc(train, shifted, md, seed=0) > 0.9


def test_precision_recall(toy, fresh):
    train, _, _, md = toy
    p, r = precision_recall(train, fresh, md, seed=0)
    assert p > 0.8 and r > 0.8
    collapsed = fresh.copy()
    collapsed[:] = fresh.iloc[[0] * len(fresh)].to_numpy()
    _, r_bad = precision_recall(train, collapsed, md, seed=0)
    assert r_bad < 0.2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `bash scripts/envrun.sh eval pytest workers/eval/tests -q`
Expected: FAIL — `No module named 'stl_eval.preprocessing'`

- [ ] **Step 4: Write implementation**

`workers/eval/src/stl_eval/preprocessing.py`:
```python
"""Align synthetic tables to real ones and encode them for distances and models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stl_core.metadata import MISSING_CATEGORY, ColumnType, TableMetadata


def align(real: pd.DataFrame, synth: pd.DataFrame, metadata: TableMetadata) -> pd.DataFrame:
    cols = metadata.model_columns
    missing = [c for c in cols if c not in synth.columns]
    if missing:
        raise ValueError(f"synthetic data is missing columns: {missing}")
    out = synth[cols].copy()
    for col in cols:
        ct = metadata.columns[col]
        if ct == ColumnType.numerical:
            values = pd.to_numeric(out[col], errors="coerce")
            if values.isna().mean() > 0.5:
                raise ValueError(f"synthetic column {col!r} is not numeric")
            out[col] = values.fillna(real[col].median()).astype(float)
        elif ct == ColumnType.categorical:
            out[col] = out[col].where(out[col].notna(), MISSING_CATEGORY).astype(str)
            out[col] = out[col].astype(object)
        elif ct == ColumnType.datetime:
            values = pd.to_datetime(out[col], errors="coerce")
            out[col] = values.fillna(real[col].median())
    return out.reset_index(drop=True)


def numeric_view(df: pd.DataFrame, metadata: TableMetadata) -> pd.DataFrame:
    out = df.copy()
    for col in metadata.of_type(ColumnType.datetime):
        if col in out:
            out[col] = pd.to_datetime(out[col]).astype("int64") / 1e9
    return out


class DistanceEncoder:
    """Min-max numericals and one-hot categoricals, both fitted on a reference table."""

    def __init__(self, metadata: TableMetadata) -> None:
        self.metadata = metadata
        cols = metadata.model_columns
        self.num_cols = [c for c in cols if metadata.columns[c] != ColumnType.categorical]
        self.cat_cols = [c for c in cols if metadata.columns[c] == ColumnType.categorical]

    def fit(self, ref: pd.DataFrame) -> DistanceEncoder:
        ref = numeric_view(ref, self.metadata)
        self.lo = {c: float(ref[c].min()) for c in self.num_cols}
        self.span = {c: (float(ref[c].max()) - self.lo[c]) or 1.0 for c in self.num_cols}
        self.categories = {c: sorted(ref[c].astype(str).unique()) for c in self.cat_cols}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = numeric_view(df, self.metadata)
        blocks = [
            np.clip((df[c].to_numpy(float) - self.lo[c]) / self.span[c], 0, 1)[:, None]
            for c in self.num_cols
        ]
        for c in self.cat_cols:
            codes = pd.Categorical(df[c].astype(str), categories=self.categories[c]).codes
            onehot = np.zeros((len(df), len(self.categories[c])))
            hit = codes >= 0
            onehot[np.flatnonzero(hit), codes[hit]] = 1.0
            blocks.append(onehot)
        return np.hstack(blocks).astype(np.float32)
```

`workers/eval/src/stl_eval/fidelity.py`:
```python
"""Fidelity: SDMetrics quality, real-vs-synthetic detection, kNN precision/recall."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sdmetrics.reports import QualityReport
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

from stl_core.metadata import TableMetadata
from stl_eval.preprocessing import DistanceEncoder


def quality(real: pd.DataFrame, synth: pd.DataFrame, metadata: TableMetadata):
    cols = metadata.model_columns
    md = {"tables": {"t": {"columns": {c: {"sdtype": metadata.columns[c].value} for c in cols}}}}
    report = QualityReport()
    report.generate({"t": real[cols]}, {"t": synth[cols]}, md, verbose=False)
    props = report.get_properties().set_index("Property")["Score"]
    return (float(report.get_score()), float(props["Column Shapes"]),
            float(props["Column Pair Trends"]))


def _paired_sample(real, synth, seed, max_rows):
    n = min(len(real), len(synth), max_rows)
    return real.sample(n, random_state=seed), synth.sample(n, random_state=seed)


def detection_auc(real, synth, metadata: TableMetadata, seed: int, max_rows: int = 5000) -> float:
    r, s = _paired_sample(real, synth, seed, max_rows)
    enc = DistanceEncoder(metadata).fit(r)
    X = np.vstack([enc.transform(r), enc.transform(s)])
    y = np.r_[np.zeros(len(r)), np.ones(len(s))]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def _knn_radii(X: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    return nn.kneighbors(X)[0][:, -1]


def _coverage(queries: np.ndarray, refs: np.ndarray, radii: np.ndarray, chunk: int = 1000):
    hits = np.zeros(len(queries), dtype=bool)
    for start in range(0, len(queries), chunk):
        d = pairwise_distances(queries[start : start + chunk], refs)
        hits[start : start + chunk] = (d <= radii[None, :]).any(axis=1)
    return float(hits.mean())


def precision_recall(real, synth, metadata: TableMetadata, seed: int, k: int = 5,
                     max_rows: int = 5000) -> tuple[float, float]:
    r, s = _paired_sample(real, synth, seed, max_rows)
    enc = DistanceEncoder(metadata).fit(r)
    R, S = enc.transform(r), enc.transform(s)
    precision = _coverage(S, R, _knn_radii(R, k))
    recall = _coverage(R, S, _knn_radii(S, k))
    return precision, recall
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `bash scripts/envrun.sh eval pytest workers/eval/tests -q`
Expected: all passed

- [ ] **Step 6: Checkpoint**

Run: `bash scripts/envrun.sh eval ruff check workers/eval` and `git status --short`.

---

### Task 9: Utility metrics (TSTR and feature-importance agreement)

**Files:**
- Create: `workers/eval/src/stl_eval/utility.py`
- Test: `workers/eval/tests/test_utility.py`

**Interfaces:**
- Consumes: `numeric_view` (Task 8); `UtilityMetrics` (Task 4)
- Produces:
  - `MODELS = ("logreg", "random_forest", "xgboost")`
  - `tstr(real_train, synth, test, metadata, seed: int) -> UtilityMetrics` — macro-F1 on TEST; `tstr[m] = synth_f1 / real_f1` (None if real_f1 == 0); synthetic tables whose target has fewer than 2 classes score 0.0 for every model and `importance_spearman=None`; raises `NotImplementedError` for regression tasks

- [ ] **Step 1: Write the failing test**

`workers/eval/tests/test_utility.py`:
```python
import numpy as np
import pytest

from stl_eval.utility import MODELS, tstr


def test_real_as_synthetic_gives_ratio_near_one(toy):
    train, _, test, md = toy
    u = tstr(train, train, test, md, seed=0)
    assert set(u.tstr) == set(MODELS)
    assert all(0.95 <= v <= 1.05 for v in u.tstr.values())
    assert u.importance_spearman > 0.9
    assert all(0 < v <= 1 for v in u.real_scores.values())


def test_shuffled_target_destroys_utility(toy, fresh):
    train, _, test, md = toy
    bad = fresh.copy()
    bad["y"] = np.random.default_rng(0).permutation(bad["y"].to_numpy())
    u = tstr(train, bad, test, md, seed=0)
    assert u.tstr["random_forest"] < 0.85


def test_single_class_synthetic_scores_zero(toy, fresh):
    train, _, test, md = toy
    mono = fresh.copy()
    mono["y"] = "pos"
    u = tstr(train, mono, test, md, seed=0)
    assert u.tstr == {m: 0.0 for m in MODELS}
    assert u.importance_spearman is None


def test_regression_not_supported(toy):
    train, _, test, md = toy
    reg = md.model_copy(update={"target": "x2", "task": "regression"})
    with pytest.raises(NotImplementedError):
        tstr(train, train, test, reg, seed=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash scripts/envrun.sh eval pytest workers/eval/tests/test_utility.py -q`
Expected: FAIL — `No module named 'stl_eval.utility'`

- [ ] **Step 3: Write implementation**

`workers/eval/src/stl_eval/utility.py`:
```python
"""Utility: train on synthetic, test on real (TSTR) and importance agreement."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from stl_core.metadata import ColumnType, TableMetadata
from stl_core.schemas import UtilityMetrics
from stl_eval.preprocessing import numeric_view

MODELS = ("logreg", "random_forest", "xgboost")


def _model(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                         random_state=seed, n_jobs=-1, eval_metric="logloss")


def _fit_predict(name, seed, X, y, X_test) -> tuple[np.ndarray, object]:
    present = np.unique(y)  # remap to contiguous labels (XGBoost requires it)
    model = _model(name, seed).fit(X, np.searchsorted(present, y))
    return present[model.predict(X_test)], model


def tstr(real_train: pd.DataFrame, synth: pd.DataFrame, test: pd.DataFrame,
         metadata: TableMetadata, seed: int) -> UtilityMetrics:
    if metadata.task != "classification":
        raise NotImplementedError("utility metrics support classification targets only")
    target, feats = metadata.target, metadata.feature_columns
    labels = {c: i for i, c in enumerate(sorted(real_train[target].astype(str).unique()))}

    def xy(df: pd.DataFrame):
        df = numeric_view(df, metadata)
        keep = df[target].astype(str).isin(labels)
        return df.loc[keep, feats], df.loc[keep, target].astype(str).map(labels).to_numpy()

    cats = [c for c in feats if metadata.columns[c] == ColumnType.categorical]
    nums = [c for c in feats if c not in cats]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=50), cats),
        ("num", StandardScaler(), nums),
    ])
    Xr_df, yr = xy(real_train)
    Xs_df, ys = xy(synth)
    Xt_df, yt = xy(test)
    pre.fit(Xr_df)
    Xr, Xs, Xt = pre.transform(Xr_df), pre.transform(Xs_df), pre.transform(Xt_df)

    real_scores: dict[str, float | None] = {}
    ratios: dict[str, float | None] = {}
    importances = {}
    single_class = len(np.unique(ys)) < 2
    for name in MODELS:
        pred_r, model_r = _fit_predict(name, seed, Xr, yr, Xt)
        real_f1 = float(f1_score(yt, pred_r, average="macro"))
        real_scores[name] = real_f1
        if single_class:
            ratios[name] = 0.0
            continue
        pred_s, model_s = _fit_predict(name, seed, Xs, ys, Xt)
        synth_f1 = float(f1_score(yt, pred_s, average="macro"))
        ratios[name] = synth_f1 / real_f1 if real_f1 > 0 else None
        if name == "random_forest":
            importances = {"real": model_r.feature_importances_,
                           "synth": model_s.feature_importances_}
    rho = None
    if importances:
        rho = float(spearmanr(importances["real"], importances["synth"]).statistic)
        rho = None if np.isnan(rho) else rho
    return UtilityMetrics(tstr=ratios, real_scores=real_scores, importance_spearman=rho)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bash scripts/envrun.sh eval pytest workers/eval/tests/test_utility.py -q`
Expected: 4 passed

- [ ] **Step 5: Checkpoint**

Run: `bash scripts/envrun.sh eval ruff check workers/eval` and `git status --short`.

---

### Task 10: Privacy metrics (DCR share, Anonymeter, membership inference)

**Files:**
- Create: `workers/eval/src/stl_eval/privacy.py`
- Test: `workers/eval/tests/test_privacy.py`

**Interfaces:**
- Consumes: `DistanceEncoder` (Task 8)
- Produces:
  - `dcr_share(train, control, synth, metadata, seed: int, max_rows: int = 5000) -> float` — synthetic rows' nearest-neighbour distance to equal-size disjoint TRAIN chunks vs to CONTROL, averaged over chunks; ties count ½. Train-copy scores ≈ 0.67 with a 60/20 split (one chunk at distance 0, the others ≈ ½), an honest generator ≈ 0.5.
  - `anonymeter_risks(train, control, synth, metadata, seed: int, n_attacks: int = 500, singling_out_mode: str = "multivariate") -> tuple[dict[str, float | None], list[str]]` — keys `singling_out_risk`, `linkability_risk`, `inference_risk`; attack count clamped to `min(n_attacks, len(control), len(synth))`; each attack failing independently yields `None` plus a warning string
  - `mia_advantage(train, control, synth, metadata, seed: int, max_rows: int = 2000) -> float` — distance-to-closest-synthetic attack on equal-size TRAIN members vs CONTROL non-members; `max(0, 2·(AUC − 0.5))`

- [ ] **Step 1: Write the failing test**

`workers/eval/tests/test_privacy.py`:
```python
from stl_eval.privacy import anonymeter_risks, dcr_share, mia_advantage


def test_dcr_share_copy_vs_fresh(toy, fresh):
    train, control, _, md = toy
    assert dcr_share(train, control, train, md, seed=0) > 0.6
    assert 0.35 < dcr_share(train, control, fresh, md, seed=0) < 0.65


def test_mia_copy_vs_fresh(toy, fresh):
    train, control, _, md = toy
    assert mia_advantage(train, control, train, md, seed=0) > 0.8
    assert mia_advantage(train, control, fresh, md, seed=0) < 0.2


def test_anonymeter_copy_riskier_than_fresh(toy, fresh):
    train, control, _, md = toy
    copy, w1 = anonymeter_risks(train, control, train, md, seed=0, n_attacks=200)
    honest, w2 = anonymeter_risks(train, control, fresh, md, seed=0, n_attacks=200)
    assert set(copy) == {"singling_out_risk", "linkability_risk", "inference_risk"}
    assert all(v is not None and 0 <= v <= 1 for v in copy.values()), w1
    assert copy["inference_risk"] > honest["inference_risk"]
    assert copy["linkability_risk"] > honest["linkability_risk"]


def test_anonymeter_clamps_attacks_on_small_control(toy, fresh):
    train, control, _, md = toy
    risks, warnings = anonymeter_risks(train, control.head(30), fresh, md, seed=0,
                                       n_attacks=500)
    assert all(v is None or 0 <= v <= 1 for v in risks.values())
    assert not any("n_attacks" in w for w in warnings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash scripts/envrun.sh eval pytest workers/eval/tests/test_privacy.py -q`
Expected: FAIL — `No module named 'stl_eval.privacy'`

- [ ] **Step 3: Write implementation**

`workers/eval/src/stl_eval/privacy.py`:
```python
"""Privacy: DCR share, Anonymeter attacks, distance-based membership inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anonymeter.evaluators import (
    InferenceEvaluator, LinkabilityEvaluator, SinglingOutEvaluator,
)
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from stl_core.metadata import TableMetadata
from stl_eval.preprocessing import DistanceEncoder


def _nn_distance(refs: np.ndarray, queries: np.ndarray) -> np.ndarray:
    return NearestNeighbors(n_neighbors=1).fit(refs).kneighbors(queries)[0][:, 0]


def dcr_share(train: pd.DataFrame, control: pd.DataFrame, synth: pd.DataFrame,
              metadata: TableMetadata, seed: int, max_rows: int = 5000) -> float:
    enc = DistanceEncoder(metadata).fit(train)
    n = min(len(control), len(train), max_rows)
    C = enc.transform(control.sample(n, random_state=seed))
    S = enc.transform(synth.sample(min(len(synth), max_rows), random_state=seed))
    shuffled = train.sample(frac=1.0, random_state=seed)
    chunks = [shuffled.iloc[i : i + n] for i in range(0, len(shuffled) - n + 1, n)]
    d_control = _nn_distance(C, S)
    shares = []
    for chunk in chunks:
        d_train = _nn_distance(enc.transform(chunk), S)
        shares.append(np.mean(d_train < d_control) + 0.5 * np.mean(d_train == d_control))
    return float(np.mean(shares))


def anonymeter_risks(train: pd.DataFrame, control: pd.DataFrame, synth: pd.DataFrame,
                     metadata: TableMetadata, seed: int, n_attacks: int = 500,
                     singling_out_mode: str = "multivariate"):
    cols = metadata.model_columns
    ori, syn, ctl = train[cols], synth[cols], control[cols]
    n = int(min(n_attacks, len(ctl), len(syn)))
    feats = metadata.feature_columns
    half = max(1, len(feats) // 2)
    risks: dict[str, float | None] = {}
    warnings: list[str] = []

    def attempt(key: str, run) -> None:
        np.random.seed(seed)
        try:
            risks[key] = float(min(1.0, max(0.0, run().risk().value)))
        except Exception as exc:  # each attack fails independently
            risks[key] = None
            warnings.append(f"{key}: {type(exc).__name__}: {exc}")

    attempt("singling_out_risk", lambda: SinglingOutEvaluator(
        ori=ori, syn=syn, control=ctl, n_attacks=n, n_cols=min(3, len(cols)),
    ).evaluate(mode=singling_out_mode))
    attempt("linkability_risk", lambda: LinkabilityEvaluator(
        ori=ori, syn=syn, control=ctl, n_attacks=n, n_neighbors=10,
        aux_cols=(feats[:half], feats[half:]),
    ).evaluate(n_jobs=-2))
    attempt("inference_risk", lambda: InferenceEvaluator(
        ori=ori, syn=syn, control=ctl, n_attacks=n, aux_cols=feats, secret=metadata.target,
        regression=metadata.task == "regression",
    ).evaluate(n_jobs=-2))
    return risks, warnings


def mia_advantage(train: pd.DataFrame, control: pd.DataFrame, synth: pd.DataFrame,
                  metadata: TableMetadata, seed: int, max_rows: int = 2000) -> float:
    enc = DistanceEncoder(metadata).fit(train)
    n = min(len(train), len(control), max_rows)
    members = enc.transform(train.sample(n, random_state=seed))
    non_members = enc.transform(control.sample(n, random_state=seed))
    S = enc.transform(synth.sample(min(len(synth), 20_000), random_state=seed))
    scores = -np.r_[_nn_distance(S, members), _nn_distance(S, non_members)]
    labels = np.r_[np.ones(n), np.zeros(n)]
    auc = roc_auc_score(labels, scores)
    return float(max(0.0, 2 * (auc - 0.5)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bash scripts/envrun.sh eval pytest workers/eval/tests/test_privacy.py -q`
Expected: 4 passed. If `test_anonymeter_copy_riskier_than_fresh` fails on linkability only, print both risk dicts and the warnings before changing anything — Anonymeter's linkability baseline is noisy at 200 attacks; raising `n_attacks` to 300 (the full control) is the permitted fix.

- [ ] **Step 5: Checkpoint**

Run: `bash scripts/envrun.sh eval ruff check workers/eval` and `git status --short`.

---

### Task 11: Job evaluation and MLflow tracking

**Files:**
- Create: `workers/eval/src/stl_eval/evaluate.py`, `packages/runner/src/stl_runner/tracking.py`
- Test: `workers/eval/tests/test_evaluate.py`

**Interfaces:**
- Consumes: Tasks 4, 6, 7, 8, 9, 10
- Produces:
  - `@dataclass EvalConfig(n_attacks: int = 500, singling_out_mode: str = "multivariate", max_rows_knn: int = 5000)`
  - `evaluate(job: JobSpec, splits: Splits, synth: pd.DataFrame, metadata: TableMetadata, cost: CostMetrics | None, config: EvalConfig = EvalConfig()) -> MetricReport` — never raises for metric errors: each failing metric is `None` with a `warnings` entry; unusable synthetic data gives `status="failed"`
  - `evaluate_stored_job(store: ArtifactStore, job: JobSpec, config: EvalConfig = EvalConfig()) -> MetricReport` — writes `report.json` and final state (`done` | `failed`); a job whose generation failed yields a failed report carrying the generation error
  - `stl_runner.tracking.log_report(report: MetricReport, job: JobSpec, experiment: str = "synth-trust-lab") -> str | None` — no-op returning `None` when `MLFLOW_TRACKING_URI` is unset

- [ ] **Step 1: Write the failing test**

`workers/eval/tests/test_evaluate.py`:
```python
from pathlib import Path

import pandas as pd

from stl_core.schemas import PRIVACY_KEYS, CostMetrics, JobSpec
from stl_core.split import Splits
from stl_eval.evaluate import EvalConfig, evaluate, evaluate_stored_job
from stl_runner import keys
from stl_runner.pipeline import (
    generate_job, job_state, plan_run, register_dataset, run_generate_with_cap,
)
from stl_runner.storage import ArtifactStore
from stl_runner.tracking import log_report

CFG = EvalConfig(n_attacks=100, singling_out_mode="univariate")
JOB = JobSpec(job_id="g__s0", run_id="r", dataset_id="toy", generator="g", seed=0)


def test_evaluate_full_report(toy, fresh):
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh, md,
                      CostMetrics(fit_seconds=1, sample_seconds=1), CFG)
    assert report.status == "done", report.warnings
    flat = report.flat()
    assert all(flat[k] is not None for k in PRIVACY_KEYS), report.warnings
    assert flat["fidelity.quality_score"] is not None
    assert flat["utility.tstr.xgboost"] is not None


def test_evaluate_rejects_unusable_synthetic(toy, fresh):
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh.drop(columns=["x1"]), md, None,
                      CFG)
    assert report.status == "failed" and "x1" in report.error


def test_evaluate_stored_job_end_to_end(tmp_path, toy):
    train, control, test, md = toy
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", pd.concat([train, control, test]), md)
    ok, bad = plan_run(store, "r1", "toy", ["independent_marginals", "ctgan"], [0])
    generate_job(store, ok)
    run_generate_with_cap(store, bad)
    report = evaluate_stored_job(store, ok, CFG)
    assert report.status == "done" and report.cost is not None
    assert store.exists(keys.report("r1", ok.job_id))
    assert job_state(store, ok)["state"] == "done"
    failed = evaluate_stored_job(store, bad, CFG)
    assert failed.status == "failed" and "q.synthcity" in failed.error


def test_log_report_noop_without_uri(monkeypatch, toy, fresh):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None, CFG)
    assert log_report(report, JOB) is None


def test_log_report_sqlite(monkeypatch, tmp_path, toy, fresh):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + Path(tmp_path, "m.db").as_posix())
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None, CFG)
    run_id = log_report(report, JOB)
    import mlflow

    metrics = mlflow.get_run(run_id).data.metrics
    assert metrics["privacy.dcr_share"] == report.privacy.dcr_share
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash scripts/envrun.sh eval pytest workers/eval/tests/test_evaluate.py -q`
Expected: FAIL — `No module named 'stl_eval.evaluate'`

- [ ] **Step 3: Write implementation**

`workers/eval/src/stl_eval/evaluate.py`:
```python
"""Compute a full MetricReport for one generated synthetic table."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stl_core.metadata import TableMetadata
from stl_core.schemas import (
    CostMetrics, FidelityMetrics, JobSpec, MetricReport, PrivacyMetrics, UtilityMetrics,
)
from stl_core.split import Splits
from stl_eval.fidelity import detection_auc, precision_recall, quality
from stl_eval.preprocessing import align
from stl_eval.privacy import anonymeter_risks, dcr_share, mia_advantage
from stl_eval.utility import tstr
from stl_runner import keys
from stl_runner.pipeline import ensure_splits, job_state, load_metadata, set_state
from stl_runner.storage import ArtifactStore


@dataclass
class EvalConfig:
    n_attacks: int = 500
    singling_out_mode: str = "multivariate"
    max_rows_knn: int = 5000


def _base(job: JobSpec) -> dict:
    return dict(job_id=job.job_id, run_id=job.run_id, dataset_id=job.dataset_id,
                generator=job.generator, seed=job.seed)


def evaluate(job: JobSpec, splits: Splits, synth: pd.DataFrame, metadata: TableMetadata,
             cost: CostMetrics | None, config: EvalConfig = EvalConfig()) -> MetricReport:
    cols = metadata.model_columns
    train, control, test = (p[cols] for p in (splits.train, splits.control, splits.test))
    try:
        synth = align(train, synth, metadata)
    except ValueError as exc:
        return MetricReport(**_base(job), status="failed", cost=cost,
                            error=f"synthetic data rejected: {exc}")
    warnings: list[str] = []

    def attempt(label, fn):
        try:
            return fn()
        except Exception as exc:
            warnings.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    seed, k = job.seed, config.max_rows_knn
    fid = FidelityMetrics()
    q = attempt("quality", lambda: quality(train, synth, metadata))
    if q:
        fid.quality_score, fid.column_shapes, fid.column_pair_trends = q
    fid.detection_auc = attempt("detection", lambda: detection_auc(train, synth, metadata, seed, k))
    pr = attempt("precision_recall",
                 lambda: precision_recall(train, synth, metadata, seed, max_rows=k))
    if pr:
        fid.alpha_precision, fid.beta_recall = pr
    util = attempt("utility", lambda: tstr(train, synth, test, metadata, seed)) or UtilityMetrics()
    priv = PrivacyMetrics()
    priv.dcr_share = attempt("dcr", lambda: dcr_share(train, control, synth, metadata, seed, k))
    risks, anon_warnings = anonymeter_risks(train, control, synth, metadata, seed,
                                            config.n_attacks, config.singling_out_mode)
    warnings.extend(anon_warnings)
    for key, value in risks.items():
        setattr(priv, key, value)
    priv.mia_advantage = attempt("mia", lambda: mia_advantage(train, control, synth, metadata,
                                                              seed))
    return MetricReport(**_base(job), status="done", warnings=warnings, cost=cost,
                        fidelity=fid, utility=util, privacy=priv)


def evaluate_stored_job(store: ArtifactStore, job: JobSpec,
                        config: EvalConfig = EvalConfig()) -> MetricReport:
    state = job_state(store, job)
    if state["state"] != "generated":
        error = state.get("error") or f"job not generated (state {state['state']!r})"
        report = MetricReport(**_base(job), status="failed", error=error)
    else:
        set_state(store, job, "evaluating")
        metadata = load_metadata(store, job.dataset_id)
        splits = ensure_splits(store, job.dataset_id, job.seed)
        synth = store.read_df(keys.synthetic(job.run_id, job.job_id))
        cost = CostMetrics.model_validate(store.read_json(keys.cost(job.run_id, job.job_id)))
        report = evaluate(job, splits, synth, metadata, cost, config)
    store.write_json(keys.report(job.run_id, job.job_id), report)
    set_state(store, job, report.status, report.error)
    return report
```

`packages/runner/src/stl_runner/tracking.py`:
```python
"""Log job reports to MLflow when MLFLOW_TRACKING_URI is set."""

from __future__ import annotations

import os

from stl_core.schemas import JobSpec, MetricReport


def log_report(report: MetricReport, job: JobSpec,
               experiment: str = "synth-trust-lab") -> str | None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    import mlflow

    mlflow.set_experiment(experiment)
    tags = {"run_id": job.run_id, "dataset_id": job.dataset_id, "status": report.status}
    with mlflow.start_run(run_name=f"{job.run_id}/{job.job_id}", tags=tags) as run:
        mlflow.log_params({"generator": job.generator, "seed": job.seed,
                           "dataset": job.dataset_id,
                           **{f"param.{k}": v for k, v in job.params.items()}})
        mlflow.log_metrics({k: v for k, v in report.flat().items() if v is not None})
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        return run.info.run_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bash scripts/envrun.sh eval pytest workers/eval/tests -q`
Expected: all passed

- [ ] **Step 5: Checkpoint**

Run: `bash scripts/envrun.sh eval ruff check workers/eval packages/runner` and `git status --short`.

---

### Task 12: SDV worker — Gaussian Copula

**Files:**
- Create: `workers/sdv/pyproject.toml`, `workers/sdv/src/stl_sdv/__init__.py`, `workers/sdv/src/stl_sdv/generators.py`
- Test: `workers/sdv/tests/test_copula.py`

**Interfaces:**
- Consumes: `Generator` (Task 3), `TableMetadata` (Task 1), catalog entry `gaussian_copula → stl_sdv.generators` (Task 3)
- Produces: `stl_sdv.generators.GaussianCopula`, `stl_sdv.generators.build(name: str, seed: int = 0, **params) -> Generator` (optional param `default_distribution`, passed to SDV)

- [ ] **Step 1: Create the environment**

`workers/sdv/pyproject.toml`:
```toml
[project]
name = "stl-sdv"
version = "0.1.0"
description = "SDV worker: Gaussian Copula"
requires-python = ">=3.12,<3.13"
dependencies = ["stl-core", "stl-runner[s3]", "sdv>=1.38"]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[tool.uv.sources]
stl-core = { path = "../../packages/core", editable = true }
stl-runner = { path = "../../packages/runner", editable = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/stl_sdv"]

[tool.pytest.ini_options]
filterwarnings = ["ignore::FutureWarning", "ignore::UserWarning"]
```

`workers/sdv/src/stl_sdv/__init__.py`:
```python
"""SDV worker for the Synthetic Data Trust Lab."""
```

Run: `bash scripts/envrun.sh sdv python -c "import sdv; print(sdv.__version__)"`
Expected: `1.38.x` or newer.

- [ ] **Step 2: Write the failing test**

`workers/sdv/tests/test_copula.py`:
```python
import numpy as np
import pandas as pd

from stl_core.catalog import load_generator
from stl_core.metadata import ColumnType, TableMetadata

MD = TableMetadata(columns={"x1": ColumnType.numerical, "x2": ColumnType.numerical,
                            "c": ColumnType.categorical, "y": ColumnType.categorical},
                   target="y")


def _train(n=800):
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": 2 * x1 + rng.normal(scale=0.3, size=n),
                         "c": np.where(x1 > 0, "hi", "lo").astype(object),
                         "y": np.where(x1 > 0.2, "pos", "neg").astype(object)})


def test_copula_via_catalog_keeps_correlation():
    train = _train()
    gen = load_generator("gaussian_copula", seed=0)
    gen.fit(train, MD)
    out = gen.sample(600)
    assert len(out) == 600 and list(out.columns) == list(train.columns)
    assert np.corrcoef(out["x1"], out["x2"])[0, 1] > 0.8
    assert set(out["c"]) <= {"hi", "lo"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bash scripts/envrun.sh sdv pytest workers/sdv/tests -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'stl_sdv.generators'`

- [ ] **Step 4: Write implementation**

`workers/sdv/src/stl_sdv/generators.py`:
```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `bash scripts/envrun.sh sdv pytest workers/sdv/tests -q`
Expected: 1 passed

- [ ] **Step 6: Checkpoint**

Run: `bash scripts/envrun.sh sdv ruff check workers/sdv` and `git status --short`.

---

### Task 13: Experiment CLI with Hydra config

**Files:**
- Create: `experiments/run.py`, `experiments/conf/config.yaml`, `experiments/conf/datasets/adult.yaml`, `experiments/conf/generators/default.yaml`
- Test: `experiments/tests/test_cli.py`

**Interfaces:**
- Consumes: everything above
- Produces: `experiments/run.py` with `main(argv: list[str] | None = None) -> int`, subcommands `prepare`, `generate`, `evaluate`, `report`, `all`; global `-o/--override KEY=VALUE` (repeatable Hydra overrides). `report --json` prints `{"recommendation": Recommendation, "aggregated": [AggregatedReport]}`. `run_id` defaults to `f"{dataset.name}-main"`.

- [ ] **Step 1: Write the config**

`experiments/conf/config.yaml`:
```yaml
defaults:
  - datasets: adult
  - generators: default
  - _self_

run_id: null
seeds: [0, 1, 2]
max_rows: 20000
storage_root: outputs/store
tracking_uri: sqlite:///outputs/mlflow.db
generators_to_run: [train_copy, independent_marginals, gaussian_copula]
budget:
  epochs: 300
  time_cap_seconds: 7200
eval:
  n_attacks: 500
  singling_out_mode: multivariate
  max_rows_knn: 5000
```

`experiments/conf/datasets/adult.yaml`:
```yaml
# @package dataset
name: adult
source: openml
openml_name: adult
version: 2
openml_id: null
path: null
target: class
drop: []
column_types: {}
```

`experiments/conf/generators/default.yaml`:
```yaml
# @package generator_params
# Per-generator parameters. Deep generators share one epoch budget (equal-budget rule).
train_copy: {}
independent_marginals: {}
marginals: {}
gaussian_copula: {}
bayesian_network: {}
ctgan: {epochs: ${budget.epochs}}
tvae: {epochs: ${budget.epochs}}
tabddpm: {epochs: ${budget.epochs}}
great: {epochs: ${budget.epochs}}
privbayes_eps0.1: {}
privbayes_eps1: {}
privbayes_eps10: {}
```

- [ ] **Step 2: Write the failing test**

`experiments/tests/test_cli.py`:
```python
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402


def _toy_csv(path: Path) -> Path:
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=1000)
    df = pd.DataFrame({"x1": x1, "x2": 2 * x1 + rng.normal(scale=0.3, size=1000),
                       "c": np.where(x1 > 0, "hi", "lo"),
                       "y": np.where(x1 + rng.normal(scale=0.5, size=1000) > 0, "p", "n")})
    p = path / "toy.csv"
    df.to_csv(p, index=False)
    return p


def test_full_local_run(tmp_path, capsys):
    csv = _toy_csv(tmp_path).as_posix()
    store = (tmp_path / "store").as_posix()
    overrides = [
        "dataset.name=toy", "dataset.source=file", f"dataset.path='{csv}'", "dataset.target=y",
        "seeds=[0]", "max_rows=1000", f"storage_root='{store}'", "tracking_uri=''",
        "generators_to_run=[train_copy,independent_marginals]",
        "eval.n_attacks=100", "eval.singling_out_mode=univariate",
        "budget.time_cap_seconds=null",
    ]
    args = [a for o in overrides for a in ("-o", o)]
    assert run.main([*args, "prepare"]) == 0
    assert run.main([*args, "generate"]) == 0
    assert run.main([*args, "evaluate"]) == 0
    capsys.readouterr()
    assert run.main([*args, "report", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    rec = out["recommendation"]
    assert rec["winner"] == "independent_marginals"
    tc = next(c for c in rec["ranking"] if c["generator"] == "train_copy")
    assert tc["gate_passed"] is False
    assert {a["generator"] for a in out["aggregated"]} == {"train_copy", "independent_marginals"}


def test_generate_skips_unavailable(tmp_path, capsys):
    csv = _toy_csv(tmp_path).as_posix()
    store = (tmp_path / "store").as_posix()
    overrides = ["dataset.name=toy", "dataset.source=file", f"dataset.path='{csv}'",
                 "dataset.target=y", "seeds=[0]", f"storage_root='{store}'", "tracking_uri=''",
                 "generators_to_run=[ctgan]", "budget.time_cap_seconds=null"]
    args = [a for o in overrides for a in ("-o", o)]
    run.main([*args, "prepare"])
    assert run.main([*args, "generate"]) == 0
    assert "skipped ctgan__s0" in capsys.readouterr().out
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bash scripts/envrun.sh eval pytest experiments/tests -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'run'`

- [ ] **Step 4: Write implementation**

`experiments/run.py`:
```python
"""Benchmark CLI. Each stage runs in the environment that has its libraries:

  bash scripts/envrun.sh eval python experiments/run.py prepare
  bash scripts/envrun.sh sdv  python experiments/run.py generate      # gaussian_copula + core baselines
  bash scripts/envrun.sh eval python experiments/run.py generate      # anything left that eval can run
  bash scripts/envrun.sh eval python experiments/run.py evaluate
  bash scripts/envrun.sh eval python experiments/run.py report --preset balanced

Hydra overrides go before the subcommand:  -o datasets=adult -o seeds=[0] -o max_rows=2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from stl_core.aggregate import aggregate
from stl_core.catalog import is_available
from stl_core.decision import recommend
from stl_core.metadata import ColumnType, infer_metadata
from stl_core.schemas import MetricReport
from stl_runner import keys
from stl_runner.datasets import load_dataset
from stl_runner.pipeline import (
    ensure_splits, job_state, load_jobs, plan_run, register_dataset, run_generate_with_cap,
)
from stl_runner.storage import ArtifactStore

CONF_DIR = Path(__file__).resolve().parent / "conf"


def load_config(overrides: list[str]) -> dict:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def _store(cfg: dict) -> ArtifactStore:
    return ArtifactStore(os.environ.get("STL_STORAGE_ROOT") or cfg["storage_root"])


def _run_id(cfg: dict, args) -> str:
    return args.run or cfg["run_id"] or f"{cfg['dataset']['name']}-main"


def cmd_prepare(cfg: dict, args) -> int:
    store = _store(cfg)
    ds = cfg["dataset"]
    df, target = load_dataset(ds)
    md = infer_metadata(df, target)
    if ds.get("column_types"):
        md = md.model_copy(update={"columns": {
            **md.columns, **{c: ColumnType(t) for c, t in ds["column_types"].items()}}})
    prepared = register_dataset(store, ds["name"], df, md, max_rows=cfg["max_rows"])
    for seed in cfg["seeds"]:
        ensure_splits(store, ds["name"], seed)
    jobs = plan_run(store, _run_id(cfg, args), ds["name"], cfg["generators_to_run"],
                    cfg["seeds"], params=cfg["generator_params"],
                    time_cap_seconds=cfg["budget"]["time_cap_seconds"])
    print(f"registered {ds['name']} ({len(prepared)} rows); planned {len(jobs)} jobs "
          f"in run {_run_id(cfg, args)}")
    return 0


def cmd_generate(cfg: dict, args) -> int:
    store = _store(cfg)
    retry = {"queued", "failed"} if args.retry_failed else {"queued"}
    for job in load_jobs(store, _run_id(cfg, args)):
        if job_state(store, job)["state"] not in retry:
            continue
        if not is_available(job.generator):
            print(f"skipped {job.job_id}: not installed in this environment")
            continue
        print(f"{job.job_id}: {run_generate_with_cap(store, job)}", flush=True)
    return 0


def cmd_evaluate(cfg: dict, args) -> int:
    from stl_eval.evaluate import EvalConfig, evaluate_stored_job
    from stl_runner.tracking import log_report

    if cfg.get("tracking_uri"):
        os.environ.setdefault("MLFLOW_TRACKING_URI", cfg["tracking_uri"])
    store = _store(cfg)
    config = EvalConfig(**cfg["eval"])
    for job in load_jobs(store, _run_id(cfg, args)):
        state = job_state(store, job)["state"]
        has_report = store.exists(keys.report(job.run_id, job.job_id))
        if state == "generated" or (state == "failed" and not has_report):
            report = evaluate_stored_job(store, job, config)
            log_report(report, job)
            print(f"{job.job_id}: {report.status}", flush=True)
    return 0


def _reports(store: ArtifactStore, run_id: str) -> list[MetricReport]:
    return [
        MetricReport.model_validate(store.read_json(keys.report(j.run_id, j.job_id)))
        for j in load_jobs(store, run_id)
        if store.exists(keys.report(j.run_id, j.job_id))
    ]


def cmd_report(cfg: dict, args) -> int:
    aggs = aggregate(_reports(_store(cfg), _run_id(cfg, args)))
    weights = tuple(float(w) for w in args.weights.split(",")) if args.weights else None
    rec = recommend(aggs, preset=args.preset, weights=weights)
    if args.json:
        print(json.dumps({"recommendation": rec.model_dump(mode="json"),
                          "aggregated": [a.model_dump(mode="json") for a in aggs]}, indent=2))
        return 0
    print(f"{'generator':<24}{'fidelity':>10}{'utility':>10}{'privacy':>10}{'score':>8}  gate")
    for c in rec.ranking:
        s = c.scores
        cells = [f"{v:>10.3f}" if v is not None else f"{'—':>10}"
                 for v in (s.fidelity, s.utility, s.privacy)]
        score = f"{c.score:>8.3f}" if c.score is not None else f"{'—':>8}"
        gate = "pass" if c.gate_passed else "FAIL: " + c.gate_failures[0]
        print(f"{c.generator:<24}{''.join(cells)}{score}  {gate}")
    print()
    print(rec.reason)
    return 0


def cmd_all(cfg: dict, args) -> int:
    for fn in (cmd_prepare, cmd_generate, cmd_evaluate, cmd_report):
        if fn(cfg, args) != 0:
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="Hydra override, e.g. -o seeds=[0]")
    parser.add_argument("--run", help="run id (default: <dataset>-main)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    gen = sub.add_parser("generate")
    gen.add_argument("--retry-failed", action="store_true")
    sub.add_parser("evaluate")
    for name in ("report", "all"):
        p = sub.add_parser(name)
        p.add_argument("--preset", default="balanced")
        p.add_argument("--weights", help="custom fidelity,utility,privacy weights")
        p.add_argument("--json", action="store_true")
    sub.choices["all"].add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.override)
    commands = {"prepare": cmd_prepare, "generate": cmd_generate, "evaluate": cmd_evaluate,
                "report": cmd_report, "all": cmd_all}
    return commands[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `bash scripts/envrun.sh eval pytest experiments/tests -q`
Expected: 2 passed

- [ ] **Step 6: Checkpoint**

Run: `bash scripts/envrun.sh eval ruff check experiments` and `git status --short`.

---

### Task 14: Sanity gate on Adult, cross-environment smoke run, and CI

**Files:**
- Create: `workers/eval/tests/test_sanity_gate.py`, `scripts/smoke_adult.sh`, `.github/workflows/ci.yml`, `README.md`
- Test: `workers/eval/tests/test_sanity_gate.py`

**Interfaces:**
- Consumes: everything above. Produces the S1 phase-gate evidence.

- [ ] **Step 1: Write the sanity-gate test**

`workers/eval/tests/test_sanity_gate.py`:
```python
"""Phase-1 gate: train-copy worst on privacy, independent marginals worst on pair trends."""

import pytest

from stl_core.aggregate import aggregate
from stl_core.decision import axis_scores, recommend
from stl_core.metadata import infer_metadata
from stl_eval.evaluate import EvalConfig, evaluate_stored_job
from stl_runner.datasets import load_dataset
from stl_runner.pipeline import generate_job, plan_run, register_dataset
from stl_runner.storage import ArtifactStore


@pytest.mark.network
@pytest.mark.slow
def test_sanity_gate_on_adult(tmp_path):
    df, target = load_dataset({"source": "openml", "openml_name": "adult", "version": 2,
                               "target": "class"})
    md = infer_metadata(df, target)
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "adult", df, md, max_rows=2000, seed=0)
    jobs = plan_run(store, "sanity", "adult", ["train_copy", "independent_marginals"], [0])
    config = EvalConfig(n_attacks=300, singling_out_mode="univariate")
    reports = {}
    for job in jobs:
        generate_job(store, job)
        reports[job.generator] = evaluate_stored_job(store, job, config)
        assert reports[job.generator].status == "done", reports[job.generator].warnings

    aggs = {a.generator: a for a in aggregate(list(reports.values()))}
    scores = {g: axis_scores(a.mean) for g, a in aggs.items()}
    assert scores["train_copy"].privacy < scores["independent_marginals"].privacy
    assert (reports["independent_marginals"].fidelity.column_pair_trends
            < reports["train_copy"].fidelity.column_pair_trends)
    rec = recommend(list(aggs.values()))
    train_copy = next(c for c in rec.ranking if c.generator == "train_copy")
    assert not train_copy.gate_passed
```

- [ ] **Step 2: Run it**

Run: `bash scripts/envrun.sh eval pytest workers/eval/tests/test_sanity_gate.py -q`
Expected: 1 passed (a few minutes on a laptop CPU). If it fails, use superpowers:systematic-debugging — a failure here means a metric is wrong, which is exactly what this gate exists to catch. Do not loosen the assertions.

- [ ] **Step 3: Write the cross-environment smoke script**

`scripts/smoke_adult.sh`:
```bash
#!/usr/bin/env bash
# Phase-1 smoke run: Adult capped at 2,000 rows, one seed, three generators across two envs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OPTS=(-o seeds=[0] -o max_rows=2000 -o run_id=adult-smoke -o eval.n_attacks=300
      -o eval.singling_out_mode=univariate -o budget.time_cap_seconds=1800)
bash scripts/envrun.sh eval python experiments/run.py "${OPTS[@]}" prepare
bash scripts/envrun.sh sdv  python experiments/run.py "${OPTS[@]}" generate
bash scripts/envrun.sh eval python experiments/run.py "${OPTS[@]}" evaluate
bash scripts/envrun.sh eval python experiments/run.py "${OPTS[@]}" report
```

Run: `bash scripts/smoke_adult.sh`
Expected: three `…__s0: generated` lines from the sdv env, three `…: done` lines from evaluate, then a table where `train_copy` shows `FAIL: …`, `gaussian_copula` has a higher fidelity than `independent_marginals`, and a reason line naming the winner. Also confirm `outputs/mlflow.db` exists. Record the printed table in the task report.

- [ ] **Step 4: Write CI**

`.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uvx ruff check .

  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        include:
          - env: core
            tests: packages/core/tests
          - env: runner
            tests: packages/runner/tests
          - env: eval
            tests: workers/eval/tests experiments/tests
          - env: sdv
            tests: workers/sdv/tests
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - uses: actions/cache@v4
        with:
          path: ~/scikit_learn_data
          key: openml-${{ hashFiles('experiments/conf/datasets/*.yaml') }}
      - run: bash scripts/envrun.sh ${{ matrix.env }} pytest ${{ matrix.tests }} -q
```

- [ ] **Step 5: Write the README**

`README.md`:
````markdown
# Synthetic Data Trust Lab

Benchmarks tabular synthetic-data generators on fidelity, utility, privacy and cost, and
recommends one through a privacy-gated decision engine. Design: `docs/superpowers/specs/`.

## Environments

Each worker has its own uv project and lockfile. Environments live outside the repo
(`$STL_VENV_ROOT`, default `~/.stl`) because the Windows path limit breaks deep installs.
Always run Python through `scripts/envrun.sh <env> …`:

| env | project | Python | contents |
|---|---|---|---|
| core | packages/core | ≥3.10 | contract only |
| runner | packages/runner | ≥3.10 | storage, pipeline |
| eval | workers/eval | 3.11 | metrics (SDMetrics, Anonymeter 1.0.0, sklearn, XGBoost) |
| sdv | workers/sdv | 3.12 | Gaussian Copula |

## Tests

```bash
bash scripts/envrun.sh core   pytest packages/core/tests -q
bash scripts/envrun.sh runner pytest packages/runner/tests -q
bash scripts/envrun.sh eval   pytest workers/eval/tests experiments/tests -q
bash scripts/envrun.sh sdv    pytest workers/sdv/tests -q
```

## Benchmark CLI

See the docstring of `experiments/run.py`. Quick smoke run: `bash scripts/smoke_adult.sh`.
Full main run on Adult (3 seeds, 20,000 rows) is the same sequence without the overrides.
````

- [ ] **Step 6: Run the whole suite and checkpoint**

Run each line of the README "Tests" block. Expected: all pass. Run `uvx ruff check .` → `All checks passed!`. Run `git status --short` and list the new files in the task report. Do not commit.

---

## Phase gate (S1 done when)

- `test_sanity_gate_on_adult` passes: train-copy has the lowest privacy score and fails the gate; independent marginals have the lowest column-pair-trends score.
- `scripts/smoke_adult.sh` prints a recommendation with Gaussian Copula above independent marginals on fidelity.
- All four environment test suites pass.

Next plans (written after S1 lands, each against the code as built): S2 synthcity generators + offline GPU path + `import-results`; S3 API/queue/storage/Compose; S4 Next.js app; S5 RQ3; S6 Data Designer; S7 analysis notebooks.
