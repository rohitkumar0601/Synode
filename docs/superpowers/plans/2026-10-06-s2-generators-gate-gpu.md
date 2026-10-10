# S2 — Statistical Privacy Gate, Synthcity Generators, Datasets, Offline GPU Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the privacy gate statistically sound, add the full synthcity generator lineup (marginals, Bayesian network, CTGAN, TVAE, TabDDPM, PrivBayes ε ∈ {0.1, 1, 10}) plus GReaT, add the Credit-Default and Diabetes-130 datasets, and ship the offline GPU path (results bundle export/import + Kaggle/Colab notebook).

**Architecture:** The gate compares Anonymeter's Bonferroni-corrected confidence lower bounds (not point estimates) against the threshold and treats too-few attacks as inconclusive (fails closed). A new `workers/synthcity` uv project (Python 3.11) implements `build(name, seed, **params)` for the synthcity generators and GReaT (via `be-great` directly). Results bundles are zip files of a run's per-job reports that `experiments/run.py export-results` writes and `import-results` merges into a local store and MLflow.

**Tech Stack:** synthcity 0.2.12, torch 2.2.2 (CPU wheels), opacus 1.4.1, pyarrow < 20, be-great 0.0.14 (GReaT), Anonymeter 1.0.0, Hydra, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-synth-trust-lab-design.md` (S1 code is the base: `packages/core`, `packages/runner`, `workers/eval`, `workers/sdv`, `experiments/run.py`).

## Global Constraints

- **No git commits or pushes.** Checkpoint steps = tests + lint + `git status --short` only.
- Run Python only via `bash scripts/envrun.sh <env> …`; envs live in `$HOME/.stl/<env>` (Windows MAX_PATH; `LongPathsEnabled=0`). Never create a `.venv` in the repo.
- `packages/core` runtime deps stay pandas, numpy, pydantic only.
- Synthcity env pins (verified 2026-10-06 on this machine): Python `>=3.11,<3.12`; `synthcity==0.2.12`; `torch==2.2.2` from the CPU index `https://download.pytorch.org/whl/cpu`; `opacus==1.4.1` (1.5.x imports `nn.RMSNorm`, absent in torch 2.2); `pyarrow>=14,<20` (`datasets` 2.14 needs `pa.PyExtensionType`).
- Synthcity plugin names: `marginal_distributions`, `bayesian_network`, `ctgan`, `tvae`, `ddpm`, `privbayes`. Fit/sample API: `Plugins().get(name, random_state=seed, **kw).fit(df)`; `.generate(count=n).dataframe()`. Epoch-like kwarg is `n_iter`. PrivBayes privacy budget kwarg is `epsilon`.
- synthcity's `great` plugin is broken with be-great 0.0.14 (passes `torch.device` where a string is expected) — implement GReaT with `be_great.GReaT` directly.
- Gate: threshold 0.10 on the **lower confidence bound** of each Anonymeter risk; Bonferroni over `m = 2 + number_of_inference_secrets` tests at family α = 0.05 → Anonymeter two-sided `confidence_level = 1 - 0.10 / m` (its lower bound is then a one-sided 1 − 0.05/m bound). Fewer than `min_attacks` (default 100) effective attacks → that risk is `None` (inconclusive → gate fails closed). DCR gate unchanged (0.6). The privacy **axis score** keeps using point estimates.
- Dataset OpenML ids (verified): Adult `name=adult, version=2, target=class`; Credit default `openml_id=46373, target=y` (30,000 rows); Diabetes 130-US `openml_id=4541, target=readmitted` (101,766 rows; drop `encounter_id`, `patient_nbr`).
- Equal budget rule: deep generators share `budget.epochs` (passed as `n_iter`/`epochs`).

## Review Focus

1. **Gate on small CONTROL sets** — effective attacks below `min_attacks` must yield `None` with a warning naming the attack count, never a point estimate. (Task 1.)
2. **Synthetic frames from synthcity with changed dtypes or column order** (e.g. categoricals as numbers, ints as floats) — the wrapper must return columns in TRAIN order with TRAIN's dtypes where castable. (Task 2.)
3. **PrivBayes at ε = 0.1 dropping minority classes** — evaluation must still run (S1 utility remap) — exercised in Task 2's smoke and Task 7.
4. **Importing a bundle made from a differently-sized dataset** — `import-results` must refuse when the bundle's dataset fingerprint differs from the locally registered one, unless `--force`. (Task 5.)
5. **GPU-only generators selected on a CPU machine** — `generate` skips `cpu_ok=False` generators unless `--allow-gpu-only` (prints why), so a CPU benchmark doesn't silently run TabDDPM for hours. (Task 4.)

---

### Task 1: Statistical privacy gate (confidence lower bounds) + S1 deferred privacy minors

**Files:**
- Modify: `workers/eval/src/stl_eval/privacy.py`, `workers/eval/src/stl_eval/evaluate.py`, `packages/core/src/stl_core/schemas.py`, `packages/core/src/stl_core/decision.py`, `experiments/run.py`, `experiments/conf/config.yaml`, `docs/superpowers/specs/2026-10-05-synth-trust-lab-design.md`
- Test: `workers/eval/tests/test_privacy.py`, `workers/eval/tests/test_evaluate.py`, `packages/core/tests/test_decision.py`, `packages/core/tests/test_schemas.py`, `experiments/tests/test_cli.py`

**Interfaces:**
- `PrivacyMetrics` gains `singling_out_risk_lower`, `linkability_risk_lower`, `inference_risk_lower` (`float | None = None`). `PRIVACY_KEYS` becomes an explicit tuple of the original five keys (it must not grow with new fields). `MetricReport.flat()` includes the three `*_lower` keys (under `privacy.`) and still excludes `inference_by_secret`.
- `anonymeter_risks(..., inference_secrets="all", min_attacks: int = 100, family_alpha: float = 0.05)` returns `(risks, warnings, by_secret)` where `risks` now also has the three `*_lower` keys. Lower bound = `risk(confidence_level=1 - 2*family_alpha/m).ci[0]` clipped to [0,1], with `m = 2 + len(secrets)`. `inference_risk_lower` = max over secrets of their lower bounds (the gate asks "is there evidence any attribute leaks?").
- Effective attacks `n < min_attacks` → all three risks and lower bounds `None`, warning `f"anonymeter: only {n} attacks possible (< min_attacks={min_attacks}); privacy inconclusive"`.
- Singling-out shortfall: iterate re-runs with `n_attacks = k` until the generated count is stable (at most 5 rounds); if the stable count is `< min_attacks` → `None` + warning `"singling_out_risk: query generation fell short of min_attacks; inconclusive"`.
- `inference_secrets` given as a single column name string (not "all"/"target") is treated as `[name]`. `EvalConfig` validates `inference_secrets`, `min_attacks >= 1`, `0 < family_alpha < 1` in `__post_init__` (ValueError); `cmd_evaluate` builds `EvalConfig` before the loop so a config error stops the run with exit code 2 and a one-line message.
- `check_gate`: for each Anonymeter check, compare `privacy.<x>_lower` when present in `agg.mean`, else the point estimate; message `f"failed the {label} check (lower bound {v:.2f} > {t:.2f})"` / unchanged point-estimate message otherwise. Order failures: risk/DCR failures first, then "missing" failures.
- `cmd_report`: invalid `--weights` prints the ValueError message and returns 2 (no traceback). The table's gate column shows all failures joined by `"; "`.
- Spec §6: replace the gate sentence with the lower-bound rule (one short paragraph: Bonferroni, min attacks, axis score still uses point estimates). Spec §2 deviations table: add rows "synthcity env uses Python 3.11 (3.10 resolves only synthcity 0.2.4)", "GReaT via be-great directly", "local platform storage uses a shared filesystem volume; optional S3-compatible storage is reserved for later".

- [ ] **Step 1: Write failing tests**

Add to `workers/eval/tests/test_privacy.py`:
```python
def test_lower_bounds_reported_and_below_point(toy, fresh):
    train, control, _, md = toy
    risks, _, _ = anonymeter_risks(train, control, train, md, seed=0, n_attacks=200,
                                   singling_out_mode="univariate", inference_secrets="target")
    for key in ("singling_out_risk", "linkability_risk", "inference_risk"):
        lo = risks[f"{key}_lower"]
        assert lo is not None and 0 <= lo <= risks[key] <= 1
    assert risks["inference_risk_lower"] > 0.10  # train-copy leaks: evidence even at the bound


def test_honest_generator_lower_bounds_pass(toy, fresh):
    train, control, _, md = toy
    risks, warnings, _ = anonymeter_risks(train, control, fresh, md, seed=0, n_attacks=200,
                                          singling_out_mode="univariate")
    assert risks["inference_risk_lower"] is not None and risks["inference_risk_lower"] <= 0.10
    assert risks["linkability_risk_lower"] <= 0.10


def test_too_few_attacks_is_inconclusive(toy, fresh):
    train, control, _, md = toy
    risks, warnings, _ = anonymeter_risks(train, control.head(30), fresh, md, seed=0,
                                          n_attacks=500, min_attacks=100)
    assert all(v is None for v in risks.values())
    assert any("only 30 attacks possible" in w for w in warnings)


def test_single_secret_string_is_one_column(toy, fresh):
    train, control, _, md = toy
    _, _, by_secret = anonymeter_risks(train, control, fresh, md, seed=0, n_attacks=100,
                                       singling_out_mode="univariate", inference_secrets="x1")
    assert set(by_secret) == {"x1"}
```
(Existing tests that use small controls must pass `min_attacks=` explicitly where they expect values; update `test_anonymeter_clamps_attacks_on_small_control` to pass `min_attacks=10` and keep its spy assertion.)

Add to `packages/core/tests/test_decision.py`:
```python
def test_gate_uses_lower_bound_when_present():
    agg = _agg("g", privacy={"privacy.inference_risk": 0.25,
                             "privacy.inference_risk_lower": 0.04})
    assert check_gate(agg, GateConfig()) == []
    agg = _agg("g", privacy={"privacy.inference_risk": 0.25,
                             "privacy.inference_risk_lower": 0.18})
    assert check_gate(agg, GateConfig()) == [
        "failed the inference check (lower bound 0.18 > 0.10)"]


def test_gate_failures_list_risks_before_missing():
    agg = _agg("g", privacy={"privacy.singling_out_risk": 0.5}, missing=["privacy.dcr_share"])
    fails = check_gate(agg, GateConfig())
    assert fails[0].startswith("failed the singling-out check")
    assert fails[-1] == "is missing the DCR share metric"
```
Add to `packages/core/tests/test_schemas.py`:
```python
def test_privacy_keys_fixed_and_lower_in_flat():
    assert PRIVACY_KEYS == ("privacy.dcr_share", "privacy.singling_out_risk",
                            "privacy.linkability_risk", "privacy.inference_risk",
                            "privacy.mia_advantage")
    flat = _report(privacy=PrivacyMetrics(dcr_share=0.5, inference_risk_lower=0.02)).flat()
    assert flat["privacy.inference_risk_lower"] == 0.02
```
Add to `experiments/tests/test_cli.py` a test that `report --weights 1,-1,1` returns 2 and prints the ValueError message, and that `-o eval.inference_secrets=[nope]` makes `evaluate` return 2 with a message containing `nope`.

- [ ] **Step 2: Run the new tests, verify they fail** (`bash scripts/envrun.sh core pytest packages/core/tests -q`, `bash scripts/envrun.sh eval pytest workers/eval/tests/test_privacy.py -q`, `bash scripts/envrun.sh eval pytest experiments/tests -q`).

- [ ] **Step 3: Implement** per Interfaces. Key code for the lower bound inside `anonymeter_risks`:
```python
m = 2 + len(secrets)
level = 1 - 2 * family_alpha / m  # two-sided level whose lower bound is one-sided 1 - alpha/m

def _bounds(evaluator) -> tuple[float, float]:
    point = evaluator.risk().value
    lower = evaluator.risk(confidence_level=level).ci[0]
    clip = lambda v: float(min(1.0, max(0.0, v)))  # noqa: E731
    return clip(point), clip(lower)
```
Use `_bounds` for singling-out (inside the fixed-point helper), linkability, and each inference secret; `inference_risk = max(points)`, `inference_risk_lower = max(lowers)` over successful secrets. Add `min_attacks`, `family_alpha` to `EvalConfig` (defaults 100, 0.05) and to `experiments/conf/config.yaml` under `eval:`; pass through in `evaluate`.

- [ ] **Step 4: Run all affected suites; all pass, output pristine.** core, eval (`workers/eval/tests`), experiments (`experiments/tests`), then `bash scripts/envrun.sh core ruff check .`.

- [ ] **Step 5: Re-run the Adult multivariate check** and paste the table into the report:
```bash
OPTS=(-o seeds=[0] -o max_rows=2000 -o run_id=adult-gate -o eval.n_attacks=500 -o eval.singling_out_mode=multivariate -o budget.time_cap_seconds=1800)
bash scripts/envrun.sh eval python experiments/run.py "${OPTS[@]}" prepare
bash scripts/envrun.sh sdv  python experiments/run.py "${OPTS[@]}" generate
bash scripts/envrun.sh eval python experiments/run.py "${OPTS[@]}" generate
bash scripts/envrun.sh eval python experiments/run.py "${OPTS[@]}" evaluate
bash scripts/envrun.sh eval python experiments/run.py "${OPTS[@]}" report
```
Expected: train_copy FAILS on a lower-bound check; gaussian_copula and independent_marginals either pass or fail only on "missing … singling-out" (inconclusive). Record which. If honest generators still fail a *lower-bound* risk check, report the per-secret lower bounds (from `inference_by_secret` and warnings) — do not change thresholds.

- [ ] **Step 6: Checkpoint** (`git status --short`).

---

### Task 2: Synthcity worker — environment and generators

**Files:**
- Create: `workers/synthcity/pyproject.toml`, `workers/synthcity/src/stl_synthcity/__init__.py`, `workers/synthcity/src/stl_synthcity/generators.py`, `workers/synthcity/tests/test_generators.py`
- Modify: `packages/core/src/stl_core/catalog.py` (marginals entry stays; nothing else unless names differ), `README.md` (env table row)

**Interfaces:**
- `stl_synthcity.generators.build(name: str, seed: int = 0, **params) -> Generator` for catalog names `marginals`, `bayesian_network`, `ctgan`, `tvae`, `tabddpm`, `privbayes_eps0.1`, `privbayes_eps1`, `privbayes_eps10` (and `great`, Task 3). Unknown name → `KeyError`.
- Mapping catalog name → synthcity plugin: `marginals→marginal_distributions`, `bayesian_network→bayesian_network`, `ctgan→ctgan`, `tvae→tvae`, `tabddpm→ddpm`, `privbayes_*→privbayes`. Param mapping: `epochs → n_iter`; `epsilon → epsilon`; everything else passed through. Always pass `random_state=seed`.
- `SynthcityGenerator.fit(train_df, metadata)`: drops `id` columns before fitting (remembers them); fits the plugin on the remaining frame.
- `SynthcityGenerator.sample(n)`: `plugin.generate(count=n).dataframe()`; re-orders columns to TRAIN order; casts each column back to TRAIN's dtype when possible (categoricals → `object` strings, ints via round); re-creates dropped id columns as `f"syn-{i}"` strings; returns exactly `n` rows (synthcity may return fewer — top up by generating again, max 3 tries, then raise `RuntimeError("synthcity returned fewer rows than requested")`).
- Silence synthcity's own logging noise during fit/sample by setting the `synthcity` loguru logger level to ERROR inside the module (one place, commented).

- [ ] **Step 1: Create the env**

`workers/synthcity/pyproject.toml`:
```toml
[project]
name = "stl-synthcity"
version = "0.1.0"
description = "Synthcity worker: marginals, Bayesian network, CTGAN, TVAE, TabDDPM, PrivBayes, GReaT"
requires-python = ">=3.11,<3.12"
dependencies = [
  "stl-core",
  "stl-runner[s3]",
  "synthcity==0.2.12",
  "torch==2.2.2",
  "opacus==1.4.1",
  "pyarrow>=14,<20",
  "be-great==0.0.14",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[tool.uv.sources]
stl-core = { path = "../../packages/core", editable = true }
stl-runner = { path = "../../packages/runner", editable = true }
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/stl_synthcity"]

[tool.pytest.ini_options]
markers = ["slow: takes more than a minute", "network: downloads a model"]
filterwarnings = ["ignore::FutureWarning", "ignore::DeprecationWarning"]
```
Run: `bash scripts/envrun.sh synthcity python -c "from synthcity.plugins import Plugins; print(sorted(Plugins().list()))"` → list includes `ctgan`, `ddpm`, `privbayes`. If the resolver changes any pin, stop and report.

- [ ] **Step 2: Write failing tests** — `workers/synthcity/tests/test_generators.py`:
```python
import numpy as np
import pandas as pd
import pytest

from stl_core.catalog import load_generator
from stl_core.metadata import ColumnType, TableMetadata

MD = TableMetadata(columns={"rid": ColumnType.id, "x1": ColumnType.numerical,
                            "k": ColumnType.numerical, "c": ColumnType.categorical,
                            "y": ColumnType.categorical}, target="y")


def _train(n=400):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({"rid": [f"r{i}" for i in range(n)], "x1": x,
                         "k": rng.integers(0, 50, n).astype(float),
                         "c": np.where(x > 0, "hi", "lo").astype(object),
                         "y": np.where(x > 0.2, "p", "n").astype(object)})


FAST = [("marginals", {}), ("bayesian_network", {}), ("privbayes_eps1", {}),
        ("ctgan", {"epochs": 2}), ("tvae", {"epochs": 2})]


@pytest.mark.parametrize("name,params", FAST)
def test_generator_contract(name, params):
    train = _train()
    gen = load_generator(name, seed=0, **params)
    gen.fit(train, MD)
    out = gen.sample(150)
    assert len(out) == 150
    assert list(out.columns) == list(train.columns)
    assert out["c"].dtype == object and set(out["c"]) <= {"hi", "lo"}
    assert out["x1"].dtype == float
    assert out["rid"].is_unique


def test_privbayes_epsilons_are_distinct_configs():
    assert load_generator("privbayes_eps0.1").params["epsilon"] == 0.1
    assert load_generator("privbayes_eps10").params["epsilon"] == 10.0


@pytest.mark.slow
def test_tabddpm_contract():
    train = _train(200)
    gen = load_generator("tabddpm", seed=0, epochs=2)
    gen.fit(train, MD)
    assert len(gen.sample(50)) == 50


def test_unknown_name():
    from stl_synthcity.generators import build
    with pytest.raises(KeyError):
        build("nope")
```

- [ ] **Step 3: Run, verify FAIL** (`bash scripts/envrun.sh synthcity pytest workers/synthcity/tests -q -m "not slow"` → `No module named 'stl_synthcity.generators'`).

- [ ] **Step 4: Implement `generators.py`** per Interfaces. Core shape:
```python
"""Synthcity-backed generators (and GReaT via be-great)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from stl_core.generator import Generator
from stl_core.metadata import ColumnType, TableMetadata

PLUGINS = {"marginals": "marginal_distributions", "bayesian_network": "bayesian_network",
           "ctgan": "ctgan", "tvae": "tvae", "tabddpm": "ddpm",
           "privbayes_eps0.1": "privbayes", "privbayes_eps1": "privbayes",
           "privbayes_eps10": "privbayes"}


class SynthcityGenerator(Generator):
    def __init__(self, name: str, seed: int = 0, **params) -> None:
        super().__init__(seed=seed, **params)
        self.name = name

    def _plugin_kwargs(self) -> dict:
        kw = dict(self.params)
        if "epochs" in kw:
            kw["n_iter"] = kw.pop("epochs")
        return {"random_state": self.seed, **kw}

    def fit(self, train_df: pd.DataFrame, metadata: TableMetadata) -> None:
        from synthcity.plugins import Plugins
        self._columns = list(train_df.columns)
        self._dtypes = train_df.dtypes.to_dict()
        self._ids = [c for c in self._columns if metadata.columns.get(c) == ColumnType.id]
        self._cats = {c for c in self._columns if metadata.columns.get(c) == ColumnType.categorical}
        self._plugin = Plugins().get(PLUGINS[self.name], **self._plugin_kwargs())
        self._plugin.fit(train_df.drop(columns=self._ids))

    def sample(self, n: int) -> pd.DataFrame:
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
        for c in self._ids:
            out[c] = [f"syn-{i}" for i in range(n)]
        return self._restore_types(out[self._columns])

    def _restore_types(self, out: pd.DataFrame) -> pd.DataFrame:
        for c, dt in self._dtypes.items():
            if c in self._cats or dt == object:
                out[c] = out[c].astype(str).astype(object)
            elif pd.api.types.is_integer_dtype(dt):
                out[c] = pd.to_numeric(out[c], errors="coerce").round().astype(dt)
            elif pd.api.types.is_float_dtype(dt):
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        return out.reset_index(drop=True)


def build(name: str, seed: int = 0, **params) -> Generator:
    if name == "great":
        from stl_synthcity.great import GReaTGenerator
        return GReaTGenerator(seed=seed, **params)
    if name not in PLUGINS:
        raise KeyError(f"stl_synthcity does not provide {name!r}")
    return SynthcityGenerator(name, seed=seed, **params)
```
Note: categorical values that synthcity returns as numbers (e.g. `1.0` for `"1"`) — when TRAIN's categorical values are all numeric-like strings, map `str(float(v))` back to the matching TRAIN category; add this to `_restore_types` with a test using a categorical column of `"1"/"2"/"3"`.

- [ ] **Step 5: Run tests → PASS** (fast set), then the slow test once. Lint. README env table: add `synthcity | workers/synthcity | 3.11 | marginals, BayesNet, CTGAN, TVAE, TabDDPM, PrivBayes, GReaT`.

- [ ] **Step 6: Checkpoint.**

---

### Task 3: GReaT via be-great

**Files:**
- Create: `workers/synthcity/src/stl_synthcity/great.py`, `workers/synthcity/tests/test_great.py`

**Interfaces:**
- `GReaTGenerator(Generator)`, `name = "great"`; params `epochs` (default 50), `llm` (default `"distilgpt2"`), `batch_size` (default 8). `fit`: drop id columns; `be_great.GReaT(llm=llm, epochs=epochs, batch_size=batch_size, experiment_dir=<tempdir>)` then `.fit(df)`. `sample(n)`: `.sample(n_samples=n, device="cuda" if torch.cuda.is_available() else "cpu")`; restore column order/dtypes and ids exactly as `SynthcityGenerator._restore_types` does (reuse it — move `_restore_types` to a module-level helper `restore_types(out, columns, dtypes, cats, ids)` shared by both).
- Seed: `torch.manual_seed(seed)`, `np.random.seed(seed)` before fit and sample.

- [ ] **Step 1: Failing test** (`network` + `slow` marked — downloads distilgpt2, ~350 MB, CPU ≈ 2 min):
```python
import numpy as np
import pandas as pd
import pytest

from stl_core.catalog import load_generator
from stl_core.metadata import ColumnType, TableMetadata


@pytest.mark.network
@pytest.mark.slow
def test_great_contract():
    rng = np.random.default_rng(0)
    x = rng.normal(size=120).round(2)
    train = pd.DataFrame({"x1": x, "c": np.where(x > 0, "hi", "lo").astype(object),
                          "y": np.where(x > 0.2, "p", "n").astype(object)})
    md = TableMetadata(columns={"x1": ColumnType.numerical, "c": ColumnType.categorical,
                                "y": ColumnType.categorical}, target="y")
    gen = load_generator("great", seed=0, epochs=1, batch_size=16)
    gen.fit(train, md)
    out = gen.sample(20)
    assert len(out) == 20 and list(out.columns) == ["x1", "c", "y"]
    assert out["x1"].dtype == float
```
- [ ] **Step 2–4:** RED → implement → GREEN (run it once; it is excluded from default CI runs by `-m "not slow"`).
- [ ] **Step 5: Checkpoint.**

---

### Task 4: Datasets, GPU-only guard, and main-benchmark script

**Files:**
- Create: `experiments/conf/datasets/credit.yaml`, `experiments/conf/datasets/diabetes.yaml`, `scripts/run_benchmark.sh`
- Modify: `experiments/run.py` (`generate --allow-gpu-only`), `experiments/conf/config.yaml` (`generators_to_run` default = full CPU lineup), `experiments/tests/test_cli.py`, `README.md`

**Interfaces:**
- `credit.yaml`: `name: credit, source: openml, openml_id: 46373, openml_name: null, version: null, path: null, target: y, drop: [], column_types: {}`.
- `diabetes.yaml`: `name: diabetes, source: openml, openml_id: 4541, …, target: readmitted, drop: [encounter_id, patient_nbr]`.
- `generators_to_run` default: `[train_copy, independent_marginals, marginals, gaussian_copula, bayesian_network, ctgan, tvae, privbayes_eps0.1, privbayes_eps1, privbayes_eps10]` (GPU-only `tabddpm`, `great` run via the offline path / explicit override).
- `cmd_generate`: if `CATALOG[g].cpu_ok is False` and not `--allow-gpu-only` and `torch` reports no CUDA (or torch not importable), print `f"skipped {job.job_id}: GPU-only generator (use --allow-gpu-only to force)"` and leave it queued.
- `scripts/run_benchmark.sh DATASET [extra -o overrides…]`: runs prepare (eval env) → generate (sdv env) → generate (synthcity env) → generate (eval env, for core baselines) → evaluate (eval env) → report (eval env), with `-o datasets=$DATASET`, failing fast with `set -euo pipefail` but continuing past non-zero generate exit codes (record them) so one failed generator doesn't block the others; prints a summary of exit codes at the end.

- [ ] **Step 1: Failing tests** in `experiments/tests/test_cli.py`: (a) `load_config(["datasets=credit"])["dataset"]["openml_id"] == 46373` and diabetes drops both id columns; (b) with `generators_to_run=[tabddpm]` in the eval env, `generate` prints `skipped tabddpm__s0: GPU-only generator` and the state stays `queued`.
- [ ] **Step 2–4:** RED → implement → GREEN. Run `bash scripts/envrun.sh eval python experiments/run.py -o datasets=credit -o max_rows=2000 -o seeds=[0] -o run_id=credit-check prepare` and `-o datasets=diabetes …` once each to prove the OpenML loads and metadata inference work; paste the printed "registered … rows" lines. If diabetes inference marks a high-cardinality code column (`diag_1/2/3`, `medical_specialty`, `payer_code`) as `id`, set it to `categorical` via `column_types` in `diabetes.yaml`.
- [ ] **Step 5: README** — "Running the main benchmark" section: `bash scripts/run_benchmark.sh adult` (and credit, diabetes), what each env does, expected CPU time warning.
- [ ] **Step 6: Checkpoint.**

---

### Task 5: Results bundles — export and import

**Files:**
- Create: `packages/runner/src/stl_runner/bundle.py`, `packages/runner/tests/test_bundle.py`
- Modify: `packages/core/src/stl_core/schemas.py` (`BundleManifest`), `experiments/run.py` (`export-results`, `import-results`), `experiments/tests/test_cli.py`

**Interfaces:**
- `BundleManifest(BaseModel)`: `format_version: int = 1`, `run_id: str`, `dataset_id: str`, `dataset_fingerprint: str`, `created_at: str` (ISO UTC), `jobs: list[JobSpec]`, `include_synthetic: bool`.
- `export_bundle(store, run_id, out_path: str, include_synthetic: bool = False) -> BundleManifest` — zip containing `bundle.json` and, for every job with a `report.json`, `jobs/<job_id>/{report.json,cost.json,status.json}` (+ `synthetic.parquet` if requested).
- `import_bundle(store, bundle_path: str, force: bool = False) -> list[MetricReport]` — reads the manifest; if `datasets/<dataset_id>/fingerprint.json` exists locally and differs from the bundle's → `ValueError("bundle dataset fingerprint … differs from local … (use --force)")` unless `force`; writes job files into `runs/<run_id>/<job_id>/`; merges the run manifest (union of jobs by `job_id`, bundle wins for duplicates); returns the imported reports.
- CLI: `export-results --run R --out PATH [--include-synthetic]`; `import-results PATH [--force]` (logs each imported report to MLflow via `log_report` when a tracking URI is configured; prints `imported N reports into run R`).

- [ ] **Step 1: Failing tests** — `test_bundle.py`: build a store with `register_dataset` + `plan_run` + `generate_job` (core baselines) + write a minimal `MetricReport` per job via `store.write_json(keys.report(...))`; export → zip exists, manifest lists jobs; import into a fresh store with the same dataset registered → reports readable and manifest merged; import into a store whose dataset was registered with a different row cap → ValueError mentioning `--force`; with `force=True` succeeds. CLI test: `export-results` then `import-results` into another `storage_root` round-trips and `report --json` shows the imported generators.
- [ ] **Step 2–4:** RED → implement → GREEN.
- [ ] **Step 5: Checkpoint.**

---

### Task 6: Offline GPU notebook (Kaggle / Colab)

**Files:**
- Create: `notebooks/gpu_offline_run.ipynb`, `notebooks/README.md`

**Interfaces:** Notebook cells (markdown + code), runnable top to bottom on Kaggle/Colab with a GPU:
1. Clone the repo (`git clone <REPO_URL>`; first cell sets `REPO_URL = "https://github.com/rohitkumar0601/Synode.git"` and `DATASET = "adult"`, `SEEDS = "[0,1,2]"`, `GENERATORS = "[tabddpm,great]"`).
2. Install uv (`pip install uv`), export `STL_VENV_ROOT=/kaggle/working/.stl` (or `/content/.stl`), `STL_STORAGE_ROOT=/kaggle/working/store`.
3. On Linux the synthcity env must use CUDA torch: run `uv` with an override env var `UV_INDEX_PYTORCH_CPU_URL=https://download.pytorch.org/whl/cu121` (document in markdown that this swaps the CPU index for CUDA 12.1 wheels).
4. `bash scripts/envrun.sh eval python experiments/run.py -o datasets=$DATASET -o seeds=$SEEDS -o generators_to_run=$GENERATORS -o run_id=$DATASET-gpu prepare`
5. `… synthcity … generate --allow-gpu-only`
6. `… eval … evaluate`
7. `… eval … export-results --run $DATASET-gpu --out /kaggle/working/$DATASET-gpu.zip`
8. Markdown: download the zip, then locally `bash scripts/envrun.sh eval python experiments/run.py -o datasets=<same> import-results <zip>` — note the dataset must be registered locally with the same `max_rows` (fingerprint check).
`notebooks/README.md`: what the notebook does, GPU time estimates are unknown until first run, how to import.

- [ ] **Step 1:** Write the notebook as valid nbformat 4 JSON (use `nbformat` in the eval env to create it programmatically from a small script in the scratchpad, or write JSON directly) — verify with `bash scripts/envrun.sh eval python -c "import json; nb=json.load(open('notebooks/gpu_offline_run.ipynb')); assert nb['nbformat']==4; print(len(nb['cells']),'cells')"`.
- [ ] **Step 2:** Verify the shell commands in the notebook locally with a CPU dry run on tiny settings: same sequence with `-o max_rows=600 -o seeds=[0] -o generators_to_run=[tabddpm] -o generator_params.tabddpm.epochs=1 -o run_id=nb-dry` and `generate --allow-gpu-only` in the synthcity env, then evaluate + export-results; confirm the zip contains `bundle.json`. Paste outputs.
- [ ] **Step 3: Checkpoint.**

---

### Task 7: Cross-environment smoke with the full CPU lineup + CI

**Files:**
- Modify: `scripts/smoke_adult.sh` (add the synthcity generate stage; low epochs), `.github/workflows/ci.yml` (synthcity matrix entry running `-m "not slow and not network"`), `README.md`

- [ ] **Step 1:** Update `smoke_adult.sh`: overrides add `-o budget.epochs=5 -o generators_to_run=[train_copy,independent_marginals,marginals,gaussian_copula,bayesian_network,ctgan,tvae,privbayes_eps0.1,privbayes_eps1,privbayes_eps10]`, and a `bash scripts/envrun.sh synthcity … generate` stage between sdv and eval-evaluate.
- [ ] **Step 2:** Run `bash scripts/smoke_adult.sh` end to end. Paste the full report table. Expected: train_copy fails the gate; every other generator has scores; record which pass the gate and why any fail (lower-bound check vs inconclusive). PrivBayes ε=0.1 should rank lowest on utility among PrivBayes variants (RQ2 sanity) — record whether it does; don't assert.
- [ ] **Step 3:** CI: add matrix entry `env: synthcity, tests: workers/synthcity/tests -m "not slow and not network"` (adjust the run line to pass `${{ matrix.args }}` if needed).
- [ ] **Step 4:** Run every suite separately (core, runner, eval workers, eval experiments, sdv, synthcity fast) + `ruff check .`; paste summary lines.
- [ ] **Step 5: Checkpoint.**

## Phase gate (S2 done when)
- All suites pass; ruff clean.
- `smoke_adult.sh` runs the full CPU lineup end to end; train-copy fails the statistical gate; at least one honest generator passes it (if none do, the report states exactly which lower-bound/inconclusive checks block them).
- A bundle exported from the notebook's dry-run imports into a fresh store and appears in `report`.
