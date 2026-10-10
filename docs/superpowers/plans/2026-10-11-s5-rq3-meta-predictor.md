# S5 — RQ3 Meta-Predictor (pymfe + leave-one-dataset-out) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer RQ3 — can a meta-model predict the best generator from a dataset's profile alone? Build the meta-dataset (OpenML-CC18 subset × fast generators × seeds → winner labels), extract pymfe meta-features, evaluate a small classifier leave-one-dataset-out against an always-pick-the-overall-winner baseline, and serve a "suggested generator" for newly registered datasets in the platform.

**Architecture:** A new `workers/meta` uv project (Python 3.11; pymfe 0.4.4 from `third_party/pymfe`, openml from `third_party/openml-python`, scikit-learn, joblib) provides `stl_meta` with: dataset selection (OpenML-CC18 filter → frozen YAML list), meta-feature extraction, meta-dataset assembly from benchmark reports, LODO evaluation, and a predictor artifact. Benchmark runs reuse the existing CLI stages per dataset (`scripts/run_meta_benchmark.sh`). In the platform a Celery task `stl.metafeatures` on queue `q.meta` computes meta-features + a prediction when a dataset is registered; the API exposes `GET /datasets/{id}/suggestion`; the web new-run page shows the suggestion.

**Tech Stack:** pymfe 0.4.4, openml 0.15.x (both editable from `third_party/`), scikit-learn, joblib, Celery 5.6.3; existing core/runner/eval/sdv/synthcity/smartnoise envs; FastAPI; Next.js.

**Spec:** `docs/superpowers/specs/2026-10-05-synth-trust-lab-design.md` §10 (RQ3).

## Global Constraints

- **No git commits or pushes.** Python only via `bash scripts/envrun.sh <env> …`; envs in `$HOME/.stl/<env>`.
- RQ3 is **exploratory** (spec): with ~25 datasets a negative result is valid and must be reported honestly; never tune on the held-out dataset.
- Generators for the meta-benchmark (fast, CPU): `independent_marginals, gaussian_copula, bayesian_network, ctgan, tvae, mst_eps1` (+ `train_copy` as a sanity row, excluded from labels). Budget: `budget.epochs=50`, `max_rows=5000`, seeds `[0,1,2]`, eval `singling_out_mode=univariate`, `n_attacks=200`, `min_attacks=100`.
- Winner label per dataset = generator with the highest **balanced** score among gate-passing candidates (aggregated over seeds via the existing `aggregate` + `recommend`); datasets with no gate-passing candidate are labelled `none` and kept (reported), but excluded from classifier training/eval accuracy (counted separately).
- Meta-features: pymfe groups `general`, `statistical`, `info-theory` (summary `mean`, `sd`); computed on the **registered (prepared, capped) TRAIN split of seed 0** with the target; non-finite → imputed by training-fold medians inside LODO (no leakage).
- Meta-model: `RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")`; LODO: for each dataset, train on the others, predict it. Metrics: accuracy, balanced accuracy, mean **regret** (true best balanced score − score of the predicted generator on that dataset), vs baseline "always predict the most frequent winner in the training folds".
- Dataset selection is deterministic and frozen to `experiments/conf/meta/datasets.yaml` (OpenML ids + targets) — criteria: CC18 tasks with 500 ≤ instances ≤ 20,000, ≤ 60 features, ≤ 20 classes, every class ≥ 5 rows after a 60/20/20 split; take the first 25 by OpenML id that satisfy them (record exclusions with reasons).

## Review Focus

1. **A dataset where every generator fails the gate** — labelled `none`, reported, excluded from accuracy (Task 4).
2. **pymfe producing NaN/inf features or raising on a dataset** — impute in-fold; a dataset whose extraction fails entirely is excluded with a recorded reason (Task 2).
3. **Leakage** — imputation/scaling and the baseline's "most frequent" must be computed on training folds only (Task 4 tests).
4. **Suggestion for a dataset when no predictor artifact exists yet** — API returns `{"available": false}` not 500 (Task 5).
5. **Very wide/huge uploaded datasets** — meta-feature extraction runs on a capped sample (≤ 5,000 rows) with a time cap (Task 5).

---

### Task 1: `workers/meta` environment + dataset selection
Create `workers/meta/{pyproject.toml,src/stl_meta/__init__.py,src/stl_meta/select.py,tests/test_select.py}` and `experiments/conf/meta/datasets.yaml`. pyproject: Python `>=3.11,<3.12`; deps `stl-core`, `stl-runner[celery]` (use the `s3` extra only when an external object store is restored), `pymfe==0.4.4`, `openml`, `scikit-learn`, `joblib`; editable sources `pymfe → ../../third_party/pymfe`, `openml → ../../third_party/openml-python` (verify versions; if the openml clone's version differs from PyPI resolution, pin to the clone). `scripts/envrun.sh`: add `meta` to the third_party guard list. `select.py`: `select_cc18(max_n=25) -> list[dict]` using `openml.study.get_suite(99)` + dataset qualities to apply the criteria; `freeze(path)` writes the YAML (id, name, target, n_instances, n_features, n_classes) + an `excluded:` list with reasons. Tests: criteria filter on a fake qualities table; YAML round-trip. Then run it once (network) and commit the YAML to the tree (no git commit — just write it). Paste the selected list.

### Task 2: Meta-feature extraction
`stl_meta/features.py`: `extract(df, target, metadata, seed, max_rows=5000) -> dict[str, float]` using pymfe (`MFE(groups=["general","statistical","info-theory"], summary=["mean","sd"], random_state=seed)`), categoricals encoded as pymfe expects (`cat_cols` from metadata), stratified sample to max_rows; returns finite floats or NaN (never raises for per-feature failures; raises `MetaFeatureError` only if pymfe fails entirely). CLI stage `experiments/run.py metafeatures` (meta env) writes `datasets/<id>/metafeatures.json` for registered datasets. Tests: toy dataset → expected keys (e.g. `nr_inst`, `nr_attr`, `nr_class`, `class_ent`), deterministic for a seed, NaN tolerated, total failure → MetaFeatureError.

### Task 3: Meta-benchmark runner
`scripts/run_meta_benchmark.sh`: for each dataset in `experiments/conf/meta/datasets.yaml`, run prepare (eval env, `-o dataset.source=openml -o dataset.openml_id=<id> -o dataset.name=cc18_<id> -o dataset.target=<target>` with `~dataset.public_bounds` so Adult bounds aren't inherited, or a dedicated `datasets/openml_generic.yaml` with null bounds), generate (sdv, synthcity, smartnoise, eval envs), metrics (synthcity), evaluate (eval), metafeatures (meta), with run id `meta-cc18_<id>`; continue on per-dataset failures and print a summary table (dataset, stage exit codes). MST needs public bounds: for OpenML datasets without codebook bounds, MST uses the private-bounds fallback (0.2·ε) — record that in the meta report. Verification: run on 2 datasets with `seeds=[0]`, `budget.epochs=2` (smoke); paste output.

### Task 4: Meta-dataset + LODO evaluation
`stl_meta/meta.py`: `build_meta_dataset(store, datasets_yaml) -> DataFrame` (one row per dataset: meta-features + `label` from `recommend(aggregate(reports), "balanced")`, `none` if no winner, plus per-generator balanced scores for regret); `lodo_evaluate(meta_df) -> dict` (per-dataset predictions, accuracy, balanced accuracy, mean regret, baseline metrics, n_none); `train_final(meta_df) -> artifact` (model + feature list + medians + label set + metadata: n_datasets, date, versions) saved with joblib to the store at `meta/predictor.joblib` and a JSON summary `meta/predictor.json`. CLI `experiments/run.py meta-eval` prints the LODO table + summary and writes `meta/lodo.json`. Tests on a synthetic meta-dataset where the label is a deterministic function of one feature (meta-model must beat the baseline) and on a random-label one (no significant win); leakage tests (fold medians, baseline from training folds only); `none` handling.

### Task 5: Platform integration (task, API, web)
- `stl_meta.tasks.metafeatures` (`stl.metafeatures`, queue `q.meta`): for a registered dataset id, compute meta-features (capped sample, time cap 300 s) → `datasets/<id>/metafeatures.json`; if `meta/predictor.joblib` exists, predict → `datasets/<id>/suggestion.json` `{generator, probabilities: {gen: p}, model: {n_datasets, trained_at}, available: true}`; never raises.
- API: dataset register (and builtin) enqueues `stl.metafeatures` (dispatcher; failure to enqueue is logged, not fatal); `GET /datasets/{id}/suggestion` → the JSON or `{"available": false, "reason": ...}`. Tests with NullDispatcher + a fixture suggestion file.
- Web: new-run page shows "Suggested: <generator> (p=…) — exploratory, trained on N datasets" with a "why exploratory" tooltip when available; MSW + tests. Compose: a `worker-meta` service (queue q.meta, Python 3.11 image, third_party pymfe/openml in context).
- Verification: API + web tests; compose build of worker-meta; register a dataset in the running stack and see the suggestion appear (after training a predictor from the smoke meta-benchmark).

### Task 6: RQ3 report artefacts + docs
`notebooks/rq3_meta.ipynb` (reads `meta/lodo.json`, plots accuracy/regret vs baseline, feature importances, confusion matrix); README "RQ3" section (how to run the meta-benchmark on owner compute, expected runtime, how to retrain the predictor); spec §10 updated with the actual protocol and the "exploratory" framing.

## Phase gate (S5 done when)
- All suites pass (incl. `workers/meta`), ruff clean, web checks green.
- Smoke meta-benchmark (2 datasets) runs end to end; `meta-eval` works on it (LODO with 2 datasets is degenerate — the full 25-dataset run happens on owner compute).
- Registering a dataset in the Docker stack yields a suggestion when a predictor exists, and `{"available": false}` otherwise.
