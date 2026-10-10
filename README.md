# Synthetic Data Trust Lab

Benchmarks tabular synthetic-data generators on fidelity, utility, privacy and cost, and
recommends one through a privacy-gated decision engine. Design: `docs/superpowers/specs/`.

## Setup

1. Fetch the upstream libraries into `third_party/` (pinned refs; idempotent):
   `bash scripts/fetch_third_party.sh`. The worker environments install synthcity, SDV,
   SDMetrics, Anonymeter and smartnoise-synth **editable from these clones** (`[tool.uv.sources]`
   in each `workers/*/pyproject.toml`), so the project runs on the upstream code directly;
   `scripts/envrun.sh` refuses to start a worker env (exit 2) while `third_party/` is missing.
2. Run anything through `bash scripts/envrun.sh <env> …` (below); the first call creates and syncs
   the env.

Docker images of the worker envs must be built with the **repository root** as build context and
`third_party/` fetched first: the editable path sources (`../../third_party/…`) are resolved from
the worker's pyproject, so a worker-directory context cannot install them.

## Environments

Each worker has its own uv project and lockfile. Environments live outside the repo
(`$STL_VENV_ROOT`, default `~/.stl`) because the Windows path limit breaks deep installs.
Always run Python through `scripts/envrun.sh <env> …`:

| env | project | Python | contents |
|---|---|---|---|
| core | packages/core | ≥3.10 | contract only |
| runner | packages/runner | ≥3.10 | storage, pipeline |
| eval | workers/eval | 3.11 | metrics (SDMetrics 0.32.0, Anonymeter 1.0.0, sklearn, XGBoost), merges synthcity metrics |
| sdv | workers/sdv | 3.12 | Gaussian Copula |
| synthcity | workers/synthcity | 3.11 | marginals, BayesNet, CTGAN, TVAE, TabDDPM, GReaT, PrivBayes (nominal-DP audit only); the `metrics` stage (synthcity detection AUC, alpha-precision/beta-recall/authenticity, DOMIAS MIA) |
| smartnoise | workers/smartnoise | 3.12 | MST (smartnoise-synth 1.0.8): the RQ2 epsilon sweep |

RQ2 (the differential-privacy epsilon trade-off) uses smartnoise-synth MST (`mst_eps0.1`,
`mst_eps1`, `mst_eps10`; the catalog epsilon is the total budget: 10% infers continuous
bounds, 90% fits MST). synthcity 0.2.12 PrivBayes is **not** epsilon-DP (numeric columns are
never binned, so synthetic numerics are exact training values; its Laplace noise is scaled
for probabilities but added to counts): `privbayes_eps*` are kept only as a nominal-DP
audit (catalog family `nominal_dp_audit`) and are not in the default lineup.

## Tests

```bash
bash scripts/envrun.sh core   pytest packages/core/tests -q
bash scripts/envrun.sh runner pytest packages/runner/tests -q
bash scripts/envrun.sh eval   pytest workers/eval/tests -q
bash scripts/envrun.sh eval   pytest experiments/tests -q
bash scripts/envrun.sh sdv    pytest workers/sdv/tests -q
bash scripts/envrun.sh synthcity pytest workers/synthcity/tests -q -m "not slow and not network"
bash scripts/envrun.sh smartnoise pytest workers/smartnoise/tests -q
```

## Benchmark CLI

See the docstring of `experiments/run.py`. Quick smoke run: `bash scripts/smoke_adult.sh` (Adult, 2,000 rows, one seed, 5 epochs, the full CPU lineup across the sdv, synthcity and smartnoise envs, plus the PrivBayes audit entries; takes tens of minutes on CPU).
Full main run on Adult (3 seeds, 20,000 rows) is the same sequence without the overrides.

## Running the main benchmark

```bash
bash scripts/run_benchmark.sh adult
bash scripts/run_benchmark.sh credit
bash scripts/run_benchmark.sh diabetes
# extra Hydra overrides follow the dataset, e.g. a reduced run:
bash scripts/run_benchmark.sh adult 'seeds=[0]' max_rows=2000
```

The script runs, in order: `prepare` (eval env: loads the dataset, infers metadata, creates splits,
plans jobs), `generate` in the sdv env (gaussian_copula, baselines), `generate` in the synthcity env
(marginals, bayesian_network, ctgan, tvae), `generate` in the smartnoise env (MST at eps
0.1/1/10), `generate` in the eval env
(core baselines), `metrics` (synthcity env: synthcity's SyntheticDetectionXGB, AlphaPrecision and
DomiasMIAKDE per generated job, written to `runs/<run>/<job>/synthcity_metrics.json`;
`metrics --recompute` re-runs jobs whose file has an error or an older metrics version and
resets their reports for re-evaluation), `evaluate`
(eval env; merges those metrics and adds a nearest-neighbour distance membership attack;
the gated MIA advantage is the max of DOMIAS and the distance attack; without the metrics stage
the fidelity detection/alpha/beta values are missing and a warning is recorded) and `report` (eval env). A failed generator does not stop
the others; exit codes are summarised at the end. Re-run to resume (finished jobs are kept).
The Celery job chain does not have the `metrics` step yet: it must run between generate and
evaluate on the synthcity queue.

## Local Platform

The local compose stack no longer starts MinIO or any S3-compatible localhost service by default.
Artifacts are shared between the API and workers through the `storedata` Docker volume at
`/stl-store`; host-run tools keep using `outputs/store`.

```bash
docker compose up -d --build --wait
bash scripts/e2e_platform.sh
```

Useful URLs: web UI `http://127.0.0.1:3010`, API docs `http://127.0.0.1:8010/docs`, MLflow
`http://127.0.0.1:5000`.

CPU time warning: the full lineup (10 generators x 3 seeds, 300 epochs for the deep models) takes
many hours on CPU; Diabetes (20,000-row cap by default) is the slowest. The GPU-only generators
`tabddpm` and `great` are not in the default lineup; `generate` skips them without CUDA unless you
pass `--allow-gpu-only` (add them via `-o generators_to_run=[...]`).

### Public bounds for the DP generators (MST)

Each dataset config (`experiments/conf/datasets/*.yaml`) lists `public_bounds` (`{column: [lower,
upper]}` from the dataset's codebook, never from the data) and `public_log_scale` for its numerical
columns; MST bins those columns at 0 epsilon. `prepare` exits 2 if a bound names a non-numerical
column, is not a numeric `[lower, upper]` pair with `lower < upper`, or a log-scale column lacks a
bound or has `lower < 0`. For your own file dataset (`dataset.source=file`), the default dataset
group is Adult and Hydra *merges* dict overrides, so Adult's bounds would be inherited: drop them
with `-o '~dataset.public_bounds' -o '~dataset.public_log_scale'` (then MST spends 20% of epsilon
inferring bounds privately) or give your own with `-o '+dataset.public_bounds={col:[lo,hi]}'`
after the removal. The resolved bounds' sha256 is recorded per job in `privacy.json`, which result
bundles carry when present.

## RQ3: Meta-Learning Best-Generator Predictor

Answers RQ3: *Can a meta-model predict the best generator from dataset meta-features alone?*

### Protocol
1. **Dataset Suite:** 25 OpenML-CC18 classification datasets (`experiments/conf/meta/datasets.yaml`) filtered by strict criteria (500–20,000 instances, ≤60 features, ≤20 classes, minority class ≥25 rows).
2. **Generators & Budget:** Fast CPU lineup (`independent_marginals`, `gaussian_copula`, `bayesian_network`, `ctgan`, `tvae`, `mst_eps1` + `train_copy` as sanity ceiling). Budget: 50 epochs, max 5,000 rows, 3 seeds.
3. **Winner Labelling:** Generator with the highest balanced fidelity-utility-privacy score among candidates passing the privacy gate. Datasets where every generator fails the gate are labelled `none` and reported, but excluded from accuracy calculations.
4. **Meta-Features:** `pymfe` groups `general`, `statistical`, and `info-theory` (mean and standard deviation summaries) computed on seed-0 TRAIN split.
5. **Evaluation:** Leave-One-Dataset-Out (LODO) cross-validation with a `RandomForestClassifier` vs an always-predict-the-most-frequent-winner baseline. Regret (loss in balanced score vs ex-post optimal) is reported alongside accuracy.

### Running on Owner Compute

Run the complete meta-benchmark across all 25 datasets:
```bash
bash scripts/run_meta_benchmark.sh --seeds 0,1,2 --epochs 50 --max-rows 5000
```
Expected runtime is approximately 3–6 hours on modern multi-core CPU. Smoke verification on a subset:
```bash
bash scripts/run_meta_benchmark.sh --limit 2 --seeds 0 --epochs 2
```

### Evaluating and Retraining the Predictor

Compute meta-features for all registered datasets:
```bash
bash scripts/envrun.sh meta python experiments/run.py metafeatures --all-registered
```

Run LODO evaluation and train the final predictor:
```bash
bash scripts/envrun.sh meta python experiments/run.py meta-eval
```
Artifacts saved to `outputs/store/`:
- `meta/lodo.json` — LODO accuracy, balanced accuracy, mean regret, and per-dataset predictions.
- `meta/predictor.joblib` — Trained model bundle (RandomForest, feature list, training fold medians, labels).
- `meta/predictor.json` — Metadata and feature list for inference.

Analyze results and plots:
Launch `notebooks/rq3_meta.ipynb` to inspect accuracy/regret comparisons, top feature importances, and the confusion matrix.

### Platform Integration
When running in Docker Compose (`worker-meta` on `q.meta`), registering an uploaded or built-in dataset enqueues `stl.metafeatures`, computing meta-features and serving the best-generator suggestion to `GET /datasets/{id}/suggestion` and the web UI's New Run page.

