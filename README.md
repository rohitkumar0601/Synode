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
| sdv | workers/sdv | 3.12 | Gaussian Copula, SDV CTGAN (`sdv_ctgan`), SDV TVAE (`sdv_tvae`) |
| synthcity | workers/synthcity | 3.11 | marginals, BayesNet, CTGAN, TVAE, TabDDPM, GReaT (`great`, `great_qwen`), PrivBayes (nominal-DP audit only); the `metrics` stage (synthcity detection AUC, alpha-precision/beta-recall/authenticity, DOMIAS MIA) |
| smartnoise | workers/smartnoise | 3.12 | MST (smartnoise-synth 1.0.8): the RQ2 epsilon sweep |
| safesynth | workers/safesynth | ≥3.10 | NVIDIA Safe-Synthesizer (`nss_smollm3`, `nss_smollm3_dp_eps8`) config and GPU-export wrapper |

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
bash scripts/envrun.sh safesynth pytest workers/safesynth/tests -q
```

## Benchmark CLI

See the docstring of `experiments/run.py`. Quick smoke run: `bash scripts/smoke_adult.sh` (Adult, 2,000 rows, one seed, 5 epochs, the full CPU lineup across the sdv, synthcity and smartnoise envs, plus the PrivBayes audit entries; takes tens of minutes on CPU).
Full main run on Adult (3 seeds, 20,000 rows) is the same sequence without the overrides.

## Training Budget Profiles

The benchmark supports three training-budget profiles (`budget.profile`, configured in `experiments/conf/budget/*.yaml`):

| Profile | Epochs | Time Cap | Use Case |
|---|---|---|---|
| `library` (default) | `null` (library defaults) | 86,400 s (24 h) | Canonical benchmark. No blanket epoch overrides; each library trains with its native settings (e.g. synthcity CTGAN `n_iter=2000` with early stopping, TVAE `n_iter=1000`, TabDDPM `1000`, SDV `300`, GReaT `100`). |
| `equal50` | 50 | 7,200 s (2 h) | CPU benchmark comparison. Gives every deep generator an identical 50-epoch budget. |
| `smoke` | 5 | 1,800 s (30 min) | Fast CI and smoke tests. |

### Measured and Estimated Runtimes (Adult, ~12,000 TRAIN rows, CPU)

| Generator | Library Default | Measured / Estimated (CPU) | Notes |
|---|---|---|---|
| `gaussian_copula` (SDV) | Analytical fit | 10–50 s | Fast algebraic estimation |
| `independent_marginals` | Analytical fit | 156–347 s | Per-column marginal empirical CDF |
| `bayesian_network` | Tree search (K2) | 128–214 s | Chow-Liu tree structure search |
| `mst_eps*` (SmartNoise) | Graphical model | 188–583 s | Private 1-way and 2-way marginal selection |
| `ctgan` (Synthcity) | `n_iter=2000` (early-stop) | 26.9 s/epoch (50 ep: ~22 min; 2000 ep max: ~15 h) | Early stops when discriminator patience >= 5 and iter >= 100 |
| `tvae` (Synthcity) | `n_iter=1000` (early-stop) | 48.2 s/epoch (50 ep: ~40 min; 1000 ep max: ~13.4 h) | Early stops on validation loss |
| `sdv_ctgan` (SDV) | `epochs=300` | ~1.5–2 h | Batch 500, Pac 10, no early stopping |
| `sdv_tvae` (SDV) | `epochs=300` | ~2–3 h | Batch 500, loss factor 2 |
| `tabddpm` (Synthcity) | `n_iter=1000` | GPU recommended (`cpu_ok=False`) | 1,000 diffusion steps; run via GPU notebook |
| `great` (be-great `distilgpt2`) | `epochs=100` | GPU recommended (`cpu_ok=False`) | Guided sampling on 12k rows is ~14.7 h on CPU |
| `great_qwen` (`Qwen3-0.3B-distil`) | `epochs=5` | GPU recommended (`cpu_ok=False`) | Tabular-pretrained LLM; fast fine-tuning on GPU |
| `nss_smollm3` (Safe-Synthesizer) | QLoRA, `SmolLM3-3B` | GPU only (Linux A100/H100) | Run via `notebooks/gpu_offline_run.ipynb`, load via `import-results` |
| `nss_smollm3_dp_eps8` | DP-SGD, ε=8.0 | GPU only (Linux A100/H100) | DP-SGD with Opacus on SmolLM3-3B |


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

Artifacts live in S3-compatible object storage. Official MinIO container images are no longer
publicly pullable, so MinIO runs natively on the host and the compose services reach it at
`host.docker.internal:9000` (any S3-compatible endpoint works: set `STL_COMPOSE_S3_ENDPOINT_URL`
for compose, `S3_ENDPOINT_URL` for host-run processes).

Start MinIO (once per boot; credentials must match `.env.example` / `docker-compose.yml`):

```bash
MINIO_ROOT_USER=stl MINIO_ROOT_PASSWORD=stl-dev-secret minio server ~/minio-data --address :9000 --console-address :9001
mc alias set stl http://127.0.0.1:9000 stl stl-dev-secret && mc mb --ignore-existing stl/stl-artifacts
```

Then start the platform and run the end-to-end check:

```bash
docker compose up -d --build --wait
bash scripts/e2e_platform.sh
```

Useful URLs: web UI `http://127.0.0.1:3010`, API docs `http://127.0.0.1:8010/docs`, MLflow
`http://127.0.0.1:5000`.

### Web Dashboard (`apps/web`)

The Next.js web application runs on port 3010. It provides:
- Dataset catalog and upload interface (enforces `STL_MAX_UPLOAD_MB`, default 200 MB).
- Run creation wizard with budget profile selection (`library`, `equal50`, `smoke`).
- Run overview with live job statuses, run control (terminate and resume), parameter disclosures, and recommendation display.
- Interactive Pareto frontier plots (Recharts / Plotly) and downloadable report summaries.
- Responsive design supporting dark mode, light mode, and mobile/phone widths.

To run the web app in standalone development mode:
```bash
cd apps/web
npm install
npm run dev
```

### API Configuration & Limits

The API (`services/api`) enforces streaming request body limits:
- `STL_MAX_UPLOAD_MB`: Maximum dataset upload size in megabytes (default: `200`). Uploads exceeding this limit receive HTTP 413.
- `STL_BUDGET_PROFILE`: Default budget profile applied when POSTing runs without an explicit profile (default: `library`).
- `STL_TIME_CAP_SECONDS`: Global execution timeout for jobs (overrides profile defaults if set).

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
RQ3 is **exploratory**: with ~25 datasets a meta-model that does not beat the majority baseline is
a valid result and is reported as such (full protocol: spec §10).

### Protocol
1. **Dataset Suite:** 25 OpenML-CC18 classification datasets (`experiments/conf/meta/datasets.yaml`, with the excluded tasks and reasons) filtered by strict criteria (500–20,000 instances, ≤60 features, ≤20 classes, minority class ≥25 rows); first 25 by OpenML id.
2. **Generators & Budget:** Fast CPU lineup (`independent_marginals`, `gaussian_copula`, `bayesian_network`, `ctgan`, `tvae`, `mst_eps1` + `train_copy` as sanity ceiling, never a label). Budget profile `equal50` (50 epochs for the deep generators, 7,200 s cap per job; independent of the library-defaults mode of the main benchmark), max 5,000 rows, seeds 0,1,2.
3. **Winner Labelling:** `recommend(aggregate(reports), "balanced")` — the highest balanced score among candidates passing the privacy gate. Datasets where every generator fails the gate are labelled `none` and reported, but excluded from training and accuracy.
4. **Meta-Features:** `pymfe` groups `general`, `statistical`, and `info-theory` (mean and standard deviation summaries) computed on the seed-0 TRAIN split.
5. **Evaluation:** Leave-One-Dataset-Out (LODO) with a `RandomForestClassifier` (imputation medians and the baseline fitted on the training folds only) vs an always-predict-the-most-frequent-winner baseline. Accuracy, balanced accuracy, mean regret (definition in `lodo.json`) and `n_gate_failing` (suggestions that failed the gate) are reported.

### Running on Owner Compute

The full run (25 datasets × 7 generators × 3 seeds) is too long for a laptop session; run it in
chunks with `--ids` (re-running a chunk resumes: finished jobs are kept). Defaults are the protocol
(seeds 0,1,2, `budget=equal50`, max 5,000 rows) — do **not** pass `--epochs` for the real run.

1. **Clean up the smoke/diagnostic artefacts first** (they would otherwise be mixed into the labels
   and served as suggestions):
   ```bash
   rm -rf outputs/store/runs/meta-cc18_37   # 2-generator diagnostic run (MST probe) -> would be labelled none
   rm -f outputs/store/meta/predictor.joblib outputs/store/meta/predictor.json outputs/store/meta/lodo.json  # 2-dataset smoke predictor
   ```
   The 1-seed/2-epoch smoke runs `meta-cc18_3` / `meta-cc18_11` are resumed by the full run only for
   their seed-0 jobs, which were trained with 2 epochs: delete them too (`rm -rf outputs/store/runs/meta-cc18_3 outputs/store/runs/meta-cc18_11`).
   With a non-local store (`STL_STORAGE_ROOT=s3://…`) delete the same keys there.
2. **Run in chunks** (dataset ids from `experiments/conf/meta/datasets.yaml`):
   ```bash
   bash scripts/run_meta_benchmark.sh --ids 3,11,15,18,22
   bash scripts/run_meta_benchmark.sh --ids 23,29,31,32,37
   bash scripts/run_meta_benchmark.sh --ids 38,44,46,50,54
   bash scripts/run_meta_benchmark.sh --ids 182,188,307,469,1049
   bash scripts/run_meta_benchmark.sh --ids 1050,1053,1063,1067,1068
   ```
   Each stage logs to `outputs/meta_logs/<id>_<stage>.log`; the exit-code table only covers the
   current invocation (check `report` output and `outputs/meta_logs/mst_bounds.txt` for failed jobs).
   Smoke check: `bash scripts/run_meta_benchmark.sh --limit 2 --seeds 0 --epochs 2`.

**Expected runtime (estimate, not a measurement of the full protocol).** Measured: the smoke run of
`cc18_3` (kr-vs-kp, 3,196 rows, 1 seed, 2 epochs) took ≈64 min end to end on a loaded laptop
(synthcity generate ≈18 min, evaluate ≈25 min), `cc18_11` (625 rows) ≈25 min. Extrapolation for the
protocol: the non-training stages scale with the number of jobs (×3 seeds → ≈2–3 h per dataset of
that size), and the deep generators' training grows with 50 instead of 2 epochs, bounded by the
`equal50` cap of 7,200 s per job (ctgan + tvae × 3 seeds ≤ 12 h worst case). Plan on roughly
3–15 h per dataset and several days for all 25 on a laptop-class CPU; a multi-core workstation and
running chunks in parallel (different `--ids`, same store) shortens this.

### Evaluating and Retraining the Predictor

After the benchmark (the runner already computes `datasets/cc18_<id>/metafeatures.json`; to recompute
meta-features for registered datasets: `bash scripts/envrun.sh meta python experiments/run.py metafeatures --all-registered`):

```bash
bash scripts/envrun.sh meta python experiments/run.py meta-eval            # LODO only -> meta/lodo.json
bash scripts/envrun.sh meta python experiments/run.py meta-eval --train    # LODO + final predictor
```
Run `meta-eval` first and read the LODO result; `--train` then writes the predictor the platform
serves. Dataset `cc18_<id>` is read from run `meta-cc18_<id>`; use `--run-prefix <prefix>` for runs named
`<prefix>cc18_<id>` (`--datasets-yaml` selects another frozen list). Datasets with missing/failed
meta-features, no run, or an incomplete run (a planned job without a report) are excluded and
listed with the reason. With fewer than 3 labelled datasets (or a single winner class) LODO reports
`insufficient_data` and `--train` refuses to write a predictor (exit 1).
Artifacts saved to `outputs/store/`:
- `meta/lodo.json` — LODO accuracy, balanced accuracy, mean regret (definition in `regret_definition`;
  gate-failing picks counted in `n_gate_failing`), the same for the baseline, per-dataset predictions,
  `none` and excluded datasets.
- `meta/predictor.joblib` — Trained model bundle (RandomForest, feature list, training medians, labels, metadata).
- `meta/predictor.json` — Metadata, feature list, medians and the LODO summary (no model).

Report: `notebooks/rq3_meta.ipynb` reads these from the store (`STL_STORAGE_ROOT` or the path in its
first cell) plus `outputs/meta_logs/mst_bounds.txt`: protocol, dataset table (labelled / `none` /
excluded with reasons, MST fallback losses), LODO metrics vs baseline, per-dataset predictions,
confusion matrix, top-15 feature importances and the label distribution; every section explains
instead of plotting when the data are insufficient. Headless:
`bash scripts/envrun.sh meta python -m jupyter nbconvert --to notebook --execute --output-dir outputs/notebooks --output rq3_meta_executed.ipynb notebooks/rq3_meta.ipynb`
(matplotlib/nbconvert/ipykernel are in the meta env's dev group). Keep the committed notebook without outputs.

### Known limitations
- **MST private-bounds fallback:** the OpenML datasets have no codebook bounds (`datasets=openml_generic`),
  so MST spends 0.2·ε inferring bounds of continuous columns privately. On small numeric datasets this
  fails (e.g. diabetes, 461 TRAIN rows, 8 continuous columns: 0.025 ε per column) and `mst_eps1` is then
  not a candidate there. This is reported, not patched with data-derived "public" bounds (that would void
  the DP claim): `outputs/meta_logs/mst_bounds.txt` and the notebook list the affected datasets, and the
  labels are conditional on it.
- **DOMIAS not applicable on all-categorical datasets** (ids 3, 11, 46, 50, 469): synthcity's DOMIAS needs a
  continuous column; there the gate's membership-inference check uses the distance-based MIA only.

### Platform Integration
When running in Docker Compose (`worker-meta` on `q.meta`), registering an uploaded or built-in dataset enqueues `stl.metafeatures`, computing meta-features and serving the best-generator suggestion to `GET /datasets/{id}/suggestion` and the web UI's New Run page.

Suggestions are computed at registration time: training or retraining the predictor does not
update suggestions of datasets registered before it. Refresh one with
`POST /datasets/{id}/suggestion/refresh` (202; works even when runs use the dataset; 409 if the
dataset is not registered, 503 if the task cannot be queued) or the New Run page's "Refresh
suggestion" link, or run the task directly against the same store:
```bash
STL_STORAGE_ROOT=<store> bash scripts/envrun.sh meta python -c "from stl_meta.tasks import metafeatures; metafeatures('<dataset_id>')"
```
Without a predictor the endpoint returns `{"available": false, "reason": ...}`. The predictor covers
**classification datasets only**: a dataset with a numeric (regression) target is never given to
pymfe and gets `{"available": false, "reason": "RQ3 predictor covers classification datasets only"}`.
pymfe runs in a child process that is terminated after 300 s (plus a Celery hard limit of 360 s on
the task), so an oversized upload cannot leave extraction work running in the worker.

## Analysis & Report (RQ1, RQ2, Cost)

The analysis package (`packages/analysis`, run via `scripts/envrun.sh analysis …`) transforms raw stored job metrics into the final thesis evidence:

1. **Collect & Tidy:** Scans one or more artifact stores (`outputs/store`, `outputs/store_integrated`) and compiles all job outcomes into a unified results table.
2. **RQ1 Benchmarks & CD Diagrams:** Computes aggregated balanced scores across seeds, ranks generators per dataset, and runs statistical tests via `autorank` (Friedman test followed by Nemenyi critical difference diagrams or Bayesian sign tests).
3. **RQ2 Differential Privacy Curves:** Generates ε-tradeoff plots for quality, utility, and privacy risks, and quantifies the Pareto frontier shift when DP synthesizers are introduced.
4. **Cost & Efficiency Analysis:** Measures median training runtime, sampling throughput, and peak RAM consumption (RSS) per generator.

### Generating the Report

```bash
# Generate all figures, tables, and reports/results.md from local stores:
bash scripts/make_report.sh

# Or run with explicit store paths:
bash scripts/make_report.sh --store outputs/store --store outputs/store_integrated --out reports
```

### Importing GPU/Owner Compute Runs

For models trained offline on GPU (such as `tabddpm`, `great_qwen`, or NVIDIA Safe-Synthesizer `nss_smollm3`):
```bash
# Import external result tarball into the local store:
bash scripts/envrun.sh runner python experiments/run.py import-results path/to/results_bundle.tar.gz

# Re-run report generation to integrate the newly imported numbers:
bash scripts/make_report.sh
```

### Interactive Notebooks
- `notebooks/rq1_benchmark.ipynb`: Statistical comparison across generators with Autorank CD diagrams.
- `notebooks/rq2_dp.ipynb`: Differential privacy ε trade-off curves and Pareto-frontier shifts.
- `notebooks/rq3_meta.ipynb`: Meta-learning predictor evaluation, confusion matrices, and feature importances.


