# Synthetic Data Trust Lab — Design Spec

Date: 2026-10-05 · Owner: Rohit Kumar · Source: "Synthetic Data Trust Lab — Architecture, Stack & Project Plan" (Oct 5, 2026)

This spec records the agreed design for building the whole project. The source plan
remains the authority for research framing (RQ1–RQ3), the generator lineup, datasets and
rigor rules; this document fixes what the plan leaves open and records the agreed
deviations.

## 1. Goal and success criteria

Benchmark tabular synthetic-data generators on fidelity, utility, privacy and cost, and
recommend one through a privacy-gated decision engine. Answers:

- **RQ1** — does the best generator change with dataset size, cardinality, imbalance, correlation?
- **RQ2** — how does differential privacy (PrivBayes ε ∈ {0.1, 1, 10}) shift the Pareto front?
- **RQ3** — can a meta-model predict the best generator from dataset meta-features alone?

Success = each phase gate from the source plan passes. The first, automated in CI:
train-copy scores worst on privacy; independent marginals score worst on column-pair trends.

## 2. Decisions

| Decision | Choice |
|---|---|
| Frontend | Next.js + TypeScript + Tailwind + shadcn/ui; Plotly (Pareto, distributions), Recharts (scorecards); TanStack Query polling |
| RQ3 | In scope, framed as exploratory |
| Data Designer track | Built; provider NVIDIA build.nvidia.com (OpenAI-compatible), key via `NVIDIA_API_KEY` |
| Safe Synthesizer | Not built; its offline results can be loaded with `import-results` |
| Compute | Owner provides CPU/GPU time later; all code paths verified locally with capped smoke runs |

### Agreed deviations from the source plan

1. **`packages/runner`** — a second shared package holding storage (local FS / S3-MinIO),
   MLflow logging, Celery task wiring and the pipeline. `packages/core` keeps only pandas,
   numpy, pydantic.
2. **synthcity-derived metrics are own code in the eval image** — detection AUC,
   α-precision/β-recall (kNN variant without the learned embedding), and a distance-based
   membership-inference attack. Keeps synthcity out of the eval image.
3. **Workers report status over HTTP** to `POST /internal/jobs/{id}`; only the API touches Postgres.
4. **Per-image Python pins** — synthcity 3.10, sdv 3.12, eval 3.11 (Anonymeter 1.0.0), datadesigner 3.12.
   Validated in the dependency spike; any change is recorded in the image's README.

## 3. Repository layout

```
Synode/  (synth-trust-lab)
├── apps/web/                 Next.js frontend
├── services/api/             FastAPI + SQLModel + Alembic
├── packages/core/            contract: metadata, split, generator interface, schemas, decision engine
├── packages/runner/          storage, mlflow, pipeline, celery app + task names
├── workers/
│   ├── sdv/                  Gaussian Copula, train-copy                     (q.sdv)
│   ├── synthcity/            marginals, BayesNet, CTGAN, TVAE, TabDDPM, GReaT, PrivBayes ε  (q.synthcity)
│   ├── eval/                 all metrics + pymfe meta-features + meta-predictor  (q.eval)
│   └── datadesigner/         schema-mode generation + rule validation        (q.datadesigner)
├── experiments/
│   ├── conf/                 Hydra: datasets/, generators/, sweep.yaml
│   ├── run.py                benchmark CLI (local backend)
│   └── meta/                 RQ3: openml pull, labelling, LODO evaluation
├── notebooks/                analysis, autorank CD diagrams, ε curves; kaggle_gpu.ipynb
├── docker-compose.yml
├── .github/workflows/ci.yml
└── report/
```

Each worker folder has its own `Dockerfile`, `pyproject.toml` and `uv.lock`, and installs
`packages/core` and `packages/runner` as path dependencies.

## 4. Core contract (`packages/core`)

- `TableMetadata`: `columns: dict[str, ColumnType]` (`numerical | categorical | datetime | id`),
  `target: str`, `task: classification | regression` (benchmark focuses on classification).
  `infer_metadata(df, target)` heuristic; user-editable in UI.
- `split(df, metadata, seed) -> Splits(train, control, test)` — stratified 60/20/20, disjoint,
  deterministic per seed; `Splits.fingerprint()` hash.
- `Generator` protocol: `name`, `fit(train_df, metadata)`, `sample(n)`; plus a registry
  `register_generator(name, factory, queue)` so the runner knows which queue owns which generator.
  Built-ins in core: `train_copy` (returns real TRAIN rows, resampled to n).
- Schemas (pydantic v2): `JobSpec`, `CostMetrics`, `FidelityMetrics`, `UtilityMetrics`,
  `PrivacyMetrics`, `MetricReport`, `AggregatedReport` (mean ± std over seeds),
  `Recommendation`, `ResultsBundle` (directory: `bundle.json` + per-job `report.json`).
- `decision`: normalisation, gate, Pareto front, weighted score, reason text (section 6).

## 5. Pipeline and data flow

Per run (dataset × generators × seeds):

1. **Ingest** — upload stored as `datasets/{id}/raw.parquet`; metadata inferred, confirmed in UI;
   stratified subsample to ≤ 20,000 rows.
2. **Split** — per seed, stored once at `datasets/{id}/splits/{seed}/{train,control,test}.parquet`.
3. **Generate** (generator's queue) — fit on TRAIN, sample `len(TRAIN)`; writes
   `runs/{run}/{job}/synthetic.parquet`; records fit/sample seconds and peak RSS. Per-generator
   epoch cap and time cap from Hydra config (equal budget rule).
4. **Evaluate** (`q.eval`) — computes all axes → `report.json`.
5. **Log** — one MLflow run per (generator, seed): params, metrics, artifacts. Worker POSTs status
   `queued → fitting → sampling → evaluating → done | failed` to the API.
6. **Decide** — when all jobs end, API aggregates over seeds and calls `core.decision`.

Backends: **local** (CLI; in-process, whatever generators are importable — the Kaggle/Colab
path) and **celery** (web app; `generate` task routed to the owning queue, chained to
`evaluate` on `q.eval`). Both call the same `runner.pipeline` functions.

`experiments/run.py import-results <bundle>` loads offline bundles into MLflow and, if
`--api` is given, into the app.

**Errors.** A generator failure or time-cap breach marks the job `failed` with traceback; the run
continues. A metric that errors is `null`; a null privacy metric fails the gate.

## 6. Metrics and decision engine

| Axis | Raw metrics |
|---|---|
| Fidelity | SDMetrics quality (shapes, pair trends); detection AUC (GBM classifier, 5-fold); α-precision, β-recall |
| Utility | TSTR ratio for LogReg, RandomForest, XGBoost (metric: macro-F1 for classification; synthetic-trained score ÷ real-trained score on TEST); Spearman of feature importances (RF) |
| Privacy | DCR share vs equal-size CONTROL; Anonymeter singling-out, linkability, inference (max over attribute targets); distance-based MIA advantage (AUC−0.5)·2 |
| Cost | fit s, sample s, peak MB |

Normalised 0–1, higher better:

- fidelity = mean(quality, 1−2·|detAUC−0.5|, α-precision, β-recall)
- utility = 0.5·mean(clip(TSTR ratios, 0, 1)) + 0.5·clip(importance ρ, 0, 1)
- privacy = mean(1 − max Anonymeter risk, 1 − 2·max(0, DCR − 0.5), 1 − MIA advantage)

Gate: reject if max Anonymeter risk > 0.10 (configurable) or DCR share > 0.6, or any privacy
metric is null. Pareto: non-dominated sort over (fidelity, utility, privacy), own code.
Presets: Balanced (⅓,⅓,⅓), Privacy-first (0.2, 0.2, 0.6), Utility-first (0.2, 0.6, 0.2), custom.
Reason: template naming the winner's strongest axis, the runner-up, and any higher-scoring
candidate that failed the gate with the failing check.

## 7. Generators

| Name | Queue | Library |
|---|---|---|
| train_copy | q.sdv (also local everywhere) | core |
| marginals | q.synthcity | synthcity `marginal_distributions` |
| gaussian_copula | q.sdv | SDV |
| bayesian_network | q.synthcity | synthcity |
| ctgan, tvae | q.synthcity | synthcity |
| tabddpm | q.synthcity (GPU offline recommended) | synthcity `ddpm` |
| great | q.synthcity (GPU offline recommended) | synthcity `great` |
| privbayes_eps{0.1,1,10} | q.synthcity | synthcity `privbayes` |

## 8. API (`services/api`)

FastAPI, SQLModel, Alembic, Postgres (SQLite in tests).
Tables: `Dataset`, `Run`, `Job`, `JobResult`. Endpoints:

- `POST /datasets` (upload), `GET /datasets`, `GET /datasets/{id}`, `PUT /datasets/{id}/metadata`
- `GET /generators` (registry: name, queue, cpu_ok)
- `POST /runs` (dataset, generators, seeds, preset/weights), `GET /runs`, `GET /runs/{id}` (status + per-job progress)
- `GET /runs/{id}/report?preset=…&weights=…` (aggregated + Pareto + recommendation; recomputed, so presets switch live)
- `GET /datasets/{id}/predict` (RQ3 meta-predictor hint)
- `POST /schema-runs`, `GET /schema-runs/{id}` (Data Designer track)
- `POST /internal/jobs/{id}` (worker status/result callback, shared-token auth)
- `POST /import-results` (bundle upload)

## 9. Web app (`apps/web`)

Pages: Datasets (upload, metadata editor), New run (generator picker with predicted-best hint,
seeds, preset), Run detail (live job progress, scorecard table mean ± std, 2D/3D Pareto plot,
preset switcher/custom weights, recommendation card with reason, gate failures), Schema mode
(define columns/samplers/rules, generate, rule-validity score).

## 10. RQ3 (`experiments/meta`)

Pull ~25 small OpenML-CC18 classification datasets; run fast generators (marginals, Copula,
BayesNet, CTGAN with a low epoch cap) × 3 seeds; label each dataset's winner by Balanced
score among gate-passing candidates; extract pymfe meta-features (general, statistical,
info-theory); train a small RandomForest; leave-one-dataset-out accuracy vs always-pick-the-
overall-winner baseline. Serialised model is served by `q.eval` for `/datasets/{id}/predict`.

## 11. Data Designer track

`workers/datadesigner`: accepts a schema (columns with sampler or LLM-text definitions, plus
rules expressed as pandas-evaluable expressions), generates N rows via NVIDIA endpoint,
scores **rule validity** = fraction of rows satisfying all rules plus per-rule rates. No key →
job fails with a clear message; no other part of the system depends on it.

## 12. Infrastructure

Docker Compose: postgres, redis, minio (+bucket init), mlflow, api, web, worker-sdv,
worker-synthcity, worker-eval, worker-datadesigner. `docker compose up` starts the demo.
`.env.example` documents all settings.

## 13. Testing and CI

- core unit tests: split disjointness/stratification/determinism, Pareto, gate, scoring, reason.
- eval integration: sanity gate on 2,000-row Adult sample (train-copy worst privacy; marginals worst pair trends).
- api tests: SQLite + mocked dispatch.
- web: typecheck + build.
- GitHub Actions: ruff + pytest per package environment; web build.

## 14. Out of scope / limitations

Full benchmark sweeps and GPU runs execute later on owner-provided compute; this build verifies
with capped smoke runs. Safe Synthesizer is import-only. No auth/multi-tenant in the app.
