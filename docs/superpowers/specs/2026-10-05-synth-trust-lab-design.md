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
- **RQ2** — how does differential privacy (smartnoise-synth MST, ε ∈ {0.1, 1, 10}) shift the Pareto front?
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

1. **`packages/runner`** — a second shared package holding storage (local FS / optional S3-compatible object storage),
   MLflow logging, Celery task wiring and the pipeline. `packages/core` keeps only pandas,
   numpy, pydantic.
2. **synthcity metrics run in the synthcity env, not the eval image** (integration ruling,
   2026-10-09; replaces the earlier own-code detection AUC and kNN precision/recall; the own
   nearest-neighbour distance MIA is kept as a second membership check, owner decision 2026-10-10).
   A `metrics` stage in the synthcity env computes synthcity's `SyntheticDetectionXGB`,
   `AlphaPrecision` and `DomiasMIAKDE` per generated job (`synthcity_metrics.json`); the eval
   image merges them, so synthcity stays out of the eval image. All library code (synthcity, SDV,
   SDMetrics, Anonymeter, smartnoise-synth) is installed editable from the pinned upstream clones
   in `third_party/`.
3. **Workers report status over HTTP** to `POST /internal/jobs/{id}`; only the API touches Postgres.
4. **Per-image Python pins** — synthcity 3.10, sdv 3.12, eval 3.11 (Anonymeter 1.0.0), datadesigner 3.12.
   Validated in the dependency spike; any change is recorded in the image's README.
5. **synthcity env uses Python 3.11** (3.10 resolves only synthcity 0.2.4).
6. **GReaT via be-great directly** (synthcity's `great` plugin is broken with be-great 0.0.14).
7. **Local platform storage uses a shared filesystem volume**. Optional S3-compatible storage remains in `packages/runner` for later deployment, but compose does not start MinIO/SeaweedFS by default.
8. **RQ2 ε sweep uses smartnoise-synth MST; synthcity PrivBayes retained as a nominal-DP audit**
   (not ε-DP: synthcity 0.2.12 `_encode` never bins numeric columns, so their domain is the exact
   training values and synthetic numerics are copies of them; its Laplace noise is scaled for
   probabilities but added to counts, about n times too little; zero cells are not noised).
   MST (`smartnoise-synth==1.0.8`) runs in its own `smartnoise` worker env (Python 3.12, queue
   `q.smartnoise`): its dependencies conflict with the synthcity env's opacus pin. The catalog ε is
   the total budget; numeric bounds come from public codebooks (0 ε, see §7), so all of it fits MST.

## 3. Repository layout

```
Synode/  (synth-trust-lab)
├── apps/web/                 Next.js frontend
├── services/api/             FastAPI + SQLModel + Alembic
├── packages/core/            contract: metadata, split, generator interface, schemas, decision engine
├── packages/runner/          storage, mlflow, pipeline, celery app + task names
├── workers/
│   ├── sdv/                  Gaussian Copula, train-copy                     (q.sdv)
│   ├── synthcity/            marginals, BayesNet, CTGAN, TVAE, TabDDPM, GReaT, PrivBayes audit (q.synthcity)
│   ├── smartnoise/           MST ε sweep (RQ2)                               (q.smartnoise)
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
| Fidelity | SDMetrics quality (shapes, pair trends); detection AUC = synthcity `SyntheticDetectionXGB` (XGBoost, 3-fold CV, mean AUC); α-precision, β-recall = synthcity `AlphaPrecision` OC variants (Alaa et al. 2022, one-class embedding — replaces the earlier kNN precision/recall proxy); synthcity authenticity reported (not scored) |
| Utility | TSTR ratio for LogReg, RandomForest, XGBoost (metric: macro-F1 for classification; synthetic-trained score ÷ real-trained score on TEST); Spearman of feature importances (RF) |
| Privacy | DCR share vs equal-size CONTROL; Anonymeter singling-out, linkability, inference (max over attribute targets); membership inference, two attacks, each advantage max(0, (AUC−0.5)·2) with members = a TRAIN sample and non-members from CONTROL: synthcity `DomiasMIAKDE` (primary; reference = a disjoint CONTROL half, synthetic density over the released table) and a nearest-neighbour distance attack (score = −distance to the nearest synthetic row; works on all-categorical data). The MIA advantage used by the privacy axis and the null check = max of the available two (null only if both are) |
| Cost | fit s, sample s, peak MB |

Normalised 0–1, higher better:

- fidelity = mean(quality, 1−2·max(0, detAUC−0.5), α-precision, β-recall) — following the
  SDMetrics convention, an AUC below 0.5 is not penalised: it typically arises when synthetic
  rows duplicate real ones (cross-validated classifiers then predict worse than chance), which
  is a privacy signal handled by the privacy axis and an evaluation warning (AUC < 0.4), not a
  fidelity defect.
- utility = 0.5·mean(clip(TSTR ratios, 0, 1)) + 0.5·clip(importance ρ, 0, 1)
- privacy = mean(1 − max Anonymeter risk, 1 − 2·max(0, DCR − 0.5), 1 − MIA advantage)

Gate: reject if the **lower confidence bound** of any Anonymeter risk > 0.10 (configurable),
DCR share > 0.6, or any privacy metric is null. The bounds are Bonferroni-corrected over the
m = 2 + (number of inference secrets) Anonymeter attacks at family α = 0.05 (Anonymeter's
two-sided interval at level 1 − 2α/m, i.e. a one-sided 1 − α/m lower bound); inference uses
the max over secrets of their lower bounds. A risk with fewer than `min_attacks` (default 100)
effective attacks is null (inconclusive), so the gate fails closed. The privacy axis score
still uses the point estimates.

Singling-out outcome rule (owner ruling, S2 follow-up 2). Anonymeter's query generator needs
synthetic rows with unique values or unique attribute combinations; on 20,000-row Adult its
multivariate mode generated only 19–41 queries for every generator, and univariate mode none
for discretised output (MST), so the fail-closed gate rejected everything. Singling-out
therefore runs the configured mode and, only if it is inconclusive (< `min_attacks` queries
after the fixed-point re-runs), the other mode; the risk and its lower bound come from the
conclusive mode, and the report records `singling_out_status` (`measured`), the mode used and
per-mode query counts. If **both** modes are inconclusive the status is `not_attackable` (risk
stays null; a crashed attack, or a query count that never stabilises, is `error` and fails
closed). The singling-out check then passes only when every successful seed is
`not_attackable` **and** all safeguards hold on the worst seed: the linkability and inference
lower bounds are conclusive in every seed and ≤ 0.10; DCR share ≤ 0.6; the per-seed exact-match
excess (share of synthetic rows equal to a row of a TRAIN sample minus the share for an
equal-size CONTROL sample), scaled by N_train / n_sample to undo the dilution of a TRAIN
sample (integration ruling: copying 3% of TRAIN must fail), is ≤ 0.01; and the mean, over 50 seeded random 3-column subsets of
the feature columns (all subsets when there are fewer), of the share of synthetic rows unique
on that subset is ≤ 0.05. Otherwise it fails ("singling-out not attackable but safeguards failed (…)"); mixed
seeds (some measured, some not) fall back to the ordinary fail-closed rule. A
not-attackable singling-out risk is never scored as 0: the privacy axis then uses
max(linkability, inference) as the Anonymeter term, and the candidate carries the note
"singling-out not attackable (safeguards passed)", repeated in the recommendation reason when
it is the winner. Cost if wrong: a low-diversity generator whose singling-out risk is
unmeasurable may pass; the safeguards bound that.

Pareto: non-dominated sort over (fidelity, utility, privacy), own code.
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
| great | q.synthcity (GPU offline recommended) | be-great `GReaT` directly (synthcity `great` plugin is broken) |
| mst_eps{0.1,1,10} | q.smartnoise | smartnoise-synth MST (ε-DP; RQ2 sweep) |
| privbayes_eps{0.1,1,10} | q.synthcity | synthcity `privbayes` — nominal-DP audit only (not ε-DP, see §2 deviation 8) |

**DP generators use public codebook bounds (0 ε), never data-derived ones.** Each dataset YAML
(`experiments/conf/datasets/*.yaml`) lists `public_bounds` ({column: [lower, upper]}) for every
numerical column, taken from the UCI pages / codebooks or, where none is documented, a generous
domain assumption marked `# assumption:` with its source URL; `public_log_scale` lists
nonnegative heavy-tailed columns binned on the log1p scale over the same bounds. `prepare`
validates them (numerical columns only, exit 2 otherwise) and `register_dataset` stores them as
`datasets/<id>/public.json`; `generate_job` passes them into the params of
`differential_privacy`-family generators only (MST), which bins each column with a
fixed-bounds smartnoise `BinTransformer` (`preprocessor_eps = 0`; values outside the bounds are
clipped into the edge bins) so the whole catalog ε goes to MST. A continuous column without a
public bound falls back to private bound inference costing 20% of ε. The job's
`runs/<run>/<job>/privacy.json` records the ε split and which path (`public`/`private`/`mixed`)
was used. Caveats (not DP-accounted): the categorical domains (sets of categories) and the row
count are taken as public, and `prepare()`'s median imputation of missing numerics uses the data.

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

## 10. RQ3 (`workers/meta`)

Pull 25 small OpenML-CC18 classification datasets (`experiments/conf/meta/datasets.yaml`);
run fast generators (`independent_marginals`, `gaussian_copula`, `bayesian_network`, `ctgan`,
`tvae`, `mst_eps1` + `train_copy` sanity ceiling) × 3 seeds with budget cap (50 epochs, 5,000 rows);
label each dataset's winner by Balanced score among gate-passing candidates (datasets with no
gate-passing candidate labelled `none` and reported, excluded from accuracy); extract `pymfe`
meta-features (groups `general`, `statistical`, `info-theory`, summary `mean` and `sd`) on the
seed-0 TRAIN split; train a `RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")`;
evaluate via leave-one-dataset-out (LODO) cross-validation with train-fold imputation and baseline
"always predict majority winner in training folds", recording accuracy, balanced accuracy, and mean regret.
Exploratory framing: reported honestly even if sample size yields modest raw accuracy vs baseline.
Serialised predictor (`meta/predictor.joblib` and `meta/predictor.json`) is served by `worker-meta`
on queue `q.meta` (`stl.metafeatures` Celery task) for `GET /datasets/{id}/suggestion` and the web UI.


## 11. Data Designer track

`workers/datadesigner`: accepts a schema (columns with sampler or LLM-text definitions, plus
rules expressed as pandas-evaluable expressions), generates N rows via NVIDIA endpoint,
scores **rule validity** = fraction of rows satisfying all rules plus per-rule rates. No key →
job fails with a clear message; no other part of the system depends on it.

## 12. Infrastructure

Docker Compose: postgres, redis, mlflow, a shared artifact volume, api, web, worker-sdv,
worker-synthcity, worker-smartnoise, worker-eval and worker-meta. `docker compose up` starts
the demo; S6 adds `worker-datadesigner`.
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
