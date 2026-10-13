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
7. **Object storage: MinIO run natively on the host** (official MinIO container images are no longer publicly pullable); compose services reach it via host.docker.internal:9000; any S3-compatible endpoint works via S3_ENDPOINT_URL.
8. **RQ2 ε sweep uses smartnoise-synth MST; synthcity PrivBayes retained as a nominal-DP audit**
   (not ε-DP: synthcity 0.2.12 `_encode` never bins numeric columns, so their domain is the exact
   training values and synthetic numerics are copies of them; its Laplace noise is scaled for
   probabilities but added to counts, about n times too little; zero cells are not noised).
   MST (`smartnoise-synth==1.0.8`) runs in its own `smartnoise` worker env (Python 3.12, queue
   `q.smartnoise`): its dependencies conflict with the synthcity env's opacus pin. The catalog ε is
   the total budget; numeric bounds come from public codebooks (0 ε, see §7), so all of it fits MST.
9. **Library-default hyperparameters replace the equal-budget rule** (owner decision 2026-10-10): each generator
   trains with its upstream library's default hyperparameters (`budget_profile=library`, default time cap 86,400 s / 24 h);
   the equal-budget rule survives as an explicit profile `equal50` (50 epochs, 7,200 s cap for CPU comparability)
   and `smoke` (5 epochs, small rows). Pretrained base LLM weights are supported via library loaders (be-great
   `tabularisai/Qwen3-0.3B-distil`, Safe-Synthesizer `HuggingFaceTB/SmolLM3-3B`).

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

**Queues and run control.** The celery chain is generate (generator's queue) → `stl.metrics`
(synthcity metrics, own queue `q.metrics` served by a synthcity-image worker, so metrics never
wait behind hours of training on `q.synthcity`) → evaluate (`q.eval`). A run can be
**cancelled**: `runs/{run}/cancel.json` is the flag; every task checks it at start and marks its
job `cancelled` (terminal, like `done`/`failed`; finished results are kept), the generate stage
polls it every 2 s and kills the training child process, and metrics/evaluate drop their output
if it appears before they write. Retrying a run clears the flag and re-runs cancelled and failed
jobs. **Resume** re-enqueues jobs whose messages were lost (broker restart): `generated` →
metrics → evaluate, `queued` → full chain, fitting/sampling/evaluating only once their status is
older than `STL_STALL_SECONDS` (default time cap + 600 s); jobs whose messages are still in the
broker (queues or the redis `unacked` hash) are skipped.

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
| sdv_ctgan, sdv_tvae | q.sdv | SDV CTGAN / TVAE (shipped defaults: 300 epochs, batch 500) |
| tabddpm | q.synthcity (GPU offline recommended) | synthcity `ddpm` |
| great | q.synthcity (GPU offline recommended) | be-great `GReaT` directly (distilgpt2 base) |
| great_qwen | q.synthcity (GPU offline recommended) | be-great on pretrained `tabularisai/Qwen3-0.3B-distil` (Apache-2.0) |
| nss_smollm3 | q.safesynth (GPU offline) | NVIDIA Safe-Synthesizer QLoRA fine-tuning (`HuggingFaceTB/SmolLM3-3B`, Apache-2.0) |
| nss_smollm3_dp_eps8 | q.safesynth (GPU offline) | NVIDIA Safe-Synthesizer DP-SGD at ε=8.0 (`HuggingFaceTB/SmolLM3-3B`) |
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
- `POST /runs` (dataset, generators, seeds, preset/weights, optional `budget_profile` ∈ `library`/`equal50`/`smoke`, 422 otherwise; default setting `STL_BUDGET_PROFILE`, itself default `library`), `GET /runs`, `GET /runs/{id}` (status + per-job progress; both echo `budget_profile`, null for runs created before it). The profile's epochs go to the epoch-taking generators (none under `library`; explicit `params` win) and its time cap applies per job unless `STL_TIME_CAP_SECONDS` is set, which overrides every profile; the run manifest and `Run.budget_profile` record it
- `POST /runs/{id}/retry` (failed + cancelled jobs), `POST /runs/{id}/cancel` (202, idempotent: `{run_id, cancelled: n}`), `POST /runs/{id}/resume` (`{run_id, resumed: [job ids]}`; 409 for a cancelled run, 503 if nothing could be dispatched); run summaries carry `cancelled: bool`
- `GET /runs/{id}/report?preset=…&weights=…` (aggregated + Pareto + recommendation; recomputed, so presets switch live); each `per_job` entry carries `generator_params` (from report.json, else the job's `generator_params.json` written before fit, else null)
- `GET /datasets/{id}/suggestion` (RQ3 meta-predictor hint: the stored `suggestion.json`, or `{"available": false, "reason": …}`; 404 for an unknown dataset; classification datasets only). `POST /datasets/{id}/suggestion/refresh` re-enqueues it (202; 404 unknown, 409 not registered, 503 dispatch failure). Register/builtin enqueue `stl.metafeatures` on `q.meta`.
- `POST /schema-runs`, `GET /schema-runs/{id}` (Data Designer track)
- `POST /internal/jobs/{id}` (worker status/result callback, shared-token auth)
- `POST /import-results` (bundle upload)

## 9. Web app (`apps/web`)

Pages: Datasets (upload, metadata editor), New run (generator picker with predicted-best hint,
seeds, preset, training budget profile with a cost hint), Run detail (budget-profile chip; live job
progress with a per-job "Parameters" disclosure: library + version, profile, overrides highlighted,
full resolved params; scorecard table mean ± std, 2D/3D Pareto plot,
preset switcher/custom weights, recommendation card with reason, gate failures), Schema mode
(define columns/samplers/rules, generate, rule-validity score).

## 10. RQ3 (`workers/meta`)

RQ3 — *can a meta-model predict the best generator from a dataset's profile alone?* — is
**exploratory**: with ~25 datasets a meta-model that does not beat the majority baseline is a valid
result and is reported as such; nothing is tuned on the held-out dataset.

**Datasets.** OpenML-CC18 (suite 99) classification tasks with 500 ≤ instances ≤ 20,000, ≤ 60
features, ≤ 20 classes and every class ≥ 5 rows after a 60/20/20 split (checked conservatively as
minority class ≥ 25 rows); the first 25 by OpenML id are frozen in
`experiments/conf/meta/datasets.yaml` together with every excluded task and its reason (too many
features/classes/instances, or beyond the cap of 25). Datasets are registered as `cc18_<id>` with
`datasets=openml_generic` (no public bounds).

**Benchmark.** `scripts/run_meta_benchmark.sh`, one run `meta-cc18_<id>` per dataset: generators
`independent_marginals, gaussian_copula, bayesian_network, ctgan, tvae, mst_eps1` + `train_copy`
(sanity ceiling, never a candidate or label); budget profile `equal50` (50 epochs for the deep
generators, 7,200 s cap per job; the meta-benchmark keeps the equal budget independent of the main
benchmark's library-defaults mode), `max_rows=5000`, seeds 0,1,2, eval
`singling_out_mode=univariate`, `n_attacks=200`, `min_attacks=100`. Stages prepare → generate →
metrics → evaluate → report → metafeatures; a failed stage never stops the loop.

**Labels.** Per dataset `recommend(aggregate(reports), "balanced")` over the run's reports with
`train_copy` dropped: the gate-passing candidate with the highest balanced score, seeds aggregated.
No gate-passing candidate → label `none`: kept and reported, excluded from training and accuracy
(`n_none`). A dataset with missing/failed meta-features, no run, or an incomplete run (a planned job
without a report) is **excluded** with its reason and never labelled `none`. A generator whose
seeds all failed is not a candidate on that dataset.

**Meta-features.** `pymfe` 0.4.4 groups `general`, `statistical`, `info-theory`, summaries `mean`
and `sd`, on the registered (prepared, capped) seed-0 TRAIN split with the target; per-feature
failures → NaN, total failure → recorded error (dataset excluded).

**Evaluation.** Leave-one-dataset-out over the labelled datasets: for each held-out dataset the
imputation medians (non-finite → NaN → training-fold median), the meta-model
`RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")` and the baseline
("always predict the most frequent winner of the training folds", ties alphabetical) are fitted on
the other datasets only. Metrics for model and baseline: accuracy, balanced accuracy, mean regret and
`n_gate_failing`. **Regret** of predicting `g`: with P the generators that passed the gate with a
finite balanced score, `max(P) − score(g)` if `g ∈ P`, else `max(P) − min(P)` (a gate-failing or
unscored pick costs as much as the worst acceptable pick); because that penalty is 0 when |P| = 1,
`n_gate_failing` counts gate-failing picks separately. Fewer than 3 labelled datasets or a single
winner class → `insufficient_data` (no metrics, no predictor). `experiments/run.py meta-eval` writes
`meta/lodo.json`; `--train` also fits the final model on all labelled datasets →
`meta/predictor.joblib` (model, features, medians, labels, metadata) + `meta/predictor.json`.
`notebooks/rq3_meta.ipynb` reports the dataset table, metrics vs baseline, per-dataset predictions,
confusion matrix, feature importances and label distribution.

**Serving.** `worker-meta` (queue `q.meta`, task `stl.metafeatures`) computes meta-features of a
newly registered classification dataset (≤ 5,000-row sample; pymfe in a child process terminated
after 300 s, Celery hard limit 360 s) and, if a predictor exists, writes
`datasets/<id>/suggestion.json`; a non-classification dataset skips pymfe and gets
`{"available": false, "reason": "RQ3 predictor covers classification datasets only"}`.
`GET /datasets/{id}/suggestion` returns it or `{"available": false, "reason"}`; the web New-run
page shows it as an exploratory hint. Suggestions are computed at registration; after retraining,
refresh with `POST /datasets/{id}/suggestion/refresh` (or the page's "Refresh suggestion" link).

**Threats to validity.** (1) At most 25 datasets (fewer after `none`/exclusions): LODO estimates
have high variance (one dataset ≈ 4 accuracy points) and no significance claim is made. (2) MST
private-bounds fallback: without public bounds MST spends 0.2·ε on privately inferred bounds of
continuous columns, which fails on small numeric datasets (e.g. diabetes, 461 TRAIN rows); there
`mst_eps1` is not a candidate, so labels are conditional on it and biased against MST on small
numeric tables — the number of affected datasets is reported (`outputs/meta_logs/mst_bounds.txt`,
notebook). Data-derived "public" bounds were rejected because they would void the DP claim.
(3) DOMIAS is not applicable on all-categorical datasets (ids 3, 11, 46, 50, 469); the gate's
membership-inference check there uses the distance-based MIA only. (4) Labels depend on the fixed
CPU budget (`equal50`) and the `balanced` preset weights.


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
