# S3 — Platform: FastAPI, Celery Queue Routing, Worker Images, Docker Compose — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the S1/S2 engine into a running backend: a typed FastAPI service (datasets, generators, runs, live job status, report with presets, bundle import) that dispatches each job to its generator's Celery queue and then to the eval queue, with one Docker image per worker, all started by `docker compose up`.

**Architecture:** The API (Python 3.12, `services/api`) owns Postgres via SQLModel/Alembic and never imports generator or metric libraries. It plans runs with `stl_runner.pipeline.plan_run` and enqueues, per job, a Celery chain by task **name**: `stl.generate` on the generator's queue (`CATALOG[g].queue`) → `stl.metrics` on `q.synthcity` (synthcity's AlphaPrecision / detection / DOMIAS, added in the integration step) → `stl.evaluate` on `q.eval`. Workers are thin: each image runs `celery -A stl_runner.celery_app worker -Q <queue>`; `stl.generate` (in `stl_runner.tasks`) wraps `run_generate_with_cap`, `stl.evaluate` (in `stl_eval.tasks`) wraps `evaluate_stored_job` + `log_report`. Job state's single source of truth is the artifact store's `status.json` (already written by the pipeline); the API reads it when polled and mirrors it into Postgres.

**Tech Stack:** FastAPI 0.142, SQLModel 0.0.48 (SQLAlchemy 2.1), Alembic 1.20, psycopg 3, Celery 5.6.3 + Redis, uvicorn, httpx (tests), Docker Compose, uv (one lockfile per image). Verified to co-resolve on 2026-10-07 (API on 3.12; Celery 5.6.3 also resolves in the synthcity and eval envs).

**Spec:** `docs/superpowers/specs/2026-10-05-synth-trust-lab-design.md` §8 (API), §12 (infra). S1/S2 code is the base.

## Global Constraints

- **No git commits or pushes.** Checkpoint = tests + lint + `git status --short`.
- Local Python via `bash scripts/envrun.sh <env> …` (envs in `$HOME/.stl/<env>`; Windows MAX_PATH). New env name: `api` (already mapped to `services/api` in `envrun.sh`).
- Inside Docker images, envs live at `/opt/venv` (`UV_PROJECT_ENVIRONMENT=/opt/venv`), built with `uv sync --frozen --no-dev --project <path>`; build context is the repo root.
- The API never imports `stl_eval`, `stl_sdv`, `stl_synthcity`, `stl_smartnoise`; it dispatches by task name.
- Task names: `stl.generate` (args: `job_json: str`, kwargs: `allow_gpu_only: bool = False`; returns `job_json`), `stl.metrics` (args: `job_json: str`; computes `synthcity_metrics.json` via `stl_synthcity.metrics`; never raises; returns `job_json`), `stl.evaluate` (args: `job_json: str`; returns report status). Queues: `q.sdv`, `q.synthcity`, `q.smartnoise`, `q.eval` (and `q.datadesigner` reserved for S6).
- `stl.generate` must never raise for generator failures (state is recorded by the pipeline) so the chain always reaches `stl.evaluate`, which writes a failed report for failed jobs.
- Host ports (taken on the dev machine: 3000, 5432, 8000): API **8010**, MLflow **5000**, Postgres **55432**, Redis **6379**. Local artifact storage is a shared compose volume, not a local S3-compatible server.
- Ruling (recorded in S3 ledger): **no worker→API HTTP callback** (spec §2 deviation 3 is superseded): status lives in the store and the API reads it; only the API writes Postgres. Spec §2/§8 updated in Task 7.
- Celery workers: `--pool=prefork --concurrency=1` (the eval worker's singling-out query counter swaps a module global — not thread-safe), `task_acks_late=True`, `worker_prefetch_multiplier=1`.
- Evaluation is idempotent (`evaluate_stored_job` returns the stored report for "done" jobs) — safe under Celery redelivery.

## Review Focus

1. **Uploading a CSV whose target has a class with < 5 rows, or an unknown target column** — the API returns 422 with the split/metadata error text, not 500. (Task 3.)
2. **Creating a run with a generator whose queue has no worker running** — the run is created, jobs stay `queued`, and `GET /runs/{id}` shows them as queued (no crash, no false "failed"). (Task 4.)
3. **Polling a run while a worker is mid-job** — partial reports must not break `/report` (it reports only finished jobs plus `jobs_reported/jobs_total`). (Task 5.)
4. **Uploading a malformed or traversal bundle** — 422 with the validation message; nothing written. (Task 5.)
5. **Large uploads** — the API rejects files above `STL_MAX_UPLOAD_MB` (default 200) with 413 before parsing. (Task 3.)

---

### Task 1: Celery app and the two task contracts

**Files:**
- Create: `packages/runner/src/stl_runner/celery_app.py`, `packages/runner/src/stl_runner/tasks.py`, `workers/eval/src/stl_eval/tasks.py`, `packages/runner/tests/test_tasks.py`, `workers/eval/tests/test_tasks.py`
- Modify: `packages/runner/pyproject.toml` (optional extra `celery = ["celery[redis]==5.6.3"]`), `workers/{eval,sdv,synthcity,smartnoise}/pyproject.toml` (depend on `stl-runner[s3,celery]`)

**Interfaces:**
- `stl_runner.celery_app.app: Celery` — broker/backend from `REDIS_URL` (default `redis://127.0.0.1:6379/0`); `include` from env `STL_CELERY_INCLUDE` (comma-separated module names, default `"stl_runner.tasks"`); conf: `task_acks_late=True`, `worker_prefetch_multiplier=1`, `task_serializer="json"`, `result_serializer="json"`, `accept_content=["json"]`, `task_default_queue="q.default"`, `broker_connection_retry_on_startup=True`. `make_app(**overrides) -> Celery` for tests (e.g. `task_always_eager=True`, memory broker).
- `stl_runner.tasks.generate` registered as `stl.generate`: `(job_json: str, allow_gpu_only: bool = False) -> str` — builds the store with `store_from_env()`, `JobSpec.model_validate_json`, calls `run_generate_with_cap(store, job, allow_gpu_only=...)`, returns `job_json`. Exceptions inside are already recorded as `failed` by the pipeline; any unexpected exception is caught, recorded with `set_state(..., "failed", ...)`, and the task still returns `job_json`.
- `stl_eval.tasks.evaluate` registered as `stl.evaluate`: `(job_json: str) -> str` — `EvalConfig` from env (`STL_EVAL_N_ATTACKS`, `STL_EVAL_SINGLING_OUT_MODE`, `STL_EVAL_MIN_ATTACKS`, `STL_EVAL_INFERENCE_SECRETS`; defaults = EvalConfig defaults), `evaluate_stored_job`, `log_report`; returns `report.status`.
- `stl_synthcity.tasks.metrics` registered as `stl.metrics` (synthcity worker `STL_CELERY_INCLUDE=stl_runner.tasks,stl_synthcity.tasks`).
- `stl_runner.tasks.job_chain(job: JobSpec, allow_gpu_only: bool = False)` → a Celery `chain(signature("stl.generate", …, queue=CATALOG[job.generator].queue, immutable=True), signature("stl.metrics", args=[job_json], queue="q.synthcity", immutable=True), signature("stl.evaluate", args=[job_json], queue="q.eval", immutable=True))` built **by name only** (no worker imports) — used by the API.

- [ ] **Step 1: Failing tests.** `packages/runner/tests/test_tasks.py`: (a) `job_chain` builds two signatures with the right task names and queues for `gaussian_copula` (`q.sdv`) and `mst_eps1` (`q.smartnoise`); (b) with `make_app(task_always_eager=True)` and a tmp store (`STL_STORAGE_ROOT` monkeypatched), calling `generate.apply(args=[job_json]).get()` on a planned `train_copy` job writes synthetic and state `generated`; a `ctgan` job in the runner env returns job_json and state `failed` (no exception escapes). `workers/eval/tests/test_tasks.py`: eager `evaluate.apply` on a generated `independent_marginals` toy job → state `done`, report exists; on a failed job → failed report; invalid `STL_EVAL_INFERENCE_SECRETS=nope` → failed report "invalid eval config".
- [ ] **Step 2: RED**, **Step 3: implement**, **Step 4: GREEN** (runner env, eval env; eval tests set `min_attacks` low via env for the toy).
- [ ] **Step 5: Checkpoint.**

---

### Task 2: API skeleton — settings, database models, Alembic, health

**Files:**
- Create: `services/api/pyproject.toml`, `services/api/src/stl_api/{__init__.py,settings.py,db.py,models.py,main.py}`, `services/api/alembic.ini`, `services/api/migrations/{env.py,script.py.mako,versions/0001_initial.py}`, `services/api/tests/{conftest.py,test_health.py,test_models.py}`

**Interfaces:**
- `pyproject.toml`: Python `>=3.12,<3.13`; deps `stl-core`, `stl-runner[s3,celery]`, `fastapi==0.142.*`, `uvicorn[standard]`, `sqlmodel==0.0.48`, `alembic`, `psycopg[binary]`, `python-multipart`; dev: pytest, ruff, httpx. uv sources for core/runner by path.
- `Settings` (pydantic-settings-free: a dataclass reading env): `DATABASE_URL` (default `sqlite:///./outputs/api.db`), `STL_STORAGE_ROOT`, `REDIS_URL`, `STL_MAX_UPLOAD_MB` (200), `STL_CORS_ORIGINS` (default `http://localhost:3010`), `STL_DISPATCH` (`celery` | `none`; tests use `none`).
- Models (SQLModel, `table=True`): `Dataset(id: str PK, name: str, source: str ["upload"|"builtin"], target: str, n_rows: int | None, metadata_json: str, fingerprint: str | None, registered: bool, created_at: datetime)`; `Run(id: str PK, dataset_id: FK, generators_json: str, seeds_json: str, preset: str, weights_json: str | None, created_at)`; `Job(id: str PK = f"{run_id}/{job_id}", run_id: FK, job_id: str, generator: str, seed: int, state: str, error: str | None, updated_at)`.
- `main.app` (FastAPI, title "Synthetic Data Trust Lab API", version "0.1.0"): CORS from settings; `GET /health` → `{"status":"ok","db":true|false,"store":true|false}`; startup creates tables when `DATABASE_URL` is SQLite (tests/dev); Postgres uses Alembic (`alembic upgrade head` in the container entrypoint).
- Alembic `0001_initial` creates the three tables (hand-written, matching models).

- [ ] Steps: failing tests (`TestClient` health returns ok with SQLite tmp DB and tmp store; models round-trip) → implement → GREEN in `api` env; also run `bash scripts/envrun.sh api alembic -c services/api/alembic.ini upgrade head` against `DATABASE_URL=postgresql+psycopg://stl:stl@127.0.0.1:55432/stl` (compose Postgres is running) and paste the output. Checkpoint.

---

### Task 3: Dataset endpoints

**Files:** Create `services/api/src/stl_api/routes/datasets.py`, `services/api/src/stl_api/deps.py` (store/session/settings dependencies), `services/api/tests/test_datasets.py`; modify `main.py` (include router).

**Interfaces:**
- `POST /datasets` (multipart: `file` CSV or Parquet, `name`, `target`) → 201 `{id, name, target, n_rows, metadata, registered: false}`. Rejects > `STL_MAX_UPLOAD_MB` with 413 (check `Content-Length` and streamed size). Reads via pandas, converts `category` → object, infers metadata (`infer_metadata`), writes upload to the store at `datasets/<id>/upload.parquet`, stores metadata JSON in Postgres. Unknown target → 422 with the error text. `id` = slug of name + 6-char random suffix, validated by `keys.validate_component`.
- `PUT /datasets/{id}/metadata` (JSON `{columns: {col: type}, target}`) → validates via `TableMetadata` (422 on error; unknown columns → 422); updates row; sets `registered=false`.
- `POST /datasets/{id}/register` (JSON `{max_rows: int = 20000, public_bounds?: {...}, public_log_scale?: [...]}`) → reads upload, calls `register_dataset` (same signature as the CLI uses after the S2 follow-up — read it), runs `ensure_splits` for seed 0 to surface split errors early (rare class → 422 with the split message), stores `fingerprint`, `n_rows`, `registered=true`.
- `POST /datasets/builtin/{name}` (`adult|credit|diabetes`) → loads the Hydra dataset YAML via `experiments/conf/datasets/<name>.yaml` (read the YAML file directly with `yaml.safe_load`; do not import Hydra in the API — add `pyyaml` dep) and `stl_runner.datasets.load_dataset`, then behaves like upload+register with the YAML's `drop`/`column_types`/`public_bounds`.
- `GET /datasets`, `GET /datasets/{id}` (includes metadata and a 5-row preview from the store), `DELETE /datasets/{id}` (409 if runs reference it).

- [ ] Steps: tests first (upload small CSV → 201; bad target → 422; rare class at register → 422 message names the class; oversize → 413; metadata edit; builtin adult with `max_rows=500` marked `network`) → implement → GREEN → checkpoint.

---

### Task 4: Generators and runs endpoints with dispatch

**Files:** Create `services/api/src/stl_api/routes/{generators.py,runs.py}`, `services/api/src/stl_api/dispatch.py`, `services/api/tests/test_runs.py`.

**Interfaces:**
- `GET /generators` → list of `{name, family, queue, cpu_ok, params}` from `CATALOG` (family `nominal_dp_audit` included but flagged `"audit": true`).
- `dispatch.Dispatcher` protocol: `enqueue(job: JobSpec, allow_gpu_only: bool) -> None`. `CeleryDispatcher` uses `job_chain(...).apply_async()`. `NullDispatcher` (STL_DISPATCH=none) records calls in memory for tests.
- `POST /runs` JSON `{dataset_id, generators: [..], seeds: [0,1,2], preset: "balanced", weights?: [f,u,p], allow_gpu_only: false, params?: {gen: {...}}}` → validates dataset registered (409 otherwise), generators in CATALOG (422), weights via `resolve_weights` (422 on error); `run_id` = `f"{dataset_id}-{yyyymmddHHMMSS}-{4 hex}"`; `plan_run(store, run_id, dataset_id, generators, seeds, params, time_cap_seconds=STL_TIME_CAP_SECONDS (default 7200))`; inserts Run + Jobs (state queued); enqueues every job; 201 `{run_id, jobs: [...]}`.
- `GET /runs` (newest first, with counts by state), `GET /runs/{id}` → `{run_id, dataset_id, generators, seeds, preset, jobs: [{job_id, generator, seed, state, error}], counts: {state: n}, done: bool}`; each GET refreshes job states from the store (`job_state`) and updates Postgres rows whose state changed.
- `POST /runs/{id}/retry` → re-enqueues jobs in `failed` state (generation failures only; reset is handled by `run_generate_with_cap`).

- [ ] Steps: tests with `NullDispatcher` (create run → jobs queued and dispatcher saw N enqueues with the right queues via `CATALOG`; unknown generator 422; unregistered dataset 409; status refresh: write `status.json` for one job to `generated` directly in the store → GET shows `generated`); implement; GREEN; checkpoint.

---

### Task 5: Report and bundle-import endpoints

**Files:** Create `services/api/src/stl_api/routes/reports.py`, `services/api/tests/test_reports.py`.

**Interfaces:**
- `GET /runs/{id}/report?preset=balanced&weights=f,u,p` → `{run_id, jobs_reported, jobs_total, recommendation: Recommendation, aggregated: [AggregatedReport], pareto: [{generator, fidelity, utility, privacy, on_front, gate_passed}], per_job: [{job_id, generator, seed, status, warnings}]}` built from stored `report.json` files with `aggregate` + `recommend`; 404 if no reports yet is NOT used — return `jobs_reported: 0` and `recommendation: null`. Bad weights → 422.
- `POST /import-results` (multipart `file` zip, `force: bool=false`) → saves to a temp file, `import_bundle(store, path, force)`; 422 with the ValueError message on validation errors (nothing written — already guaranteed by import_bundle); inserts/updates Run + Job rows from the manifest; returns `{run_id, imported: n, warnings: [...]}`.

- [ ] Steps: tests (report on a run with two finished jobs written directly via `evaluate`-free fixtures: write `MetricReport` JSON for train_copy & independent_marginals into the store → train_copy fails gate, winner independent_marginals; partial run → jobs_reported < jobs_total; bad weights 422; import a bundle exported by `export_bundle` in-test → 200 and run visible in `GET /runs`; traversal bundle → 422) → implement → GREEN → checkpoint.

---

### Task 6: Docker images, Compose wiring, end-to-end test

**Files:** Create `docker/api.Dockerfile`, `docker/worker.Dockerfile` (one parametrised Dockerfile, `ARG WORKER=sdv|synthcity|smartnoise|eval`, `ARG PYTHON=3.11|3.12`), `.dockerignore`, `scripts/e2e_platform.sh`; modify `docker-compose.yml`, `.env.example`.

**Interfaces:**
- Images: base `ghcr.io/astral-sh/uv:python<PY>-bookworm-slim`; copy `packages/`, the worker/service dir, `experiments/conf` (API only); `uv sync --frozen --no-dev --project <dir>` into `/opt/venv`; synthcity image uses the CPU torch index already in its pyproject. Non-root user.
- Compose services (in addition to postgres/redis and the shared `storedata` artifact volume): `mlflow` (image `ghcr.io/mlflow/mlflow:v3.x` or `python:3.12-slim` + `pip install mlflow`; backend `sqlite:////mlflow/mlflow.db` on a volume; proxied artifacts at `/mlflow/artifacts`; port 5000); `api` (port `8010:8000`, entrypoint runs `alembic upgrade head` then `uvicorn stl_api.main:app --host 0.0.0.0 --port 8000`); `worker-sdv`, `worker-synthcity`, `worker-smartnoise`, `worker-eval` (command `celery -A stl_runner.celery_app worker -Q <queue> --pool=prefork --concurrency=1 -n <name>@%h`; eval worker sets `STL_CELERY_INCLUDE=stl_runner.tasks,stl_eval.tasks`). API and workers share env: `STL_STORAGE_ROOT=/stl-store`, `REDIS_URL=redis://redis:6379/1`, `MLFLOW_TRACKING_URI=http://mlflow:5000`, `DATABASE_URL=postgresql+psycopg://stl:stl@postgres:5432/stl` (api only). `depends_on` with health conditions.
- `scripts/e2e_platform.sh`: `docker compose up -d --build --wait` → upload a 1,000-row toy CSV (generated by a small inline Python in the `core` env) via `curl` to `http://127.0.0.1:8010/datasets` → register → `POST /runs` with `[train_copy, independent_marginals, gaussian_copula, ctgan, mst_eps10]`, seeds `[0]`, `params: {ctgan: {epochs: 2}}` → poll `GET /runs/{id}` until done (timeout 30 min) → `GET /runs/{id}/report` → assert with `python -c` that train_copy failed the gate and `jobs_reported == jobs_total`; print the ranking. Uses lowered eval settings via worker env for speed (`STL_EVAL_N_ATTACKS=200`, `STL_EVAL_MIN_ATTACKS=100`, univariate).

- [ ] Steps: build all images (`docker compose build`; paste image sizes); run `scripts/e2e_platform.sh`; paste output; confirm MLflow UI at http://127.0.0.1:5000 lists the runs. If an image fails to build, fix the Dockerfile — do not change library pins without recording why. Checkpoint.

---

### Task 7: Docs, CI, spec

**Files:** Modify `README.md` (Platform section: `docker compose up -d --build`, URLs, API docs at `http://127.0.0.1:8010/docs`, e2e script), `.github/workflows/ci.yml` (add `env: api, tests: services/api/tests` with `-m "not network"`), `docs/superpowers/specs/2026-10-05-synth-trust-lab-design.md` (§2: callback deviation superseded by store-as-source-of-truth; §8 endpoints as built; §12 shared artifact volume + ports), `.env.example`.

- [ ] Steps: update; run every suite separately + ruff; paste lines. Checkpoint.

## Phase gate (S3 done when)
- All suites (incl. `services/api/tests`) pass; ruff clean.
- `docker compose up -d --build --wait` brings up postgres, redis, mlflow, api, shared artifact storage and 4 workers healthy.
- `scripts/e2e_platform.sh` completes: a run created over HTTP is routed to sdv/synthcity/smartnoise/eval queues, finishes, and `/report` returns a recommendation where train_copy fails the gate.
