# Synthetic Data Trust Lab — API Service (`services/api`)

FastAPI backend orchestrating dataset registrations, run planning, Celery dispatch, artifact persistence, and report recommendations.

## Configuration & Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./outputs/api.db` | PostgreSQL or SQLite connection string. |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery message broker and result backend. |
| `STL_MAX_UPLOAD_MB` | `200` | Maximum dataset upload size in megabytes. Enforced on the streaming HTTP body by `UploadLimitMiddleware` (returns HTTP 413 if exceeded). |
| `STL_DISPATCH` | `celery` | Task dispatch mode: `celery` (submits jobs to worker queues) or `none` (offline/test mode). |
| `STL_BUDGET_PROFILE` | `library` | Default training budget profile for runs: `library`, `equal50`, or `smoke`. |
| `STL_TIME_CAP_SECONDS` | `null` | Global execution time cap per job in seconds. Overrides the budget profile's default cap if set. |
| `STL_STALL_SECONDS` | `null` | Stall timeout for detecting lost or unresponsive jobs during run resumption. Defaults to job time cap + 600 s. |
| `STL_CORS_ORIGINS` | `http://localhost:3010` | Comma-separated list of allowed CORS origins. |
| `STL_STORAGE_ROOT` | `outputs/store` | Local directory or S3 bucket (`s3://stl-artifacts`) for artifact storage. |

## Running the API Locally

```bash
# Run database migrations
bash scripts/envrun.sh api alembic -c services/api/alembic.ini upgrade head

# Start development server on port 8010
bash scripts/envrun.sh api uvicorn stl_api.main:app --host 0.0.0.0 --port 8010 --reload
```

Interactive API documentation is served at `http://localhost:8010/docs`.

