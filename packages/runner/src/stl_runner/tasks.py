"""Celery task `stl.generate` (the pipeline's generate stage for one job)."""

from __future__ import annotations

import traceback

from stl_core.schemas import JobSpec

from stl_runner.celery_app import app
from stl_runner.pipeline import run_generate_with_cap, set_state
from stl_runner.storage import store_from_env


@app.task(name="stl.generate")
def generate(job_json: str, allow_gpu_only: bool = False) -> str:
    """Generate one job's synthetic table. Never raises, so the chain reaches evaluate."""
    store = store_from_env()
    job = JobSpec.model_validate_json(job_json)
    try:
        run_generate_with_cap(store, job, allow_gpu_only=allow_gpu_only)
    except Exception:
        set_state(store, job, "failed", traceback.format_exc())
    return job_json
