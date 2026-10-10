"""Celery task `stl.metrics` (synthcity metrics for one generated job)."""

from __future__ import annotations

import logging

from stl_core.schemas import JobSpec
from stl_runner.celery_app import app
from stl_runner.control import cancel_if_requested
from stl_runner.pipeline import job_state
from stl_runner.storage import store_from_env

from stl_synthcity.metrics import compute_stored_job


@app.task(name="stl.metrics")
def metrics(job_json: str) -> str:
    """Write synthcity_metrics.json for a generated job; a no-op otherwise. Never raises."""
    try:
        store = store_from_env()
        job = JobSpec.model_validate_json(job_json)
        if cancel_if_requested(store, job):
            return job_json
        if job_state(store, job)["state"] == "generated":
            compute_stored_job(store, job)
    except Exception:  # evaluate reports a missing metrics file; the chain must go on
        logging.getLogger(__name__).exception("stl.metrics failed")
    return job_json
