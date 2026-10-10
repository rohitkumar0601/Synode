"""Celery task `stl.evaluate` (evaluation of one job)."""

from __future__ import annotations

import os

from stl_core.schemas import JobSpec
from stl_runner import keys
from stl_runner.celery_app import app
from stl_runner.pipeline import job_state
from stl_runner.storage import store_from_env
from stl_runner.tracking import log_report

from stl_eval.evaluate import EvalConfig, evaluate_stored_job, fail_job


def _secrets(raw: str | None, default):
    if not raw:
        return default
    names = [c.strip() for c in raw.split(",") if c.strip()]
    return names or default


def eval_config_from_env() -> EvalConfig:
    env, cfg = os.environ, EvalConfig()
    secrets = env.get("STL_EVAL_INFERENCE_SECRETS")
    return EvalConfig(
        n_attacks=int(env.get("STL_EVAL_N_ATTACKS", cfg.n_attacks)),
        singling_out_mode=env.get("STL_EVAL_SINGLING_OUT_MODE", cfg.singling_out_mode),
        min_attacks=int(env.get("STL_EVAL_MIN_ATTACKS", cfg.min_attacks)),
        inference_secrets=_secrets(secrets, cfg.inference_secrets),
    )


@app.task(name="stl.evaluate")
def evaluate(job_json: str) -> str:
    """Evaluate a job (idempotent), store and log its report; returns the report status."""
    store = store_from_env()
    job = JobSpec.model_validate_json(job_json)
    if job_state(store, job)["state"] == "done" and store.exists(keys.report(job.run_id,
                                                                            job.job_id)):
        return evaluate_stored_job(store, job).status  # idempotent redelivery
    try:
        config = eval_config_from_env()
    except (ValueError, TypeError) as exc:
        report = fail_job(store, job, f"invalid eval config: {exc}")
    else:
        report = evaluate_stored_job(store, job, config)
    log_report(report, job)
    return report.status
