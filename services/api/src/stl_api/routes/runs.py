# ruff: noqa: B008  (FastAPI Depends defaults)
"""Run endpoints: create + dispatch, list/get with state refresh from the store, retry,
cancel and resume.

Cancel (POST /runs/{id}/cancel, 202) writes the run's cancel flag to the store and marks
every unfinished job cancelled (finished results are kept); it is idempotent (a repeat, or a
finished run, answers ``cancelled: 0``). Retry re-runs failed and cancelled jobs and clears
the flag. Resume (POST /runs/{id}/resume) re-enqueues jobs whose Celery messages were lost
(e.g. a broker restart): ``generated`` jobs get metrics -> evaluate, ``queued`` jobs the full
chain, and jobs stuck in fitting/sampling/evaluating for longer than STL_STALL_SECONDS are
treated as lost. Jobs whose messages are still in the broker are left alone.

Training budget: a run is planned under a budget profile (stl_core.budget) taken from the
body's ``budget_profile`` or else STL_BUDGET_PROFILE (default ``library``). The profile's
epoch count goes into the params of the epoch-taking generators (EPOCH_GENERATORS, as the
Hydra config wires ``${budget.epochs}``; ``library`` passes none, so library defaults apply;
explicit ``params`` win). The per-job time cap is the profile's, unless STL_TIME_CAP_SECONDS
is set: an explicit operator cap overrides every profile (it is a deployment safety bound).
"""
from __future__ import annotations

import json
import secrets
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator
from sqlmodel import Session, select
from stl_core.budget import BUDGET_PROFILES, check_profile
from stl_core.catalog import CATALOG
from stl_core.decision import resolve_weights
from stl_core.schemas import JobSpec
from stl_runner import keys
from stl_runner.control import (
    CANCEL_ERROR,
    CANCELLED,
    TERMINAL_STATES,
    clear_cancel,
    is_cancelled,
    mark_cancelled,
    request_cancel,
)
from stl_runner.pipeline import job_state, load_jobs, plan_run, set_state
from stl_runner.storage import ArtifactStore

from stl_api.deps import get_session, get_settings, get_store
from stl_api.dispatch import METRICS_QUEUE, Dispatcher
from stl_api.models import Dataset, Job, Run
from stl_api.settings import Settings

router = APIRouter(prefix="/runs", tags=["runs"])
TERMINAL = TERMINAL_STATES
RUNNING = ("fitting", "sampling", "evaluating")
# Generators whose epoch count comes from the budget profile; must match the generators
# experiments/conf/generators/default.yaml wires to ${budget.epochs} (tested).
EPOCH_GENERATORS = frozenset({
    "ctgan", "tvae", "sdv_ctgan", "sdv_tvae", "tabddpm", "great", "great_qwen",
})


def get_dispatcher(request: Request) -> Dispatcher:
    return request.app.state.dispatcher


class RunBody(BaseModel):
    dataset_id: str
    generators: list[str]
    seeds: list[Annotated[StrictInt, Field(ge=0)]] = [0, 1, 2]
    preset: str = "balanced"
    weights: list[float] | None = None
    allow_gpu_only: bool = False
    params: dict[str, dict] | None = None
    # stl_core.budget profile; None means the STL_BUDGET_PROFILE setting.
    budget_profile: str | None = None

    @field_validator("budget_profile")
    @classmethod
    def _budget_profile(cls, v: str | None) -> str | None:
        return None if v is None else check_profile(v)

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id(cls, v: str) -> str:
        return keys.validate_component(v, "dataset id")

    @field_validator("generators")
    @classmethod
    def _generators(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("generators must not be empty")
        if len(set(v)) != len(v):
            raise ValueError("generators must be unique")
        unknown = [g for g in v if g not in CATALOG]
        if unknown:
            raise ValueError(f"unknown generators {unknown}; known: {sorted(CATALOG)}")
        return v

    @field_validator("seeds")
    @classmethod
    def _seeds(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("seeds must not be empty")
        if len(set(v)) != len(v):
            raise ValueError("seeds must be unique")
        return v

    @model_validator(mode="after")
    def _params(self) -> RunBody:
        extra = sorted(set(self.params or {}) - set(self.generators))
        if extra:
            raise ValueError(f"params name generators not in this run: {extra}")
        return self


class RetryBody(BaseModel):
    allow_gpu_only: bool | None = None


def _spec(run: Run, row: Job) -> JobSpec:
    return JobSpec(job_id=row.job_id, run_id=run.id, dataset_id=run.dataset_id,
                   generator=row.generator, seed=row.seed)


def _set(session: Session, store: ArtifactStore, run: Run, row: Job, state: str,
         error: str | None = None) -> None:
    """Record a state in the store and the DB (caller commits)."""
    set_state(store, _spec(run, row), state, error)
    row.state, row.error = state, error
    row.updated_at = datetime.now(UTC)
    session.add(row)


def _refresh(session: Session, store: ArtifactStore, run: Run) -> tuple[list[Job], bool]:
    """Pull job states from the store. Returns (jobs, stale); stale if the store was unreachable."""
    rows = list(session.exec(select(Job).where(Job.run_id == run.id)))
    if all(r.state in TERMINAL for r in rows):
        return rows, False
    stale, changed = False, False
    try:
        cancelled = is_cancelled(store, run.id)
    except Exception:
        cancelled, stale = False, True
    for row in rows:
        if row.state in TERMINAL:
            continue
        try:
            st = job_state(store, _spec(run, row))
        except Exception:
            stale = True
            continue
        state, error = st.get("state", "queued"), st.get("error")
        if cancelled and state not in TERMINAL:  # a worker wrote after the cancel
            state, error = CANCELLED, CANCEL_ERROR
        if (state, error) != (row.state, row.error):
            row.state, row.error = state, error
            row.updated_at = datetime.now(UTC)
            session.add(row)
            changed = True
    if changed:
        session.commit()
    return rows, stale


def _summary(run: Run, jobs: list[Job], stale: bool) -> dict:
    return {"run_id": run.id, "dataset_id": run.dataset_id,
            "generators": json.loads(run.generators_json), "seeds": json.loads(run.seeds_json),
            "preset": run.preset, "budget_profile": run.budget_profile,
            "weights": json.loads(run.weights_json) if run.preset == "custom" else None,
            "counts": dict(Counter(j.state for j in jobs)),
            "done": bool(jobs) and all(j.state in TERMINAL for j in jobs), "stale": stale,
            "cancelled": any(j.state == CANCELLED for j in jobs)}


def _detail(run: Run, jobs: list[Job], stale: bool) -> dict:
    return {**_summary(run, jobs, stale),
            "jobs": [{"job_id": j.job_id, "generator": j.generator, "seed": j.seed,
                      "state": j.state, "error": j.error}
                     for j in sorted(jobs, key=lambda j: j.job_id)]}


def _get(session: Session, run_id: str) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id!r} not found")
    return run


def _dispatch(session: Session, store: ArtifactStore, dispatcher: Dispatcher, run: Run,
              pairs: list[tuple[JobSpec, Job]], allow_gpu_only: bool) -> int:
    """Enqueue each job; a failed enqueue marks only that job failed. Returns the failure count."""
    failures = 0
    for spec, row in pairs:
        try:
            dispatcher.enqueue(spec, allow_gpu_only)
        except Exception as e:
            failures += 1
            _set(session, store, run, row, "failed",
                 f"dispatch failed: {type(e).__name__}: {e}")
    session.commit()
    return failures


def _budget_params(generators: list[str], params: dict[str, dict] | None,
                   epochs: int | None) -> dict[str, dict]:
    """Per-generator params with the profile's epochs on the epoch-taking generators.
    ``epochs`` None (``library``) adds nothing; explicit params take precedence."""
    out = {g: dict(p) for g, p in (params or {}).items()}
    if epochs is not None:
        for g in generators:
            if g in EPOCH_GENERATORS:
                out[g] = {"epochs": epochs, **out.get(g, {})}
    return out


@router.post("", status_code=201, response_model=None)
def create_run(body: RunBody, session: Session = Depends(get_session),
               store: ArtifactStore = Depends(get_store),
               dispatcher: Dispatcher = Depends(get_dispatcher),
               settings: Settings = Depends(get_settings)) -> dict | JSONResponse:
    ds = session.get(Dataset, body.dataset_id)
    if ds is None or not ds.registered:
        raise HTTPException(409, f"dataset {body.dataset_id!r} is not registered")
    custom = tuple(body.weights) if body.weights is not None else None
    try:
        preset, weights = resolve_weights(body.preset, custom)
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e.args[0])) from e
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    try:
        run_id = keys.validate_component(f"{body.dataset_id}-{stamp}-{secrets.token_hex(2)}",
                                         "run id")
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    profile = body.budget_profile or settings.budget_profile
    budget = BUDGET_PROFILES[profile]
    cap = settings.time_cap_seconds or budget["time_cap_seconds"]
    try:
        jobs = plan_run(store, run_id, body.dataset_id, body.generators, body.seeds,
                        _budget_params(body.generators, body.params, budget["epochs"]),
                        time_cap_seconds=cap, budget_profile=profile)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    run = Run(id=run_id, dataset_id=body.dataset_id,
              generators_json=json.dumps(body.generators), seeds_json=json.dumps(body.seeds),
              preset=preset, weights_json=json.dumps(list(weights)),
              allow_gpu_only=body.allow_gpu_only, budget_profile=profile)
    session.add(run)
    session.flush()
    rows = [Job(id=f"{run_id}/{j.job_id}", run_id=run_id, job_id=j.job_id,
                generator=j.generator, seed=j.seed, state="queued") for j in jobs]
    session.add_all(rows)
    session.commit()
    failures = _dispatch(session, store, dispatcher, run, list(zip(jobs, rows, strict=True)),
                         body.allow_gpu_only)
    payload = {"run_id": run_id,
               "jobs": [{"job_id": r.job_id, "generator": r.generator, "seed": r.seed,
                         "state": r.state, "error": r.error} for r in rows]}
    if failures == len(jobs):
        return JSONResponse(status_code=503, content={
            "detail": f"run {run_id} created but no job could be dispatched: {rows[0].error}",
            **payload})
    return payload


@router.get("")
def list_runs(session: Session = Depends(get_session),
              store: ArtifactStore = Depends(get_store)) -> list[dict]:
    runs = session.exec(select(Run).order_by(Run.created_at.desc())).all()
    out = []
    for r in runs:
        jobs, stale = _refresh(session, store, r)
        out.append(_summary(r, jobs, stale))
    return out


@router.get("/{run_id}")
def get_run(run_id: str, session: Session = Depends(get_session),
            store: ArtifactStore = Depends(get_store)) -> dict:
    run = _get(session, run_id)
    jobs, stale = _refresh(session, store, run)
    return _detail(run, jobs, stale)


@router.post("/{run_id}/retry", response_model=None)
def retry_run(run_id: str, body: RetryBody | None = None,
              session: Session = Depends(get_session),
              store: ArtifactStore = Depends(get_store),
              dispatcher: Dispatcher = Depends(get_dispatcher)) -> dict | JSONResponse:
    run = _get(session, run_id)
    allow = run.allow_gpu_only
    if body is not None and body.allow_gpu_only is not None:
        allow = body.allow_gpu_only
    jobs, _ = _refresh(session, store, run)
    try:
        clear_cancel(store, run_id)  # a cancelled run can be retried
    except Exception as e:
        raise HTTPException(503, f"artifact store unavailable: {e}") from e
    failed = {j.job_id: j for j in jobs if j.state in ("failed", CANCELLED)}
    if not failed:
        return {"run_id": run_id, "retried": []}
    try:
        specs = [s for s in load_jobs(store, run_id) if s.job_id in failed]
    except Exception as e:
        raise HTTPException(409, f"run {run_id} has no readable manifest: {e}") from e
    pairs = [(s, failed[s.job_id]) for s in specs]
    try:
        for _, row in pairs:  # mark queued first so a repeated retry finds nothing failed
            _set(session, store, run, row, "queued")
    except Exception as e:
        session.rollback()
        raise HTTPException(503, f"artifact store unavailable: {e}") from e
    session.commit()
    failures = _dispatch(session, store, dispatcher, run, pairs, allow)
    retried = [s.job_id for s, _ in pairs]
    if failures == len(pairs):
        return JSONResponse(status_code=503, content={
            "detail": f"run {run_id}: no retried job could be dispatched: {pairs[0][1].error}",
            "run_id": run_id, "retried": retried})
    return {"run_id": run_id, "retried": retried}


@router.post("/{run_id}/cancel", status_code=202)
def cancel_run(run_id: str, session: Session = Depends(get_session),
               store: ArtifactStore = Depends(get_store)) -> dict:
    """Stop a run: queued jobs are skipped, running training is killed, finished results stay.
    Idempotent: answers ``cancelled: 0`` when nothing was left to cancel."""
    run = _get(session, run_id)
    rows, _ = _refresh(session, store, run)
    n = 0
    try:
        request_cancel(store, run_id)
        for row in rows:
            if row.state in TERMINAL:
                continue
            spec = _spec(run, row)
            mark_cancelled(store, spec)  # keeps a job that finished in the meantime
            st = job_state(store, spec)
            row.state, row.error = st.get("state", CANCELLED), st.get("error")
            row.updated_at = datetime.now(UTC)
            session.add(row)
            n += row.state == CANCELLED
    except Exception as e:
        session.rollback()
        raise HTTPException(503, f"artifact store unavailable: {e}") from e
    session.commit()
    return {"run_id": run_id, "cancelled": n}


def _stall_seconds(settings: Settings, spec: JobSpec) -> float:
    """STL_STALL_SECONDS (or explicit cap + 600), else the job's own time cap + 600 s."""
    if settings.stall_seconds is not None:
        return settings.stall_seconds
    cap = spec.time_cap_seconds or BUDGET_PROFILES["library"]["time_cap_seconds"]
    return cap + 600


def _in_flight(waiting: set[tuple[str, str]]) -> bool:
    """A job message is still in the broker. A stl.metrics message on another queue than
    METRICS_QUEUE (enqueued before it existed) waits behind training, so it does not count."""
    return any(task != "stl.metrics" or queue == METRICS_QUEUE for task, queue in waiting)


@router.post("/{run_id}/resume", response_model=None)
def resume_run(run_id: str, session: Session = Depends(get_session),
               store: ArtifactStore = Depends(get_store),
               dispatcher: Dispatcher = Depends(get_dispatcher),
               settings: Settings = Depends(get_settings)) -> dict | JSONResponse:
    """Re-enqueue jobs whose Celery messages were lost (see the module docstring)."""
    run = _get(session, run_id)
    rows, stale = _refresh(session, store, run)
    if stale:
        raise HTTPException(503, "artifact store unavailable: job states cannot be read")
    if is_cancelled(store, run_id):
        raise HTTPException(409, f"run {run_id} was cancelled; retry it to run the cancelled "
                                 "jobs again")
    open_ids = {r.job_id for r in rows if r.state not in TERMINAL}
    if not open_ids:
        return {"run_id": run_id, "resumed": []}
    try:
        specs = [s for s in load_jobs(store, run_id) if s.job_id in open_ids]
    except Exception as e:
        raise HTTPException(409, f"run {run_id} has no readable manifest: {e}") from e
    pending_jobs = getattr(dispatcher, "pending_jobs", None)
    try:
        pending = (pending_jobs() if pending_jobs else None) or {}
    except Exception:  # broker not inspectable: decide from the store states alone
        pending = {}
    now, plan = time.time(), []
    for spec in specs:
        if _in_flight(pending.get((run_id, spec.job_id), set())):
            continue
        state = job_state(store, spec)["state"]
        if state in RUNNING:
            age = now - store.modified(keys.status(run_id, spec.job_id))
            if age < _stall_seconds(settings, spec):
                continue
        if state in ("generated", "evaluating"):
            plan.append((spec, "eval"))
        elif state in ("queued", "fitting", "sampling"):
            plan.append((spec, "full"))
    resumed, errors = [], []
    for spec, kind in plan:
        try:
            if kind == "eval":
                dispatcher.enqueue_eval(spec)
            else:
                dispatcher.enqueue(spec, run.allow_gpu_only)
            resumed.append(spec.job_id)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    if plan and not resumed:
        return JSONResponse(status_code=503, content={
            "detail": f"run {run_id}: no stalled job could be dispatched: {errors[0]}",
            "run_id": run_id, "resumed": []})
    return {"run_id": run_id, "resumed": resumed}
