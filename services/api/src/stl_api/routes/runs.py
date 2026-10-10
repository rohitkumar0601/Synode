# ruff: noqa: B008  (FastAPI Depends defaults)
"""Run endpoints: create + dispatch, list/get with state refresh from the store, retry."""
from __future__ import annotations

import json
import secrets
from collections import Counter
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator
from sqlmodel import Session, select
from stl_core.catalog import CATALOG
from stl_core.decision import resolve_weights
from stl_core.schemas import JobSpec
from stl_runner import keys
from stl_runner.pipeline import job_state, load_jobs, plan_run, set_state
from stl_runner.storage import ArtifactStore

from stl_api.deps import get_session, get_settings, get_store
from stl_api.dispatch import Dispatcher
from stl_api.models import Dataset, Job, Run
from stl_api.settings import Settings

router = APIRouter(prefix="/runs", tags=["runs"])
TERMINAL = {"done", "failed"}


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
    for row in rows:
        if row.state in TERMINAL:
            continue
        try:
            st = job_state(store, _spec(run, row))
        except Exception:
            stale = True
            continue
        state, error = st.get("state", "queued"), st.get("error")
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
            "preset": run.preset,
            "weights": json.loads(run.weights_json) if run.preset == "custom" else None,
            "counts": dict(Counter(j.state for j in jobs)),
            "done": bool(jobs) and all(j.state in TERMINAL for j in jobs), "stale": stale}


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
    try:
        jobs = plan_run(store, run_id, body.dataset_id, body.generators, body.seeds,
                        body.params, time_cap_seconds=settings.time_cap_seconds)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    run = Run(id=run_id, dataset_id=body.dataset_id,
              generators_json=json.dumps(body.generators), seeds_json=json.dumps(body.seeds),
              preset=preset, weights_json=json.dumps(list(weights)),
              allow_gpu_only=body.allow_gpu_only)
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
    failed = {j.job_id: j for j in jobs if j.state == "failed"}
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
