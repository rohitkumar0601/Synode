# ruff: noqa: B008  (FastAPI Depends/File defaults)
"""Report endpoint (aggregate + recommend over stored reports) and results-bundle import."""
from __future__ import annotations

import json
import math
import os
import tempfile
import zipfile
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session, select
from stl_core.aggregate import aggregate
from stl_core.decision import recommend, resolve_weights
from stl_core.schemas import MetricReport
from stl_runner import keys
from stl_runner.bundle import import_bundle, read_manifest
from stl_runner.pipeline import job_state, load_jobs
from stl_runner.storage import ArtifactStore

from stl_api.deps import get_session, get_store
from stl_api.models import Dataset, Job, Run

router = APIRouter(tags=["reports"])


def _clean(x: Any) -> Any:
    """Make a JSON-able structure strict: non-finite floats become null."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _parse_weights(raw: str | None) -> tuple[float, float, float] | None:
    if raw is None:
        return None
    try:
        parts = tuple(float(p) for p in raw.split(","))
    except ValueError as e:
        raise HTTPException(
            422, f"weights must be 'fidelity,utility,privacy' numbers: {raw!r}") from e
    return parts  # type: ignore[return-value]  (length checked by resolve_weights)


def _stored_params(store: ArtifactStore, run_id: str, job_id: str) -> dict | None:
    """The job's generator_params.json (written before fit), or None when absent/unreadable;
    used when the job has no report yet or its report predates the field."""
    key = keys.generator_params(run_id, job_id)
    try:
        if store.exists(key):
            got = store.read_json(key)
            return got if isinstance(got, dict) else None
    except Exception:
        return None
    return None


@router.get("/runs/{run_id}/report")
def run_report(run_id: str, preset: str = "balanced", weights: str | None = None,
               session: Session = Depends(get_session),
               store: ArtifactStore = Depends(get_store)) -> dict:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id!r} not found")
    custom = _parse_weights(weights)
    try:
        resolve_weights(preset, custom)
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e.args[0])) from e
    try:
        has_manifest = store.exists(keys.run_manifest(run_id))
        if has_manifest:
            jobs = load_jobs(store, run_id)
    except (ValueError, KeyError) as e:  # corrupt manifest
        raise HTTPException(409, f"run {run_id} has no readable manifest: {e}") from e
    except Exception as e:
        raise HTTPException(503, f"artifact store unavailable: {e}") from e
    if not has_manifest:
        raise HTTPException(409, f"run {run_id} has no readable manifest")
    reports: list[MetricReport] = []
    per_job = []
    for job in sorted(jobs, key=lambda j: j.job_id):
        report, unreadable = None, None
        try:
            if store.exists(keys.report(run_id, job.job_id)):
                report = MetricReport.model_validate(
                    store.read_json(keys.report(run_id, job.job_id)))
        except ValueError as e:  # corrupt JSON / schema mismatch (pydantic errors included)
            unreadable = f"report unreadable: {type(e).__name__}: {e}"
        except Exception as e:
            raise HTTPException(503, f"artifact store unavailable: {e}") from e
        if report is not None:
            reports.append(report)
            status, warnings = report.status, list(report.warnings)
        else:
            try:
                status = job_state(store, job).get("state", "queued")
            except Exception:
                status = "unknown"
            warnings = [unreadable] if unreadable else []
        params = report.generator_params if report is not None else None
        per_job.append({"job_id": job.job_id, "generator": job.generator, "seed": job.seed,
                        "status": status, "warnings": warnings,
                        "generator_params": params or _stored_params(store, run_id, job.job_id)})
    run_warnings: list[str] = []
    missing_synthcity = [
        r.job_id for r in reports
        if r.status == "done" and any("synthcity metrics missing" in w for w in (r.warnings or []))
    ]
    if missing_synthcity:
        run_warnings.append(
            f"{len(missing_synthcity)} job(s) missing synthcity metrics: run `metrics` stage"
        )
    out: dict = {"run_id": run_id, "jobs_reported": len(reports), "jobs_total": len(jobs),
                 "warnings": run_warnings,
                 "recommendation": None, "aggregated": [], "pareto": [], "per_job": per_job}
    if reports:
        aggs = aggregate(reports)
        rec = recommend(aggs, preset, custom)
        out["recommendation"] = rec.model_dump(mode="json")
        out["aggregated"] = [a.model_dump(mode="json") for a in aggs]
        out["pareto"] = [{"generator": c.generator, "fidelity": c.scores.fidelity,
                          "utility": c.scores.utility, "privacy": c.scores.privacy,
                          "on_front": c.on_pareto_front, "gate_passed": c.gate_passed}
                         for c in rec.ranking]
    return _clean(out)


@router.post("/import-results")
def import_results(file: UploadFile = File(...), force: bool = Form(False),
                   session: Session = Depends(get_session),
                   store: ArtifactStore = Depends(get_store)) -> dict:
    fd, path = tempfile.mkstemp(suffix=".zip")
    try:
        try:
            out = os.fdopen(fd, "wb")
        except Exception:
            os.close(fd)
            raise
        with out:
            while chunk := file.file.read(1024 * 1024):
                out.write(chunk)
        warnings: list[str] = []
        try:
            manifest = read_manifest(path)
            reports = import_bundle(store, path, force, warnings=warnings)
        except (ValueError, zipfile.BadZipFile) as e:
            raise HTTPException(422, str(e)) from e
        except OSError as e:
            raise HTTPException(503, f"artifact store unavailable: {e}") from e
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    run_id = manifest.run_id
    try:
        merged = sorted(load_jobs(store, run_id), key=lambda j: j.job_id)  # store is the truth
        dataset_id = store.read_json(keys.run_manifest(run_id))["dataset_id"]
    except Exception as e:
        raise HTTPException(503, f"artifact store unavailable: {e}") from e
    if session.get(Dataset, dataset_id) is None:
        session.add(Dataset(id=dataset_id, name=dataset_id, source="import", target="",
                            fingerprint=manifest.dataset_fingerprint, registered=False))
        session.flush()
    gens = list(dict.fromkeys(j.generator for j in merged))
    seeds = sorted({j.seed for j in merged})
    run = session.get(Run, run_id)
    if run is None:
        run = Run(id=run_id, dataset_id=dataset_id, generators_json=json.dumps(gens),
                  seeds_json=json.dumps(seeds), preset="balanced")
    else:
        run.dataset_id = dataset_id
        run.generators_json, run.seeds_json = json.dumps(gens), json.dumps(seeds)
    session.add(run)
    session.flush()
    existing = {j.job_id: j for j in session.exec(select(Job).where(Job.run_id == run_id))}
    for spec in merged:
        try:
            st = job_state(store, spec)
        except Exception:
            st = {}
        state, error = st.get("state", "done"), st.get("error")
        row = existing.get(spec.job_id)
        if row is None:
            row = Job(id=f"{run_id}/{spec.job_id}", run_id=run_id, job_id=spec.job_id,
                      generator=spec.generator, seed=spec.seed)
        row.generator, row.seed = spec.generator, spec.seed
        row.state, row.error = state, error
        row.updated_at = datetime.now(UTC)
        session.add(row)
    session.commit()
    return {"run_id": run_id, "imported": len(reports), "warnings": warnings}
