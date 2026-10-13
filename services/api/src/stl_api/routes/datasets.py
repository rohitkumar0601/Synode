# ruff: noqa: B008  (FastAPI Depends/File defaults)
"""Dataset endpoints: upload, metadata edit, register, built-ins, list/get/delete."""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, ValidationError
from sqlmodel import Session, select
from stl_core.bounds import bound_value_problems
from stl_core.metadata import ColumnType, TableMetadata, infer_metadata
from stl_runner import keys
from stl_runner.datasets import load_dataset
from stl_runner.pipeline import dataset_fingerprint, ensure_splits, register_dataset
from stl_runner.storage import ArtifactStore

from stl_api.deps import get_session, get_store
from stl_api.models import Dataset, Run

router = APIRouter(prefix="/datasets", tags=["datasets"])
log = logging.getLogger(__name__)

BUILTINS = ("adult", "credit", "diabetes")


def upload_key(dataset_id: str) -> str:
    return f"datasets/{dataset_id}/upload.parquet"


def conf_dir() -> Path:
    override = os.environ.get("STL_CONF_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[5] / "experiments" / "conf"


def _slug(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40].strip("-") or "dataset"
    return f"{base}-{secrets.token_hex(3)}"


def _task(columns: dict[str, ColumnType], target: str) -> str:
    return "classification" if columns.get(target) == ColumnType.categorical else "regression"


class DatasetOut(BaseModel):
    """The one response shape shared by every dataset endpoint."""
    id: str
    name: str
    source: str
    target: str
    n_rows: int | None = None
    metadata: dict
    registered: bool
    fingerprint: str | None = None


class DatasetDetail(DatasetOut):
    created_at: str
    preview: list


def _out(row: Dataset) -> dict:
    return DatasetOut(id=row.id, name=row.name, source=row.source, target=row.target,
                      n_rows=row.n_rows, metadata=json.loads(row.metadata_json),
                      registered=row.registered, fingerprint=row.fingerprint).model_dump()


def _get(session: Session, dataset_id: str) -> Dataset:
    row = session.get(Dataset, dataset_id)
    if row is None:
        raise HTTPException(404, f"dataset {dataset_id!r} not found")
    return row


def _guard_runs(session: Session, dataset_id: str) -> None:
    existing = session.get(Dataset, dataset_id)
    if existing is not None and existing.source == "import":
        return  # placeholder created by a bundle import: registering it promotes it
    n = len(session.exec(select(Run.id).where(Run.dataset_id == dataset_id)).all())
    if n:
        raise HTTPException(409, f"dataset {dataset_id} is used by {n} run(s); "
                                 "create a new dataset instead")


def _read_upload(file: UploadFile) -> pd.DataFrame:
    # Size is bounded by UploadLimitMiddleware; pandas reads the spooled file directly.
    data = file.file
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".csv", ".parquet"):
        raise HTTPException(422, f"unsupported file type {suffix!r}; upload CSV or Parquet")
    try:
        df = pd.read_parquet(data) if suffix == ".parquet" else pd.read_csv(data)
    except Exception as e:
        raise HTTPException(422, f"could not read the file: {e}") from e
    if df.empty:
        raise HTTPException(422, "the file has no rows")
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(object)
    df.columns = [str(c) for c in df.columns]
    return df


def _validate_public(md: TableMetadata, bounds: dict, log_scale: list) -> dict:
    """The same checks as the CLI's prepare; returns the public_info to store."""
    numerical = set(md.of_type(ColumnType.numerical))
    not_numerical = sorted((set(bounds) | set(log_scale)) - numerical)
    if not_numerical:
        raise HTTPException(422, "public_bounds/public_log_scale name columns that are not "
                                 f"numerical: {', '.join(repr(c) for c in not_numerical)}")
    unbounded = sorted(set(log_scale) - set(bounds))
    if unbounded:
        raise HTTPException(422, "public_log_scale names columns without public_bounds: "
                                 f"{', '.join(repr(c) for c in unbounded)}")
    bad = bound_value_problems(bounds, log_scale)
    if bad:
        raise HTTPException(422, f"public_bounds has invalid values: {'; '.join(bad)}")
    return {"public_bounds": dict(bounds), "public_log_scale": list(log_scale)}


def _register(store: ArtifactStore, session: Session, row: Dataset, df: pd.DataFrame,
              md: TableMetadata, max_rows: int, public_info: dict) -> Dataset:
    try:
        prepared = register_dataset(store, row.id, df, md, max_rows=max_rows,
                                    public_info=public_info)
        ensure_splits(store, row.id, 0)
    except ValueError as e:
        row.registered = False
        row.fingerprint = None
        session.add(row)
        session.commit()
        raise HTTPException(422, str(e)) from e
    row.n_rows = len(prepared)
    row.metadata_json = md.model_dump_json()
    row.fingerprint = dataset_fingerprint(store, row.id)
    row.registered = True
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _clear_metafeatures(store: ArtifactStore, dataset_id: str) -> None:
    """Remove the dataset's meta-features and suggestion (logged, never raises)."""
    try:
        store.delete(keys.dataset_suggestion(dataset_id))
        store.delete(keys.dataset_metafeatures(dataset_id))
    except Exception:
        log.exception("could not clear stale meta-features of dataset %s", dataset_id)


def _enqueue_metafeatures(request: Request, store: ArtifactStore, dataset_id: str) -> None:
    """After a (re-)registration: drop meta-features/suggestion computed from older data and
    queue stl.metafeatures (q.meta). Failures are logged; registration still succeeds."""
    _clear_metafeatures(store, dataset_id)
    try:
        request.app.state.dispatcher.enqueue_metafeatures(dataset_id)
    except Exception:
        log.exception("could not enqueue meta-feature task for dataset %s", dataset_id)


@router.post("", status_code=201)
def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    target: str = Form(...),
    session: Session = Depends(get_session),
    store: ArtifactStore = Depends(get_store),
) -> dict:
    if not name.strip():
        raise HTTPException(422, "name must not be empty")
    df = _read_upload(file)
    try:
        md = infer_metadata(df, target)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    dataset_id = keys.validate_component(_slug(name), "dataset id")
    store.write_df(upload_key(dataset_id), df)
    row = Dataset(id=dataset_id, name=name.strip(), source="upload", target=target,
                  n_rows=len(df), metadata_json=md.model_dump_json())
    session.add(row)
    session.commit()
    session.refresh(row)
    return _out(row)


class MetadataBody(BaseModel):
    columns: dict[str, ColumnType]
    target: str


@router.put("/{dataset_id}/metadata")
def update_metadata(dataset_id: str, body: MetadataBody,
                    session: Session = Depends(get_session),
                    store: ArtifactStore = Depends(get_store)) -> dict:
    row = _get(session, dataset_id)
    _guard_runs(session, dataset_id)
    if not store.exists(upload_key(dataset_id)):
        raise HTTPException(409, "dataset has no stored upload")
    actual = set(store.read_df(upload_key(dataset_id)).columns)
    unknown = sorted(set(body.columns) - actual)
    if unknown:
        raise HTTPException(422, f"unknown columns: {', '.join(repr(c) for c in unknown)}")
    try:
        md = TableMetadata(columns=body.columns, target=body.target,
                           task=_task(body.columns, body.target))
    except ValidationError as e:
        raise HTTPException(422, "; ".join(str(err["msg"]) for err in e.errors())) from e
    row.metadata_json = md.model_dump_json()
    row.target = md.target
    row.registered = False
    row.fingerprint = None
    session.add(row)
    session.commit()
    session.refresh(row)
    return _out(row)


class RegisterBody(BaseModel):
    max_rows: int = Field(20_000, ge=1)
    public_bounds: dict[str, Any] | None = None
    public_log_scale: list[str] | None = None


def _promote(session: Session, row: Dataset, imported_fp: str | None, source: str) -> list[str]:
    """Turn an import placeholder into a real dataset; warn if its data differs."""
    row.source = source
    session.add(row)
    session.commit()
    session.refresh(row)
    if imported_fp and imported_fp != row.fingerprint:
        return [f"fingerprint differs from imported run(s) of dataset {row.id}; "
                "results not comparable"]
    return []


@router.post("/builtin/{name}", status_code=201)
def builtin_dataset(name: str, request: Request, body: RegisterBody | None = None,
                    session: Session = Depends(get_session),
                    store: ArtifactStore = Depends(get_store)) -> dict:
    if name not in BUILTINS:
        raise HTTPException(404, f"unknown built-in dataset {name!r}; known: {list(BUILTINS)}")
    body = body or RegisterBody()
    _guard_runs(session, name)
    path = conf_dir() / "datasets" / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(404, f"no dataset config for {name!r} at {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        df, target = load_dataset(cfg)
    except Exception as e:
        raise HTTPException(502, f"could not load built-in dataset {name!r}: {e}") from e
    try:
        md = infer_metadata(df, target)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    column_types = cfg.get("column_types") or {}
    unknown = sorted(set(column_types) - set(df.columns))
    if unknown:
        raise HTTPException(422, "column_types names columns not in the dataset: "
                                 f"{', '.join(repr(c) for c in unknown)}")
    columns = {**md.columns, **{c: ColumnType(t) for c, t in column_types.items()}}
    try:
        md = TableMetadata(columns=columns, target=target, task=_task(columns, target))
    except ValidationError as e:
        raise HTTPException(422, "; ".join(str(err["msg"]) for err in e.errors())) from e
    public_info = _validate_public(md, dict(cfg.get("public_bounds") or {}),
                                   list(cfg.get("public_log_scale") or []))
    store.write_df(upload_key(name), df)
    row = session.get(Dataset, name) or Dataset(id=name, name=name, source="builtin",
                                                target=target)
    placeholder = row.source == "import"
    imported_fp = row.fingerprint if placeholder else None
    row.target = target
    row.n_rows = len(df)
    row.registered = False
    row.metadata_json = md.model_dump_json()
    out = _out(_register(store, session, row, df, md, body.max_rows, public_info))
    _enqueue_metafeatures(request, store, name)
    if placeholder:
        warnings = _promote(session, row, imported_fp, "builtin")
        if warnings:
            out["warnings"] = warnings
    return out


@router.post("/{dataset_id}/register")
def register(dataset_id: str, request: Request, body: RegisterBody | None = None,
             session: Session = Depends(get_session),
             store: ArtifactStore = Depends(get_store)) -> dict:
    body = body or RegisterBody()
    row = _get(session, dataset_id)
    _guard_runs(session, dataset_id)
    if not store.exists(upload_key(dataset_id)):
        raise HTTPException(409, "dataset has no stored upload")
    df = store.read_df(upload_key(dataset_id))
    md = TableMetadata.model_validate(json.loads(row.metadata_json))
    public_info = _validate_public(md, body.public_bounds or {}, body.public_log_scale or [])
    placeholder = row.source == "import"
    imported_fp = row.fingerprint if placeholder else None
    out = _out(_register(store, session, row, df, md, body.max_rows, public_info))
    _enqueue_metafeatures(request, store, dataset_id)
    if placeholder:
        warnings = _promote(session, row, imported_fp, "upload")
        if warnings:
            out["warnings"] = warnings
    return out


def _unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def _read_suggestion(store: ArtifactStore, key: str) -> dict:
    """The stored suggestion if it is well formed, else an 'unreadable' answer."""
    try:
        obj = json.loads(store.read_bytes(key))
    except Exception:
        log.warning("unreadable suggestion file %s", key, exc_info=True)
        return _unavailable("stored suggestion is unreadable")
    ok = isinstance(obj, dict) and isinstance(obj.get("available"), bool)
    if ok and obj["available"]:
        probs = obj.get("probabilities")
        ok = (isinstance(obj.get("generator"), str) and isinstance(probs, dict)
              and all(isinstance(p, int | float) for p in probs.values())
              and isinstance(obj.get("model", {}), dict))
    if not ok:
        log.warning("malformed suggestion file %s", key)
        return _unavailable("stored suggestion is unreadable")
    return obj


def _metafeature_error(store: ArtifactStore, dataset_id: str) -> str | None:
    key = keys.dataset_metafeatures(dataset_id)
    try:
        if not store.exists(key):
            return None
        rec = json.loads(store.read_bytes(key))
    except Exception:
        return None
    err = rec.get("error") if isinstance(rec, dict) else None
    return str(err) if err else None


CLASSIFICATION_ONLY = "RQ3 predictor covers classification datasets only"


def _is_classification(row: Dataset) -> bool:
    try:
        return json.loads(row.metadata_json).get("task", "classification") == "classification"
    except Exception:
        return True  # unreadable metadata: let the stored files decide


@router.get("/{dataset_id}/suggestion")
def get_suggestion(dataset_id: str, session: Session = Depends(get_session),
                   store: ArtifactStore = Depends(get_store)) -> dict:
    """RQ3 best-generator suggestion (exploratory). 404 for an unknown dataset (as every
    /datasets/{id} endpoint); otherwise always 200 with the stored
    ``{"available": true, "generator", "probabilities", "model"}`` or
    ``{"available": false, "reason"}`` (not registered, not a classification dataset,
    extraction failed, no predictor trained, not computed yet, or an unreadable file)."""
    row = _get(session, dataset_id)
    if not row.registered:
        return _unavailable("dataset is not registered")
    if not _is_classification(row):
        return _unavailable(CLASSIFICATION_ONLY)
    key = keys.dataset_suggestion(dataset_id)
    if store.exists(key):
        return _read_suggestion(store, key)
    err = _metafeature_error(store, dataset_id)
    if err:
        return _unavailable(f"meta-feature extraction failed: {err}")
    if not store.exists(keys.meta_predictor()):
        return _unavailable("no meta-predictor has been trained yet (RQ3 meta-benchmark)")
    return _unavailable("suggestion has not been computed yet")


@router.post("/{dataset_id}/suggestion/refresh", status_code=202)
def refresh_suggestion(dataset_id: str, request: Request,
                       session: Session = Depends(get_session),
                       store: ArtifactStore = Depends(get_store)) -> dict:
    """Recompute meta-features and the suggestion (e.g. after the meta-predictor was
    (re)trained). Allowed even when runs use the dataset: nothing about the dataset itself
    changes. 404 unknown dataset, 409 not registered, 503 if the task cannot be queued."""
    row = _get(session, dataset_id)
    if not row.registered:
        raise HTTPException(409, f"dataset {dataset_id} is not registered")
    _clear_metafeatures(store, dataset_id)
    try:
        request.app.state.dispatcher.enqueue_metafeatures(dataset_id)
    except Exception as e:
        log.exception("could not enqueue meta-feature task for dataset %s", dataset_id)
        raise HTTPException(503, f"could not queue the meta-feature task: {e}") from e
    return {"dataset_id": dataset_id, "queued": True}


@router.get("")
def list_datasets(session: Session = Depends(get_session)) -> list[dict]:
    rows = session.exec(select(Dataset).order_by(Dataset.created_at.desc())).all()
    return [_out(r) for r in rows]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, session: Session = Depends(get_session),
                store: ArtifactStore = Depends(get_store)) -> dict:
    row = _get(session, dataset_id)
    preview: list = []
    if store.exists(upload_key(dataset_id)):
        head = store.read_df(upload_key(dataset_id)).head(5)
        preview = json.loads(head.to_json(orient="records", date_format="iso"))
    return DatasetDetail(**_out(row), created_at=row.created_at.isoformat(),
                         preview=preview).model_dump()


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, session: Session = Depends(get_session),
                   store: ArtifactStore = Depends(get_store)) -> None:
    """Delete the row and the dataset's meta-features/suggestion. The registered data
    (prepared table, splits, upload) is left in the store; dataset ids are never reused
    for uploads, and a built-in re-registration overwrites it."""
    row = _get(session, dataset_id)
    if session.exec(select(Run).where(Run.dataset_id == dataset_id)).first() is not None:
        raise HTTPException(409, "dataset is referenced by runs")
    session.delete(row)
    session.commit()
    _clear_metafeatures(store, dataset_id)
