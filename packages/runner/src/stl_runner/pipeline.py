"""Pipeline stages shared by the CLI (local backend) and the Celery workers."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import traceback

import pandas as pd
from stl_core.catalog import CATALOG, load_generator
from stl_core.metadata import TableMetadata, prepare
from stl_core.schemas import CostMetrics, JobSpec
from stl_core.split import Splits, split, stratified_subsample

from stl_runner import keys
from stl_runner.cost import measure
from stl_runner.storage import ArtifactStore

PARTS = ("train", "control", "test")


def register_dataset(
    store: ArtifactStore,
    dataset_id: str,
    df: pd.DataFrame,
    metadata: TableMetadata,
    max_rows: int = 20_000,
    seed: int = 0,
    public_info: dict | None = None,
) -> pd.DataFrame:
    """Prepare, cap and store a dataset.

    ``public_info`` is information taken from public documentation, never from the data
    (``{"public_bounds": {column: [lower, upper]}, "public_log_scale": [column]}``). It is
    stored as ``datasets/<id>/public.json`` (``{}`` when not given, so re-registering clears
    stale info) and passed by generate_job to differentially private generators only.
    """
    prepared = stratified_subsample(prepare(df, metadata), metadata, max_rows, seed)
    store.write_df(keys.dataset_raw(dataset_id), prepared)
    store.write_json(keys.dataset_metadata(dataset_id), metadata)
    store.write_json(keys.dataset_public(dataset_id), dict(public_info or {}))
    store.write_json(keys.dataset_fingerprint(dataset_id),
                     {"fingerprint": _compute_fingerprint(store, dataset_id)})
    return prepared


def _compute_fingerprint(store: ArtifactStore, dataset_id: str) -> str:
    """Hash of the stored prepared table (values, columns, dtypes) and its metadata."""
    raw = store.read_df(keys.dataset_raw(dataset_id))
    h = hashlib.sha256()
    h.update(json.dumps([[str(c), str(t)] for c, t in raw.dtypes.items()]).encode())
    h.update(pd.util.hash_pandas_object(raw, index=True).to_numpy().tobytes())
    h.update(json.dumps(store.read_json(keys.dataset_metadata(dataset_id)),
                        sort_keys=True).encode())
    return h.hexdigest()


def dataset_fingerprint(store: ArtifactStore, dataset_id: str) -> str:
    key = keys.dataset_fingerprint(dataset_id)
    if not store.exists(key):  # registered before fingerprints existed
        store.write_json(key, {"fingerprint": _compute_fingerprint(store, dataset_id)})
    return store.read_json(key)["fingerprint"]


def load_metadata(store: ArtifactStore, dataset_id: str) -> TableMetadata:
    return TableMetadata.model_validate(store.read_json(keys.dataset_metadata(dataset_id)))


def ensure_splits(store: ArtifactStore, dataset_id: str, seed: int) -> Splits:
    """Load the seed's splits, recomputing them if missing or made from an older
    registration of the dataset (fingerprint mismatch)."""
    part_keys = [keys.split_part(dataset_id, seed, p) for p in PARTS]
    source_key = keys.split_source(dataset_id, seed)
    fingerprint = dataset_fingerprint(store, dataset_id)
    current = (
        store.exists(source_key)
        and store.read_json(source_key).get("dataset_fingerprint") == fingerprint
        and all(store.exists(k) for k in part_keys)
    )
    if current:
        return Splits(*(store.read_df(k) for k in part_keys))
    df = store.read_df(keys.dataset_raw(dataset_id))
    splits = split(df, load_metadata(store, dataset_id), seed)
    for k, part in zip(part_keys, (splits.train, splits.control, splits.test), strict=True):
        store.write_df(k, part)
    store.write_json(source_key, {"dataset_fingerprint": fingerprint})
    return Splits(*(store.read_df(k) for k in part_keys))


def set_state(store: ArtifactStore, job: JobSpec, state: str, error: str | None = None) -> None:
    store.write_json(keys.status(job.run_id, job.job_id), {"state": state, "error": error})


def job_state(store: ArtifactStore, job: JobSpec) -> dict:
    key = keys.status(job.run_id, job.job_id)
    return store.read_json(key) if store.exists(key) else {"state": "queued", "error": None}


def plan_run(
    store: ArtifactStore,
    run_id: str,
    dataset_id: str,
    generators: list[str],
    seeds: list[int],
    params: dict[str, dict] | None = None,
    time_cap_seconds: float | None = None,
) -> list[JobSpec]:
    unknown = [g for g in generators if g not in CATALOG]
    if unknown:
        raise ValueError(f"unknown generators {unknown}; known: {sorted(CATALOG)}")
    params = params or {}
    jobs = [
        JobSpec(job_id=f"{g}__s{s}", run_id=run_id, dataset_id=dataset_id, generator=g, seed=s,
                params=dict(params.get(g, {})), time_cap_seconds=time_cap_seconds)
        for g in generators
        for s in seeds
    ]
    store.write_json(keys.run_manifest(run_id), {
        "run_id": run_id, "dataset_id": dataset_id,
        "jobs": [j.model_dump(mode="json") for j in jobs],
    })
    for job in jobs:
        if not store.exists(keys.status(run_id, job.job_id)):
            set_state(store, job, "queued")
    return jobs


def load_jobs(store: ArtifactStore, run_id: str) -> list[JobSpec]:
    manifest = store.read_json(keys.run_manifest(run_id))
    return [JobSpec.model_validate(j) for j in manifest["jobs"]]


def generator_params(store: ArtifactStore, job: JobSpec) -> dict:
    """The job's params; differentially private generators also get the dataset's public
    info (codebook bounds), so bounds cost no epsilon. Job params take precedence."""
    params = dict(job.params)
    public_key = keys.dataset_public(job.dataset_id)
    if CATALOG[job.generator].family == "differential_privacy" and store.exists(public_key):
        params = {**store.read_json(public_key), **params}
    return params


def generate_job(store: ArtifactStore, job: JobSpec) -> CostMetrics:
    # Outputs derived from an earlier synthetic table must not outlive it.
    for key in (keys.synthcity_metrics(job.run_id, job.job_id),
                keys.report(job.run_id, job.job_id)):
        store.delete(key)
    metadata = load_metadata(store, job.dataset_id)
    splits = ensure_splits(store, job.dataset_id, job.seed)
    generator = load_generator(job.generator, seed=job.seed, **generator_params(store, job))
    set_state(store, job, "fitting")
    with measure() as fit_m:
        generator.fit(splits.train, metadata)
    set_state(store, job, "sampling")
    with measure() as sample_m:
        synthetic = generator.sample(len(splits.train))
    cost = CostMetrics(fit_seconds=fit_m.seconds, sample_seconds=sample_m.seconds,
                       peak_memory_mb=max(fit_m.peak_mb, sample_m.peak_mb),
                       baseline_memory_mb=fit_m.start_mb)
    store.write_df(keys.synthetic(job.run_id, job.job_id), synthetic)
    store.write_json(keys.cost(job.run_id, job.job_id), cost)
    if getattr(generator, "privacy_spent", None) is not None:
        store.write_json(keys.privacy(job.run_id, job.job_id), generator.privacy_spent)
    set_state(store, job, "generated")
    return cost


def _generate_entry(root: str, storage_options: dict, job_json: str) -> None:
    store = ArtifactStore(root, storage_options)
    job = JobSpec.model_validate_json(job_json)
    try:
        generate_job(store, job)
    except BaseException:
        set_state(store, job, "failed", traceback.format_exc())
        raise SystemExit(1) from None


def cuda_available() -> bool:
    """True only if torch is importable and reports CUDA (no torch means no CUDA)."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def gpu_only_blocked(generator: str, allow: bool = False) -> bool:
    """True if `generator` is GPU-only, CUDA is unavailable and running it anyway was not
    allowed (by `allow` or the environment variable STL_ALLOW_GPU_ONLY=1)."""
    if CATALOG[generator].cpu_ok or allow or os.environ.get("STL_ALLOW_GPU_ONLY") == "1":
        return False
    return not cuda_available()


def run_generate_with_cap(store: ArtifactStore, job: JobSpec,
                          allow_gpu_only: bool = False) -> str:
    """Run generate_job with optional time cap.

    A GPU-only generator without CUDA (see gpu_only_blocked) is not run: the job is left
    `queued` and "queued" is returned. The time cap includes child process start-up and
    import time.
    """
    set_state(store, job, "queued")  # Reset to clear any stale state
    if gpu_only_blocked(job.generator, allow_gpu_only):
        return "queued"
    if job.time_cap_seconds is None:
        try:
            generate_job(store, job)
        except KeyboardInterrupt:
            set_state(store, job, "failed", "interrupted")
            raise
        except Exception:
            set_state(store, job, "failed", traceback.format_exc())
        return job_state(store, job)["state"]
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_generate_entry,
                       args=(store.root, store.storage_options, job.model_dump_json()))
    proc.start()
    proc.join(job.time_cap_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        set_state(store, job, "failed", f"time cap of {job.time_cap_seconds}s exceeded")
    else:
        # Child exited; check if it succeeded
        state = job_state(store, job)["state"]
        if state != "generated" and state != "failed":
            msg = (
                f"worker exited with code {proc.exitcode} without finishing "
                f"(last state {state!r})"
            )
            set_state(store, job, "failed", msg)
    return job_state(store, job)["state"]
