"""Pipeline stages shared by the CLI (local backend) and the Celery workers."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback

import pandas as pd
from stl_core.budget import check_profile
from stl_core.catalog import CATALOG, load_generator
from stl_core.generator import Generator
from stl_core.metadata import TableMetadata, prepare
from stl_core.schemas import CostMetrics, JobSpec
from stl_core.split import Splits, split, stratified_subsample

from stl_runner import keys
from stl_runner.control import cancel_if_requested, is_cancelled, mark_cancelled
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
    budget_profile: str | None = None,
) -> list[JobSpec]:
    """Plan and store a run. ``budget_profile`` (stl_core.budget) is recorded in the
    manifest when given; manifests without it (older runs, callers that do not pass it)
    stay valid."""
    unknown = [g for g in generators if g not in CATALOG]
    if unknown:
        raise ValueError(f"unknown generators {unknown}; known: {sorted(CATALOG)}")
    if budget_profile is not None:
        check_profile(budget_profile)
    params = params or {}
    jobs = [
        JobSpec(job_id=f"{g}__s{s}", run_id=run_id, dataset_id=dataset_id, generator=g, seed=s,
                params=dict(params.get(g, {})), time_cap_seconds=time_cap_seconds)
        for g in generators
        for s in seeds
    ]
    manifest = {"run_id": run_id, "dataset_id": dataset_id,
                "jobs": [j.model_dump(mode="json") for j in jobs]}
    if budget_profile is not None:
        manifest["budget_profile"] = budget_profile
    store.write_json(keys.run_manifest(run_id), manifest)
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


def budget_profile(store: ArtifactStore, job: JobSpec) -> str | None:
    """The run's budget profile from its manifest (None for runs planned without one)."""
    key = keys.run_manifest(job.run_id)
    return store.read_json(key).get("budget_profile") if store.exists(key) else None


def generate_job(store: ArtifactStore, job: JobSpec) -> CostMetrics:
    # Outputs derived from an earlier synthetic table must not outlive it.
    for key in (keys.synthcity_metrics(job.run_id, job.job_id),
                keys.report(job.run_id, job.job_id)):
        store.delete(key)
    metadata = load_metadata(store, job.dataset_id)
    splits = ensure_splits(store, job.dataset_id, job.seed)
    generator = load_generator(job.generator, seed=job.seed, **generator_params(store, job))
    # Before fit, so a job that fails or hits the time cap still records what it ran with.
    # Duck-typed generators without resolved_params get the base class's empty record.
    resolved = getattr(generator, "resolved_params", None)
    record = resolved() if resolved else Generator.resolved_params(generator)
    store.write_json(keys.generator_params(job.run_id, job.job_id),
                     {**record, "profile": budget_profile(store, job)})
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
                          allow_gpu_only: bool = False, poll_seconds: float = 2.0) -> str:
    """Run generate_job in a child process, with the job's optional time cap.

    A GPU-only generator without CUDA (see gpu_only_blocked) is not run: the job is left
    `queued` and "queued" is returned. A cancelled run (stl_runner.control) is not run
    either ("cancelled"). While the child runs, the parent polls every ``poll_seconds``
    and kills it when the run is cancelled (state "cancelled") or the time cap, which
    includes child start-up and import time, is exceeded (state "failed").
    """
    if cancel_if_requested(store, job):
        return job_state(store, job)["state"]
    set_state(store, job, "queued")  # Reset to clear any stale state
    if gpu_only_blocked(job.generator, allow_gpu_only):
        return "queued"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_generate_entry,
                       args=(store.root, store.storage_options, job.model_dump_json()))
    proc.start()
    try:
        stop = _watch_child(store, job, proc, poll_seconds)
    except BaseException:  # KeyboardInterrupt etc.: do not leave the child training
        _kill(proc)
        set_state(store, job, "failed", "interrupted")
        raise
    if stop is not None:
        _kill(proc)
    if stop == "cancelled":
        mark_cancelled(store, job)
    elif stop == "cap":
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


def _watch_child(store: ArtifactStore, job: JobSpec, proc, poll_seconds: float) -> str | None:
    """Join the child in short slices. Returns "cancelled" or "cap" if it must be killed,
    None once it has exited by itself."""
    cap = job.time_cap_seconds
    deadline = None if cap is None else time.monotonic() + cap
    while True:
        wait = poll_seconds if deadline is None else min(poll_seconds,
                                                         max(0.0, deadline - time.monotonic()))
        proc.join(wait)
        if not proc.is_alive():
            return None
        try:
            if is_cancelled(store, job.run_id):
                return "cancelled"
        except Exception:  # a store hiccup must not kill the job; check again next slice
            pass
        if deadline is not None and time.monotonic() >= deadline:
            return "cap"


def _kill(proc) -> None:
    proc.terminate()
    proc.join(10)
    if proc.is_alive():
        proc.kill()
        proc.join()
