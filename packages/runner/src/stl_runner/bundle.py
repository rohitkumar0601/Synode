"""Results bundles: move job results between stores (e.g. offline GPU runs back home)."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone

from stl_core.catalog import CATALOG
from stl_core.schemas import BundleManifest, CostMetrics, JobSpec, MetricReport

from stl_runner import keys
from stl_runner.pipeline import dataset_fingerprint, load_jobs
from stl_runner.storage import ArtifactStore

MANIFEST_NAME = "bundle.json"
FORMAT_VERSION = 1
MAX_MEMBER_BYTES = 512 * 1024**2  # per member, uncompressed (ZipInfo.file_size)
MAX_TOTAL_BYTES = 2 * 1024**3  # whole bundle, uncompressed


def _job_files(run_id: str, job_id: str, include_synthetic: bool) -> dict[str, str]:
    """Bundle member name -> store key."""
    files = {
        f"jobs/{job_id}/report.json": keys.report(run_id, job_id),
        f"jobs/{job_id}/cost.json": keys.cost(run_id, job_id),
        f"jobs/{job_id}/status.json": keys.status(run_id, job_id),
        # optional: only DP generators write it (epsilon split, bounds path and hash)
        f"jobs/{job_id}/privacy.json": keys.privacy(run_id, job_id),
    }
    if include_synthetic:
        files[f"jobs/{job_id}/synthetic.parquet"] = keys.synthetic(run_id, job_id)
    return files


def export_bundle(store: ArtifactStore, run_id: str, out_path: str,
                  include_synthetic: bool = False) -> BundleManifest:
    """Write a zip with the run's reported jobs (jobs without a report.json are skipped)."""
    jobs = [j for j in load_jobs(store, run_id) if store.exists(keys.report(run_id, j.job_id))]
    dataset_id = store.read_json(keys.run_manifest(run_id))["dataset_id"]
    manifest = BundleManifest(
        format_version=FORMAT_VERSION, run_id=run_id, dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint(store, dataset_id),
        created_at=datetime.now(timezone.utc).isoformat(), jobs=jobs,
        include_synthetic=include_synthetic)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST_NAME, manifest.model_dump_json(indent=2))
        for job in jobs:
            for name, key in _job_files(run_id, job.job_id, include_synthetic).items():
                if store.exists(key):
                    z.writestr(name, store.read_bytes(key))
    return manifest


def read_manifest(bundle_path: str) -> BundleManifest:
    """Parse and validate a bundle's manifest (ids are checked before any use)."""
    with zipfile.ZipFile(bundle_path) as z:
        return _parse_manifest(z, bundle_path)


def _check_sizes(z: zipfile.ZipFile, bundle_path: str) -> None:
    """Refuse zip bombs: cap declared uncompressed sizes before reading any member."""
    total = 0
    for info in z.infolist():
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"{bundle_path}: member {info.filename!r} is {info.file_size} "
                             f"bytes uncompressed (limit {MAX_MEMBER_BYTES})")
        total += info.file_size
    if total > MAX_TOTAL_BYTES:
        raise ValueError(f"{bundle_path} is {total} bytes uncompressed "
                         f"(limit {MAX_TOTAL_BYTES})")


def _parse_manifest(z: zipfile.ZipFile, bundle_path: str) -> BundleManifest:
    _check_sizes(z, bundle_path)
    if MANIFEST_NAME not in z.namelist():
        raise ValueError(f"{bundle_path} is not a results bundle (no {MANIFEST_NAME})")
    manifest = BundleManifest.model_validate_json(z.read(MANIFEST_NAME))
    if manifest.format_version != FORMAT_VERSION:
        raise ValueError(f"unsupported bundle format_version {manifest.format_version}")
    keys.validate_component(manifest.run_id, "run_id")
    keys.validate_component(manifest.dataset_id, "dataset_id")
    for job in manifest.jobs:
        keys.validate_component(job.job_id, "job_id")
        if job.generator not in CATALOG:
            raise ValueError(f"job {job.job_id!r} has unknown generator {job.generator!r}")
        if job.run_id != manifest.run_id or job.dataset_id != manifest.dataset_id:
            raise ValueError(f"job {job.job_id!r} belongs to run {job.run_id!r} / dataset "
                             f"{job.dataset_id!r}, not the bundle's {manifest.run_id!r} / "
                             f"{manifest.dataset_id!r}")
    return manifest


def import_bundle(store: ArtifactStore, bundle_path: str, force: bool = False,
                  warnings: list[str] | None = None) -> list[MetricReport]:
    """Import a bundle into the store; returns the imported reports.

    Everything is read and validated before the first write. Non-fatal notices are
    appended to `warnings` when a list is given.
    """
    with zipfile.ZipFile(bundle_path) as z:
        manifest = _parse_manifest(z, bundle_path)
        run_id = manifest.run_id
        names = set(z.namelist())
        fp_key = keys.dataset_fingerprint(manifest.dataset_id)
        if store.exists(fp_key):
            local = store.read_json(fp_key)["fingerprint"]
            if not force and local != manifest.dataset_fingerprint:
                raise ValueError(
                    f"bundle dataset fingerprint {manifest.dataset_fingerprint[:12]} differs "
                    f"from local {local[:12]} for dataset {manifest.dataset_id!r} "
                    "(use --force)")
        elif warnings is not None:
            warnings.append(f"dataset {manifest.dataset_id} not registered locally; "
                            "fingerprint not verified")
        manifest_key = keys.run_manifest(run_id)
        if store.exists(manifest_key) and not force:
            local_ds = store.read_json(manifest_key).get("dataset_id")
            if local_ds != manifest.dataset_id:
                raise ValueError(f"local run {run_id!r} is for dataset {local_ds!r}, bundle is "
                                 f"for {manifest.dataset_id!r} (use --force)")
        writes: list[tuple[str, bytes]] = []
        reports: list[MetricReport] = []
        for job in manifest.jobs:
            for name, key in _job_files(run_id, job.job_id, manifest.include_synthetic).items():
                kind = name.rsplit("/", 1)[1]
                if name not in names:
                    if kind in ("report.json", "status.json"):
                        raise ValueError(f"bundle is missing {name}")
                    # cost.json (failed job), privacy.json (non-DP generator) and
                    # synthetic.parquet may be absent
                    continue
                data = z.read(name)
                if kind == "report.json":
                    report = MetricReport.model_validate_json(data)
                    if (report.run_id, report.job_id) != (run_id, job.job_id):
                        raise ValueError(f"{name} is for {report.run_id}/{report.job_id}, "
                                         f"expected {run_id}/{job.job_id}")
                    for field in ("generator", "seed", "dataset_id"):
                        if getattr(report, field) != getattr(job, field):
                            raise ValueError(
                                f"{name} has {field} {getattr(report, field)!r}, but job "
                                f"{job.job_id!r} has {getattr(job, field)!r}")
                    if store.exists(key) and not force:
                        raise ValueError(f"job {job.job_id!r} already has a local report in "
                                         f"run {run_id!r}; refusing to overwrite (use --force)")
                    reports.append(report)
                elif kind == "cost.json":
                    CostMetrics.model_validate_json(data)
                elif kind in ("status.json", "privacy.json"):
                    if not isinstance(json.loads(data), dict):
                        raise ValueError(f"{name} is not a JSON object")
                writes.append((key, data))
    for key, data in writes:
        store.write_bytes(key, data)
    _merge_manifest(store, manifest)
    return reports


def _merge_manifest(store: ArtifactStore, manifest: BundleManifest) -> None:
    merged: dict[str, JobSpec] = {}
    if store.exists(keys.run_manifest(manifest.run_id)):
        merged = {j.job_id: j for j in load_jobs(store, manifest.run_id)}
    merged.update({j.job_id: j for j in manifest.jobs})  # bundle wins
    store.write_json(keys.run_manifest(manifest.run_id), {
        "run_id": manifest.run_id, "dataset_id": manifest.dataset_id,
        "jobs": [j.model_dump(mode="json") for j in merged.values()],
    })
