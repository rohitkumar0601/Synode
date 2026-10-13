"""Collect stored benchmark results into one tidy table (one row per job).

Nothing is recomputed: metrics come from ``report.json`` (``cost.json`` for jobs that never
got a report), and every gate result and score comes from the decision engine in stl_core
(``recommend(aggregate([report]))``, the same calls as the API report route), evaluated on
the single job, i.e. per seed. Seed-aggregated gates/scores are the RQ1 stage's job.
"""

from __future__ import annotations

import os
import platform
from collections import Counter
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from typing import Any

import pandas as pd
from pydantic import ValidationError
from stl_core.aggregate import aggregate
from stl_core.budget import BUDGET_PROFILES
from stl_core.catalog import CATALOG
from stl_core.decision import recommend
from stl_core.schemas import JobSpec, MetricReport
from stl_runner import keys
from stl_runner.storage import ArtifactStore

PRESET = "balanced"
MIN_SEEDS = 3  # fewer seeds than this: "smoke — not for the thesis"
LOW_EPOCHS = 5  # with no known budget profile, this many epochs or fewer: smoke
META_PREFIX = "meta-"
ID_COLUMNS = [
    "store", "run_id", "dataset_id", "generator", "family", "seed", "job_id", "state",
    "report_status", "error_last_line", "report_error", "budget_profile", "epochs",
    "n_seeds_in_run", "metrics_version", "generator_library", "generator_library_version",
    "dp_epsilon_preprocessor", "dp_epsilon_synthesizer", "dp_bounds",
    "gate_passed", "gate_reasons", "score", "score_fidelity", "score_utility", "score_privacy",
    "privacy.singling_out_status", "is_smoke", "smoke_reasons", "is_meta",
]
_EPOCH_KEYS = ("epochs", "n_iter")


def open_store(root: str) -> ArtifactStore:
    """An ArtifactStore for ``root``; S3 credentials come from the environment, as in
    stl_runner.storage.store_from_env."""
    options: dict = {}
    if root.startswith("s3://"):
        options = {"key": os.environ.get("AWS_ACCESS_KEY_ID"),
                   "secret": os.environ.get("AWS_SECRET_ACCESS_KEY"),
                   "client_kwargs": {"endpoint_url": os.environ.get("S3_ENDPOINT_URL")}}
    store = ArtifactStore(root, options)
    # ArtifactStore has no public "root exists" / glob helper (``list`` walks every file of
    # the prefix, i.e. every artifact of every run), so the private ``_path`` is used here
    # and in run_ids_in.
    if not store.fs.exists(store._path("")):
        raise FileNotFoundError(f"artifact store {root!r} does not exist")
    return store


def run_ids_in(store: ArtifactStore) -> list[str]:
    paths = store.fs.glob(store._path("runs/*/manifest.json"))
    return sorted(str(p).replace("\\", "/").rstrip("/").split("/")[-2] for p in paths)


def _read(store: ArtifactStore, key: str) -> dict | None:
    return store.read_json(key) if store.exists(key) else None


def _last_line(text: str | None) -> str | None:
    if not text:
        return None
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _epochs(job: JobSpec, gen_params: dict | None) -> int | None:
    for source in (job.params, (gen_params or {}).get("params") or {}):
        for k in _EPOCH_KEYS:
            if source.get(k) is not None:
                return int(source[k])
    return None


def smoke_reasons(profile: str | None, epochs: int | None, n_seeds: int) -> list[str]:
    """Why a result is "smoke — not for the thesis" (empty list: it is not)."""
    reasons = []
    if profile == "smoke":
        reasons.append("profile smoke")
    default = BUDGET_PROFILES.get(profile or "", {}).get("epochs")
    if epochs is not None and default is not None and epochs < default:
        reasons.append(f"epochs {epochs} < {profile} default {default}")
    if profile is None and epochs is not None and epochs <= LOW_EPOCHS:
        reasons.append("unknown profile, low epochs")
    if n_seeds < MIN_SEEDS:
        reasons.append(f"{n_seeds} seed{'s' if n_seeds != 1 else ''} < {MIN_SEEDS}")
    return reasons


def _metrics(report: MetricReport) -> dict[str, Any]:
    out: dict[str, Any] = dict(report.flat())
    out.update({f"utility.real_scores.{k}": v for k, v in report.utility.real_scores.items()})
    out["privacy.singling_out_status"] = report.privacy.singling_out_status
    return out


def _decision(report: MetricReport) -> dict[str, Any]:
    cand = recommend(aggregate([report]), preset=PRESET).ranking[0]
    return {"gate_passed": cand.gate_passed, "gate_reasons": "; ".join(cand.gate_failures),
            "score": cand.score, "score_fidelity": cand.scores.fidelity,
            "score_utility": cand.scores.utility, "score_privacy": cand.scores.privacy}


def _job_row(store: ArtifactStore, root: str, manifest: dict, job: JobSpec,
             n_seeds: int) -> dict[str, Any]:
    run_id, job_id = job.run_id, job.job_id
    status = _read(store, keys.status(run_id, job_id)) or {"state": "queued", "error": None}
    gen_params = _read(store, keys.generator_params(run_id, job_id))
    privacy = _read(store, keys.privacy(run_id, job_id)) or {}
    report, report_error = None, None
    raw = _read(store, keys.report(run_id, job_id))
    if raw is not None:
        try:
            report = MetricReport.model_validate(raw)
        except (ValidationError, ValueError) as e:
            report_error = f"{type(e).__name__}: {_last_line(str(e))}"
    if gen_params is None and report is not None:
        gen_params = report.generator_params
    profile = manifest.get("budget_profile") or (gen_params or {}).get("profile")
    epochs = _epochs(job, gen_params)
    reasons = smoke_reasons(profile, epochs, n_seeds)
    info = CATALOG.get(job.generator)
    row: dict[str, Any] = {
        "store": root, "run_id": run_id, "dataset_id": job.dataset_id,
        "generator": job.generator, "family": info.family if info else None,
        "seed": job.seed, "job_id": job_id, "state": status.get("state"),
        "report_status": report.status if report else None,
        "error_last_line": _last_line(status.get("error"))
        or _last_line(report.error if report else None),
        "report_error": report_error, "budget_profile": profile, "epochs": epochs,
        "n_seeds_in_run": n_seeds,
        "metrics_version": report.metrics_version if report else None,
        "generator_library": (gen_params or {}).get("library"),
        "generator_library_version": (gen_params or {}).get("library_version"),
        "dp_epsilon_preprocessor": privacy.get("preprocessor"),
        "dp_epsilon_synthesizer": privacy.get("synthesizer"),
        "dp_bounds": privacy.get("bounds"),
        "is_smoke": bool(reasons), "smoke_reasons": "; ".join(reasons),
        "is_meta": run_id.startswith(META_PREFIX),
    }
    if report is not None:
        row.update(_metrics(report))
        row.update(_decision(report))
    else:
        cost = _read(store, keys.cost(run_id, job_id))
        if cost:
            row.update({f"cost.{k}": v for k, v in cost.items()})
    return row


def collect(stores: list[str], run_ids: list[str] | None = None,
            include_meta: bool = True) -> pd.DataFrame:
    """One row per job of every run in ``stores`` (optionally only ``run_ids``; meta-benchmark
    runs ``meta-*`` dropped unless ``include_meta``). Jobs without a report (failed,
    cancelled, unfinished) are kept with NaN metrics. A run id present in several stores
    gives rows for each, told apart by the ``store`` column."""
    rows: list[dict[str, Any]] = []
    for root in stores:
        store = open_store(root)
        for run_id in run_ids_in(store):
            if run_ids is not None and run_id not in run_ids:
                continue
            if not include_meta and run_id.startswith(META_PREFIX):
                continue
            manifest = store.read_json(keys.run_manifest(run_id))
            jobs = [JobSpec.model_validate(j) for j in manifest["jobs"]]
            seeds: dict[str, set[int]] = {}
            for j in jobs:
                seeds.setdefault(j.generator, set()).add(j.seed)
            for job in sorted(jobs, key=lambda j: (j.generator, j.seed)):
                rows.append(_job_row(store, root, manifest, job, len(seeds[job.generator])))
    return _frame(rows)


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in ID_COLUMNS:
        if c not in df.columns:
            df[c] = None
    metric_cols = sorted(c for c in df.columns if c not in ID_COLUMNS)
    df = df[ID_COLUMNS + metric_cols]
    for c in metric_cols + ["score", "score_fidelity", "score_utility", "score_privacy",
                            "dp_epsilon_preprocessor", "dp_epsilon_synthesizer"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ("epochs", "metrics_version", "n_seeds_in_run", "seed"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("gate_passed", "is_smoke", "is_meta"):
        df[c] = df[c].astype("boolean")
    str_cols = [c for c in ID_COLUMNS if df[c].dtype == object]
    df[str_cols] = df[str_cols].astype("string")
    return df.reset_index(drop=True)


def _version(dist: str) -> str | None:
    try:
        return importlib_metadata.version(dist)
    except importlib_metadata.PackageNotFoundError:
        return None


def provenance(df: pd.DataFrame, stores: list[str], run_ids: list[str] | None,
               include_meta: bool) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {s: {} for s in stores}
    for (store, run_id), g in df.groupby(["store", "run_id"], sort=True):
        runs[store][run_id] = {
            "dataset_id": sorted(g.dataset_id.dropna().unique().tolist()),
            "n_jobs": int(len(g)),
            "states": dict(sorted(Counter(g.state.fillna("unknown")).items())),
            "budget_profile": sorted(g.budget_profile.dropna().unique().tolist()),
            "n_seeds": int(g.seed.nunique()),
            "is_smoke": bool(g.is_smoke.any()), "is_meta": bool(g.is_meta.any()),
        }
    libs: dict[str, list[str]] = {}
    for lib, ver in df[["generator_library", "generator_library_version"]].dropna() \
            .drop_duplicates().itertuples(index=False):
        libs.setdefault(lib, []).append(ver)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "stores": list(stores), "run_ids_filter": run_ids, "include_meta": include_meta,
        "n_rows": int(len(df)),
        "n_reports": int(df.report_status.notna().sum()),
        "runs": runs,
        "metrics_versions": sorted({int(v) for v in df.metrics_version.dropna()}),
        "reports_without_metrics_version": int(
            (df.report_status.notna() & df.metrics_version.isna()).sum()),
        "generator_libraries": {k: sorted(v) for k, v in sorted(libs.items())},
        "budget_profiles": sorted(df.budget_profile.dropna().unique().tolist()),
        "versions": {"python": platform.python_version(),
                     **{d.replace("-", "_"): _version(d) for d in
                        ("stl-analysis", "stl-core", "stl-runner", "pandas", "numpy",
                         "scipy", "matplotlib", "autorank")}},
    }
