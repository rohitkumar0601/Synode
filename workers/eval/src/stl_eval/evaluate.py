"""Compute a full MetricReport for one generated synthetic table.

Metrics computed here (eval env): SDMetrics quality, TSTR utility, DCR share, Anonymeter,
exact-match and QI-uniqueness safeguards. synthcity's detection AUC, alpha-precision /
beta-recall / authenticity and DOMIAS membership inference are computed by the `metrics`
stage in the synthcity env (stl_synthcity.metrics) and merged from synthcity_metrics.json;
a nearest-neighbour distance membership attack runs here, and privacy.mia_advantage is the
max of the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from stl_core.metadata import TableMetadata
from stl_core.schemas import (
    CostMetrics,
    FidelityMetrics,
    JobSpec,
    MetricReport,
    PrivacyMetrics,
    UtilityMetrics,
)
from stl_core.split import Splits
from stl_runner import keys
from stl_runner.pipeline import ensure_splits, job_state, load_metadata, set_state
from stl_runner.storage import ArtifactStore

from stl_eval.fidelity import quality
from stl_eval.preprocessing import align, unseen_category_shares
from stl_eval.privacy import (
    ANONYMETER_KEYS,
    anonymeter_risks,
    dcr_share,
    exact_match_rates,
    mia_distance_advantage,
    qi_unique_share,
    resolve_secrets,
)
from stl_eval.utility import tstr

UNSEEN_CATEGORY_WARN = 0.10
# Metric-set version written into every MetricReport (2: synthcity metrics + distance MIA,
# scaled exact-match excess); `report` warns when a run mixes versions.
METRICS_VERSION = 2
SYNTHCITY_MISSING = "synthcity metrics missing: run `metrics` stage"
# synthcity_metrics.json key -> (report section, field)
SYNTHCITY_FIELDS = {
    "detection_auc": ("fidelity", "detection_auc"),
    "alpha_precision": ("fidelity", "alpha_precision"),
    "beta_recall": ("fidelity", "beta_recall"),
    "authenticity": ("privacy", "authenticity"),
    "mia_auc": ("privacy", "mia_auc"),
    "mia_domias_advantage": ("privacy", "mia_domias_advantage"),
}


@dataclass
class EvalConfig:
    n_attacks: int = 500
    singling_out_mode: str = "multivariate"
    max_rows_knn: int = 5000
    inference_secrets: str | list[str] = "all"
    min_attacks: int = 100  # fewer effective Anonymeter attacks -> risk None (inconclusive)
    family_alpha: float = 0.05  # Bonferroni family error rate for the gate's lower bounds

    def __post_init__(self) -> None:
        s = self.inference_secrets
        if not (isinstance(s, str) and s) and not (
                isinstance(s, (list, tuple)) and s and all(isinstance(c, str) for c in s)):
            raise ValueError("eval.inference_secrets must be 'all', 'target', a column name "
                             f"or a non-empty list of column names; got {s!r}")
        if self.singling_out_mode not in ("multivariate", "univariate"):
            raise ValueError("eval.singling_out_mode must be 'multivariate' or 'univariate'; "
                             f"got {self.singling_out_mode!r}")
        if not self.min_attacks >= 1:
            raise ValueError(f"eval.min_attacks must be >= 1; got {self.min_attacks}")
        if self.n_attacks < self.min_attacks:
            raise ValueError(f"eval.n_attacks ({self.n_attacks}) is below eval.min_attacks "
                             f"({self.min_attacks}): every Anonymeter risk would be "
                             "inconclusive; raise n_attacks or lower min_attacks")
        if not 0 < self.family_alpha < 1:
            raise ValueError(f"eval.family_alpha must be in (0, 1); got {self.family_alpha}")

    def check(self, metadata: TableMetadata) -> None:
        """Raise ValueError if the config does not fit the dataset (e.g. unknown secrets)."""
        resolve_secrets(self.inference_secrets, metadata)


def _base(job: JobSpec) -> dict:
    return dict(job_id=job.job_id, run_id=job.run_id, dataset_id=job.dataset_id,
                generator=job.generator, seed=job.seed)


def _merge_synthcity(sc: dict | None, fid: FidelityMetrics, priv: PrivacyMetrics,
                     warnings: list[str]) -> None:
    """Fill the synthcity fields from the `metrics` stage output. Missing file or a whole-job
    error leaves them None (mia_advantage is a PRIVACY_KEY: the gate then fails closed)."""
    if sc is None or "error" in sc:
        reason = f" (error: {sc['error']})" if sc is not None else ""
        warnings.append(SYNTHCITY_MISSING + reason)
        return
    for name, error in (sc.get("errors") or {}).items():
        warnings.append(f"synthcity metric {name} failed: {error}")
    sections = {"fidelity": fid, "privacy": priv}
    for key, (section, field) in SYNTHCITY_FIELDS.items():
        setattr(sections[section], field, sc.get(key))
    if "mia_domias_advantage" not in sc:  # metrics_version 1 files
        priv.mia_domias_advantage = sc.get("mia_advantage")


def evaluate(job: JobSpec, splits: Splits, synth: pd.DataFrame, metadata: TableMetadata,
             cost: CostMetrics | None, config: EvalConfig | None = None,
             synthcity_metrics: dict | None = None) -> MetricReport:
    """``synthcity_metrics``: the job's synthcity_metrics.json content (None if missing)."""
    config = config or EvalConfig()
    cols = metadata.model_columns
    train, control, test = (p[cols] for p in (splits.train, splits.control, splits.test))
    try:
        synth = align(train, synth, metadata)
    except Exception as exc:
        return MetricReport(**_base(job), status="failed", cost=cost,
                            error=f"synthetic data rejected: {exc}")
    warnings: list[str] = [
        f"column {col!r}: {share:.0%} of synthetic categories unseen in real data"
        for col, share in unseen_category_shares(train, synth, metadata).items()
        if share > UNSEEN_CATEGORY_WARN
    ]

    def attempt(label, fn):
        try:
            return fn()
        except Exception as exc:
            warnings.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    seed, k = job.seed, config.max_rows_knn
    fid = FidelityMetrics()
    q = attempt("quality", lambda: quality(train, synth, metadata))
    if q:
        fid.quality_score, fid.column_shapes, fid.column_pair_trends = q
    priv = PrivacyMetrics()
    _merge_synthcity(synthcity_metrics, fid, priv, warnings)
    if fid.detection_auc is not None and fid.detection_auc < 0.4:
        warnings.append(f"detection_auc {fid.detection_auc:.3f} < 0.4: synthetic rows likely "
                        "duplicate real rows (memorisation signal)")
    util = attempt("utility", lambda: tstr(train, synth, test, metadata, seed)) or UtilityMetrics()
    priv.dcr_share = attempt("dcr", lambda: dcr_share(train, control, synth, metadata, seed, k))
    details: dict = {}
    try:
        risks, anon_warnings, by_secret = anonymeter_risks(
            train, control, synth, metadata, seed, config.n_attacks, config.singling_out_mode,
            config.inference_secrets, config.min_attacks, config.family_alpha, details=details)
    except Exception as exc:
        risks, anon_warnings, by_secret = {}, [f"anonymeter: {type(exc).__name__}: {exc}"], {}
        details = {"singling_out_status": "error"}  # fails closed in the gate
    warnings.extend(anon_warnings)
    for key in ANONYMETER_KEYS:
        setattr(priv, key, risks.get(key))
        setattr(priv, f"{key}_lower", risks.get(f"{key}_lower"))
    priv.inference_by_secret = by_secret
    for key, value in details.items():
        setattr(priv, key, value)
    matches = attempt("exact_match",
                      lambda: exact_match_rates(train, control, synth, metadata, seed))
    for key, value in (matches or {}).items():
        setattr(priv, key, value)
    priv.qi_unique_share = attempt("qi_unique", lambda: qi_unique_share(synth, metadata, seed))
    priv.mia_distance_advantage = attempt(
        "mia_distance", lambda: mia_distance_advantage(train, control, synth, metadata, seed))
    # The gated MIA metric: the stronger of the two attacks; None only if both are missing.
    attacks = [v for v in (priv.mia_domias_advantage, priv.mia_distance_advantage)
               if v is not None]
    priv.mia_advantage = max(attacks) if attacks else None
    return MetricReport(**_base(job), status="done", metrics_version=METRICS_VERSION,
                        warnings=warnings, cost=cost,
                        fidelity=fid, utility=util, privacy=priv)


def _config_error(config: EvalConfig, metadata: TableMetadata) -> str | None:
    try:
        config.check(metadata)
    except ValueError as exc:
        return f"invalid eval config: {exc}"
    return None


def fail_job(store: ArtifactStore, job: JobSpec, error: str) -> MetricReport:
    """Write a failed report for a job and mark it failed."""
    report = MetricReport(**_base(job), status="failed", error=error)
    store.write_json(keys.report(job.run_id, job.job_id), report)
    set_state(store, job, "failed", error)
    return report


def evaluate_stored_job(store: ArtifactStore, job: JobSpec,
                        config: EvalConfig | None = None) -> MetricReport:
    """Evaluate a generated job and store its report.

    Idempotent: a job already "done" with a stored report returns that report unchanged.
    A job left in "evaluating" (process killed mid-evaluation) is evaluated again. On an
    interrupt (KeyboardInterrupt etc.) the job is marked failed/"interrupted", no report
    is written, and the interrupt propagates.
    """
    state = job_state(store, job)
    report_key = keys.report(job.run_id, job.job_id)
    if state["state"] == "done" and store.exists(report_key):
        return MetricReport.model_validate(store.read_json(report_key))
    if state["state"] not in ("generated", "evaluating"):
        error = state.get("error") or f"job not generated (state {state['state']!r})"
        report = MetricReport(**_base(job), status="failed", error=error)
    else:
        set_state(store, job, "evaluating")
        try:
            metadata = load_metadata(store, job.dataset_id)
            config = config or EvalConfig()
            config_error = _config_error(config, metadata)
            if config_error is not None:
                report = MetricReport(**_base(job), status="failed", error=config_error)
            else:
                splits = ensure_splits(store, job.dataset_id, job.seed)
                synth = store.read_df(keys.synthetic(job.run_id, job.job_id))
                cost = CostMetrics.model_validate(
                    store.read_json(keys.cost(job.run_id, job.job_id)))
                sc_key = keys.synthcity_metrics(job.run_id, job.job_id)
                sc = store.read_json(sc_key) if store.exists(sc_key) else None
                report = evaluate(job, splits, synth, metadata, cost, config, sc)
        except Exception as exc:
            report = MetricReport(**_base(job), status="failed",
                                  error=f"evaluation crashed: {type(exc).__name__}: {exc}")
        except BaseException:
            set_state(store, job, "failed", "interrupted")
            raise
    store.write_json(report_key, report)
    set_state(store, job, report.status, report.error)
    return report
