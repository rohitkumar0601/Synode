"""Pydantic schemas shared by workers, CLI, API and frontend."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def finite_or_none(v: Any) -> Any:
    """Map non-finite numbers (nan, inf) to None; leave everything else unchanged."""
    if v is None or isinstance(v, (str, bool)):
        return v
    try:
        return v if math.isfinite(float(v)) else None
    except (TypeError, ValueError):
        return v


class _Metrics(BaseModel):
    """Metric containers: non-finite floats are stored as None (missing), on init and on
    assignment, so in-memory values agree with their JSON form."""

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("*", mode="before")
    @classmethod
    def _finite(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: finite_or_none(x) for k, x in v.items()}
        return finite_or_none(v)


class JobSpec(BaseModel):
    job_id: str
    run_id: str
    dataset_id: str
    generator: str
    seed: int
    params: dict[str, Any] = Field(default_factory=dict)
    time_cap_seconds: float | None = None


class BundleManifest(BaseModel):
    """Index of a results bundle exported from one run (see stl_runner.bundle)."""

    format_version: int = 1
    run_id: str
    dataset_id: str
    dataset_fingerprint: str
    created_at: str  # ISO 8601, UTC
    jobs: list[JobSpec]
    include_synthetic: bool


class CostMetrics(_Metrics):
    fit_seconds: float
    sample_seconds: float
    peak_memory_mb: float | None = None
    baseline_memory_mb: float | None = None  # process RSS before fit


class FidelityMetrics(_Metrics):
    quality_score: float | None = None
    column_shapes: float | None = None
    column_pair_trends: float | None = None
    detection_auc: float | None = None
    alpha_precision: float | None = None
    beta_recall: float | None = None


class UtilityMetrics(_Metrics):
    tstr: dict[str, float | None] = Field(default_factory=dict)
    real_scores: dict[str, float | None] = Field(default_factory=dict)
    importance_spearman: float | None = None


class PrivacyMetrics(_Metrics):
    dcr_share: float | None = None
    singling_out_risk: float | None = None
    linkability_risk: float | None = None
    inference_risk: float | None = None
    # Two membership-inference attacks (members = a TRAIN sample, non-members = CONTROL),
    # each advantage = max(0, 2 * (AUC - 0.5)): synthcity DomiasMIAKDE (primary, from
    # synthcity_metrics.json; mia_auc is its AUC) and a nearest-neighbour distance attack
    # (eval worker). mia_advantage (a PRIVACY_KEY) = max of the available two, None only if
    # both are None. mia_auc, the per-attack values and synthcity AlphaPrecision's
    # authenticity are reported only (not PRIVACY_KEYS, not gated).
    mia_advantage: float | None = None
    mia_domias_advantage: float | None = None
    mia_distance_advantage: float | None = None
    mia_auc: float | None = None
    authenticity: float | None = None
    # Bonferroni-corrected lower confidence bounds of the Anonymeter risks: the gate
    # compares these (when present); the privacy axis score uses the point estimates.
    singling_out_risk_lower: float | None = None
    linkability_risk_lower: float | None = None
    inference_risk_lower: float | None = None
    # Per-secret Anonymeter inference risks (inference_risk is their max). Detail only:
    # excluded from PRIVACY_KEYS and MetricReport.flat(), so the gate and scores ignore it.
    inference_by_secret: dict[str, float | None] = Field(default_factory=dict)
    # Singling-out outcome (detail, not in flat()): "measured" when a mode was conclusive,
    # "not_attackable" when both modes generated fewer than min_attacks queries, "error"
    # when the attack crashed; None in older reports or when Anonymeter could not run.
    singling_out_status: Literal["measured", "not_attackable", "error"] | None = None
    # Mode behind the outcome: the conclusive mode, or "both" when both were tried and
    # neither was conclusive (not_attackable).
    singling_out_mode_used: Literal["multivariate", "univariate", "both"] | None = None
    singling_out_queries: dict[str, int | None] = Field(default_factory=dict)  # per mode
    # Safeguards of the "not attackable" singling-out rule (not PRIVACY_KEYS). Shares of
    # synthetic rows (model columns) equal to some row of equal-size TRAIN / CONTROL
    # references, their per-seed difference, that difference scaled by N_train / n_sample
    # (the gated value: a TRAIN sample of n_sample rows only catches n_sample / N_train of
    # the copied TRAIN rows), the rate against the full TRAIN split (reported only), and
    # the mean share of synthetic rows unique on random 3-column subsets of the features.
    exact_match_train: float | None = None
    exact_match_control: float | None = None
    exact_match_excess: float | None = None
    exact_match_excess_scaled: float | None = None
    exact_match_train_full: float | None = None
    qi_unique_share: float | None = None


_PRIVACY_DETAIL = {"inference_by_secret", "singling_out_status", "singling_out_mode_used",
                   "singling_out_queries"}
# The metrics every report must have (a missing one fails the gate); fixed on purpose so
# that adding detail fields such as the *_lower bounds does not make them mandatory.
PRIVACY_KEYS: tuple[str, ...] = (
    "privacy.dcr_share",
    "privacy.singling_out_risk",
    "privacy.linkability_risk",
    "privacy.inference_risk",
    "privacy.mia_advantage",
)
# Keys the gate thresholds; aggregation also keeps their worst (max) value over seeds.
WORST_KEYS: tuple[str, ...] = (
    "privacy.singling_out_risk_lower",
    "privacy.linkability_risk_lower",
    "privacy.inference_risk_lower",
    "privacy.singling_out_risk",
    "privacy.linkability_risk",
    "privacy.inference_risk",
    "privacy.dcr_share",
    "privacy.exact_match_train",
    "privacy.exact_match_control",
    "privacy.exact_match_excess",
    "privacy.exact_match_excess_scaled",
    "privacy.qi_unique_share",
)


class MetricReport(BaseModel):
    job_id: str
    run_id: str
    dataset_id: str
    generator: str
    seed: int
    status: Literal["done", "failed"]
    # Version of the metric set behind this report (stl_eval.evaluate.METRICS_VERSION);
    # None in reports written before versioning. Not a metric: kept out of flat().
    metrics_version: int | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    cost: CostMetrics | None = None
    fidelity: FidelityMetrics = Field(default_factory=FidelityMetrics)
    utility: UtilityMetrics = Field(default_factory=UtilityMetrics)
    privacy: PrivacyMetrics = Field(default_factory=PrivacyMetrics)
    # The hyperparameters the generator's library was called with (runs/<run>/<job>/
    # generator_params.json: library, library_version, params = signature defaults merged
    # with what we passed, overrides, profile). None in reports written before S2b and for
    # jobs that never got that far. Not a metric: kept out of flat().
    generator_params: dict[str, Any] | None = None

    def flat(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        out.update({f"fidelity.{k}": v for k, v in self.fidelity.model_dump().items()})
        out.update({f"utility.tstr.{k}": v for k, v in self.utility.tstr.items()})
        out["utility.importance_spearman"] = self.utility.importance_spearman
        out.update({f"privacy.{k}": v for k, v in
                    self.privacy.model_dump(exclude=_PRIVACY_DETAIL).items()})
        if self.cost is not None:
            out.update({f"cost.{k}": v for k, v in self.cost.model_dump().items()})
        return out


class AxisScores(BaseModel):
    fidelity: float | None = None
    utility: float | None = None
    privacy: float | None = None


class AggregatedReport(BaseModel):
    generator: str
    n_seeds: int
    n_failed: int
    mean: dict[str, float | None]
    std: dict[str, float | None]
    missing: list[str] = Field(default_factory=list)
    # Per-key max over successful seeds for WORST_KEYS: the gate must not average lower
    # bounds across seeds. Empty in reports written before it existed (gate uses mean).
    worst: dict[str, float | None] = Field(default_factory=dict)
    # privacy.singling_out_status of each successful seed (empty in older reports).
    singling_out_status: list[str | None] = Field(default_factory=list)


class RankedCandidate(BaseModel):
    generator: str
    scores: AxisScores
    score: float | None
    gate_passed: bool
    gate_failures: list[str]
    on_pareto_front: bool
    notes: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    preset: str
    weights: tuple[float, float, float]
    winner: str | None
    reason: str
    ranking: list[RankedCandidate]
