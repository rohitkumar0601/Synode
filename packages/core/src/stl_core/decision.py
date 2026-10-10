"""Privacy-gated decision engine: gate, Pareto front, weighted score, reason."""

from __future__ import annotations

import math
from dataclasses import dataclass

from stl_core.schemas import (
    PRIVACY_KEYS,
    AggregatedReport,
    AxisScores,
    RankedCandidate,
    Recommendation,
)

AXES = ("fidelity", "utility", "privacy")
PRESETS: dict[str, tuple[float, float, float]] = {
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "privacy_first": (0.2, 0.2, 0.6),
    "utility_first": (0.2, 0.6, 0.2),
}
ANONYMETER_CHECKS = {
    "privacy.singling_out_risk": "singling-out",
    "privacy.linkability_risk": "linkability",
    "privacy.inference_risk": "inference",
}
METRIC_LABELS = {
    "privacy.dcr_share": "DCR share",
    "privacy.singling_out_risk": "singling-out risk",
    "privacy.linkability_risk": "linkability risk",
    "privacy.inference_risk": "inference risk",
    "privacy.mia_advantage": "membership-inference advantage",
}


NOT_ATTACKABLE_NOTE = "singling-out not attackable (safeguards passed)"
SINGLING_OUT = "privacy.singling_out_risk"


@dataclass(frozen=True)
class GateConfig:
    max_anonymeter_risk: float = 0.10
    max_dcr_share: float = 0.6
    # Safeguards when singling-out is not attackable in every seed (see check_gate).
    exact_match_margin: float = 0.01
    max_qi_unique_share: float = 0.05


def singling_out_not_attackable(agg: AggregatedReport) -> bool:
    """True iff every successful seed reported singling-out status "not_attackable"."""
    statuses = agg.singling_out_status
    return (bool(statuses) and len(statuses) == agg.n_seeds - agg.n_failed
            and all(s == "not_attackable" for s in statuses))


def _finite(x: float | None) -> float | None:
    """None for missing or non-finite values (NaN must never count as a score)."""
    return x if x is not None and math.isfinite(x) else None


def _clip01(x: float) -> float:
    # max(0, nan) would return 0: propagate NaN instead; callers pass finite values only.
    x = float(x)
    return x if math.isnan(x) else min(1.0, max(0.0, x))


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if _finite(v) is not None]
    return sum(present) / len(present) if present else None


def axis_scores(flat: dict[str, float | None],
                singling_out_not_attackable: bool = False) -> AxisScores:
    """Axis scores from mean metrics. With ``singling_out_not_attackable`` the privacy axis
    leaves singling-out out (anon = max(linkability, inference)); it is never scored as 0."""
    def get(key: str) -> float | None:
        return _finite(flat.get(key))

    det = get("fidelity.detection_auc")
    fidelity = _mean([
        get("fidelity.quality_score"),
        # SDMetrics convention: AUC below 0.5 is not penalised (it signals near-copies,
        # which the privacy metrics catch); only distinguishability above chance costs.
        None if det is None else 1 - 2 * max(0.0, det - 0.5),
        get("fidelity.alpha_precision"),
        get("fidelity.beta_recall"),
    ])
    tstr = [_clip01(v) for k in flat if k.startswith("utility.tstr.")
            if (v := get(k)) is not None]
    t = sum(tstr) / len(tstr) if tstr else None
    imp = get("utility.importance_spearman")
    imp = None if imp is None else _clip01(imp)
    if t is not None and imp is not None:
        utility = 0.5 * t + 0.5 * imp
    else:
        utility = t if t is not None else imp
    privacy = None
    skip = {SINGLING_OUT} if singling_out_not_attackable else set()
    if all(get(k) is not None for k in PRIVACY_KEYS if k not in skip):
        anon = max(get(k) for k in ANONYMETER_CHECKS if k not in skip)
        privacy = (
            (1 - _clip01(anon))
            + (1 - 2 * max(0.0, get("privacy.dcr_share") - 0.5))
            + (1 - _clip01(get("privacy.mia_advantage")))
        ) / 3
    return AxisScores(
        fidelity=None if fidelity is None else _clip01(fidelity),
        utility=None if utility is None else _clip01(utility),
        privacy=None if privacy is None else _clip01(privacy),
    )


def check_gate(agg: AggregatedReport, gate: GateConfig) -> list[str]:
    if agg.n_failed >= agg.n_seeds:
        return ["had every seed fail"]
    failures = []
    t = gate.max_anonymeter_risk

    def gated(key: str) -> float | None:
        # Worst (max) value over seeds: one leaky seed must not be averaged away.
        # Reports aggregated before `worst` existed fall back to the mean.
        return _finite(agg.worst[key] if key in agg.worst else agg.mean.get(key))

    not_attackable = singling_out_not_attackable(agg)
    for key, label in ANONYMETER_CHECKS.items():
        if key == SINGLING_OUT and not_attackable:
            failed = _not_attackable_safeguards(agg, gate, gated)
            if failed:
                failures.append("singling-out not attackable but safeguards failed "
                                f"({'; '.join(failed)})")
            continue
        # Gate on the lower confidence bound when the report has one: fail only on
        # statistical evidence of risk, not on attack noise around the point estimate.
        lower = gated(f"{key}_lower")
        if lower is not None:
            if lower > t:
                failures.append(f"failed the {label} check (lower bound {lower:.2f} > {t:.2f})")
            continue
        v = gated(key)
        if v is not None and v > t:
            failures.append(f"failed the {label} check (risk {v:.2f} > {t:.2f})")
    dcr = gated("privacy.dcr_share")
    if dcr is not None and dcr > gate.max_dcr_share:
        failures.append(f"failed the DCR check (share {dcr:.2f} > {gate.max_dcr_share:.2f})")
    for k in PRIVACY_KEYS:
        if k == SINGLING_OUT and not_attackable:
            continue  # judged by the safeguards above (which fail closed on missing values)
        v = agg.mean.get(k)
        if v is None or k in agg.missing or (isinstance(v, float) and math.isnan(v)):
            failures.append(f"is missing the {METRIC_LABELS[k]} metric")
    return failures


def _not_attackable_safeguards(agg: AggregatedReport, gate: GateConfig, gated) -> list[str]:
    """Which safeguards fail when singling-out was not attackable in every seed.

    The singling-out check then passes only if linkability and inference are conclusive in
    every seed and pass on their lower bounds, DCR passes, in every seed synthetic rows copy
    TRAIN no more often than CONTROL (+ margin; the excess measured against an equal-size
    TRAIN sample is scaled by N_train / n_sample), and few synthetic
    rows are unique on 3-column feature subsets. Values are
    the worst (max) over seeds; a value missing in any seed fails its safeguard.
    """
    def value(key: str) -> float | None:
        return None if key in agg.missing else gated(key)

    t = gate.max_anonymeter_risk
    failed = []
    for name in ("linkability", "inference"):
        lower = value(f"privacy.{name}_risk_lower")
        if lower is None:
            failed.append(f"{name} inconclusive")
        elif lower > t:
            failed.append(f"{name} lower bound {lower:.2f} > {t:.2f}")
    dcr = value("privacy.dcr_share")
    if dcr is None:
        failed.append("DCR share missing")
    elif dcr > gate.max_dcr_share:
        failed.append(f"DCR share {dcr:.2f} > {gate.max_dcr_share:.2f}")
    # Per-seed excess (train rate - control rate) scaled by N_train / n_sample, worst over
    # seeds: comparing the worst train rate with the worst control rate could pair values
    # from different seeds, and the unscaled excess is diluted by the sampling fraction
    # (copying 3% of TRAIN shows as ~1% against a third-size TRAIN sample).
    excess = value("privacy.exact_match_excess_scaled")
    if excess is None:
        failed.append("exact-match excess missing")
    elif excess > gate.exact_match_margin:
        failed.append(f"exact-match excess over control, scaled to TRAIN, {excess:.3f} > "
                      f"{gate.exact_match_margin:.2f}")
    qi = value("privacy.qi_unique_share")
    if qi is None:
        failed.append("QI-unique share missing")
    elif qi > gate.max_qi_unique_share:
        failed.append(f"QI-unique share {qi:.3f} > {gate.max_qi_unique_share:.2f}")
    return failed


def pareto_front(points: dict[str, tuple[float, ...]]) -> set[str]:
    front = set()
    for a, pa in points.items():
        dominated = any(
            all(qb >= qa for qa, qb in zip(pa, pb, strict=False))
            and any(qb > qa for qa, qb in zip(pa, pb, strict=False))
            for b, pb in points.items()
            if b != a
        )
        if not dominated:
            front.add(a)
    return front


def weighted_score(scores: AxisScores, weights: tuple[float, float, float]) -> float | None:
    values = [getattr(scores, a) for a in AXES]
    if any(v is None for v in values):
        return None
    total = sum(weights)
    if not total > 0:
        return None
    return sum(w * v for w, v in zip(weights, values, strict=False)) / total


def resolve_weights(
    preset: str = "balanced", weights: tuple[float, float, float] | None = None
) -> tuple[str, tuple[float, float, float]]:
    if weights is not None:
        values = tuple(float(w) for w in weights)
        if (len(values) != 3 or not all(math.isfinite(w) and w >= 0 for w in values)
                or sum(values) <= 0):
            raise ValueError(
                "custom weights must be exactly 3 finite non-negative numbers "
                f"(fidelity, utility, privacy) with a positive sum; got {tuple(weights)}"
            )
        total = sum(values)
        return "custom", tuple(w / total for w in values)  # type: ignore[return-value]
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return preset, PRESETS[preset]


def _reason(winner, passing, ranking, label) -> str:
    if winner is None:
        parts = [
            f"{c.generator} {c.gate_failures[0] if c.gate_failures else 'has incomplete scores'}"
            for c in ranking
        ]
        any_passed_gate = any(c.gate_passed for c in ranking)
        prefix = (
            "No generator passed the privacy gate with complete scores. "
            if any_passed_gate
            else "No generator passed the privacy gate. "
        )
        return prefix + "; ".join(parts) + "."
    best_axis = max(AXES, key=lambda a: getattr(winner.scores, a))
    text = (
        f"{winner.generator}: highest {label} score ({winner.score:.2f}) among models passing "
        f"the privacy gate; strongest on {best_axis} ({getattr(winner.scores, best_axis):.2f})."
    )
    for note in winner.notes:
        text += f" Note: {note}."
    if len(passing) > 1:
        text += f" Runner-up: {passing[1].generator} ({passing[1].score:.2f})."
    for c in ranking:
        if c.gate_passed:
            continue
        better = [
            a for a in ("fidelity", "utility")
            if getattr(c.scores, a) is not None
            and getattr(c.scores, a) > getattr(winner.scores, a)
        ]
        if better:
            text += f" {c.generator} had higher {better[0]} but {c.gate_failures[0]}."
    return text


def recommend(
    aggs: list[AggregatedReport],
    preset: str = "balanced",
    weights: tuple[float, float, float] | None = None,
    gate: GateConfig | None = None,
) -> Recommendation:
    if gate is None:
        gate = GateConfig()
    label, w = resolve_weights(preset, weights)
    candidates = []
    for agg in aggs:
        not_attackable = singling_out_not_attackable(agg)
        scores = axis_scores(agg.mean, singling_out_not_attackable=not_attackable)
        failures = check_gate(agg, gate)
        notes = [NOT_ATTACKABLE_NOTE] if not_attackable and not failures else []
        candidates.append(dict(generator=agg.generator, scores=scores,
                               score=weighted_score(scores, w), gate_failures=failures,
                               gate_passed=not failures, notes=notes))
    eligible = {
        c["generator"]: tuple(getattr(c["scores"], a) for a in AXES)
        for c in candidates
        if c["gate_passed"] and c["score"] is not None
    }
    front = pareto_front(eligible)
    ranking = sorted(
        (RankedCandidate(**c, on_pareto_front=c["generator"] in front) for c in candidates),
        key=lambda c: (not c.gate_passed, c.score is None, -(c.score or 0.0), c.generator),
    )
    passing = [c for c in ranking if c.gate_passed and c.score is not None]
    winner = passing[0] if passing else None
    return Recommendation(preset=label, weights=w, winner=winner.generator if winner else None,
                          reason=_reason(winner, passing, ranking, label), ranking=ranking)
