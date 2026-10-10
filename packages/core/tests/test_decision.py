import math

import pytest
from stl_core.decision import (
    PRESETS,
    GateConfig,
    axis_scores,
    check_gate,
    pareto_front,
    recommend,
    resolve_weights,
    weighted_score,
)
from stl_core.schemas import AggregatedReport, AxisScores

GOOD_PRIVACY = {
    "privacy.dcr_share": 0.5, "privacy.singling_out_risk": 0.0,
    "privacy.linkability_risk": 0.0, "privacy.inference_risk": 0.0,
    "privacy.mia_advantage": 0.0,
}


def _agg(gen, fid=0.8, tstr=0.9, imp=0.9, privacy=None, n_failed=0, missing=()):
    mean = {"fidelity.quality_score": fid, "fidelity.detection_auc": 0.5,
            "fidelity.alpha_precision": fid, "fidelity.beta_recall": fid,
            "utility.tstr.logreg": tstr, "utility.importance_spearman": imp,
            **GOOD_PRIVACY, **(privacy or {})}
    return AggregatedReport(generator=gen, n_seeds=3, n_failed=n_failed, mean=mean,
                            std={k: 0.0 for k in mean}, missing=list(missing))


def test_axis_scores_formulas():
    s = axis_scores(_agg("g", fid=0.8, tstr=1.4, imp=-0.2,
                         privacy={"privacy.dcr_share": 0.7,
                                  "privacy.singling_out_risk": 0.2,
                                  "privacy.mia_advantage": 0.4}).mean)
    assert math.isclose(s.fidelity, (0.8 + 1.0 + 0.8 + 0.8) / 4)
    assert math.isclose(s.utility, 0.5 * 1.0 + 0.5 * 0.0)  # both clipped
    assert math.isclose(s.privacy, ((1 - 0.2) + (1 - 0.4) + (1 - 0.4)) / 3)


def test_privacy_none_when_any_privacy_metric_missing():
    flat = dict(_agg("g").mean)
    flat["privacy.mia_advantage"] = None
    assert axis_scores(flat).privacy is None


def test_gate_failures_read_naturally():
    agg = _agg("g", privacy={"privacy.singling_out_risk": 0.34, "privacy.dcr_share": 0.71})
    fails = check_gate(agg, GateConfig())
    assert "failed the singling-out check (risk 0.34 > 0.10)" in fails
    assert "failed the DCR check (share 0.71 > 0.60)" in fails
    assert check_gate(_agg("ok"), GateConfig()) == []


def test_gate_missing_and_all_failed():
    assert check_gate(_agg("g", missing=["privacy.dcr_share"]), GateConfig()) == [
        "is missing the DCR share metric"]
    assert check_gate(_agg("g", n_failed=3), GateConfig()) == ["had every seed fail"]


def test_gate_nan_privacy_metric():
    flat = dict(_agg("g").mean)
    flat["privacy.dcr_share"] = float("nan")
    agg = AggregatedReport(generator="g", n_seeds=3, n_failed=0, mean=flat,
                           std={k: 0.0 for k in flat}, missing=[])
    assert check_gate(agg, GateConfig()) == ["is missing the DCR share metric"]


def test_pareto_front():
    pts = {"a": (1, 1, 0), "b": (0, 0, 1), "c": (0.5, 0.5, 0), "d": (1, 1, 0)}
    assert pareto_front(pts) == {"a", "b", "d"}


def test_weights():
    assert resolve_weights("privacy_first") == ("privacy_first", PRESETS["privacy_first"])
    label, w = resolve_weights(weights=(2, 1, 1))
    assert label == "custom" and math.isclose(w[0], 0.5)
    with pytest.raises(KeyError, match="unknown preset"):
        resolve_weights("nope")
    assert weighted_score(AxisScores(fidelity=1, utility=0, privacy=None), (1, 1, 1)) is None


def test_recommend_gates_then_ranks_with_reason():
    aggs = [
        _agg("tvae", fid=0.7, tstr=0.95, imp=0.9),
        _agg("ctgan", fid=0.95, tstr=0.9, imp=0.9,
             privacy={"privacy.singling_out_risk": 0.3}),
        _agg("copula", fid=0.6, tstr=0.7, imp=0.6),
    ]
    rec = recommend(aggs, preset="balanced")
    assert rec.winner == "tvae"
    assert [c.generator for c in rec.ranking if c.gate_passed] == ["tvae", "copula"]
    assert [c.generator for c in rec.ranking] == ["tvae", "copula", "ctgan"]
    assert rec.ranking[0].generator == rec.winner
    assert rec.reason.startswith("tvae: highest balanced score")
    assert "Runner-up: copula" in rec.reason
    assert "ctgan had higher fidelity but failed the singling-out check" in rec.reason
    front = {c.generator for c in rec.ranking if c.on_pareto_front}
    assert "tvae" in front and "ctgan" not in front


def test_recommend_no_winner():
    aggs = [_agg("train_copy", privacy={"privacy.dcr_share": 0.9}),
            _agg("x", missing=["privacy.inference_risk"])]
    rec = recommend(aggs)
    assert rec.winner is None
    assert rec.reason.startswith("No generator passed the privacy gate.")
    assert "train_copy failed the DCR check" in rec.reason
    assert "x is missing the inference risk metric" in rec.reason


def test_recommend_no_winner_some_passed_gate():
    """Test 'with complete scores' phrasing when gate passes but score is None."""
    # Create a candidate with good privacy but no fidelity and no utility metrics
    flat = dict(GOOD_PRIVACY)
    flat.update({
        "fidelity.quality_score": None,
        "fidelity.detection_auc": None,
        "fidelity.alpha_precision": None,
        "fidelity.beta_recall": None,
        "utility.tstr.logreg": None,
        "utility.importance_spearman": None,
    })
    incomplete = AggregatedReport(
        generator="incomplete", n_seeds=3, n_failed=0, mean=flat,
        std={k: 0.0 for k in flat}, missing=[]
    )
    aggs = [incomplete, _agg("fail_gate", privacy={"privacy.dcr_share": 0.9})]
    rec = recommend(aggs)
    assert rec.winner is None
    assert rec.reason.startswith("No generator passed the privacy gate with complete scores.")
    assert "incomplete has incomplete scores" in rec.reason


def test_axis_scores_nan_is_missing_not_zero():
    nan = float("nan")
    s = axis_scores({"fidelity.quality_score": nan})
    assert s.fidelity is None
    s = axis_scores({"fidelity.quality_score": nan, "fidelity.alpha_precision": 0.6})
    assert math.isclose(s.fidelity, 0.6)
    s = axis_scores({"utility.tstr.logreg": nan, "utility.tstr.xgboost": 0.8,
                     "utility.importance_spearman": nan})
    assert math.isclose(s.utility, 0.8)
    flat = dict(GOOD_PRIVACY)
    flat["privacy.mia_advantage"] = nan
    assert axis_scores(flat).privacy is None


def test_detection_auc_below_half_is_not_penalised():
    low = axis_scores({"fidelity.detection_auc": 0.04})
    assert math.isclose(low.fidelity, 1.0)
    high = axis_scores({"fidelity.detection_auc": 0.8})
    assert math.isclose(high.fidelity, 1 - 2 * 0.3)


@pytest.mark.parametrize("bad", [
    (1, 1), (1, 1, 1, 1), (-1, 1, 1), (0, 0, 0), (float("nan"), 1, 1),
    (float("inf"), 1, 1), (2, -1, 0),
])
def test_custom_weights_validated(bad):
    with pytest.raises(ValueError, match="weights"):
        resolve_weights(weights=bad)


def test_custom_weights_allow_zeros():
    label, w = resolve_weights(weights=(0, 0, 1))
    assert label == "custom" and w == (0.0, 0.0, 1.0)


def test_weighted_score_zero_weights_no_division():
    assert weighted_score(AxisScores(fidelity=1, utility=1, privacy=1), (0, 0, 0)) is None


def test_gate_uses_lower_bound_when_present():
    agg = _agg("g", privacy={"privacy.inference_risk": 0.25,
                             "privacy.inference_risk_lower": 0.04})
    assert check_gate(agg, GateConfig()) == []
    agg = _agg("g", privacy={"privacy.inference_risk": 0.25,
                             "privacy.inference_risk_lower": 0.18})
    assert check_gate(agg, GateConfig()) == [
        "failed the inference check (lower bound 0.18 > 0.10)"]


def test_gate_failures_list_risks_before_missing():
    agg = _agg("g", privacy={"privacy.singling_out_risk": 0.5}, missing=["privacy.dcr_share"])
    fails = check_gate(agg, GateConfig())
    assert fails[0].startswith("failed the singling-out check")
    assert fails[-1] == "is missing the DCR share metric"


def test_gate_uses_worst_seed_lower_bound_not_mean():
    from stl_core.aggregate import aggregate
    from stl_core.schemas import FidelityMetrics, MetricReport, PrivacyMetrics, UtilityMetrics

    def rep(seed, lower):
        return MetricReport(job_id=f"g{seed}", run_id="r", dataset_id="d", generator="g",
                            seed=seed, status="done", fidelity=FidelityMetrics(),
                            utility=UtilityMetrics(),
                            privacy=PrivacyMetrics(dcr_share=0.5, singling_out_risk=lower,
                                                   linkability_risk=0.0, inference_risk=0.0,
                                                   mia_advantage=0.0,
                                                   singling_out_risk_lower=lower))

    agg = aggregate([rep(0, 0.25), rep(1, 0.02), rep(2, 0.02)])[0]
    assert agg.mean["privacy.singling_out_risk_lower"] < 0.10  # the mean would pass
    assert check_gate(agg, GateConfig()) == [
        "failed the singling-out check (lower bound 0.25 > 0.10)"]


def test_gate_falls_back_to_mean_without_worst():
    agg = _agg("g", privacy={"privacy.dcr_share": 0.7})
    assert agg.worst == {}
    assert check_gate(agg, GateConfig()) == ["failed the DCR check (share 0.70 > 0.60)"]


# --- singling-out "not attackable" rule (follow-up 2, C2 ruling) ---

SAFE = {"privacy.linkability_risk_lower": 0.0, "privacy.inference_risk_lower": 0.02,
        "privacy.exact_match_train": 0.01, "privacy.exact_match_control": 0.01,
        "privacy.exact_match_excess": 0.0, "privacy.exact_match_excess_scaled": 0.0,
        "privacy.qi_unique_share": 0.01}
NOTE = "singling-out not attackable (safeguards passed)"


def _na_agg(gen="g", statuses=("not_attackable",) * 3, override=None, missing=(),
            worst=None, fid=0.8):
    agg = _agg(gen, fid=fid, privacy={"privacy.singling_out_risk": None,
                                      "privacy.singling_out_risk_lower": None,
                                      **SAFE, **(override or {})},
               missing=["privacy.singling_out_risk", "privacy.singling_out_risk_lower",
                        *missing])
    keys = ("privacy.linkability_risk_lower", "privacy.inference_risk_lower",
            "privacy.dcr_share", "privacy.exact_match_excess_scaled",
            "privacy.qi_unique_share")
    agg.worst = {k: agg.mean.get(k) for k in keys} | (worst or {})
    agg.singling_out_status = list(statuses)
    return agg


def test_not_attackable_passes_when_safeguards_pass():
    agg = _na_agg()
    assert check_gate(agg, GateConfig()) == []


@pytest.mark.parametrize("override, which", [
    ({"privacy.linkability_risk_lower": None}, "linkability inconclusive"),
    ({"privacy.linkability_risk_lower": 0.2}, "linkability lower bound 0.20 > 0.10"),
    ({"privacy.inference_risk_lower": None}, "inference inconclusive"),
    ({"privacy.inference_risk_lower": 0.15}, "inference lower bound 0.15 > 0.10"),
    ({"privacy.dcr_share": 0.7}, "DCR share 0.70 > 0.60"),
    ({"privacy.dcr_share": None}, "DCR share missing"),
    ({"privacy.exact_match_excess_scaled": 0.02},
     "exact-match excess over control, scaled to TRAIN, 0.020 > 0.01"),
    ({"privacy.exact_match_excess_scaled": None}, "exact-match excess missing"),
    ({"privacy.qi_unique_share": 0.06}, "QI-unique share 0.060 > 0.05"),
    ({"privacy.qi_unique_share": None}, "QI-unique share missing"),
])
def test_not_attackable_fails_when_a_safeguard_fails(override, which):
    fails = check_gate(_na_agg(override=override), GateConfig())
    assert f"singling-out not attackable but safeguards failed ({which})" in fails
    assert "is missing the singling-out risk metric" not in fails


def test_not_attackable_safeguard_uses_worst_seed_and_missing():
    # one seed's linkability lower bound missing -> inconclusive even if the worst exists
    fails = check_gate(_na_agg(missing=["privacy.linkability_risk_lower"]), GateConfig())
    assert ("singling-out not attackable but safeguards failed (linkability inconclusive)"
            in fails)
    fails = check_gate(_na_agg(worst={"privacy.qi_unique_share": 0.2}), GateConfig())
    assert ("singling-out not attackable but safeguards failed (QI-unique share 0.200 > 0.05)"
            in fails)


@pytest.mark.parametrize("statuses", [
    ("not_attackable", "measured", "not_attackable"),
    ("not_attackable", "error", "not_attackable"),
    ("not_attackable", None, "not_attackable"),
    ("not_attackable", "not_attackable"),  # one seed's report has no status (n_seeds=3)
    (),
])
def test_not_attackable_needs_every_successful_seed(statuses):
    fails = check_gate(_na_agg(statuses=statuses), GateConfig())
    assert "is missing the singling-out risk metric" in fails


def test_not_attackable_ignores_failed_seeds():
    agg = _na_agg(statuses=("not_attackable", "not_attackable"))
    agg.n_failed = 1
    assert check_gate(agg, GateConfig()) == []


def test_not_attackable_axis_privacy_without_singling_out():
    agg = _na_agg(override={"privacy.linkability_risk": 0.03, "privacy.inference_risk": 0.08,
                            "privacy.dcr_share": 0.55, "privacy.mia_advantage": 0.1})
    assert axis_scores(agg.mean).privacy is None  # unflagged: singling-out missing
    s = axis_scores(agg.mean, singling_out_not_attackable=True)
    assert math.isclose(s.privacy, ((1 - 0.08) + (1 - 0.1) + (1 - 0.1)) / 3)


def test_recommend_notes_not_attackable_winner():
    rec = recommend([_na_agg("mst", fid=0.95), _agg("copula", fid=0.6)])
    assert rec.winner == "mst"
    top = rec.ranking[0]
    assert top.gate_passed and top.scores.privacy is not None
    assert top.notes == [NOTE]
    assert rec.ranking[1].notes == []
    assert NOTE in rec.reason


def test_recommend_no_note_when_safeguards_fail():
    rec = recommend([_na_agg("mst", override={"privacy.qi_unique_share": 0.5}),
                     _agg("copula", fid=0.6)])
    assert rec.winner == "copula"
    mst = next(c for c in rec.ranking if c.generator == "mst")
    assert not mst.gate_passed and mst.notes == []
    assert NOTE not in rec.reason


def test_not_attackable_exact_match_excess_is_per_seed_worst():
    # Reviewer's example: worst train 0.05 <= worst control 0.05 + 0.01 would pass, but seed B
    # copies TRAIN 0.04 more often than CONTROL -> per-seed excess must fail the gate.
    from stl_core.aggregate import aggregate
    from stl_core.schemas import MetricReport, PrivacyMetrics

    def rep(seed, em_train, em_control):
        return MetricReport(
            job_id=f"g{seed}", run_id="r", dataset_id="d", generator="g", seed=seed,
            status="done", privacy=PrivacyMetrics(
                dcr_share=0.5, linkability_risk=0.0, inference_risk=0.0, mia_advantage=0.0,
                linkability_risk_lower=0.0, inference_risk_lower=0.0,
                singling_out_status="not_attackable", exact_match_train=em_train,
                exact_match_control=em_control,
                exact_match_excess=em_train - em_control,
                exact_match_excess_scaled=em_train - em_control,  # scale 1 for simplicity
                qi_unique_share=0.01))

    agg = aggregate([rep(0, 0.05, 0.05), rep(1, 0.04, 0.00)])[0]
    assert agg.worst["privacy.exact_match_train"] <= agg.worst["privacy.exact_match_control"]
    assert check_gate(agg, GateConfig()) == [
        "singling-out not attackable but safeguards failed "
        "(exact-match excess over control, scaled to TRAIN, 0.040 > 0.01)"]
    ok = aggregate([rep(0, 0.05, 0.05), rep(1, 0.01, 0.00)])[0]
    assert check_gate(ok, GateConfig()) == []


def test_not_attackable_fails_when_three_percent_of_train_is_copied():
    # 3% of TRAIN copied, references of n = N_train / 3 rows: the unscaled excess is
    # diluted to ~0.01 (would pass the 0.01 margin); scaled by N_train / n it is 0.03.
    agg = _na_agg(override={"privacy.exact_match_excess": 0.01,
                            "privacy.exact_match_excess_scaled": 0.03})
    assert check_gate(agg, GateConfig()) == [
        "singling-out not attackable but safeguards failed "
        "(exact-match excess over control, scaled to TRAIN, 0.030 > 0.01)"]
