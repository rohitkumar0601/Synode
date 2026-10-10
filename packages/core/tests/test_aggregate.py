import math

from stl_core.aggregate import aggregate
from stl_core.schemas import FidelityMetrics, MetricReport, PrivacyMetrics, UtilityMetrics


def _r(gen, seed, q, dcr=0.5, status="done"):
    return MetricReport(job_id=f"{gen}{seed}", run_id="r", dataset_id="d", generator=gen,
                        seed=seed, status=status, fidelity=FidelityMetrics(quality_score=q),
                        utility=UtilityMetrics(), privacy=PrivacyMetrics(dcr_share=dcr))


def test_mean_std_over_seeds():
    aggs = aggregate([_r("a", 0, 0.6), _r("a", 1, 0.8), _r("b", 0, 0.5)])
    assert [a.generator for a in aggs] == ["a", "b"]
    a = aggs[0]
    assert a.n_seeds == 2 and a.n_failed == 0
    assert math.isclose(a.mean["fidelity.quality_score"], 0.7)
    assert math.isclose(a.std["fidelity.quality_score"], 0.1414, rel_tol=1e-3)
    assert aggs[1].std["fidelity.quality_score"] == 0.0


def test_missing_metric_recorded():
    aggs = aggregate([_r("a", 0, 0.6, dcr=0.5), _r("a", 1, 0.6, dcr=None)])
    assert "privacy.dcr_share" in aggs[0].missing
    assert aggs[0].mean["privacy.dcr_share"] == 0.5


def test_failed_seeds_excluded():
    aggs = aggregate([_r("a", 0, 0.6), _r("a", 1, 0.0, status="failed")])
    assert aggs[0].n_failed == 1 and aggs[0].mean["fidelity.quality_score"] == 0.6


def test_all_failed():
    agg = aggregate([_r("a", 0, 0.6, status="failed")])[0]
    assert agg.n_failed == 1 and agg.mean == {}


def test_non_finite_values_treated_as_missing():
    # model_construct bypasses validation, simulating a report built without the contract
    bad = _r("a", 1, 0.6)
    bad.fidelity = FidelityMetrics.model_construct(quality_score=float("nan"))
    agg = aggregate([_r("a", 0, 0.8), bad])[0]
    assert agg.mean["fidelity.quality_score"] == 0.8
    assert "fidelity.quality_score" in agg.missing
    only_bad = aggregate([bad])[0]
    assert only_bad.mean["fidelity.quality_score"] is None


def _rp(seed, so_lower, status="done"):
    return MetricReport(job_id=f"a{seed}", run_id="r", dataset_id="d", generator="a",
                        seed=seed, status=status, fidelity=FidelityMetrics(quality_score=0.5),
                        utility=UtilityMetrics(),
                        privacy=PrivacyMetrics(dcr_share=0.5 + seed / 100,
                                               singling_out_risk=so_lower + 0.01,
                                               singling_out_risk_lower=so_lower))


def test_worst_is_max_over_successful_seeds():
    agg = aggregate([_rp(0, 0.25), _rp(1, 0.02), _rp(2, 0.02), _rp(3, 0.9, status="failed")])[0]
    assert math.isclose(agg.worst["privacy.singling_out_risk_lower"], 0.25)
    assert math.isclose(agg.worst["privacy.singling_out_risk"], 0.26)
    assert math.isclose(agg.worst["privacy.dcr_share"], 0.52)
    assert agg.worst["privacy.linkability_risk_lower"] is None
    assert "fidelity.quality_score" not in agg.worst
    # axis scores keep using means
    assert math.isclose(agg.mean["privacy.singling_out_risk_lower"], 0.29 / 3)


def test_singling_out_status_and_safeguard_worst():
    def r(seed, status, em, qi, run_status="done"):
        return MetricReport(job_id=f"a{seed}", run_id="r", dataset_id="d", generator="a",
                            seed=seed, status=run_status, privacy=PrivacyMetrics(
                                singling_out_status=status, exact_match_train=em,
                                exact_match_control=em / 2, exact_match_excess=em / 2,
                                qi_unique_share=qi))

    agg = aggregate([r(0, "not_attackable", 0.02, 0.01), r(1, "measured", 0.04, 0.03),
                     r(2, "error", 0.9, 0.9, run_status="failed")])[0]
    assert agg.singling_out_status == ["not_attackable", "measured"]
    assert math.isclose(agg.worst["privacy.exact_match_train"], 0.04)
    assert math.isclose(agg.worst["privacy.exact_match_control"], 0.02)
    assert math.isclose(agg.worst["privacy.exact_match_excess"], 0.02)
    assert math.isclose(agg.worst["privacy.qi_unique_share"], 0.03)


def test_scaled_exact_match_excess_worst_over_seeds():
    def r(seed, scaled):
        return MetricReport(job_id=f"a{seed}", run_id="r", dataset_id="d", generator="a",
                            seed=seed, status="done",
                            privacy=PrivacyMetrics(exact_match_excess_scaled=scaled))

    agg = aggregate([r(0, 0.0), r(1, 0.03)])[0]
    assert math.isclose(agg.worst["privacy.exact_match_excess_scaled"], 0.03)
