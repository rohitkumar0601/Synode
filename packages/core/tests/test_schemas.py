from stl_core.schemas import (
    PRIVACY_KEYS,
    CostMetrics,
    FidelityMetrics,
    MetricReport,
    PrivacyMetrics,
    UtilityMetrics,
)


def _report(**kw):
    base = dict(job_id="j", run_id="r", dataset_id="d", generator="g", seed=0, status="done",
                fidelity=FidelityMetrics(quality_score=0.8),
                utility=UtilityMetrics(tstr={"logreg": 0.9}, real_scores={"logreg": 0.7}),
                privacy=PrivacyMetrics(dcr_share=0.5),
                cost=CostMetrics(fit_seconds=1.0, sample_seconds=0.5))
    base.update(kw)
    return MetricReport(**base)


def test_flat_keys():
    flat = _report().flat()
    assert flat["fidelity.quality_score"] == 0.8
    assert flat["utility.tstr.logreg"] == 0.9
    assert flat["privacy.dcr_share"] == 0.5
    assert flat["privacy.mia_advantage"] is None
    assert flat["cost.fit_seconds"] == 1.0
    assert not any(k.startswith("utility.real_scores") for k in flat)


def test_privacy_keys():
    assert len(PRIVACY_KEYS) == 5 and all(k.startswith("privacy.") for k in PRIVACY_KEYS)


def test_round_trip_json():
    r = _report()
    assert MetricReport.model_validate_json(r.model_dump_json()) == r


def test_non_finite_metrics_become_none():
    nan, inf = float("nan"), float("inf")
    fid = FidelityMetrics(quality_score=nan, detection_auc=inf, column_shapes=0.5)
    assert fid.quality_score is None and fid.detection_auc is None
    assert fid.column_shapes == 0.5
    fid.alpha_precision = nan  # assignment is validated too (evaluate sets fields)
    assert fid.alpha_precision is None
    util = UtilityMetrics(tstr={"logreg": nan, "xgboost": 0.9}, real_scores={"logreg": -inf},
                          importance_spearman=nan)
    assert util.tstr == {"logreg": None, "xgboost": 0.9}
    assert util.real_scores == {"logreg": None} and util.importance_spearman is None
    priv = PrivacyMetrics(dcr_share=nan)
    priv.mia_advantage = inf
    assert priv.dcr_share is None and priv.mia_advantage is None
    cost = CostMetrics(fit_seconds=1.0, sample_seconds=1.0, peak_memory_mb=nan)
    assert cost.peak_memory_mb is None


def test_nan_report_round_trips_as_none():
    r = _report(fidelity=FidelityMetrics(quality_score=float("nan")))
    again = MetricReport.model_validate_json(r.model_dump_json())
    assert again.fidelity.quality_score is None and again == r


def test_inference_by_secret_kept_out_of_flat_and_privacy_keys():
    r = _report(privacy=PrivacyMetrics(inference_risk=0.2,
                                       inference_by_secret={"y": 0.2, "x": float("nan")}))
    assert r.privacy.inference_by_secret == {"y": 0.2, "x": None}
    assert not any("inference_by_secret" in k for k in r.flat())
    assert "privacy.inference_by_secret" not in PRIVACY_KEYS
    assert all(isinstance(v, (float, type(None))) for v in r.flat().values())


def test_cost_baseline_memory():
    c = CostMetrics(fit_seconds=1.0, sample_seconds=1.0, baseline_memory_mb=120.0)
    assert c.baseline_memory_mb == 120.0
    assert CostMetrics(fit_seconds=1.0, sample_seconds=1.0).baseline_memory_mb is None


def test_privacy_keys_fixed_and_lower_in_flat():
    assert PRIVACY_KEYS == ("privacy.dcr_share", "privacy.singling_out_risk",
                            "privacy.linkability_risk", "privacy.inference_risk",
                            "privacy.mia_advantage")
    flat = _report(privacy=PrivacyMetrics(dcr_share=0.5, inference_risk_lower=0.02)).flat()
    assert flat["privacy.inference_risk_lower"] == 0.02


def test_singling_out_detail_fields_kept_out_of_flat():
    p = PrivacyMetrics(singling_out_status="not_attackable", singling_out_mode_used="both",
                       singling_out_queries={"multivariate": 20, "univariate": 1},
                       exact_match_train=0.01, exact_match_control=0.02, qi_unique_share=0.03,
                       exact_match_excess=-0.01, exact_match_train_full=0.02)
    flat = _report(privacy=p).flat()
    assert not any(k in flat for k in ("privacy.singling_out_status",
                                       "privacy.singling_out_mode_used",
                                       "privacy.singling_out_queries"))
    assert flat["privacy.exact_match_train"] == 0.01
    assert flat["privacy.exact_match_excess"] == -0.01
    assert flat["privacy.qi_unique_share"] == 0.03
    assert all(isinstance(v, (float, type(None))) for v in flat.values())
    assert not any("exact_match" in k or "qi_unique" in k for k in PRIVACY_KEYS)
    r = _report(privacy=p)
    assert MetricReport.model_validate_json(r.model_dump_json()) == r


def test_singling_out_status_values_validated():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PrivacyMetrics(singling_out_status="maybe")
    with pytest.raises(ValidationError):
        PrivacyMetrics(singling_out_mode_used="sideways")
    assert PrivacyMetrics().singling_out_status is None


def test_synthcity_and_scaled_exact_match_fields():
    from stl_core.schemas import WORST_KEYS

    p = PrivacyMetrics(mia_advantage=0.1, mia_auc=0.55, authenticity=0.48,
                       exact_match_excess=0.01, exact_match_excess_scaled=0.03)
    flat = _report(privacy=p).flat()
    assert flat["privacy.mia_auc"] == 0.55 and flat["privacy.authenticity"] == 0.48
    assert flat["privacy.exact_match_excess_scaled"] == 0.03
    # reported, not gated: neither is a mandatory privacy metric
    assert "privacy.authenticity" not in PRIVACY_KEYS and "privacy.mia_auc" not in PRIVACY_KEYS
    assert "privacy.exact_match_excess_scaled" in WORST_KEYS


def test_both_mia_attacks_recorded_and_report_metrics_version():
    p = PrivacyMetrics(mia_advantage=0.4, mia_domias_advantage=0.3,
                       mia_distance_advantage=0.4)
    r = _report(privacy=p, metrics_version=2)
    flat = r.flat()
    assert flat["privacy.mia_domias_advantage"] == 0.3
    assert flat["privacy.mia_distance_advantage"] == 0.4
    assert "metrics_version" not in flat and "privacy.metrics_version" not in flat
    assert MetricReport.model_validate_json(r.model_dump_json()).metrics_version == 2
    assert _report().metrics_version is None  # reports written before versioning
