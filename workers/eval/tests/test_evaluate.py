from pathlib import Path

import pandas as pd
import pytest
from stl_core.schemas import PRIVACY_KEYS, CostMetrics, JobSpec
from stl_core.split import Splits
from stl_eval.evaluate import METRICS_VERSION, EvalConfig, evaluate, evaluate_stored_job
from stl_runner import keys
from stl_runner.pipeline import (
    generate_job,
    job_state,
    plan_run,
    register_dataset,
    run_generate_with_cap,
)
from stl_runner.storage import ArtifactStore
from stl_runner.tracking import log_report

CFG = EvalConfig(n_attacks=100, singling_out_mode="univariate")
JOB = JobSpec(job_id="g__s0", run_id="r", dataset_id="toy", generator="g", seed=0)
# What the synthcity-env `metrics` stage writes (stl_synthcity.metrics); eval only merges it.
SC = {"alpha_precision": 0.9, "beta_recall": 0.5, "authenticity": 0.45, "detection_auc": 0.52,
      "mia_auc": 0.51, "mia_domias_advantage": 0.02}
MISSING = "synthcity metrics missing: run `metrics` stage"


def test_evaluate_full_report(toy, fresh):
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh, md,
                      CostMetrics(fit_seconds=1, sample_seconds=1), CFG, synthcity_metrics=SC)
    assert report.status == "done", report.warnings
    assert MISSING not in report.warnings
    f, p = report.fidelity, report.privacy
    assert (f.alpha_precision, f.beta_recall, f.detection_auc) == (0.9, 0.5, 0.52)
    assert (p.mia_domias_advantage, p.mia_auc, p.authenticity) == (0.02, 0.51, 0.45)
    assert p.mia_distance_advantage is not None
    assert p.mia_advantage == max(p.mia_domias_advantage, p.mia_distance_advantage)
    assert report.metrics_version == METRICS_VERSION == 2
    flat = report.flat()
    assert all(flat[k] is not None for k in PRIVACY_KEYS), report.warnings
    assert flat["fidelity.quality_score"] is not None
    assert flat["utility.tstr.xgboost"] is not None
    assert set(report.privacy.inference_by_secret) == set(md.model_columns)
    assert report.privacy.inference_risk == max(report.privacy.inference_by_secret.values())


def test_evaluate_rejects_unusable_synthetic(toy, fresh):
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh.drop(columns=["x1"]), md, None,
                      CFG)
    assert report.status == "failed" and "x1" in report.error


def test_evaluate_stored_job_end_to_end(tmp_path, toy):
    train, control, test, md = toy
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", pd.concat([train, control, test]), md)
    ok, bad = plan_run(store, "r1", "toy", ["independent_marginals", "ctgan"], [0])
    generate_job(store, ok)
    run_generate_with_cap(store, bad)
    report = evaluate_stored_job(store, ok, CFG)
    assert report.status == "done" and report.cost is not None
    assert store.exists(keys.report("r1", ok.job_id))
    assert job_state(store, ok)["state"] == "done"
    failed = evaluate_stored_job(store, bad, CFG)
    assert failed.status == "failed" and "q.synthcity" in failed.error


def test_log_report_noop_without_uri(monkeypatch, toy, fresh):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None, CFG)
    assert log_report(report, JOB) is None


def test_log_report_sqlite(monkeypatch, tmp_path, toy, fresh):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + Path(tmp_path, "m.db").as_posix())
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None, CFG)
    run_id = log_report(report, JOB)
    import mlflow

    metrics = mlflow.get_run(run_id).data.metrics
    assert metrics["privacy.dcr_share"] == report.privacy.dcr_share


def test_evaluate_isolates_metric_crashes(monkeypatch, toy, fresh):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("stl_eval.evaluate.anonymeter_risks", boom)
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None, CFG)
    assert report.status == "done"
    assert report.privacy.singling_out_risk is None
    assert report.privacy.singling_out_status == "error"  # fails closed in the gate
    assert report.privacy.linkability_risk is None
    assert report.privacy.inference_risk is None
    assert any(w.startswith("anonymeter: RuntimeError") for w in report.warnings)
    assert report.privacy.dcr_share is not None
    assert report.fidelity.quality_score is not None


def test_evaluate_stored_job_load_failure_is_failed(tmp_path, toy):
    train, control, test, md = toy
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", pd.concat([train, control, test]), md)
    (ok,) = plan_run(store, "r2", "toy", ["independent_marginals"], [0])
    generate_job(store, ok)
    store.write_bytes(keys.cost(ok.run_id, ok.job_id), b"not json")
    report = evaluate_stored_job(store, ok, CFG)
    assert report.status == "failed" and "evaluation crashed" in report.error
    assert store.exists(keys.report("r2", ok.job_id))
    assert job_state(store, ok)["state"] == "failed"


def test_low_detection_auc_warns(toy, fresh):
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None, CFG,
                      synthcity_metrics={**SC, "detection_auc": 0.1})
    assert ("detection_auc 0.100 < 0.4: synthetic rows likely duplicate real rows "
            "(memorisation signal)") in report.warnings


def test_unseen_categories_warn(toy, fresh):
    train, control, test, md = toy
    synth = fresh.copy()
    synth.loc[: int(0.2 * len(synth)) - 1, "c"] = "zzz"
    report = evaluate(JOB, Splits(train, control, test), synth, md, None, CFG)
    assert report.status == "done"
    assert "column 'c': 20% of synthetic categories unseen in real data" in report.warnings
    assert not any(w.startswith("column 'y'") for w in report.warnings)


def _generated(tmp_path, toy, run_id="r3"):
    train, control, test, md = toy
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", pd.concat([train, control, test]), md)
    (job,) = plan_run(store, run_id, "toy", ["independent_marginals"], [0])
    generate_job(store, job)
    return store, job


def test_evaluate_stored_job_is_idempotent(tmp_path, toy, monkeypatch):
    store, job = _generated(tmp_path, toy)
    first = evaluate_stored_job(store, job, CFG)
    stored = store.read_bytes(keys.report(job.run_id, job.job_id))

    def boom(*args, **kwargs):
        raise AssertionError("must not re-evaluate")

    monkeypatch.setattr("stl_eval.evaluate.evaluate", boom)
    again = evaluate_stored_job(store, job, CFG)
    assert again == first
    assert store.read_bytes(keys.report(job.run_id, job.job_id)) == stored


def test_interrupted_evaluation_marks_failed(tmp_path, toy, monkeypatch):
    store, job = _generated(tmp_path, toy)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("stl_eval.evaluate.evaluate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        evaluate_stored_job(store, job, CFG)
    assert job_state(store, job) == {"state": "failed", "error": "interrupted"}
    assert not store.exists(keys.report(job.run_id, job.job_id))


def test_evaluating_state_is_rerunnable(tmp_path, toy):
    from stl_runner.pipeline import set_state

    store, job = _generated(tmp_path, toy)
    set_state(store, job, "evaluating")  # left behind by a killed process
    report = evaluate_stored_job(store, job, CFG)
    assert report.status == "done"
    assert job_state(store, job)["state"] == "done"


def test_eval_config_rejects_fewer_attacks_than_min_attacks():
    with pytest.raises(ValueError, match="n_attacks"):
        EvalConfig(n_attacks=50)  # default min_attacks=100
    assert EvalConfig(n_attacks=50, min_attacks=50).n_attacks == 50


def test_evaluate_stored_job_invalid_config_is_failed_not_stuck(tmp_path, toy):
    store, job = _generated(tmp_path, toy, run_id="r5")
    report = evaluate_stored_job(store, job, EvalConfig(n_attacks=100,
                                                        inference_secrets=["nope"]))
    assert report.status == "failed"
    assert report.error.startswith("invalid eval config:") and "nope" in report.error
    assert store.exists(keys.report("r5", job.job_id))
    assert job_state(store, job)["state"] == "failed"


def test_evaluate_records_singling_out_outcome_and_safeguards(toy, fresh):
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None,
                      EvalConfig(n_attacks=100))  # multivariate falls short -> univariate
    p = report.privacy
    assert p.singling_out_status == "measured" and p.singling_out_mode_used == "univariate"
    assert p.singling_out_queries["multivariate"] < 100 <= p.singling_out_queries["univariate"]
    assert p.exact_match_train == 0.0 and p.exact_match_control == 0.0
    assert p.exact_match_excess == 0.0 and p.exact_match_train_full == 0.0
    assert p.qi_unique_share == 1.0
    copy = evaluate(JOB, Splits(train, control, test), train, md, None, CFG).privacy
    assert copy.exact_match_train_full == 1.0
    assert copy.exact_match_excess == pytest.approx(len(control) / len(train))
    assert copy.exact_match_excess_scaled == pytest.approx(1.0)
    assert p.exact_match_excess_scaled == 0.0


SC_FIELDS = (("fidelity", "alpha_precision"), ("fidelity", "beta_recall"),
             ("fidelity", "detection_auc"), ("privacy", "mia_domias_advantage"),
             ("privacy", "mia_auc"), ("privacy", "authenticity"))


def _sc_values(report):
    return {k: getattr(getattr(report, axis), k) for axis, k in SC_FIELDS}


def test_missing_synthcity_metrics_warns_and_falls_back_to_distance_mia(toy, fresh):
    train, control, test, md = toy
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None, CFG)
    assert report.status == "done"
    assert MISSING in report.warnings
    assert all(v is None for v in _sc_values(report).values())
    # mia_advantage (a PRIVACY_KEY) = max of the available attacks: here the distance one
    p = report.privacy
    assert p.mia_distance_advantage is not None
    assert p.mia_advantage == p.mia_distance_advantage


def test_evaluate_stored_job_reads_synthcity_metrics(tmp_path, toy):
    store, job = _generated(tmp_path, toy, run_id="r6")
    store.write_json(keys.synthcity_metrics(job.run_id, job.job_id), SC)
    report = evaluate_stored_job(store, job, CFG)
    assert report.status == "done" and MISSING not in report.warnings
    assert _sc_values(report) == SC


def test_evaluate_stored_job_without_synthcity_metrics(tmp_path, toy):
    store, job = _generated(tmp_path, toy, run_id="r7")
    report = evaluate_stored_job(store, job, CFG)
    assert report.status == "done" and MISSING in report.warnings
    assert report.privacy.mia_domias_advantage is None
    assert report.privacy.mia_advantage == report.privacy.mia_distance_advantage


def test_synthcity_error_file_is_missing_with_reason(tmp_path, toy):
    store, job = _generated(tmp_path, toy, run_id="r8")
    store.write_json(keys.synthcity_metrics(job.run_id, job.job_id),
                     {"error": "ValueError: boom"})
    report = evaluate_stored_job(store, job, CFG)
    assert report.status == "done"
    assert f"{MISSING} (error: ValueError: boom)" in report.warnings
    assert all(v is None for v in _sc_values(report).values())


def test_synthcity_partial_failure_keeps_other_metrics(toy, fresh):
    train, control, test, md = toy
    partial = {**SC, "mia_auc": None, "mia_domias_advantage": None,
               "errors": {"domias": "ValueError: there needs to be at least one continuous "
                                    "feature"}}
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None, CFG,
                      synthcity_metrics=partial)
    assert report.fidelity.alpha_precision == 0.9
    assert report.privacy.mia_domias_advantage is None
    assert report.privacy.mia_advantage == report.privacy.mia_distance_advantage
    assert ("synthcity metric domias failed: ValueError: there needs to be at least one "
            "continuous feature") in report.warnings


def test_mia_advantage_is_max_of_attacks_and_none_only_if_both_none(toy, fresh, monkeypatch):
    train, control, test, md = toy
    high = {**SC, "mia_domias_advantage": 0.9}
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None, CFG,
                      synthcity_metrics=high)
    assert report.privacy.mia_advantage == 0.9  # domias above the distance attack

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("stl_eval.evaluate.mia_distance_advantage", boom)
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None, CFG)
    assert report.privacy.mia_distance_advantage is None
    assert report.privacy.mia_advantage is None  # both missing: the gate fails closed
    assert any(w.startswith("mia_distance: RuntimeError") for w in report.warnings)


def test_version_1_synthcity_file_key_is_read(toy, fresh):
    # metrics_version 1 files named the DOMIAS advantage "mia_advantage"
    train, control, test, md = toy
    v1 = {k: v for k, v in SC.items() if k != "mia_domias_advantage"} | {
        "mia_advantage": 0.7, "metrics_version": 1}
    report = evaluate(JOB, Splits(train, control, test), fresh, md, None, CFG,
                      synthcity_metrics=v1)
    assert report.privacy.mia_domias_advantage == 0.7
