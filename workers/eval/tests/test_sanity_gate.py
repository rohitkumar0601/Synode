"""Phase-1 gate: train-copy worst on privacy, independent marginals worst on pair trends."""

import pytest
from stl_core.aggregate import aggregate
from stl_core.decision import axis_scores, recommend
from stl_core.metadata import infer_metadata
from stl_eval.evaluate import EvalConfig, evaluate_stored_job
from stl_runner import keys
from stl_runner.datasets import load_dataset
from stl_runner.pipeline import generate_job, plan_run, register_dataset
from stl_runner.storage import ArtifactStore


@pytest.mark.network
@pytest.mark.slow
def test_sanity_gate_on_adult(tmp_path):
    df, target = load_dataset({"source": "openml", "openml_name": "adult", "version": 2,
                               "target": "class"})
    md = infer_metadata(df, target)
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "adult", df, md, max_rows=2000, seed=0)
    jobs = plan_run(store, "sanity", "adult", ["train_copy", "independent_marginals"], [0])
    config = EvalConfig(n_attacks=300, singling_out_mode="univariate")
    reports = {}
    for job in jobs:
        generate_job(store, job)
        # synthcity metrics need the synthcity env (`metrics` stage); the same neutral values
        # for both generators keep this check about the eval env's own privacy metrics.
        store.write_json(keys.synthcity_metrics(job.run_id, job.job_id),
                         {"alpha_precision": 0.9, "beta_recall": 0.5, "authenticity": 0.5,
                          "detection_auc": 0.5, "mia_auc": 0.5, "mia_domias_advantage": 0.0})
        reports[job.generator] = evaluate_stored_job(store, job, config)
        assert reports[job.generator].status == "done", reports[job.generator].warnings

    aggs = {a.generator: a for a in aggregate(list(reports.values()))}
    scores = {g: axis_scores(a.mean) for g, a in aggs.items()}
    assert scores["train_copy"].privacy < scores["independent_marginals"].privacy
    assert (reports["independent_marginals"].fidelity.column_pair_trends
            < reports["train_copy"].fidelity.column_pair_trends)
    rec = recommend(list(aggs.values()))
    train_copy = next(c for c in rec.ranking if c.generator == "train_copy")
    assert not train_copy.gate_passed
