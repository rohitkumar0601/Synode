import json

import pandas as pd
import pytest
from stl_eval.tasks import evaluate
from stl_runner import keys
from stl_runner.pipeline import (
    generate_job,
    job_state,
    plan_run,
    register_dataset,
    run_generate_with_cap,
)
from stl_runner.storage import ArtifactStore


@pytest.fixture
def env(tmp_path, toy, monkeypatch):
    monkeypatch.setenv("STL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("STL_EVAL_N_ATTACKS", "100")
    monkeypatch.setenv("STL_EVAL_MIN_ATTACKS", "50")
    monkeypatch.setenv("STL_EVAL_SINGLING_OUT_MODE", "univariate")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    train, control, test, md = toy
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", pd.concat([train, control, test]), md)
    return store


def _run(job):
    result = evaluate.apply(args=[job.model_dump_json()])
    return result.get()


def test_task_name():
    assert evaluate.name == "stl.evaluate"


def test_evaluate_generated_job_with_metrics_file(env):
    (job,) = plan_run(env, "r1", "toy", ["independent_marginals"], [0])
    generate_job(env, job)
    env.write_json(keys.synthcity_metrics("r1", job.job_id),
                   {"alpha_precision": 0.9, "beta_recall": 0.5, "authenticity": 0.45,
                    "detection_auc": 0.52, "mia_auc": 0.51, "mia_domias_advantage": 0.02})
    assert _run(job) == "done"
    assert job_state(env, job)["state"] == "done"
    assert env.read_json(keys.report("r1", job.job_id))["status"] == "done"


def test_evaluate_failed_job_writes_failed_report(env):
    (job,) = plan_run(env, "r2", "toy", ["ctgan"], [0])
    run_generate_with_cap(env, job)
    assert job_state(env, job)["state"] == "failed"
    assert _run(job) == "failed"
    assert env.read_json(keys.report("r2", job.job_id))["status"] == "failed"


def test_bad_inference_secrets_is_failed_report(env, monkeypatch):
    monkeypatch.setenv("STL_EVAL_INFERENCE_SECRETS", "nope")
    (job,) = plan_run(env, "r3", "toy", ["independent_marginals"], [0])
    generate_job(env, job)
    assert _run(job) == "failed"
    report = json.loads(json.dumps(env.read_json(keys.report("r3", job.job_id))))
    assert "invalid eval config" in report["error"]


@pytest.mark.parametrize("n_attacks", ["10", "abc"])
def test_invalid_env_config_is_failed_report(env, monkeypatch, n_attacks):
    monkeypatch.setenv("STL_EVAL_N_ATTACKS", n_attacks)
    (job,) = plan_run(env, "r4", "toy", ["independent_marginals"], [0])
    generate_job(env, job)
    assert _run(job) == "failed"
    assert job_state(env, job)["state"] == "failed"
    assert "invalid eval config" in env.read_json(keys.report("r4", job.job_id))["error"]


def test_done_job_unchanged_under_bad_config(env, monkeypatch):
    (job,) = plan_run(env, "r5", "toy", ["independent_marginals"], [0])
    generate_job(env, job)
    assert _run(job) == "done"
    before = env.read_json(keys.report("r5", job.job_id))
    monkeypatch.setenv("STL_EVAL_N_ATTACKS", "abc")
    assert _run(job) == "done"
    assert env.read_json(keys.report("r5", job.job_id)) == before


def test_inference_secrets_whitespace_and_empties(monkeypatch):
    from stl_eval.tasks import eval_config_from_env

    monkeypatch.setenv("STL_EVAL_INFERENCE_SECRETS", " x1 , ,y,")
    assert eval_config_from_env().inference_secrets == ["x1", "y"]
    monkeypatch.setenv("STL_EVAL_INFERENCE_SECRETS", " , ")
    assert eval_config_from_env().inference_secrets == "all"


def test_cancelled_run_skips_evaluation(env):
    from stl_runner.control import request_cancel

    (job,) = plan_run(env, "r6", "toy", ["independent_marginals"], [0])
    generate_job(env, job)
    request_cancel(env, "r6")
    assert _run(job) == "cancelled"
    assert job_state(env, job)["state"] == "cancelled"
    assert not env.exists(keys.report("r6", job.job_id))


def test_cancel_during_evaluation_drops_report(env, monkeypatch):
    from stl_eval import evaluate as ev
    from stl_runner.control import request_cancel

    (job,) = plan_run(env, "r7", "toy", ["independent_marginals"], [0])
    generate_job(env, job)
    real = ev.evaluate

    def cancel_midway(*a, **k):
        report = real(*a, **k)
        request_cancel(env, "r7")
        return report

    monkeypatch.setattr(ev, "evaluate", cancel_midway)
    assert _run(job) == "cancelled"
    assert job_state(env, job)["state"] == "cancelled"
    assert not env.exists(keys.report("r7", job.job_id))
