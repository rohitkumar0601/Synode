import pytest
from stl_core.schemas import JobSpec
from stl_runner import keys
from stl_runner.pipeline import generate_job, plan_run, register_dataset, set_state
from stl_runner.storage import ArtifactStore
from stl_synthcity.tasks import metrics
from test_metrics import MD, draw


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("STL_STORAGE_ROOT", str(tmp_path))
    s = ArtifactStore(str(tmp_path))
    register_dataset(s, "toy", draw(1200, 0), MD)
    return s


def test_task_name():
    assert metrics.name == "stl.metrics"


def test_metrics_task_writes_file_for_generated_job(store):
    (job,) = plan_run(store, "r1", "toy", ["independent_marginals"], [0])
    generate_job(store, job)
    assert metrics.apply(args=[job.model_dump_json()]).get() == job.model_dump_json()
    result = store.read_json(keys.synthcity_metrics("r1", job.job_id))
    assert "error" not in result and "alpha_precision" in result


def test_metrics_task_noop_when_not_generated(store):
    (job,) = plan_run(store, "r2", "toy", ["independent_marginals"], [0])
    set_state(store, job, "failed", "boom")
    assert metrics.apply(args=[job.model_dump_json()]).get() == job.model_dump_json()
    assert not store.exists(keys.synthcity_metrics("r2", job.job_id))


def test_metrics_task_never_raises(store):
    job = JobSpec(job_id="x__s0", run_id="nope", dataset_id="missing",
                  generator="independent_marginals", seed=0)
    assert metrics.apply(args=[job.model_dump_json()]).get() == job.model_dump_json()


def test_metrics_task_skips_cancelled_run(store):
    from stl_runner.control import request_cancel
    from stl_runner.pipeline import job_state

    (job,) = plan_run(store, "r3", "toy", ["independent_marginals"], [0])
    generate_job(store, job)
    request_cancel(store, "r3")
    assert metrics.apply(args=[job.model_dump_json()]).get() == job.model_dump_json()
    assert job_state(store, job)["state"] == "cancelled"
    assert not store.exists(keys.synthcity_metrics("r3", job.job_id))


def test_compute_stored_job_does_not_write_after_cancel(store, monkeypatch):
    from stl_runner.control import request_cancel
    from stl_synthcity import metrics as m

    (job,) = plan_run(store, "r4", "toy", ["independent_marginals"], [0])
    generate_job(store, job)

    def cancel_midway(*a, **k):
        request_cancel(store, "r4")
        return {"alpha_precision": 0.5}

    monkeypatch.setattr(m, "compute_synthcity_metrics", cancel_midway)
    m.compute_stored_job(store, job)
    assert not store.exists(keys.synthcity_metrics("r4", job.job_id))
