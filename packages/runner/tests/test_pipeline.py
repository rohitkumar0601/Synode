import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import ColumnType, TableMetadata
from stl_core.schemas import JobSpec
from stl_runner import keys
from stl_runner.pipeline import (
    ensure_splits,
    generate_job,
    job_state,
    load_jobs,
    load_metadata,
    plan_run,
    register_dataset,
    run_generate_with_cap,
)
from stl_runner.storage import ArtifactStore

MD = TableMetadata(columns={"x": ColumnType.numerical, "c": ColumnType.categorical,
                            "y": ColumnType.categorical}, target="y")


def _df(n=500):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=n), "c": rng.choice(["a", "b"], n),
                       "y": rng.choice(["p", "n"], n)})
    df.loc[0, "x"] = np.nan
    return df


def _store(tmp_path):
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", _df(), MD, max_rows=400)
    return store


def test_register_caps_and_prepares(tmp_path):
    store = _store(tmp_path)
    raw = store.read_df(keys.dataset_raw("toy"))
    assert len(raw) == 400 and raw.isna().sum().sum() == 0
    assert load_metadata(store, "toy") == MD


def test_ensure_splits_is_stable(tmp_path):
    store = _store(tmp_path)
    a = ensure_splits(store, "toy", 0)
    b = ensure_splits(store, "toy", 0)
    assert a.fingerprint() == b.fingerprint()
    assert store.exists(keys.split_part("toy", 0, "control"))


def test_plan_and_generate(tmp_path):
    store = _store(tmp_path)
    jobs = plan_run(store, "r1", "toy", ["train_copy", "independent_marginals"], [0, 1])
    assert [j.job_id for j in jobs][:2] == ["train_copy__s0", "train_copy__s1"]
    assert load_jobs(store, "r1") == jobs
    assert job_state(store, jobs[0])["state"] == "queued"
    cost = generate_job(store, jobs[0])
    assert cost.fit_seconds >= 0 and cost.peak_memory_mb > 0
    synth = store.read_df(keys.synthetic("r1", "train_copy__s0"))
    assert len(synth) == len(ensure_splits(store, "toy", 0).train)
    assert job_state(store, jobs[0])["state"] == "generated"


def test_generate_failure_is_recorded(tmp_path):
    store = _store(tmp_path)
    job = JobSpec(job_id="ctgan__s0", run_id="r1", dataset_id="toy", generator="ctgan", seed=0)
    assert run_generate_with_cap(store, job) == "failed"
    assert "q.synthcity" in job_state(store, job)["error"]


def test_time_cap_breach_fails_job(tmp_path):
    store = _store(tmp_path)
    job = JobSpec(job_id="train_copy__s0", run_id="r1", dataset_id="toy",
                  generator="train_copy", seed=0, time_cap_seconds=0.05)
    assert run_generate_with_cap(store, job) == "failed"
    assert "time cap" in job_state(store, job)["error"]


def test_capped_run_succeeds_with_room(tmp_path):
    store = _store(tmp_path)
    job = JobSpec(job_id="train_copy__s0", run_id="r1", dataset_id="toy",
                  generator="train_copy", seed=0, time_cap_seconds=120)
    assert run_generate_with_cap(store, job) == "generated"


def test_capped_child_failure_is_recorded(tmp_path):
    store = _store(tmp_path)
    job = JobSpec(job_id="ctgan__s0", run_id="r1", dataset_id="toy", generator="ctgan", seed=0,
                  time_cap_seconds=120)
    assert run_generate_with_cap(store, job) == "failed"
    assert "q.synthcity" in job_state(store, job)["error"]


def test_rerun_does_not_keep_stale_generated(tmp_path):
    store = _store(tmp_path)
    job1 = JobSpec(job_id="train_copy__s0", run_id="r1", dataset_id="toy",
                   generator="train_copy", seed=0)
    assert generate_job(store, job1) is not None
    assert job_state(store, job1)["state"] == "generated"
    job2 = JobSpec(job_id="train_copy__s0", run_id="r1", dataset_id="toy", generator="ctgan",
                   seed=0, time_cap_seconds=120)
    assert run_generate_with_cap(store, job2) == "failed"


def test_reregistering_dataset_invalidates_splits(tmp_path):
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", _df(500), MD)
    assert len(ensure_splits(store, "toy", 0).train) == 300
    register_dataset(store, "toy", _df(1000), MD)
    assert len(ensure_splits(store, "toy", 0).train) == 600


def test_metadata_change_invalidates_splits(tmp_path):
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", _df(500), MD)
    assert pd.api.types.is_float_dtype(ensure_splits(store, "toy", 0).train["x"])
    md2 = TableMetadata(columns={**MD.columns, "x": ColumnType.categorical}, target="y")
    register_dataset(store, "toy", _df(500), md2)
    assert not pd.api.types.is_numeric_dtype(ensure_splits(store, "toy", 0).train["x"])


def test_same_registration_keeps_fingerprint_and_splits(tmp_path):
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", _df(500), MD)
    ensure_splits(store, "toy", 0)
    fp = store.read_json(keys.dataset_fingerprint("toy"))
    part = keys.split_part("toy", 0, "train")
    before = store.fs.modified(store._path(part))
    register_dataset(store, "toy", _df(500), MD)
    assert store.read_json(keys.dataset_fingerprint("toy")) == fp
    ensure_splits(store, "toy", 0)
    assert store.fs.modified(store._path(part)) == before  # reused, not rewritten


def test_plan_run_rejects_unknown_generators(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="nope.*typo|typo.*nope"):
        plan_run(store, "r9", "toy", ["train_copy", "nope", "typo"], [0])
    assert not store.exists(keys.run_manifest("r9"))


def test_generate_records_baseline_memory(tmp_path):
    store = _store(tmp_path)
    (job,) = plan_run(store, "r1", "toy", ["train_copy"], [0])
    cost = generate_job(store, job)
    assert cost.baseline_memory_mb is not None and 0 < cost.baseline_memory_mb
    assert cost.baseline_memory_mb <= cost.peak_memory_mb


def test_gpu_only_blocked(monkeypatch):
    from stl_runner import pipeline
    monkeypatch.delenv("STL_ALLOW_GPU_ONLY", raising=False)
    monkeypatch.setattr(pipeline, "cuda_available", lambda: False)
    assert pipeline.gpu_only_blocked("tabddpm", allow=False) is True
    assert pipeline.gpu_only_blocked("tabddpm", allow=True) is False
    assert pipeline.gpu_only_blocked("ctgan", allow=False) is False
    monkeypatch.setenv("STL_ALLOW_GPU_ONLY", "1")
    assert pipeline.gpu_only_blocked("tabddpm", allow=False) is False
    monkeypatch.delenv("STL_ALLOW_GPU_ONLY")
    monkeypatch.setattr(pipeline, "cuda_available", lambda: True)
    assert pipeline.gpu_only_blocked("tabddpm", allow=False) is False


def test_cuda_available_without_torch_is_false(monkeypatch):
    import sys

    from stl_runner import pipeline
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    assert pipeline.cuda_available() is False


def test_run_generate_leaves_gpu_only_job_queued_without_cuda(tmp_path, monkeypatch, capsys):
    from stl_runner import pipeline
    monkeypatch.delenv("STL_ALLOW_GPU_ONLY", raising=False)
    monkeypatch.setattr(pipeline, "cuda_available", lambda: False)
    store = _store(tmp_path)
    job = JobSpec(job_id="tabddpm__s0", run_id="r1", dataset_id="toy", generator="tabddpm",
                  seed=0, time_cap_seconds=30)
    assert run_generate_with_cap(store, job) == "queued"
    assert job_state(store, job) == {"state": "queued", "error": None}
    assert not store.exists(keys.synthetic("r1", "tabddpm__s0"))
    assert capsys.readouterr().out == ""
    # allowed: it is attempted (and fails here: synthcity is not installed in this env)
    assert run_generate_with_cap(store, job, allow_gpu_only=True) == "failed"


PUBLIC = {"public_bounds": {"x": [-10.0, 10.0]}, "public_log_scale": []}


def test_register_writes_public_info(tmp_path):
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", _df(), MD, max_rows=400, public_info=PUBLIC)
    assert store.read_json(keys.dataset_public("toy")) == PUBLIC
    register_dataset(store, "toy", _df(), MD, max_rows=400)  # re-registering clears it
    assert store.read_json(keys.dataset_public("toy")) == {}


class _SpyGenerator:
    """Records the params generate_job passes; samples TRAIN rows back."""

    def __init__(self, params):
        self.params = params
        self.privacy_spent = {"preprocessor": 0.0, "synthesizer": 1.0, "bounds": "public"}

    def fit(self, train, metadata):
        self._train = train

    def sample(self, n):
        return self._train.head(n)


def _spy(monkeypatch):
    seen = {}

    def fake_load(name, seed=0, **params):
        seen[name] = params
        return _SpyGenerator(params)

    monkeypatch.setattr("stl_runner.pipeline.load_generator", fake_load)
    return seen


def test_generate_passes_public_info_only_to_dp_generators(tmp_path, monkeypatch):
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", _df(), MD, max_rows=400, public_info=PUBLIC)
    seen = _spy(monkeypatch)
    jobs = plan_run(store, "r1", "toy", ["mst_eps1", "train_copy"], [0],
                    params={"mst_eps1": {"epsilon": 1.0}})
    for job in jobs:
        generate_job(store, job)
    assert seen["mst_eps1"] == {"epsilon": 1.0, **PUBLIC}
    assert seen["train_copy"] == {}
    assert store.read_json(keys.privacy("r1", "mst_eps1__s0")) == {
        "preprocessor": 0.0, "synthesizer": 1.0, "bounds": "public"}


def test_generate_without_public_info_file(tmp_path, monkeypatch):
    # datasets registered before public.json existed (or imported) still generate
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", _df(), MD, max_rows=400, public_info=PUBLIC)
    (tmp_path / keys.dataset_public("toy")).unlink()
    seen = _spy(monkeypatch)
    generate_job(store, plan_run(store, "r1", "toy", ["mst_eps1"], [0])[0])
    assert "public_bounds" not in seen["mst_eps1"]


@pytest.mark.parametrize("runner", [generate_job, run_generate_with_cap])
def test_regeneration_drops_stale_metrics_and_report(tmp_path, runner):
    store = _store(tmp_path)
    (job,) = plan_run(store, "r1", "toy", ["train_copy"], [0])
    generate_job(store, job)
    stale = [keys.synthcity_metrics("r1", job.job_id), keys.report("r1", job.job_id)]
    for key in stale:
        store.write_json(key, {"stale": True})
    runner(store, job)
    assert job_state(store, job)["state"] == "generated"
    assert not any(store.exists(k) for k in stale)


def test_plan_run_records_budget_profile(tmp_path):
    store = _store(tmp_path)
    plan_run(store, "r1", "toy", ["train_copy"], [0], budget_profile="equal50")
    assert store.read_json(keys.run_manifest("r1"))["budget_profile"] == "equal50"
    plan_run(store, "r2", "toy", ["train_copy"], [0])  # callers that predate profiles
    manifest = store.read_json(keys.run_manifest("r2"))
    assert manifest.get("budget_profile") is None
    assert [j.job_id for j in load_jobs(store, "r2")] == ["train_copy__s0"]


def test_plan_run_rejects_unknown_budget_profile(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="budget profile"):
        plan_run(store, "r1", "toy", ["train_copy"], [0], budget_profile="fast")
    assert not store.exists(keys.run_manifest("r1"))


def test_generate_writes_generator_params(tmp_path):
    store = _store(tmp_path)
    (job,) = plan_run(store, "r1", "toy", ["train_copy"], [0], budget_profile="library")
    generate_job(store, job)
    gp = store.read_json(keys.generator_params("r1", job.job_id))
    assert gp == {"library": "stl_core", "library_version": gp["library_version"],
                  "params": {}, "overrides": [], "profile": "library"}
    assert gp["library_version"]


def test_generator_params_profile_is_none_for_old_manifests(tmp_path):
    store = _store(tmp_path)
    (job,) = plan_run(store, "r1", "toy", ["train_copy"], [0])
    generate_job(store, job)
    assert store.read_json(keys.generator_params("r1", job.job_id))["profile"] is None


def test_generator_params_written_before_fit(tmp_path, monkeypatch):
    """A job that fails (or is killed by the time cap) during fit still has its record."""
    from stl_core.baselines import TrainCopy

    def boom(self, *a):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(TrainCopy, "fit", boom)
    store = _store(tmp_path)
    (job,) = plan_run(store, "r1", "toy", ["train_copy"], [0], budget_profile="smoke")
    with pytest.raises(RuntimeError):
        generate_job(store, job)
    assert store.read_json(keys.generator_params("r1", job.job_id))["profile"] == "smoke"
