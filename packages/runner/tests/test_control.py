"""Run cancellation: the store flag, task short-circuits and killing a running generate."""

import threading
import time

import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import ColumnType, TableMetadata
from stl_core.schemas import JobSpec
from stl_runner import keys, pipeline
from stl_runner.control import (
    CANCEL_ERROR,
    CANCELLED,
    TERMINAL_STATES,
    cancel_if_requested,
    clear_cancel,
    is_cancelled,
    request_cancel,
)
from stl_runner.pipeline import job_state, plan_run, register_dataset, run_generate_with_cap, set_state
from stl_runner.storage import ArtifactStore

MD = TableMetadata(columns={"x": ColumnType.numerical, "c": ColumnType.categorical,
                            "y": ColumnType.categorical}, target="y")


def _store(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=300), "c": rng.choice(["a", "b"], 300),
                       "y": rng.choice(["p", "n"], 300)})
    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", df, MD)
    return store


def sleepy_entry(root: str, storage_options: dict, job_json: str) -> None:
    """Stand-in for pipeline._generate_entry (spawned child): 'fits' for a long time."""
    store = ArtifactStore(root, storage_options)
    set_state(store, JobSpec.model_validate_json(job_json), "fitting")
    time.sleep(120)


def test_flag_roundtrip(tmp_path):
    store = ArtifactStore(str(tmp_path))
    assert not is_cancelled(store, "r1")
    request_cancel(store, "r1")
    assert is_cancelled(store, "r1") and not is_cancelled(store, "r2")
    assert "requested_at" in store.read_json(keys.cancel("r1"))
    request_cancel(store, "r1")  # idempotent
    clear_cancel(store, "r1")
    assert not is_cancelled(store, "r1")
    clear_cancel(store, "r1")  # missing flag is fine
    assert TERMINAL_STATES == {"done", "failed", "cancelled"}


def test_cancel_if_requested_keeps_finished_jobs(tmp_path):
    store = _store(tmp_path)
    queued, done = plan_run(store, "r1", "toy", ["train_copy"], [0, 1])
    assert cancel_if_requested(store, queued) is False
    set_state(store, done, "done")
    request_cancel(store, "r1")
    assert cancel_if_requested(store, queued) is True
    assert job_state(store, queued) == {"state": CANCELLED, "error": CANCEL_ERROR}
    assert cancel_if_requested(store, done) is True
    assert job_state(store, done)["state"] == "done"  # finished results are kept


def test_generate_skipped_when_run_cancelled(tmp_path):
    store = _store(tmp_path)
    (job,) = plan_run(store, "r1", "toy", ["train_copy"], [0])
    request_cancel(store, "r1")
    assert run_generate_with_cap(store, job) == CANCELLED
    assert not store.exists(keys.synthetic("r1", job.job_id))


def _cancel_when_fitting(store, job):
    def go():
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if job_state(store, job)["state"] == "fitting":
                request_cancel(store, job.run_id)
                return
            time.sleep(0.05)
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


@pytest.mark.parametrize("cap", [None, 600])
def test_cancel_kills_running_generate(tmp_path, monkeypatch, cap):
    monkeypatch.setattr(pipeline, "_generate_entry", sleepy_entry)
    store = _store(tmp_path)
    job = JobSpec(job_id="train_copy__s0", run_id="r1", dataset_id="toy",
                  generator="train_copy", seed=0, time_cap_seconds=cap)
    t = _cancel_when_fitting(store, job)
    start = time.monotonic()
    assert run_generate_with_cap(store, job, poll_seconds=0.1) == CANCELLED
    assert time.monotonic() - start < 100  # killed, not slept out
    t.join(1)
    assert job_state(store, job) == {"state": CANCELLED, "error": CANCEL_ERROR}


def test_time_cap_still_enforced_by_poll_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_generate_entry", sleepy_entry)
    store = _store(tmp_path)
    job = JobSpec(job_id="train_copy__s0", run_id="r1", dataset_id="toy",
                  generator="train_copy", seed=0, time_cap_seconds=1)
    assert run_generate_with_cap(store, job, poll_seconds=0.1) == "failed"
    assert "time cap of 1.0s exceeded" in job_state(store, job)["error"]


def test_uncapped_generate_runs_in_child(tmp_path):
    store = _store(tmp_path)
    (job,) = plan_run(store, "r1", "toy", ["train_copy"], [0])
    assert job.time_cap_seconds is None
    assert run_generate_with_cap(store, job) == "generated"


def test_store_modified(tmp_path):
    store = ArtifactStore(str(tmp_path))
    store.write_json("a/b.json", {})
    assert abs(store.modified("a/b.json") - time.time()) < 60
