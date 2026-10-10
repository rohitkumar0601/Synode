import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import ColumnType, TableMetadata
from stl_runner import keys
from stl_runner.celery_app import app, make_app
from stl_runner.pipeline import job_state, plan_run, register_dataset
from stl_runner.storage import ArtifactStore
from stl_runner.tasks import generate

MD = TableMetadata(columns={"x": ColumnType.numerical, "c": ColumnType.categorical,
                            "y": ColumnType.categorical}, target="y")
DISPATCH = Path(__file__).parents[3] / "services/api/src/stl_api/dispatch.py"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("STL_STORAGE_ROOT", str(tmp_path))
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=400), "c": rng.choice(["a", "b"], 400),
                       "y": rng.choice(["p", "n"], 400)})
    s = ArtifactStore(str(tmp_path))
    register_dataset(s, "toy", df, MD)
    return s


def test_app_conf_and_include(monkeypatch):
    assert app.conf.task_acks_late and app.conf.worker_prefetch_multiplier == 1
    assert app.conf.task_default_queue == "q.default"
    assert list(app.conf.include) == ["stl_runner.tasks"]
    monkeypatch.setenv("STL_CELERY_INCLUDE", "a.b, c.d")
    assert list(make_app().conf.include) == ["a.b", "c.d"]


def test_generate_matches_api_chain():
    src = DISPATCH.read_text()
    gen = re.search(r'signature\(\s*"stl\.generate",\s*args=\[job_json\],\s*'
                    r'kwargs=\{"allow_gpu_only": allow_gpu_only\},\s*'
                    r'queue=CATALOG\[job\.generator\]\.queue,\s*immutable=True', src)
    met = re.search(r'signature\(\s*"stl\.metrics",\s*args=\[job_json\],\s*'
                    r'queue="q\.synthcity",\s*immutable=True', src)
    ev = re.search(r'signature\(\s*"stl\.evaluate",\s*args=\[job_json\],\s*'
                   r'queue="q\.eval",\s*immutable=True', src)
    assert gen and met and ev
    assert generate.name == "stl.generate"
    assert list(inspect.signature(generate.run).parameters) == ["job_json", "allow_gpu_only"]


def test_generate_eager_train_copy(store):
    (job,) = plan_run(store, "r1", "toy", ["train_copy"], [0])
    out = generate.apply(args=[job.model_dump_json()]).get()
    assert out == job.model_dump_json()
    assert job_state(store, job)["state"] == "generated"
    assert store.exists(keys.synthetic("r1", job.job_id))


def test_generate_failure_does_not_escape(store):
    (job,) = plan_run(store, "r2", "toy", ["ctgan"], [0])
    result = generate.apply(args=[job.model_dump_json()], kwargs={"allow_gpu_only": True})
    assert result.get() == job.model_dump_json()
    assert job_state(store, job)["state"] == "failed"
