import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import ColumnType, TableMetadata
from stl_meta.features import MetaFeatureError
from stl_meta.meta import save_predictor, train_final
from stl_meta.tasks import metafeatures
from stl_runner import keys
from stl_runner.pipeline import register_dataset
from stl_runner.storage import ArtifactStore

STALE = {"available": True, "generator": "stale", "probabilities": {"stale": 1.0},
         "model": {"n_datasets": 1, "trained_at": "2000-01-01T00:00:00+00:00"}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("STL_STORAGE_ROOT", str(tmp_path / "store"))
    store = ArtifactStore(str(tmp_path / "store"))
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "age": rng.integers(18, 90, n).astype(float),
        "income": rng.random(n) * 1000,
        "y": ["a", "b"] * (n // 2),
    })
    md = TableMetadata(
        columns={
            "age": ColumnType.numerical,
            "income": ColumnType.numerical,
            "y": ColumnType.categorical,
        },
        target="y",
    )
    register_dataset(store, "toy", df, md)
    return store


def _train_predictor(store):
    meta_df = pd.DataFrame([
        {"dataset_id": "d1", "label": "gaussian_copula", "nr_inst": 100.0, "nr_attr": 2.0},
        {"dataset_id": "d2", "label": "mst_eps1", "nr_inst": 200.0, "nr_attr": 2.0},
    ])
    save_predictor(store, train_final(meta_df))


def test_task_name():
    assert metafeatures.name == "stl.metafeatures"


def test_default_caps_passed_to_extraction(env, monkeypatch):
    seen = {}

    def fake(store, dataset_id, **kwargs):
        seen.update(kwargs, dataset_id=dataset_id)
        return {"features": {"nr_inst": 1.0}}

    monkeypatch.setattr("stl_meta.tasks.compute_for_dataset", fake)
    assert metafeatures.apply(args=["toy"]).get() == "toy"
    assert seen == {"dataset_id": "toy", "seed": 0, "max_rows": 5000, "timeout_s": 300.0}


def test_metafeatures_task_no_predictor(env):
    env.write_json(keys.dataset_suggestion("toy"), STALE)
    res = metafeatures.apply(args=["toy"]).get()
    assert res == "toy"
    assert "features" in env.read_json(keys.dataset_metafeatures("toy"))
    # suggestion.json is only written when a predictor exists; a stale one is removed
    assert not env.exists(keys.dataset_suggestion("toy"))


def test_metafeatures_task_with_predictor(env):
    _train_predictor(env)
    res = metafeatures.apply(args=["toy"]).get()
    assert res == "toy"
    suggestion = env.read_json(keys.dataset_suggestion("toy"))
    assert suggestion["available"] is True
    assert suggestion["generator"] in ("gaussian_copula", "mst_eps1")
    assert set(suggestion["probabilities"]) == {"gaussian_copula", "mst_eps1"}
    assert sum(suggestion["probabilities"].values()) == pytest.approx(1.0)
    assert suggestion["model"]["n_datasets"] == 2
    assert suggestion["model"]["trained_at"]


def test_metafeatures_task_extraction_error(env, monkeypatch):
    _train_predictor(env)
    env.write_json(keys.dataset_suggestion("toy"), STALE)

    def boom(*args, **kwargs):
        raise MetaFeatureError("extraction failed completely")

    monkeypatch.setattr("stl_meta.tasks.compute_for_dataset", boom)
    assert metafeatures.apply(args=["toy"]).get() == "toy"
    assert not env.exists(keys.dataset_suggestion("toy"))


def test_metafeatures_task_never_raises(env, monkeypatch):
    _train_predictor(env)

    def broken(store):
        raise OSError("store unreachable")

    monkeypatch.setattr("stl_meta.tasks.load_predictor", broken)
    assert metafeatures.apply(args=["toy"]).get() == "toy"
    assert not env.exists(keys.dataset_suggestion("toy"))


def test_metafeatures_task_unknown_or_invalid_dataset(env, tmp_path):
    assert metafeatures.apply(args=["ghost"]).get() == "ghost"
    assert not env.exists(keys.dataset_suggestion("ghost"))
    assert metafeatures.apply(args=["../escape"]).get() == "../escape"
    assert not (tmp_path / "escape").exists()
