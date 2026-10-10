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


def test_task_name():
    assert metafeatures.name == "stl.metafeatures"


def test_metafeatures_task_no_predictor(env):
    res = metafeatures.apply(args=["toy"]).get()
    assert res == "toy"
    assert env.exists(keys.dataset_metafeatures("toy"))
    suggestion = env.read_json(keys.dataset_suggestion("toy"))
    assert suggestion["available"] is False
    assert "no predictor artifact" in suggestion["reason"]


def test_metafeatures_task_with_predictor(env):
    # Train and save a small predictor
    meta_df = pd.DataFrame([
        {"dataset_id": "d1", "label": "gaussian_copula", "nr_inst": 100.0, "nr_attr": 2.0},
        {"dataset_id": "d2", "label": "mst_eps1", "nr_inst": 200.0, "nr_attr": 2.0},
    ])
    artifact = train_final(meta_df)
    save_predictor(env, artifact)

    res = metafeatures.apply(args=["toy"]).get()
    assert res == "toy"
    suggestion = env.read_json(keys.dataset_suggestion("toy"))
    assert suggestion["available"] is True
    assert suggestion["generator"] in ("gaussian_copula", "mst_eps1")
    assert "probabilities" in suggestion
    assert suggestion["model"]["n_datasets"] == 2


def test_metafeatures_task_handles_error(env, monkeypatch):
    def boom(*args, **kwargs):
        raise MetaFeatureError("extraction failed completely")

    monkeypatch.setattr("stl_meta.tasks.compute_for_dataset", boom)
    res = metafeatures.apply(args=["toy"]).get()
    assert res == "toy"
    suggestion = env.read_json(keys.dataset_suggestion("toy"))
    assert suggestion["available"] is False
    assert "extraction failed" in suggestion["reason"]
