"""The `metafeatures` stage of experiments/run.py and its store helper (meta env)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import infer_metadata
from stl_meta import features
from stl_meta.features import MetaFeatureError, compute_for_dataset, metafeatures_key
from stl_runner import keys
from stl_runner.pipeline import ensure_splits, register_dataset
from stl_runner.storage import ArtifactStore

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments"))
import run  # noqa: E402


def _toy_df(n=500):
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": 2 * x1 + rng.normal(scale=0.3, size=n),
                         "c": np.where(x1 > 0, "hi", "lo"),
                         "y": np.where(x1 + rng.normal(scale=0.5, size=n) > 0, "p", "n")})


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    monkeypatch.delenv("STL_STORAGE_ROOT", raising=False)


@pytest.fixture
def store(tmp_path):
    s = ArtifactStore((tmp_path / "store").as_posix())
    df = _toy_df()
    register_dataset(s, "toy", df, infer_metadata(df, "y"))
    return s


def test_compute_for_dataset_writes_json(store):
    rec = compute_for_dataset(store, "toy", seed=0)
    on_disk = json.loads(store.read_bytes(metafeatures_key("toy")))
    assert on_disk == rec
    assert on_disk["seed"] == 0
    assert on_disk["n_rows_used"] == len(ensure_splits(store, "toy", 0).train)
    assert on_disk["groups"] == ["general", "statistical", "info-theory"]
    assert on_disk["pymfe_version"] == "0.4.4"
    assert on_disk["features"]["nr_inst"] == on_disk["n_rows_used"]
    assert "computed_at" in on_disk


def test_nan_features_serialised_as_null(store, monkeypatch):
    monkeypatch.setattr(features, "extract", lambda *a, **k: {"a": 1.0, "b": float("nan")})
    compute_for_dataset(store, "toy")
    text = store.read_bytes(metafeatures_key("toy")).decode()
    assert "NaN" not in text
    assert json.loads(text)["features"] == {"a": 1.0, "b": None}


def test_total_failure_records_reason(store, monkeypatch):
    def boom(*a, **k):
        raise MetaFeatureError("pymfe failed: boom")

    monkeypatch.setattr(features, "extract", boom)
    with pytest.raises(MetaFeatureError):
        compute_for_dataset(store, "toy")
    rec = json.loads(store.read_bytes(metafeatures_key("toy")))
    assert "features" not in rec
    assert rec["error"] == "pymfe failed: boom"


def _args(store, *extra):
    overrides = ["~dataset.public_bounds", "~dataset.public_log_scale", "dataset.name=toy",
                 f"storage_root='{store.root}'", *extra]
    return [a for o in overrides for a in ("-o", o)]


def test_cli_stage(store, capsys):
    assert run.main([*_args(store), "metafeatures"]) == 0
    out = capsys.readouterr().out
    assert "toy:" in out and "meta-features" in out
    assert store.exists(metafeatures_key("toy"))


def test_cli_stage_unregistered_dataset_exits_2(store, capsys):
    assert run.main([*_args(store, "dataset.name=nope"), "metafeatures"]) == 2
    assert "not registered" in capsys.readouterr().err


def test_cli_stage_all_registered(store, capsys):
    df = _toy_df(300)
    register_dataset(store, "toy2", df, infer_metadata(df, "y"))
    assert run.main([*_args(store), "metafeatures", "--all-registered"]) == 0
    assert store.exists(metafeatures_key("toy")) and store.exists(metafeatures_key("toy2"))


def test_cli_stage_failure_exit_1(store, capsys, monkeypatch):
    def boom(*a, **k):
        raise MetaFeatureError("pymfe failed: boom")

    monkeypatch.setattr(features, "extract", boom)
    assert run.main([*_args(store), "metafeatures"]) == 1
    assert "toy: meta-features failed: pymfe failed: boom" in capsys.readouterr().out


def test_keys_reject_bad_id():
    with pytest.raises(ValueError):
        metafeatures_key("../x")
    assert metafeatures_key("toy") == keys.dataset_metadata("toy").replace(
        "metadata.json", "metafeatures.json")
