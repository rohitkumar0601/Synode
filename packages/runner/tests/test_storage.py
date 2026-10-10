import pandas as pd
from pydantic import BaseModel
from stl_runner.storage import ArtifactStore, store_from_env


class M(BaseModel):
    a: int


def test_df_round_trip(tmp_path):
    store = ArtifactStore(str(tmp_path))
    df = pd.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]})
    store.write_df("datasets/d/raw.parquet", df)
    assert store.exists("datasets/d/raw.parquet")
    pd.testing.assert_frame_equal(store.read_df("datasets/d/raw.parquet"), df)


def test_json_and_models(tmp_path):
    store = ArtifactStore(str(tmp_path))
    store.write_json("a/b.json", {"k": 1})
    store.write_json("a/m.json", M(a=3))
    assert store.read_json("a/b.json") == {"k": 1}
    assert M.model_validate(store.read_json("a/m.json")).a == 3
    assert store.list("a") == ["a/b.json", "a/m.json"]
    assert not store.exists("a/zzz.json")
    assert store.list("missing") == []


def test_store_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STL_STORAGE_ROOT", str(tmp_path / "s"))
    store = store_from_env()
    store.write_json("x.json", {"ok": True})
    assert (tmp_path / "s" / "x.json").exists()


def test_synthcity_metrics_key():
    from stl_runner import keys

    assert keys.synthcity_metrics("r", "j") == "runs/r/j/synthcity_metrics.json"


def test_delete_removes_key_and_ignores_missing(tmp_path):
    store = ArtifactStore(str(tmp_path))
    store.write_json("runs/r/j/x.json", {"a": 1})
    store.delete("runs/r/j/x.json")
    assert not store.exists("runs/r/j/x.json")
    store.delete("runs/r/j/x.json")  # already gone: no error
