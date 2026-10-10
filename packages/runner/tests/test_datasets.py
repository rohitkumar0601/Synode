import pandas as pd
import pytest
from stl_runner.datasets import load_dataset


def test_load_file_csv(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "junk": [0, 0]}).to_csv(p, index=False)
    df, target = load_dataset({"source": "file", "path": str(p), "target": "b", "drop": ["junk"]})
    assert target == "b" and list(df.columns) == ["a", "b"]


def test_category_columns_become_object(tmp_path):
    p = tmp_path / "d.parquet"
    pd.DataFrame({"c": pd.Categorical(["a", "b"]), "y": [0, 1]}).to_parquet(p)
    df, _ = load_dataset({"source": "file", "path": str(p), "target": "y"})
    assert df["c"].dtype == object


def test_unknown_source():
    with pytest.raises(ValueError, match="source"):
        load_dataset({"source": "ftp", "target": "y"})


@pytest.mark.network
def test_openml_adult():
    df, target = load_dataset({"source": "openml", "openml_name": "adult", "version": 2,
                               "target": "class"})
    assert target == "class" and len(df) > 40_000
