import math

import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import ColumnType, TableMetadata
from stl_meta import features
from stl_meta.features import MetaFeatureError, extract, sample_rows


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    df = pd.DataFrame({
        "x1": x1,
        "x2": 2 * x1 + rng.normal(scale=0.3, size=n),
        "c": np.where(x1 > 0, "hi", "lo").astype(object),
        "y": np.where(x1 + rng.normal(scale=0.5, size=n) > 0, "p", "n").astype(object),
    })
    md = TableMetadata(columns={"x1": ColumnType.numerical, "x2": ColumnType.numerical,
                                "c": ColumnType.categorical, "y": ColumnType.categorical},
                       target="y")
    return df, md


def test_expected_keys_and_values():
    df, md = _toy()
    out = extract(df, "y", md, seed=0)
    for k in ("nr_inst", "nr_attr", "nr_class", "class_ent"):
        assert k in out
    assert out["nr_inst"] == 300
    assert out["nr_attr"] == 3
    assert out["nr_class"] == 2
    assert all(isinstance(v, float) for v in out.values())
    assert all(math.isfinite(v) or math.isnan(v) for v in out.values())


def test_categorical_string_column_counted_as_nominal():
    df, md = _toy()
    out = extract(df, "y", md, seed=0)
    assert out["nr_cat"] == 1
    assert out["nr_num"] == 2


def test_deterministic_for_seed():
    df, md = _toy(n=800)
    a = extract(df, "y", md, seed=3, max_rows=200)
    b = extract(df, "y", md, seed=3, max_rows=200)
    assert a.keys() == b.keys()
    for k in a:
        assert (math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k], k


def test_constant_column_nan_features_tolerated():
    df, md = _toy()
    df["const"] = 1.0  # zero variance: correlation/skewness-type features become NaN
    df.loc[0:9, "x1"] = np.nan  # missing values are imputed, not fatal
    md = md.model_copy(update={"columns": {**md.columns, "const": ColumnType.numerical}})
    out = extract(df, "y", md, seed=0)
    assert out["nr_attr"] == 4
    assert all(isinstance(v, float) for v in out.values())
    assert all(math.isfinite(v) or math.isnan(v) for v in out.values())


def test_non_finite_values_become_nan(monkeypatch):
    df, md = _toy()

    class FakeMFE:
        def __init__(self, **kw):
            pass

        def fit(self, X, y, **kw):
            return self

        def extract(self, **kw):
            return ["a", "b", "c", "d"], [1, np.inf, None, "x"]

    monkeypatch.setattr(features, "MFE", FakeMFE)
    out = extract(df, "y", md, seed=0)
    assert out["a"] == 1.0
    assert math.isnan(out["b"]) and math.isnan(out["c"]) and math.isnan(out["d"])


def test_total_failure_raises(monkeypatch):
    df, md = _toy()

    def boom(self, *a, **kw):
        raise RuntimeError("pymfe broke")

    monkeypatch.setattr(features.MFE, "fit", boom)
    with pytest.raises(MetaFeatureError, match="pymfe broke"):
        extract(df, "y", md, seed=0)


def test_zero_features_raises(monkeypatch):
    df, md = _toy()
    monkeypatch.setattr(features.MFE, "extract", lambda self, **kw: ([], []))
    with pytest.raises(MetaFeatureError, match="no meta-features"):
        extract(df, "y", md, seed=0)


def test_timeout_raises(monkeypatch):
    import time

    df, md = _toy()

    def slow(self, **kw):
        time.sleep(2)
        return ["a"], [1.0]

    monkeypatch.setattr(features.MFE, "extract", slow)
    with pytest.raises(MetaFeatureError, match="timed out"):
        extract(df, "y", md, seed=0, timeout_s=0.2)


def test_sampling_caps_rows_and_keeps_all_classes():
    rng = np.random.default_rng(1)
    n = 2000
    y = np.array(["a"] * 1900 + ["b"] * 90 + ["c"] * 10, dtype=object)
    df = pd.DataFrame({"x": rng.normal(size=n), "y": y})
    md = TableMetadata(columns={"x": ColumnType.numerical, "y": ColumnType.categorical},
                       target="y")
    s = sample_rows(df, md, max_rows=100, seed=0)
    assert len(s) == 100
    assert set(s["y"]) == {"a", "b", "c"}
    assert sample_rows(df, md, max_rows=100, seed=0).equals(s)
    out = extract(df, "y", md, seed=0, max_rows=100)
    assert out["nr_inst"] == 100
    assert out["nr_class"] == 3


def test_id_columns_dropped():
    df, md = _toy()
    df["rid"] = [f"r{i}" for i in range(len(df))]
    md = md.model_copy(update={"columns": {**md.columns, "rid": ColumnType.id}})
    out = extract(df, "y", md, seed=0)
    assert out["nr_attr"] == 3
