import numpy as np
import pytest
from stl_eval.preprocessing import DistanceEncoder, align, unseen_category_shares


def test_align_coerces_and_orders(toy):
    train, _, _, md = toy
    synth = train[["y", "c", "x2", "x1"]].copy()
    synth["x1"] = synth["x1"].astype(str)
    synth.loc[0, "c"] = "never-seen"
    out = align(train, synth, md)
    assert list(out.columns) == md.model_columns
    assert out["x1"].dtype == float
    assert out.loc[0, "c"] == "never-seen"


def test_align_missing_column_named(toy):
    train, _, _, md = toy
    with pytest.raises(ValueError, match="x2"):
        align(train, train.drop(columns=["x2"]), md)


def test_align_non_numeric_column_named(toy):
    train, _, _, md = toy
    bad = train.copy()
    bad["x1"] = "abc"
    with pytest.raises(ValueError, match="x1"):
        align(train, bad, md)


def test_distance_encoder_ranges_and_unseen(toy):
    train, _, _, md = toy
    enc = DistanceEncoder(md).fit(train)
    X = enc.transform(train)
    assert X.dtype == np.float32
    assert X.min() >= 0 and X.max() <= 1
    other = train.head(3).copy()
    other["c"] = "zzz"
    Xo = enc.transform(other)
    assert np.isfinite(Xo).all()
    # layout: x1, x2, then one-hot c (hi, lo, mid), then one-hot y (neg, pos)
    assert X.shape[1] == 7
    assert Xo[:, 2:5].sum() == 0  # unseen category encodes as all zeros


def test_datetime_units_encode_identically():
    import pandas as pd
    from stl_core.metadata import ColumnType, TableMetadata

    md = TableMetadata(
        columns={"t": ColumnType.datetime, "y": ColumnType.categorical}, target="y"
    )
    dates = pd.to_datetime(["2020-01-01", "2021-06-15", "2022-12-31"])
    ns = pd.DataFrame({"t": dates.astype("datetime64[ns]"), "y": ["a", "b", "a"]})
    us = pd.DataFrame({"t": dates.astype("datetime64[us]"), "y": ["a", "b", "a"]})
    enc = DistanceEncoder(md).fit(ns)
    assert np.array_equal(enc.transform(ns), enc.transform(us))
    assert enc.transform(us)[:, 0].tolist() == [0.0, pytest.approx(0.5, abs=0.2), 1.0]


def _numeric_like_md():
    from stl_core.metadata import ColumnType, TableMetadata

    return TableMetadata(columns={"k": ColumnType.categorical, "y": ColumnType.categorical},
                         target="y")


def test_align_maps_numeric_like_categories():
    import pandas as pd

    md = _numeric_like_md()
    real = pd.DataFrame({"k": ["1", "2", "10", "x"], "y": ["a", "b", "a", "b"]}, dtype=object)
    synth = pd.DataFrame({"k": [1.0, "2.0", 10, "x", "3.0", "zz"],
                          "y": ["a", "b", "a", "b", "a", "b"]})
    out = align(real, synth, md)
    assert out["k"].tolist() == ["1", "2", "10", "x", "3.0", "zz"]


def test_unseen_category_shares():
    import pandas as pd

    md = _numeric_like_md()
    real = pd.DataFrame({"k": ["1", "2"], "y": ["a", "b"]}, dtype=object)
    synth = align(real, pd.DataFrame({"k": ["1.0", "2", "7", "8"], "y": ["a", "a", "b", "c"]}),
                  md)
    assert unseen_category_shares(real, synth, md) == {"k": 0.5, "y": 0.25}
