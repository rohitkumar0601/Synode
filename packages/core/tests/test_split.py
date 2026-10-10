import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import ColumnType, TableMetadata
from stl_core.split import split, stratified_subsample

MD = TableMetadata(
    columns={"x": ColumnType.numerical, "y": ColumnType.categorical}, target="y"
)


def _frame(n=1000, pos_rate=0.2):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"x": rng.normal(size=n), "y": np.where(rng.random(n) < pos_rate, "pos", "neg")}
    )


def test_split_sizes_and_disjoint():
    df = _frame().reset_index().rename(columns={"index": "rid"})
    md = TableMetadata(columns={**MD.columns, "rid": ColumnType.id}, target="y")
    s = split(df, md, seed=0)
    assert abs(len(s.train) - 600) <= 2
    assert abs(len(s.control) - 200) <= 2
    assert len(s.train) + len(s.control) + len(s.test) == 1000
    ids = [set(p["rid"]) for p in (s.train, s.control, s.test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_split_is_stratified():
    df = _frame()
    s = split(df, MD, seed=0)
    overall = (df["y"] == "pos").mean()
    for part in (s.train, s.control, s.test):
        assert abs((part["y"] == "pos").mean() - overall) < 0.02


def test_split_deterministic_per_seed():
    df = _frame()
    a, b, c = split(df, MD, seed=3), split(df, MD, seed=3), split(df, MD, seed=4)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_rare_class_raises_clear_error():
    df = _frame(100)
    df.loc[:2, "y"] = "rare"
    with pytest.raises(ValueError, match="'rare'"):
        split(df, MD, seed=0)


def test_regression_target_uses_quantile_bins():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=500), "t": rng.normal(size=500)})
    md = TableMetadata(
        columns={"x": ColumnType.numerical, "t": ColumnType.numerical},
        target="t",
        task="regression",
    )
    s = split(df, md, seed=0)
    assert len(s.train) + len(s.control) + len(s.test) == 500


def test_stratified_subsample_keeps_ratio():
    df = _frame(5000)
    sub = stratified_subsample(df, MD, n=1000, seed=0)
    assert len(sub) == 1000
    assert abs((sub["y"] == "pos").mean() - (df["y"] == "pos").mean()) < 0.01
    assert stratified_subsample(df, MD, n=10_000, seed=0).shape == df.shape
