import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import ColumnType, TableMetadata
from stl_core.split import split
from stl_eval.utility import MODELS, tstr


def test_real_as_synthetic_gives_ratio_near_one(toy):
    train, _, test, md = toy
    u = tstr(train, train, test, md, seed=0)
    assert set(u.tstr) == set(MODELS)
    assert all(0.95 <= v <= 1.05 for v in u.tstr.values())
    assert u.importance_spearman > 0.9
    assert all(0 < v <= 1 for v in u.real_scores.values())


def test_shuffled_target_destroys_utility(toy, fresh):
    train, _, test, md = toy
    bad = fresh.copy()
    bad["y"] = np.random.default_rng(0).permutation(bad["y"].to_numpy())
    u = tstr(train, bad, test, md, seed=0)
    assert u.tstr["random_forest"] < 0.85


def test_single_class_synthetic_scores_zero(toy, fresh):
    train, _, test, md = toy
    mono = fresh.copy()
    mono["y"] = "pos"
    u = tstr(train, mono, test, md, seed=0)
    assert u.tstr == {m: 0.0 for m in MODELS}
    assert u.importance_spearman is None


def test_regression_not_supported(toy):
    train, _, test, md = toy
    reg = md.model_copy(update={"target": "x2", "task": "regression"})
    with pytest.raises(NotImplementedError):
        tstr(train, train, test, reg, seed=0)


def test_synthetic_missing_a_real_class():
    rng = np.random.default_rng(0)
    n = 1500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = np.select([x1 > 0.5, x1 < -0.5], ["hi", "lo"], "mid").astype(object)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    md = TableMetadata(columns={"x1": ColumnType.numerical, "x2": ColumnType.numerical,
                                "y": ColumnType.categorical}, target="y")
    s = split(df, md, seed=0)
    synth = s.train[s.train["y"] != "mid"].reset_index(drop=True)
    u = tstr(s.train, synth, s.test, md, seed=0)
    assert set(u.tstr) == set(MODELS)
    assert all(v is not None and np.isfinite(v) and v > 0 for v in u.tstr.values())
    assert all(v is not None and v > 0 for v in u.real_scores.values())
