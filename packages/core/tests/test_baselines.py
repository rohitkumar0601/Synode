import numpy as np
import pandas as pd
from stl_core.baselines import IndependentMarginals, TrainCopy, build
from stl_core.metadata import ColumnType, TableMetadata

MD = TableMetadata(
    columns={"a": ColumnType.numerical, "b": ColumnType.numerical, "y": ColumnType.categorical},
    target="y",
)


def _train(n=500):
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": 3 * a + rng.normal(scale=0.1, size=n),
                         "y": np.where(a > 0, "p", "n")})


def test_train_copy_returns_real_rows():
    train = _train()
    g = TrainCopy(seed=0)
    g.fit(train, MD)
    out = g.sample(200)
    assert len(out) == 200
    merged = out.merge(train.drop_duplicates(), how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
    assert len(g.sample(1000)) == 1000  # oversampling with replacement


def test_independent_marginals_breaks_correlation_keeps_shapes():
    train = _train(2000)
    g = IndependentMarginals(seed=0)
    g.fit(train, MD)
    out = g.sample(2000)
    assert list(out.columns) == list(train.columns)
    assert abs(np.corrcoef(out["a"], out["b"])[0, 1]) < 0.1
    assert abs(out["a"].mean() - train["a"].mean()) < 0.1
    assert set(out["y"]) <= set(train["y"])
    assert out["a"].isin(train["a"]).mean() < 0.05  # numerics are not copied values


def test_build_by_name_and_seeded():
    train = _train()
    g1, g2 = build("independent_marginals", seed=5), build("independent_marginals", seed=5)
    g1.fit(train, MD)
    g2.fit(train, MD)
    pd.testing.assert_frame_equal(g1.sample(50), g2.sample(50))
    assert build("train_copy", seed=0).name == "train_copy"
