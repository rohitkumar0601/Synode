import numpy as np
import pandas as pd
from stl_core.catalog import load_generator
from stl_core.metadata import ColumnType, TableMetadata

MD = TableMetadata(columns={"x1": ColumnType.numerical, "x2": ColumnType.numerical,
                            "c": ColumnType.categorical, "y": ColumnType.categorical},
                   target="y")


def _train(n=800):
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": 2 * x1 + rng.normal(scale=0.3, size=n),
                         "c": np.where(x1 > 0, "hi", "lo").astype(object),
                         "y": np.where(x1 > 0.2, "pos", "neg").astype(object)})


def test_copula_via_catalog_keeps_correlation():
    train = _train()
    gen = load_generator("gaussian_copula", seed=0)
    gen.fit(train, MD)
    out = gen.sample(600)
    assert len(out) == 600 and list(out.columns) == list(train.columns)
    assert np.corrcoef(out["x1"], out["x2"])[0, 1] > 0.8
    assert set(out["c"]) <= {"hi", "lo"}
