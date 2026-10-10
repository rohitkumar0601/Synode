import numpy as np
import pandas as pd
import pytest
from stl_core.catalog import load_generator
from stl_core.metadata import ColumnType, TableMetadata


@pytest.mark.network
@pytest.mark.slow
def test_great_contract():
    rng = np.random.default_rng(0)
    x = rng.normal(size=120).round(2)
    train = pd.DataFrame({"x1": x, "c": np.where(x > 0, "hi", "lo").astype(object),
                          "y": np.where(x > 0.2, "p", "n").astype(object)})
    md = TableMetadata(columns={"x1": ColumnType.numerical, "c": ColumnType.categorical,
                                "y": ColumnType.categorical}, target="y")
    gen = load_generator("great", seed=0, epochs=1, batch_size=16)
    gen.fit(train, md)
    out = gen.sample(20)
    assert len(out) == 20 and list(out.columns) == ["x1", "c", "y"]
    assert out["x1"].dtype == float
    # A 1-epoch model emits free-text categories by design; the evaluator reports unseen
    # categories, so we only check the restored dtype here.
    assert out["c"].dtype == object
    assert not out.isna().any().any()
