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


def test_great_without_epochs_uses_be_great_default():
    """Construct only (no download, no training): budget=library passes epochs=None."""
    gen = load_generator("great", seed=0, epochs=None)
    kw = gen._great_kwargs()
    assert "epochs" not in kw and "batch_size" not in kw
    rp = gen.resolved_params()
    assert rp["library"] == "be-great" and rp["library_version"] == "0.0.14"
    assert rp["params"]["epochs"] == 100 and rp["params"]["batch_size"] == 8
    assert rp["params"]["llm"] == "distilgpt2"
    assert rp["params"]["save_strategy"] == "no"
    assert rp["params"]["sample"]["guided_sampling"] is True
    assert rp["params"]["sample"]["temperature"] == 0.7  # be-great sample() default
    assert rp["overrides"] == ["experiment_dir", "llm", "sample.guided_sampling",
                               "save_strategy"]
    assert "epochs" not in gen.params


def test_great_epochs_override_is_recorded():
    rp = load_generator("great", seed=0, epochs=5, batch_size=16).resolved_params()
    assert rp["params"]["epochs"] == 5 and rp["params"]["batch_size"] == 16
    assert {"epochs", "batch_size"} <= set(rp["overrides"])
