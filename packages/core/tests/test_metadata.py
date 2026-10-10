import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import (
    ColumnType,
    TableMetadata,
    infer_metadata,
    prepare,
)


def _frame(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "age": rng.integers(18, 90, n),
            "income": rng.normal(50_000, 10_000, n),
            "city": rng.choice(["a", "b", "c"], n),
            "children": rng.integers(0, 4, n),
            "joined": pd.date_range("2020-01-01", periods=n, freq="D"),
            "label": rng.choice(["yes", "no"], n),
        }
    )


def test_infer_metadata_types():
    md = infer_metadata(_frame(), target="label")
    assert md.columns["customer_id"] == ColumnType.id
    assert md.columns["age"] == ColumnType.numerical
    assert md.columns["income"] == ColumnType.numerical
    assert md.columns["city"] == ColumnType.categorical
    assert md.columns["children"] == ColumnType.categorical  # <= 10 distinct integers
    assert md.columns["joined"] == ColumnType.datetime
    assert md.columns["label"] == ColumnType.categorical
    assert md.task == "classification"


def test_feature_and_model_columns():
    md = infer_metadata(_frame(), target="label")
    assert "customer_id" not in md.model_columns
    assert "label" in md.model_columns
    assert "label" not in md.feature_columns


def test_target_must_exist_and_not_be_id():
    with pytest.raises(ValueError, match="target"):
        TableMetadata(columns={"a": ColumnType.numerical}, target="b")
    with pytest.raises(ValueError, match="target"):
        TableMetadata(columns={"a": ColumnType.id}, target="a")


def test_infer_missing_target_raises():
    with pytest.raises(ValueError, match="nope"):
        infer_metadata(_frame(), target="nope")


def test_to_sdv_dict():
    md = infer_metadata(_frame(), target="label")
    d = md.to_sdv_dict()
    assert d["columns"]["age"] == {"sdtype": "numerical"}
    assert d["columns"]["customer_id"] == {"sdtype": "id"}


def test_prepare_imputes_and_drops_missing_target():
    df = _frame(20)
    df.loc[0, "income"] = np.nan
    df.loc[1, "city"] = None
    df.loc[2, "label"] = None
    md = infer_metadata(_frame(), target="label")
    out = prepare(df, md)
    assert len(out) == 19
    assert out.isna().sum().sum() == 0
    assert (out["city"] == "__missing__").sum() == 1
    assert out["city"].dtype == object
    assert out.index.tolist() == list(range(19))


def test_prepare_rejects_all_missing_numerical_or_datetime():
    df = _frame(20)
    df["income"] = np.nan
    df["joined"] = pd.NaT
    md = infer_metadata(_frame(), target="label")
    with pytest.raises(ValueError, match="income.*joined|joined.*income"):
        prepare(df, md)


def test_prepare_all_missing_after_dropping_target_rows():
    df = _frame(20)
    df["income"] = np.nan
    df.loc[0, "income"] = 1.0
    df.loc[0, "label"] = None  # the only non-missing income is on a dropped row
    md = infer_metadata(_frame(), target="label")
    with pytest.raises(ValueError, match="income"):
        prepare(df, md)
