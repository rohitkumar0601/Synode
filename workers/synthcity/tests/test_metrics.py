import os

import numpy as np
import pandas as pd
import pytest
from stl_core.metadata import ColumnType, TableMetadata
from stl_core.split import split
from stl_synthcity.metrics import compute_synthcity_metrics

MD = TableMetadata(
    columns={"rid": ColumnType.id, "x1": ColumnType.numerical, "x2": ColumnType.numerical,
             "c": ColumnType.categorical, "y": ColumnType.categorical},
    target="y",
)


def draw(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 2 * x1 + rng.normal(scale=0.3, size=n)
    c = np.where(x1 + rng.normal(scale=0.5, size=n) > 0, "hi", "lo")
    c = np.where(rng.random(n) < 0.1, "mid", c)
    y = np.where(x1 + 0.5 * (c == "hi") + rng.normal(scale=0.5, size=n) > 0.3, "pos", "neg")
    return pd.DataFrame({"rid": [f"r{seed}-{i}" for i in range(n)], "x1": x1, "x2": x2,
                         "c": c.astype(object), "y": y.astype(object)})


@pytest.fixture(scope="module")
def toy():
    s = split(draw(2000, 0), MD, seed=0)
    return s.train, s.control


@pytest.fixture(scope="module")
def results(toy):
    train, control = toy
    fresh = draw(len(train), 99)
    return (compute_synthcity_metrics(train, control, fresh, MD, seed=0),
            compute_synthcity_metrics(train, control, train, MD, seed=0))


KEYS = {"alpha_precision", "beta_recall", "authenticity", "detection_auc", "mia_auc",
        "mia_domias_advantage"}


def test_reports_all_metrics_in_range(results):
    for r in results:
        assert KEYS <= set(r)
        assert all(0.0 <= r[k] <= 1.0 for k in KEYS), r


def test_train_copy_vs_fresh_ordering(results):
    fresh, copy = results
    assert copy["alpha_precision"] > 0.95
    assert copy["authenticity"] < 0.05 < fresh["authenticity"]
    # duplicate rows invert the cross-validated detector (see decision's 1-2*max(0, AUC-0.5))
    assert copy["detection_auc"] < 0.4 < fresh["detection_auc"] < 0.65
    # Secondary check (the strong copy test is the eval worker's distance attack):
    # DOMIAS-KDE separates a verbatim copy only weakly (Scott-bandwidth KDE smooths away
    # single rows; toy: AUC ~0.54 vs ~0.47 fresh) -- see integration-report.md.
    assert copy["mia_auc"] > fresh["mia_auc"] + 0.03
    assert copy["mia_domias_advantage"] > fresh["mia_domias_advantage"]
    assert fresh["mia_domias_advantage"] < 0.2
    assert "errors" not in fresh and "errors" not in copy
    assert fresh["alpha_precision"] > 0.8 and fresh["beta_recall"] > 0.3


def test_categorical_only_table(toy):
    md = TableMetadata(columns={"c": ColumnType.categorical, "y": ColumnType.categorical},
                       target="y")
    train, control = (d[["c", "y"]] for d in toy)
    synth = draw(len(train), 7)[["c", "y"]]
    synth.loc[:9, "c"] = "unseen"  # a category absent from TRAIN must still encode
    r = compute_synthcity_metrics(train, control, synth, md, seed=0)
    fid = ("alpha_precision", "beta_recall", "authenticity", "detection_auc")
    assert all(np.isfinite(r[k]) for k in fid), r
    # synthcity's DOMIAS needs at least one continuous column (>= 10 distinct values): its
    # failure is recorded, the other metrics are kept
    assert r["mia_domias_advantage"] is None and r["mia_auc"] is None
    assert "at least one continuous feature" in r["errors"]["domias"]


def test_deterministic_and_leaves_no_cache(toy, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train, control = toy
    fresh = draw(len(train), 5)
    a = compute_synthcity_metrics(train, control, fresh, MD, seed=3, max_rows=400)
    b = compute_synthcity_metrics(train, control, fresh, MD, seed=3, max_rows=400)
    assert a == b
    assert a["n_rows"] == 400 and a["n_mia_members"] == 200  # len(control) // 2
    assert os.listdir(tmp_path) == []  # no ./workspace cache written


def test_compute_stored_job_writes_metrics_or_error(tmp_path, toy):
    from stl_core.schemas import JobSpec
    from stl_runner import keys
    from stl_runner.pipeline import generate_job, plan_run, register_dataset
    from stl_runner.storage import ArtifactStore
    from stl_synthcity.metrics import compute_stored_job

    store = ArtifactStore(str(tmp_path))
    register_dataset(store, "toy", draw(1000, 1), MD)
    (job,) = plan_run(store, "r", "toy", ["independent_marginals"], [0])
    generate_job(store, job)
    out = compute_stored_job(store, job, max_rows=300)
    assert store.read_json(keys.synthcity_metrics("r", job.job_id)) == out
    assert out["n_rows"] == 300 and 0 <= out["alpha_precision"] <= 1
    missing = JobSpec(job_id="nope__s0", run_id="r", dataset_id="toy", generator="ctgan",
                      seed=0)
    err = compute_stored_job(store, missing)
    assert set(err) == {"error"} and "synthetic.parquet" in err["error"]



@pytest.mark.network
def test_alpha_precision_not_degenerate_on_adult():
    # On Adult's raw codes (fnlwgt ~1e5, capital-gain up to 99,999) the one-class embedding
    # collapsed: every generator, even a copy of TRAIN, scored alpha = beta = authenticity = 0.
    from stl_core.metadata import infer_metadata, prepare
    from stl_runner.datasets import load_dataset

    df, target = load_dataset({"source": "openml", "openml_name": "adult", "version": 2,
                               "target": "class"})
    md = infer_metadata(df, target)
    s = split(prepare(df, md).sample(4000, random_state=0), md, seed=0)
    other = split(prepare(df, md).sample(4000, random_state=1), md, seed=0).train
    r = compute_synthcity_metrics(s.train, s.control, other, md, seed=0, max_rows=1200)
    assert r["alpha_precision"] > 0.5 and r["beta_recall"] > 0.2, r
    assert r["authenticity"] > 0.2, r


def test_scaling_bounds_come_from_real_rows_only(toy):
    from stl_synthcity.metrics import encode_scaled

    train, control = toy
    cols = MD.model_columns
    synth = draw(len(train), 3)
    outlier = synth.copy()
    outlier.loc[0, "x1"] = 1e6  # one extreme synthetic value
    (r1, c1), (s1,) = encode_scaled([train, control], [synth], cols)
    (r2, c2), (s2,) = encode_scaled([train, control], [outlier], cols)
    pd.testing.assert_frame_equal(r1, r2)
    pd.testing.assert_frame_equal(c1, c2)
    real = pd.concat([r1, c1])
    assert real["x1"].min() == 0.0 and real["x1"].max() == 1.0
    assert s2.loc[0, "x1"] > 1.0  # synthetic values may fall outside [0, 1]
