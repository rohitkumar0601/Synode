import math

import numpy as np
import pandas as pd
import pytest
from stl_core.catalog import load_generator
from stl_core.metadata import ColumnType, TableMetadata

MD = TableMetadata(columns={"rid": ColumnType.id, "age": ColumnType.numerical,
                            "x1": ColumnType.numerical, "k": ColumnType.numerical,
                            "when": ColumnType.datetime, "c": ColumnType.categorical,
                            "y": ColumnType.categorical}, target="y")


def _train(n=400):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({"rid": [f"r{i}" for i in range(n)],
                         "age": rng.integers(18, 90, n), "x1": x,
                         "k": rng.integers(0, 50, n).astype(float),
                         "when": pd.Timestamp("2020-01-01")
                         + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
                         "c": np.where(x > 0, "hi", "lo").astype(object),
                         "y": np.where(x > 0.2, "p", "n").astype(object)})


def _fit(name="mst_eps1", seed=0, train=None, md=MD, **params):
    gen = load_generator(name, seed=seed, **params)
    gen.fit(_train() if train is None else train, md)
    return gen


def test_mst_contract():
    train = _train()
    out = _fit("mst_eps10", train=train).sample(150)
    assert len(out) == 150
    assert list(out.columns) == list(train.columns)
    assert out.dtypes.to_dict() == train.dtypes.to_dict()
    assert set(out["c"]) <= {"hi", "lo"} and set(out["y"]) <= {"p", "n"}
    assert out["rid"].is_unique
    assert not out.isna().any().any()


def test_mst_catalog_epsilons():
    assert load_generator("mst_eps0.1").params["epsilon"] == 0.1
    assert load_generator("mst_eps10").params["epsilon"] == 10.0


def test_mst_total_budget_equals_catalog_epsilon():
    # Without public bounds, 20% of epsilon infers continuous bounds (subtracted by
    # smartnoise from the synthesizer's budget), the remaining 80% fits MST: the total is
    # the catalog epsilon.
    gen = _fit("mst_eps10")
    spent = gen.privacy_spent
    assert math.isclose(spent["preprocessor"], 2.0)
    assert math.isclose(spent["synthesizer"], 8.0)
    assert math.isclose(spent["preprocessor"] + spent["synthesizer"], 10.0)
    assert spent["bounds"] == "private"


def test_mst_categorical_only_spends_everything_on_mst():
    md = TableMetadata(columns={"c": ColumnType.categorical, "y": ColumnType.categorical},
                       target="y")
    gen = _fit("mst_eps1", train=_train()[["c", "y"]], md=md)
    assert gen.privacy_spent == {"preprocessor": 0.0, "synthesizer": 1.0, "bounds": "none",
                                 "bounds_sha256": _bounds_hash({}, [])}


def test_mst_numeric_output_is_not_a_copy_of_training_values():
    """DP regression: synthetic numerics must not be (a subset of) the exact training values,
    which is how synthcity's PrivBayes leaks (see the synthcity nominal-DP audit test)."""
    train = _train()
    out = _fit("mst_eps10", train=train).sample(400)
    assert out["x1"].isin(set(train["x1"])).mean() < 0.5


def _tv(a: pd.Series, b: pd.Series) -> float:
    p, q = a.value_counts(normalize=True), b.value_counts(normalize=True)
    cats = p.index.union(q.index)
    return 0.5 * float((p.reindex(cats, fill_value=0) - q.reindex(cats, fill_value=0))
                       .abs().sum())


def test_mst_marginal_error_shrinks_with_epsilon():
    """DP regression: a categorical marginal is clearly noisier at epsilon=0.01 than at 100."""
    rng = np.random.default_rng(1)
    n = 1000
    toy = pd.DataFrame({
        "c": rng.choice(list("abcde"), n, p=[0.6, 0.2, 0.1, 0.05, 0.05]).astype(object),
        "y": rng.choice(["p", "n"], n, p=[0.7, 0.3]).astype(object)})
    md = TableMetadata(columns={"c": ColumnType.categorical, "y": ColumnType.categorical},
                       target="y")

    def mean_tv(eps):
        tvs = []
        for seed in range(3):
            gen = load_generator("mst_eps1", seed=seed, epsilon=eps)
            gen.fit(toy, md)
            tvs.append(_tv(toy["c"], gen.sample(n)["c"]))
        return float(np.mean(tvs))

    noisy, accurate = mean_tv(0.01), mean_tv(100.0)
    assert accurate < 0.05
    assert noisy > accurate + 0.1, (noisy, accurate)


@pytest.mark.xfail(
    strict=True,
    reason="MST's Gaussian measurement noise comes from OpenDP (snsynth.utils.gaussian_noise), "
           "which draws from the OS CSPRNG and cannot be seeded; seeding numpy/random only "
           "fixes the exponential-mechanism and decompression draws")
def test_mst_is_reproducible():
    outs = [_fit("mst_eps10", seed=0).sample(50) for _ in range(2)]
    pd.testing.assert_frame_equal(outs[0], outs[1])


def test_restore_types_null_categoricals_are_missing_category():
    from stl_smartnoise.generators import capture_train_stats, restore_types
    train = pd.DataFrame({"c": pd.Series(["hi", "lo"], dtype=object),
                          "g": pd.Series(["1", "2"], dtype=object), "n": [1.0, 2.0]})
    medians, cat_values = capture_train_stats(train, {"c", "g"})
    out = restore_types(pd.DataFrame({"c": ["hi", None], "g": [1.0, np.nan],
                                      "n": [1.5, None]}),
                        ["c", "g", "n"], train.dtypes.to_dict(), {"c", "g"}, [], medians,
                        cat_values)
    assert list(out["c"]) == ["hi", "__missing__"]
    assert list(out["g"]) == ["1", "__missing__"]
    assert list(out["n"]) == [1.5, 1.5]


def test_build_rejects_unknown_name():
    from stl_smartnoise.generators import build
    with pytest.raises(KeyError, match="nope"):
        build("nope")


def test_mst_too_little_budget_for_bounds_fails_clearly():
    # 0.1 * 0.1 epsilon split over 4 continuous columns cannot privately locate bounds in
    # 400 rows: smartnoise's approx_bounds finds no histogram bin above its noise threshold.
    with pytest.raises(ValueError, match="could not privately infer bounds"):
        _fit("mst_eps0.1")


# --- public (codebook) bounds: 0 epsilon on preprocessing ---

PUB_MD = TableMetadata(columns={"rid": ColumnType.id, "age": ColumnType.numerical,
                                "gain": ColumnType.numerical, "c": ColumnType.categorical,
                                "y": ColumnType.categorical}, target="y")
PUB_BOUNDS = {"age": [17, 90], "gain": [0, 99999]}


def _pub_train(n=1000):
    rng = np.random.default_rng(3)
    age = rng.integers(17, 91, n)
    gain = np.where(rng.random(n) < 0.9, 0, rng.integers(1, 20000, n))
    return pd.DataFrame({"rid": [f"r{i}" for i in range(n)], "age": age, "gain": gain,
                         "c": np.where(age > 40, "old", "young").astype(object),
                         "y": rng.choice(["p", "n"], n).astype(object)})


def test_mst_public_bounds_spend_no_epsilon_on_preprocessing():
    # All continuous columns have public bounds: nothing is spent inferring bounds and MST
    # gets the full catalog epsilon -- which makes epsilon=0.1 fit on 1,000 rows.
    gen = _fit("mst_eps0.1", train=_pub_train(), md=PUB_MD, public_bounds=PUB_BOUNDS)
    assert gen.privacy_spent == {"preprocessor": 0.0, "synthesizer": 0.1, "bounds": "public",
                                 "bounds_sha256": _bounds_hash(PUB_BOUNDS, [])}
    out = gen.sample(500)
    assert out["age"].between(17, 90).all()
    assert out["gain"].between(0, 99999).all()


def test_mst_public_bounds_clip_out_of_range_values():
    train = _pub_train()
    train.loc[:9, "age"] = 120  # outside the public bounds: clipped into the top bin
    out = _fit("mst_eps10", train=train, md=PUB_MD, public_bounds=PUB_BOUNDS).sample(500)
    assert out["age"].between(17, 90).all()


def test_mst_public_log_scale_bins_on_log1p():
    # Linear 10 bins over [0, 99999] can only emit midpoints >= ~5000; on the log1p scale
    # the lowest bins resolve small gains.
    train = _pub_train()
    lin = _fit("mst_eps10", train=train, md=PUB_MD, public_bounds=PUB_BOUNDS).sample(1000)
    log = _fit("mst_eps10", train=train, md=PUB_MD, public_bounds=PUB_BOUNDS,
               public_log_scale=["gain"]).sample(1000)
    assert (lin["gain"] >= 4999).all()
    assert (log["gain"] < 100).mean() > 0.5
    assert log["gain"].between(0, 99999).all()


def test_mst_partial_public_bounds_fall_back_to_private_share():
    # 'gain' has no public bound: 20% of epsilon is spent privately on it alone.
    gen = _fit("mst_eps10", train=_pub_train(), md=PUB_MD, public_bounds={"age": [17, 90]})
    spent = gen.privacy_spent
    assert spent["bounds"] == "mixed"
    assert math.isclose(spent["preprocessor"], 2.0)
    assert math.isclose(spent["synthesizer"], 8.0)


@pytest.mark.parametrize("params, match", [
    ({"public_bounds": {"c": [0, 1]}}, "not a continuous column"),
    ({"public_bounds": {"nope": [0, 1]}}, "not a continuous column"),
    ({"public_bounds": {"age": [90, 17]}}, "lower < upper"),
    ({"public_bounds": PUB_BOUNDS, "public_log_scale": ["c"]}, "public_log_scale"),
    ({"public_bounds": {"age": [-5, 90], "gain": [0, 99999]}, "public_log_scale": ["age"]},
     "nonnegative"),
])
def test_mst_public_bounds_validation(params, match):
    with pytest.raises(ValueError, match=match):
        _fit("mst_eps1", train=_pub_train(), md=PUB_MD, **params)


def _bounds_hash(bounds, log_scale):
    import hashlib
    import json

    resolved = {"public_bounds": {c: [float(lo), float(hi)] for c, (lo, hi) in bounds.items()},
                "public_log_scale": sorted(log_scale)}
    return hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()


def test_mst_records_resolved_bounds_hash():
    gen = _fit("mst_eps1", train=_pub_train(), md=PUB_MD, public_bounds=PUB_BOUNDS,
               public_log_scale=["gain"])
    assert gen.privacy_spent["bounds_sha256"] == _bounds_hash(PUB_BOUNDS, ["gain"])
    assert gen.privacy_spent["bounds_sha256"] != _bounds_hash(PUB_BOUNDS, [])


def test_mst_log_scale_lowest_bin_decodes_to_integer_lower_bound():
    # ~90% of training gains are exactly 0 (the public lower bound): the lowest log1p bin
    # decodes to 0, so zeros are reproducible instead of becoming expm1(midpoint) ~ 0.8 -> 1.
    train = _pub_train()
    assert (train["gain"] == 0).mean() > 0.85
    out = _fit("mst_eps10", train=train, md=PUB_MD, public_bounds=PUB_BOUNDS,
               public_log_scale=["gain"]).sample(1000)
    assert (out["gain"] == 0).mean() > 0.6
    assert out["gain"].between(0, 99999).all()


def test_mst_log_scale_lower_bound_decode_only_for_integer_bounds():
    from stl_smartnoise.generators import _log1p_bin_transformer

    t = _log1p_bin_transformer(0.5, 1000.0, "g")
    t.fit([0.5, 10.0, 1000.0])
    assert t._inverse_transform(0) != 0.5  # non-integer lower bound: midpoint decode


def test_mst_log_scale_nan_fails_clearly():
    train = _pub_train().astype({"gain": float})
    train.loc[5, "gain"] = np.nan
    with pytest.raises(ValueError, match="NaN in log-scale column 'gain'"):
        _fit("mst_eps1", train=train, md=PUB_MD, public_bounds=PUB_BOUNDS,
             public_log_scale=["gain"])


def test_mst_resolved_params():
    import json

    rp = load_generator("mst_eps1", seed=0,
                        public_bounds={"age": [0, 100]}).resolved_params()
    assert rp["library"] == "smartnoise-synth" and rp["library_version"]
    assert rp["params"]["epsilon"] == 1.0
    assert rp["params"]["delta"] == 1e-9  # library default
    assert rp["params"]["fit"] == {"bins": 10, "preprocessor_share": 0.2,
                                   "public_bounds": {"age": [0, 100]},
                                   "public_log_scale": []}
    assert rp["overrides"] == ["epsilon", "fit.transformer"]
    json.dumps(rp)
