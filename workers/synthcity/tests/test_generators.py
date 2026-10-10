import numpy as np
import pandas as pd
import pytest
from stl_core.catalog import load_generator
from stl_core.metadata import ColumnType, TableMetadata

MD = TableMetadata(columns={"rid": ColumnType.id, "age": ColumnType.numerical,
                            "x1": ColumnType.numerical, "k": ColumnType.numerical,
                            "c": ColumnType.categorical,
                            "y": ColumnType.categorical}, target="y")


def _train(n=400):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({"rid": [f"r{i}" for i in range(n)],
                         "age": rng.integers(18, 90, n), "x1": x,
                         "k": rng.integers(0, 50, n).astype(float),
                         "c": np.where(x > 0, "hi", "lo").astype(object),
                         "y": np.where(x > 0.2, "p", "n").astype(object)})


FAST = [("marginals", {}), ("bayesian_network", {}), ("privbayes_eps1", {}),
        ("ctgan", {"epochs": 2}), ("tvae", {"epochs": 2})]


@pytest.mark.parametrize("name,params", FAST)
def test_generator_contract(name, params):
    train = _train()
    gen = load_generator(name, seed=0, **params)
    gen.fit(train, MD)
    out = gen.sample(150)
    assert len(out) == 150
    assert list(out.columns) == list(train.columns)
    assert out["c"].dtype == object and set(out["c"]) <= {"hi", "lo"}
    assert out["x1"].dtype == float
    assert out["age"].dtype == train["age"].dtype
    assert out["rid"].is_unique


def test_privbayes_epsilons_are_distinct_configs():
    assert load_generator("privbayes_eps0.1").params["epsilon"] == 0.1
    assert load_generator("privbayes_eps10").params["epsilon"] == 10.0


def test_privbayes_defaults_fix_degree_and_bins_and_overrides_win():
    kw = load_generator("privbayes_eps0.1")._plugin_kwargs()
    assert kw["K"] == 2 and kw["n_bins"] == 20 and kw["epsilon"] == 0.1
    kw = load_generator("privbayes_eps0.1", K=3, n_bins=10)._plugin_kwargs()
    assert kw["K"] == 3 and kw["n_bins"] == 10
    assert "K" not in load_generator("ctgan")._plugin_kwargs()


def test_privbayes_eps0_1_fit_sample():
    train = _train()
    gen = load_generator("privbayes_eps0.1", seed=0)
    gen.fit(train, MD)
    out = gen.sample(100)
    assert len(out) == 100 and list(out.columns) == list(train.columns)


_NOT_REPRODUCIBLE = pytest.mark.xfail(
    strict=True,
    reason="marginal_distributions builds a fresh unseeded np.random.default_rng() for every "
           "generate() call (Schema.from_constraints), so neither global seeding nor "
           "random_state reaches it")
REPRO = [pytest.param(n, p, marks=_NOT_REPRODUCIBLE) if n == "marginals" else (n, p)
         for n, p in FAST]


@pytest.mark.parametrize("name,params", REPRO)
def test_generator_is_reproducible(name, params):
    train = _train()
    outs = []
    for _ in range(2):
        gen = load_generator(name, seed=0, **params)
        gen.fit(train, MD)
        outs.append(gen.sample(50))
    pd.testing.assert_frame_equal(outs[0], outs[1])


def _fitted_marginals(train, md):
    gen = load_generator("marginals", seed=0)
    gen.fit(train, md)
    return gen


def test_restore_types_handles_nan_and_garbage_numerics():
    train = _train(100)
    gen = _fitted_marginals(train, MD)
    real = gen._plugin.generate

    class _Out:
        def __init__(self, df):
            self._df = df

        def dataframe(self):
            return self._df

    def fake(count):
        df = real(count=count).dataframe().copy()
        df["age"] = df["age"].astype(object)
        df.loc[0, "age"] = np.nan
        df.loc[1, "age"] = "oops"
        df.loc[2, "x1"] = np.nan
        return _Out(df)

    gen._plugin.generate = fake
    out = gen.sample(20)
    assert out["age"].dtype == train["age"].dtype and not out["age"].isna().any()
    assert out["age"].iloc[0] == round(train["age"].median())
    assert out["x1"].dtype == float and not out["x1"].isna().any()


def test_numeric_like_categoricals_map_back_to_train_categories():
    md = TableMetadata(columns={"g": ColumnType.categorical, "n": ColumnType.numerical},
                       target="g")
    train = pd.DataFrame({"g": pd.Series(["1", "2", "3", "1"], dtype=object),
                          "n": [1, 2, 3, 4]})
    gen = _fitted_marginals(train, md)
    out = gen._restore_types(pd.DataFrame({"g": [1.0, 2.0, 3.0], "n": [1.2, 2.7, 3.4]}))
    assert list(out["g"]) == ["1", "2", "3"] and out["g"].dtype == object
    assert list(out["n"]) == [1, 3, 3]


def test_restore_types_preserves_unseen_categories():
    from stl_synthcity.generators import capture_train_stats, restore_types
    train = pd.DataFrame({"c": pd.Series(["hi", "lo", "hi"], dtype=object), "n": [1.0, 2.0, 3.0]})
    cats = {"c"}
    medians, cat_values = capture_train_stats(train, cats)
    out = restore_types(pd.DataFrame({"c": ["hi", "free text", None], "n": [1.5, None, 2.0]}),
                        ["c", "n"], train.dtypes.to_dict(), cats, [], medians, cat_values)
    assert list(out["c"]) == ["hi", "free text", "__missing__"] and out["c"].dtype == object
    assert list(out["n"]) == [1.5, 2.0, 2.0]


@pytest.mark.slow
def test_tabddpm_contract():
    train = _train(200)
    gen = load_generator("tabddpm", seed=0, epochs=2)
    gen.fit(train, MD)
    assert len(gen.sample(50)) == 50


def test_unknown_name():
    from stl_synthcity.generators import build
    with pytest.raises(KeyError):
        build("nope")


def test_restore_types_null_numeric_like_categoricals_are_missing_category():
    from stl_synthcity.generators import capture_train_stats, restore_types
    train = pd.DataFrame({"g": pd.Series(["1", "2"], dtype=object)})
    medians, cat_values = capture_train_stats(train, {"g"})
    out = restore_types(pd.DataFrame({"g": [1.0, np.nan]}), ["g"], train.dtypes.to_dict(),
                        {"g"}, [], medians, cat_values)
    assert list(out["g"]) == ["1", "__missing__"]


def test_synthcity_privbayes_nominal_dp_audit():
    """Documents the RQ2 audit finding: synthcity 0.2.12 PrivBayes is NOT epsilon-DP.

    Its _encode never bins numeric columns (`or` where `and` was meant), so each numeric
    column's domain is the exact set of training values (n_bins is a no-op), and its
    Laplace noise is scaled for probabilities but added to counts. Even at a nominal
    epsilon of 0.1 every synthetic numeric value is a training value. This is why the
    catalog lists privbayes_eps* as family "nominal_dp_audit" and the RQ2 epsilon sweep
    uses smartnoise MST instead. If this test starts failing, synthcity changed: re-audit.
    """
    train = _train()
    gen = load_generator("privbayes_eps0.1", seed=0)
    gen.fit(train, MD)
    out = gen.sample(200)
    for col in ("x1", "age"):
        assert out[col].isin(set(train[col])).mean() == 1.0, col
