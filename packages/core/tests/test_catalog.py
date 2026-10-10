import pytest
from stl_core.catalog import CATALOG, is_available, load_generator


def test_catalog_has_full_lineup():
    expected = {
        "train_copy", "independent_marginals", "marginals", "gaussian_copula",
        "bayesian_network", "ctgan", "tvae", "tabddpm", "great",
        "privbayes_eps0.1", "privbayes_eps1", "privbayes_eps10",
        "mst_eps0.1", "mst_eps1", "mst_eps10",
    }
    assert expected <= set(CATALOG)
    assert CATALOG["gaussian_copula"].queue == "q.sdv"
    assert CATALOG["ctgan"].queue == "q.synthcity"
    assert CATALOG["privbayes_eps1"].params == {"epsilon": 1.0}
    assert CATALOG["tabddpm"].cpu_ok is False


def test_load_core_generator():
    assert load_generator("train_copy", seed=1).seed == 1
    assert is_available("train_copy")


def test_unknown_generator():
    with pytest.raises(KeyError, match="nope"):
        load_generator("nope")


def test_missing_module_message():
    if is_available("ctgan"):
        pytest.skip("synthcity installed in this environment")
    with pytest.raises(ImportError, match="q.synthcity"):
        load_generator("ctgan")


def test_rq2_epsilon_sweep_uses_mst_and_privbayes_is_audit_only():
    for name, eps in [("mst_eps0.1", 0.1), ("mst_eps1", 1.0), ("mst_eps10", 10.0)]:
        info = CATALOG[name]
        assert info.family == "differential_privacy"
        assert info.queue == "q.smartnoise"
        assert info.module == "stl_smartnoise.generators"
        assert info.params == {"epsilon": eps}
    for name in ("privbayes_eps0.1", "privbayes_eps1", "privbayes_eps10"):
        assert CATALOG[name].family == "nominal_dp_audit"
        assert CATALOG[name].queue == "q.synthcity"
    assert [n for n, i in CATALOG.items() if i.family == "differential_privacy"] == [
        "mst_eps0.1", "mst_eps1", "mst_eps10"]
