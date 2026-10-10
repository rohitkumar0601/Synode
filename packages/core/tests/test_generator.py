import json
import math
from pathlib import Path

import pandas as pd
import pytest
from stl_core.baselines import build
from stl_core.budget import BUDGET_PROFILES, DEFAULT_PROFILE
from stl_core.generator import Generator, json_safe, resolve_params, signature_defaults


class _Bare(Generator):
    name = "bare"

    def fit(self, train_df, metadata) -> None: ...

    def sample(self, n: int) -> pd.DataFrame:
        return pd.DataFrame()


def _lib(a, b=2, *, c=(1, 2), d=None, **kw): ...


def test_base_generator_resolved_params_is_empty_but_well_formed():
    rp = _Bare(seed=0, x=1).resolved_params()
    assert rp == {"library": None, "library_version": None, "params": {}, "overrides": []}


def test_signature_defaults_skips_required_and_var_args():
    assert signature_defaults(_lib) == {"b": 2, "c": (1, 2), "d": None}
    assert signature_defaults(_lib, skip=("b",)) == {"c": (1, 2), "d": None}


def test_json_safe_stringifies_what_json_cannot_hold():
    out = json_safe({"t": (1, 2), "p": Path("w"), "n": float("nan"), "i": math.inf,
                     "o": object, 3: {"x": [None, True, 1.5]}})
    assert out["t"] == [1, 2] and out["p"] == "w" and out["n"] == "nan"
    assert out["i"] == "inf" and isinstance(out["o"], str)
    assert out["3"] == {"x": [None, True, 1.5]}
    json.dumps(out)  # serialisable


def test_resolve_params_merges_defaults_and_lists_overrides():
    rp = resolve_params("lib", "1.0", _lib, {"b": 50, "d": "x"}, skip=())
    assert rp == {"library": "lib", "library_version": "1.0",
                  "params": {"b": 50, "c": [1, 2], "d": "x"}, "overrides": ["b", "d"]}


@pytest.mark.parametrize("name", ["train_copy", "independent_marginals"])
def test_baselines_resolved_params(name):
    rp = build(name).resolved_params()
    assert rp["library"] == "stl_core" and rp["library_version"]
    assert rp["params"] == {} and rp["overrides"] == []


def test_budget_profiles():
    assert DEFAULT_PROFILE == "library"
    assert BUDGET_PROFILES == {
        "library": {"epochs": None, "time_cap_seconds": 86400},
        "equal50": {"epochs": 50, "time_cap_seconds": 7200},
        "smoke": {"epochs": 5, "time_cap_seconds": 1800},
    }
