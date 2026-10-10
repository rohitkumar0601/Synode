import pandas as pd
import pytest
from stl_eval.privacy import (
    anonymeter_risks,
    dcr_share,
    exact_match_rates,
    mia_distance_advantage,
)


def test_dcr_share_copy_vs_fresh(toy, fresh):
    train, control, _, md = toy
    assert dcr_share(train, control, train, md, seed=0) > 0.6
    assert 0.35 < dcr_share(train, control, fresh, md, seed=0) < 0.65


def test_mia_distance_copy_vs_fresh(toy, fresh):
    # The strong membership check (DOMIAS, the primary attack, is only weakly sensitive to a
    # verbatim copy; see workers/synthcity/tests/test_metrics.py).
    train, control, _, md = toy
    assert mia_distance_advantage(train, control, train, md, seed=0) > 0.8
    assert mia_distance_advantage(train, control, fresh, md, seed=0) < 0.2


def test_mia_distance_works_on_categorical_only_tables(toy, fresh):
    from stl_core.metadata import ColumnType, TableMetadata

    train, control, _, _ = toy
    md = TableMetadata(columns={"c": ColumnType.categorical, "y": ColumnType.categorical},
                       target="y")
    adv = mia_distance_advantage(train[["c", "y"]], control[["c", "y"]],
                                 fresh[["c", "y"]], md, seed=0)
    assert 0.0 <= adv <= 1.0


def test_anonymeter_copy_riskier_than_fresh(toy, fresh):
    train, control, _, md = toy
    # Multivariate queries on this 4-column toy reach only ~70 of 200: below the default
    # min_attacks=100 singling-out would be inconclusive (None); lower it to keep coverage.
    copy, w1, _ = anonymeter_risks(train, control, train, md, seed=0, n_attacks=200,
                                   inference_secrets="target", min_attacks=50)
    honest, w2, _ = anonymeter_risks(train, control, fresh, md, seed=0, n_attacks=200,
                                     inference_secrets="target", min_attacks=50)
    assert set(copy) == {"singling_out_risk", "linkability_risk", "inference_risk",
                         "singling_out_risk_lower", "linkability_risk_lower",
                         "inference_risk_lower"}
    assert all(v is not None and 0 <= v <= 1 for v in copy.values()), w1
    assert copy["inference_risk"] > honest["inference_risk"]
    assert copy["linkability_risk"] > honest["linkability_risk"]


def test_anonymeter_clamps_attacks_on_small_control(toy, fresh, monkeypatch):
    import stl_eval.privacy as privacy

    seen: list[int] = []

    def recording(cls):
        class Wrapper(cls):
            def __init__(self, *args, **kwargs):
                seen.append(kwargs["n_attacks"])
                super().__init__(*args, **kwargs)
        return Wrapper

    for name in ("SinglingOutEvaluator", "LinkabilityEvaluator", "InferenceEvaluator"):
        monkeypatch.setattr(privacy, name, recording(getattr(privacy, name)))
    train, control, _, md = toy
    risks, _, _ = anonymeter_risks(train, control.head(30), fresh, md, seed=0, n_attacks=500,
                                   inference_secrets="target", min_attacks=10)
    assert len(seen) >= 3 and max(seen) <= 30
    assert all(v is None or 0 <= v <= 1 for v in risks.values())


def test_anonymeter_singling_out_is_deterministic(toy, fresh):
    train, control, _, md = toy
    # univariate directly: the multivariate toy falls short and would fall back anyway
    a, wa, _ = anonymeter_risks(train, control, fresh, md, seed=3, n_attacks=100,
                                singling_out_mode="univariate", inference_secrets="target")
    b, _, _ = anonymeter_risks(train, control, fresh, md, seed=3, n_attacks=100,
                               singling_out_mode="univariate", inference_secrets="target")
    assert a["singling_out_risk"] is not None, wa
    assert a["singling_out_risk"] == b["singling_out_risk"]
    assert a["singling_out_risk_lower"] == b["singling_out_risk_lower"]


def test_inference_all_secrets_is_max_over_columns(toy):
    train, control, _, md = toy
    risks, warnings, by_secret = anonymeter_risks(train, control, train, md, seed=0,
                                                  n_attacks=100, inference_secrets="all")
    assert set(by_secret) == set(md.model_columns)
    assert all(v is not None for v in by_secret.values()), warnings
    assert risks["inference_risk"] == max(by_secret.values())
    _, _, only_target = anonymeter_risks(train, control, train, md, seed=0, n_attacks=100,
                                         inference_secrets="target")
    assert set(only_target) == {md.target}
    _, _, listed = anonymeter_risks(train, control, train, md, seed=0, n_attacks=100,
                                    inference_secrets=["x1", "c"])
    assert set(listed) == {"x1", "c"}


def test_inference_secret_failures(toy, fresh, monkeypatch):
    import stl_eval.privacy as privacy

    real_cls = privacy.InferenceEvaluator

    def failing_for(bad):
        class Flaky(real_cls):
            def __init__(self, *args, **kwargs):
                if kwargs["secret"] in bad:
                    raise RuntimeError("attack broke")
                super().__init__(*args, **kwargs)
        return Flaky

    train, control, _, md = toy
    monkeypatch.setattr(privacy, "InferenceEvaluator", failing_for({"x1"}))
    risks, warnings, by_secret = anonymeter_risks(train, control, fresh, md, seed=0,
                                                  n_attacks=100, inference_secrets="all")
    assert by_secret["x1"] is None
    assert risks["inference_risk"] == max(v for v in by_secret.values() if v is not None)
    assert [w for w in warnings if w.startswith("inference_risk")] == [
        "inference_risk[x1]: RuntimeError: attack broke"]
    monkeypatch.setattr(privacy, "InferenceEvaluator", failing_for(set(md.model_columns)))
    risks, warnings, _ = anonymeter_risks(train, control, fresh, md, seed=0, n_attacks=100,
                                          inference_secrets="all")
    assert risks["inference_risk"] is None
    assert sum(w.startswith("inference_risk[") for w in warnings) == len(md.model_columns)


def test_inference_secrets_unknown_column(toy, fresh):
    train, control, _, md = toy
    with pytest.raises(ValueError, match="nope"):
        anonymeter_risks(train, control, fresh, md, seed=0, n_attacks=50,
                         inference_secrets=["nope"])


def test_lower_bounds_reported_and_below_point(toy, fresh):
    train, control, _, md = toy
    risks, _, _ = anonymeter_risks(train, control, train, md, seed=0, n_attacks=200,
                                   singling_out_mode="univariate", inference_secrets="target")
    for key in ("singling_out_risk", "linkability_risk", "inference_risk"):
        lo = risks[f"{key}_lower"]
        assert lo is not None and 0 <= lo <= risks[key] <= 1
    assert risks["inference_risk_lower"] > 0.10  # train-copy leaks: evidence even at the bound


def test_honest_generator_lower_bounds_pass(toy, fresh):
    train, control, _, md = toy
    risks, warnings, _ = anonymeter_risks(train, control, fresh, md, seed=0, n_attacks=200,
                                          singling_out_mode="univariate")
    assert risks["inference_risk_lower"] is not None and risks["inference_risk_lower"] <= 0.10
    assert risks["linkability_risk_lower"] <= 0.10


def test_too_few_attacks_is_inconclusive(toy, fresh):
    train, control, _, md = toy
    risks, warnings, _ = anonymeter_risks(train, control.head(30), fresh, md, seed=0,
                                          n_attacks=500, min_attacks=100)
    assert all(v is None for v in risks.values())
    assert any("only 30 attacks possible" in w for w in warnings)


def test_single_secret_string_is_one_column(toy, fresh):
    train, control, _, md = toy
    _, _, by_secret = anonymeter_risks(train, control, fresh, md, seed=0, n_attacks=100,
                                       singling_out_mode="univariate", inference_secrets="x1")
    assert set(by_secret) == {"x1"}


# --- singling-out mode fallback and "not attackable" outcome (follow-up 2) ---

def test_singling_out_falls_back_to_univariate_when_multivariate_short(toy, fresh):
    train, control, _, md = toy
    details: dict = {}
    # Multivariate queries on this 4-column toy reach only ~70 of the default min_attacks=100.
    risks, warnings, _ = anonymeter_risks(train, control, fresh, md, seed=3, n_attacks=100,
                                          inference_secrets="target", details=details)
    assert details["singling_out_status"] == "measured"
    assert details["singling_out_mode_used"] == "univariate"
    q = details["singling_out_queries"]
    assert q["multivariate"] < 100 <= q["univariate"]
    assert risks["singling_out_risk"] is not None and risks["singling_out_risk_lower"] is not None
    uni, _, _ = anonymeter_risks(train, control, fresh, md, seed=3, n_attacks=100,
                                 singling_out_mode="univariate", inference_secrets="target")
    assert risks["singling_out_risk"] == uni["singling_out_risk"]
    assert ("singling_out_risk[multivariate]: query generation fell short of min_attacks; "
            "inconclusive" in warnings)
    assert any(w.startswith("singling_out_risk: status measured, mode univariate") and
               f"multivariate {q['multivariate']} queries" in w for w in warnings), warnings


def test_singling_out_configured_mode_conclusive_skips_fallback(toy, fresh):
    train, control, _, md = toy
    details: dict = {}
    anonymeter_risks(train, control, fresh, md, seed=3, n_attacks=100,
                     singling_out_mode="univariate", inference_secrets="target",
                     details=details)
    assert details["singling_out_status"] == "measured"
    assert details["singling_out_mode_used"] == "univariate"
    assert set(details["singling_out_queries"]) == {"univariate"}


def _coarse_grid():
    # Anonymeter de-duplicates the synthetic table, so duplicating rows is not enough: every
    # value and every 3-column combination must occur in >= 2 distinct rows. A full grid
    # over two values per numeric column does that (24 distinct rows, repeated).
    import itertools

    rows = list(itertools.product([0.0, 1.0], [0.0, 1.0], ["hi", "lo", "mid"],
                                  ["pos", "neg"]))
    return pd.DataFrame(rows * 13, columns=["x1", "x2", "c", "y"])


def test_singling_out_both_modes_short_is_not_attackable(toy):
    train, control, _, md = toy
    details: dict = {}
    risks, warnings, _ = anonymeter_risks(train, control, _coarse_grid(), md, seed=0,
                                          n_attacks=100, inference_secrets="target",
                                          details=details)
    assert details["singling_out_status"] == "not_attackable"
    assert details["singling_out_mode_used"] == "both"
    assert all(v < 100 for v in details["singling_out_queries"].values())
    assert set(details["singling_out_queries"]) == {"multivariate", "univariate"}
    assert risks["singling_out_risk"] is None and risks["singling_out_risk_lower"] is None
    assert risks["linkability_risk"] is not None
    assert any(w.startswith("singling_out_risk: status not_attackable") for w in warnings)


def test_singling_out_crash_is_error(toy, fresh, monkeypatch):
    import stl_eval.privacy as privacy

    class Broken(privacy.SinglingOutEvaluator):
        def evaluate(self, *args, **kwargs):
            raise RuntimeError("attack broke")

    monkeypatch.setattr(privacy, "SinglingOutEvaluator", Broken)
    train, control, _, md = toy
    details: dict = {}
    risks, warnings, _ = anonymeter_risks(train, control, fresh, md, seed=0, n_attacks=100,
                                          inference_secrets="target", details=details)
    assert details["singling_out_status"] == "error"
    assert risks["singling_out_risk"] is None
    assert "singling_out_risk[multivariate]: RuntimeError: attack broke" in warnings


def test_exact_match_rates_use_equal_size_references(toy, fresh):
    train, control, _, md = toy
    share = len(control) / len(train)  # TRAIN is subsampled to len(CONTROL) rows
    copy = exact_match_rates(train, control, train, md, seed=0)
    assert copy["exact_match_train"] == pytest.approx(share)
    assert copy["exact_match_train_full"] == pytest.approx(1.0)
    assert copy["exact_match_control"] == pytest.approx(0.0)
    assert copy["exact_match_excess"] == pytest.approx(share)  # clearly above the margin
    assert copy["exact_match_excess_scaled"] == pytest.approx(1.0)  # all of TRAIN copied
    honest = exact_match_rates(train, control, fresh, md, seed=0)
    assert honest["exact_match_train"] == honest["exact_match_control"] == 0.0
    assert honest["exact_match_excess"] == 0.0 and honest["exact_match_train_full"] == 0.0
    assert honest["exact_match_excess_scaled"] == 0.0
    # control rows copied into the synthetic table match CONTROL, not TRAIN
    ctl = exact_match_rates(train, control, control, md, seed=0)
    assert ctl["exact_match_control"] == pytest.approx(1.0)
    assert ctl["exact_match_train"] == 0.0 and ctl["exact_match_excess"] == pytest.approx(-1.0)
    again = exact_match_rates(train, control, train, md, seed=0)
    assert again == copy  # the TRAIN subsample is seeded


def test_exact_match_survives_dtype_differences_through_align():
    from stl_core.metadata import ColumnType, TableMetadata
    from stl_eval.preprocessing import align

    md = TableMetadata(columns={"n": ColumnType.numerical, "k": ColumnType.categorical,
                                "y": ColumnType.categorical}, target="y")
    real = pd.DataFrame({"n": [1, 2, 3, 4], "k": [10, 20, 10, 20],
                         "y": ["a", "b", "a", "b"]})
    assert real["n"].dtype.kind == "i" and real["k"].dtype.kind == "i"
    # synthetic numerical as float, categorical ints as strings ("10.0" is numerically 10)
    raw = pd.DataFrame({"n": [1.0, 2.0, 3.0, 4.0], "k": ["10", "20.0", "10.0", "20"],
                        "y": ["a", "b", "a", "b"]})
    synth = align(real, raw, md)
    rates = exact_match_rates(real, real, synth, md, seed=0)
    assert rates["exact_match_train_full"] == 1.0
    assert rates["exact_match_control"] == 1.0


def test_qi_unique_share_is_mean_over_three_column_subsets():
    from stl_core.metadata import ColumnType, TableMetadata
    from stl_eval.privacy import qi_unique_share

    md = TableMetadata(columns={"a": ColumnType.categorical, "b": ColumnType.categorical,
                                "c": ColumnType.categorical, "d": ColumnType.numerical,
                                "y": ColumnType.categorical}, target="y")
    n = 40
    df = pd.DataFrame({"a": ["x"] * n, "b": ["p", "q"] * (n // 2), "c": ["u"] * n,
                       "d": [float(i) for i in range(n)], "y": ["s"] * n})
    # C(4, 3) = 4 subsets: the 3 containing the unique column d give 1.0, {a, b, c} gives 0
    assert qi_unique_share(df, md, seed=0) == pytest.approx(0.75)


def test_qi_unique_share_samples_fifty_subsets_when_many(toy, fresh):
    import itertools

    from stl_core.metadata import ColumnType, TableMetadata
    from stl_eval.privacy import qi_subsets, qi_unique_share

    cols = [f"f{i}" for i in range(8)]  # C(8, 3) = 56 > 50
    md = TableMetadata(columns={**{c: ColumnType.categorical for c in cols},
                                "y": ColumnType.categorical}, target="y")
    subsets = qi_subsets(md.feature_columns, seed=0)
    assert len(subsets) == 50 and len(set(subsets)) == 50
    assert all(len(s) == 3 for s in subsets)
    assert set(subsets) <= set(itertools.combinations(cols, 3))
    assert subsets == qi_subsets(md.feature_columns, seed=0)  # seeded
    assert len(qi_subsets(["a", "b"], seed=0)) == 1  # fewer than 3 features: one subset
    train, control, _, tmd = toy  # 3 features -> the single subset of all three
    assert qi_unique_share(fresh, tmd, seed=0) == pytest.approx(1.0)
    assert qi_unique_share(_coarse_grid(), tmd, seed=0) == 0.0


def test_singling_out_unstable_query_count_is_error(toy, fresh, monkeypatch):
    import stl_eval.privacy as privacy

    counts = iter(range(299, 0, -1))  # each round generates one fewer than requested

    def shrinking(*args, **kwargs):
        return [f"x1 == {i}" for i in range(next(counts))]

    monkeypatch.setattr(privacy._so_module, "_generate_singling_out_queries", shrinking)
    train, control, _, md = toy
    details: dict = {}
    risks, warnings, _ = anonymeter_risks(train, control, fresh, md, seed=0, n_attacks=300,
                                          inference_secrets="target", details=details)
    assert details["singling_out_status"] == "error"
    assert details["singling_out_mode_used"] is None
    assert set(details["singling_out_queries"]) == {"multivariate"}  # no fallback on error
    assert risks["singling_out_risk"] is None
    assert any(w.startswith("singling_out_risk[multivariate]: query count did not stabilise")
               for w in warnings), warnings


def test_exact_match_excess_scaled_undoes_sample_dilution(toy, fresh):
    # A generator copying 3% of TRAIN: against a len(CONTROL)-row TRAIN sample only
    # ~len(CONTROL)/len(TRAIN) of the copies are caught, so the raw excess is diluted to
    # ~1% (would pass the 0.01 margin); scaled by len(TRAIN)/n it is ~3% (fails).
    train, control, _, md = toy
    synth = fresh.sample(len(train), replace=True, random_state=1).reset_index(drop=True)
    k = int(round(0.03 * len(synth)))
    synth.iloc[:k] = train.sample(k, random_state=2).to_numpy()
    rates = exact_match_rates(train, control, synth, md, seed=0)
    scale = len(train) / min(len(train), len(control))
    assert rates["exact_match_excess_scaled"] == pytest.approx(
        rates["exact_match_excess"] * scale)
    assert rates["exact_match_excess"] < 0.015
    assert rates["exact_match_excess_scaled"] > 0.02
