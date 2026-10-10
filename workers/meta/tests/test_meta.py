"""RQ3 meta-dataset, leave-one-dataset-out evaluation and predictor artifact (meta env)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stl_core.schemas import FidelityMetrics, MetricReport, PrivacyMetrics, UtilityMetrics
from stl_meta import meta
from stl_meta.meta import (
    build_meta_dataset,
    load_artifact,
    lodo_evaluate,
    predict,
    regret,
    save_artifact,
    train_final,
)
from stl_runner import keys
from stl_runner.storage import ArtifactStore

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments"))
import run  # noqa: E402


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    monkeypatch.delenv("STL_STORAGE_ROOT", raising=False)


# ---------------------------------------------------------------- synthetic meta-datasets


def _row(i, label, feats, scores, passed=None):
    row = {"dataset_id": f"d{i}", "name": f"d{i}", "label": label, **feats}
    for gen, s in scores.items():
        row[f"score__{gen}"] = s
        row[f"pass__{gen}"] = True if passed is None else gen in passed
    return row


def _signal_df(n=30, seed=0):
    """label = gen_a if `signal` < 0 else gen_b; `noise` is irrelevant."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        x = float(rng.uniform(-3, 3))
        if abs(x) < 0.2:
            x = 0.5 if x >= 0 else -0.5
        label = "gen_a" if x < 0 else "gen_b"
        scores = {"gen_a": 0.9 if label == "gen_a" else 0.5,
                  "gen_b": 0.9 if label == "gen_b" else 0.5}
        rows.append(_row(i, label, {"signal": x, "noise": float(rng.normal())}, scores))
    # imbalance so that the majority baseline is not degenerate
    rows += [_row(n + j, "gen_b", {"signal": 2.0 + j / 10, "noise": 0.0},
                  {"gen_a": 0.5, "gen_b": 0.9}) for j in range(5)]
    return pd.DataFrame(rows)


def test_signal_model_beats_baseline():
    out = lodo_evaluate(_signal_df())
    assert out["insufficient_data"] is False
    assert out["n_labelled"] == 35 and out["n_none"] == 0
    assert out["accuracy"] >= 0.9
    assert out["accuracy"] > out["baseline_accuracy"]
    assert out["balanced_accuracy"] > out["baseline_balanced_accuracy"]
    assert out["mean_regret"] < out["baseline_mean_regret"]
    assert len(out["predictions"]) == 35
    assert set(out["features"]) == {"signal", "noise"}
    assert out["regret_definition"] == meta.REGRET_DEFINITION


def test_random_labels_no_significant_win():
    rng = np.random.default_rng(123)
    n = 40
    rows = [_row(i, str(rng.choice(["gen_a", "gen_b"])),
                 {"f1": float(rng.normal()), "f2": float(rng.normal()),
                  "f3": float(rng.normal())},
                 {"gen_a": 0.6, "gen_b": 0.6}) for i in range(n)]
    out = lodo_evaluate(pd.DataFrame(rows))
    assert out["insufficient_data"] is False
    # chance is ~0.5 for two classes; 0.75 on 40 rows would be p < 0.001 under chance
    assert out["accuracy"] < 0.75
    assert out["balanced_accuracy"] < 0.75


def test_lodo_is_deterministic():
    a = lodo_evaluate(_signal_df(), random_state=0)
    b = lodo_evaluate(_signal_df(), random_state=0)
    assert a["predictions"] == b["predictions"]


# ---------------------------------------------------------------- leakage


def test_imputer_medians_come_from_training_folds_only(monkeypatch):
    rows = [_row(i, "gen_a" if i % 2 else "gen_b", {"f": float(i), "g": float(i % 3)},
                 {"gen_a": 0.5, "gen_b": 0.5}) for i in range(6)]
    rows.append(_row(6, "gen_a", {"f": 1e9, "g": np.nan}, {"gen_a": 0.5, "gen_b": 0.5}))
    rows.append(_row(7, "gen_b", {"f": np.nan, "g": np.inf}, {"gen_a": 0.5, "gen_b": 0.5}))
    df = pd.DataFrame(rows)
    seen = {}
    real = meta.fit_imputer

    def spy(train_x):
        med = real(train_x)
        held_out = sorted(set(df.index) - set(train_x.index))
        seen[tuple(held_out)] = med
        return med

    monkeypatch.setattr(meta, "fit_imputer", spy)
    lodo_evaluate(df)
    assert len(seen) == 8
    finite_f = {i: float(v) for i, v in enumerate(df["f"]) if np.isfinite(v)}
    for (held,), med in seen.items():
        expected = np.median([v for i, v in finite_f.items() if i != held])
        assert med["f"] == pytest.approx(expected)
    # the extreme held-out value never enters its own fold's medians
    assert seen[(6,)]["f"] == pytest.approx(np.median([0, 1, 2, 3, 4, 5]))


def test_imputed_test_row_uses_training_medians():
    train_x = pd.DataFrame({"f": [1.0, 2.0, 3.0, np.inf], "g": [np.nan, np.nan, 5.0, 7.0]})
    med = meta.fit_imputer(train_x)
    assert med["f"] == 2.0 and med["g"] == 6.0
    test_x = meta.apply_imputer(pd.DataFrame({"f": [np.nan], "g": [-np.inf]}), med)
    assert test_x.iloc[0].tolist() == [2.0, 6.0]
    # a column that is all-missing in the training fold falls back to 0.0
    assert meta.fit_imputer(pd.DataFrame({"h": [np.nan, np.nan]}))["h"] == 0.0


def test_baseline_is_mode_of_training_folds_only():
    rows = [_row(0, "A", {"f": 1.0}, {"A": 0.8, "B": 0.4}),
            _row(1, "A", {"f": 2.0}, {"A": 0.8, "B": 0.4}),
            _row(2, "B", {"f": 3.0}, {"A": 0.4, "B": 0.8}),
            _row(3, "B", {"f": 4.0}, {"A": 0.4, "B": 0.8}),
            _row(4, "B", {"f": 5.0}, {"A": 0.4, "B": 0.8})]
    out = lodo_evaluate(pd.DataFrame(rows))
    base = {p["dataset_id"]: p["baseline"] for p in out["predictions"]}
    # holding out a B leaves A,A,B,B -> tie broken alphabetically -> A
    assert base == {"d0": "B", "d1": "B", "d2": "A", "d3": "A", "d4": "A"}
    assert out["baseline_accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------- none / degenerate


def test_none_rows_excluded_and_counted():
    df = _signal_df()
    extra = pd.DataFrame([_row(100, "none", {"signal": -1.0, "noise": 0.0},
                               {"gen_a": 0.9, "gen_b": 0.9}, passed=set()),
                          _row(101, "none", {"signal": 1.0, "noise": 0.0},
                               {"gen_a": 0.9, "gen_b": 0.9}, passed=set())])
    out = lodo_evaluate(pd.concat([df, extra], ignore_index=True))
    assert out["n_none"] == 2
    assert out["none_datasets"] == ["d100", "d101"]
    assert out["n_labelled"] == 35
    assert {p["dataset_id"] for p in out["predictions"]}.isdisjoint({"d100", "d101"})
    assert "none" not in {p["predicted"] for p in out["predictions"]}
    assert out["accuracy"] == lodo_evaluate(df)["accuracy"]


@pytest.mark.parametrize("labels", [["none", "none"], ["gen_a", "gen_b"], ["gen_a"], []])
def test_too_few_labelled_datasets_is_insufficient(labels):
    rows = [_row(i, lab, {"f": float(i)}, {"gen_a": 0.5, "gen_b": 0.6})
            for i, lab in enumerate(labels)]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["dataset_id", "label"])
    out = lodo_evaluate(df)
    assert out["insufficient_data"] is True
    assert "labelled" in out["reason"]
    assert out["predictions"] == [] and out["accuracy"] is None
    assert out["n_none"] == labels.count("none")
    json.dumps(out, allow_nan=False)


def test_single_class_overall_is_insufficient():
    rows = [_row(i, "gen_a", {"f": float(i)}, {"gen_a": 0.5}) for i in range(5)]
    out = lodo_evaluate(pd.DataFrame(rows))
    assert out["insufficient_data"] is True
    assert "class" in out["reason"]


def test_single_class_in_a_training_fold_does_not_crash():
    rows = [_row(i, "gen_a", {"f": float(i)}, {"gen_a": 0.6, "gen_b": 0.5}) for i in range(4)]
    rows.append(_row(4, "gen_b", {"f": 9.0}, {"gen_a": 0.5, "gen_b": 0.6}))
    out = lodo_evaluate(pd.DataFrame(rows))
    assert out["insufficient_data"] is False
    p4 = next(p for p in out["predictions"] if p["dataset_id"] == "d4")
    assert p4["predicted"] == "gen_a" and p4["baseline"] == "gen_a"
    json.dumps(out, allow_nan=False)


# ---------------------------------------------------------------- regret


def test_regret_definition():
    row = pd.Series({"label": "a", "score__a": 0.9, "pass__a": True,
                     "score__b": 0.7, "pass__b": True,
                     "score__c": 0.95, "pass__c": False,
                     "score__d": None, "pass__d": True})
    assert regret(row, "a") == pytest.approx(0.0)
    assert regret(row, "b") == pytest.approx(0.2)
    # gate-failing pick: best minus worst gate-passing score
    assert regret(row, "c") == pytest.approx(0.2)
    # passed the gate but no complete score, or not run at all: same maximal penalty
    assert regret(row, "d") == pytest.approx(0.2)
    assert regret(row, "zzz") == pytest.approx(0.2)


# ---------------------------------------------------------------- build_meta_dataset


def _report(rid, did, gen, seed, q, dcr=0.5, status="done"):
    return MetricReport(
        job_id=f"{gen}__s{seed}", run_id=rid, dataset_id=did, generator=gen, seed=seed,
        status=status, fidelity=FidelityMetrics(quality_score=q),
        utility=UtilityMetrics(importance_spearman=q),
        privacy=PrivacyMetrics(dcr_share=dcr, singling_out_risk=0.0, linkability_risk=0.0,
                               inference_risk=0.0, mia_advantage=0.0))


def _write_run(store, did, spec, seeds=(0, 1)):
    """spec: {generator: (quality, dcr)}; writes a manifest + one report per job."""
    rid = f"meta-{did}"
    jobs, reports = [], []
    for gen, (q, dcr) in spec.items():
        for s in seeds:
            jobs.append({"job_id": f"{gen}__s{s}", "run_id": rid, "dataset_id": did,
                         "generator": gen, "seed": s})
            reports.append(_report(rid, did, gen, s, q + s / 100, dcr))
    store.write_json(keys.run_manifest(rid), {"run_id": rid, "dataset_id": did, "jobs": jobs})
    for r in reports:
        store.write_json(keys.report(rid, r.job_id), r)


def _write_features(store, did, feats):
    store.write_json(keys.dataset_metafeatures(did), {"features": feats, "seed": 0})


@pytest.fixture
def meta_store(tmp_path):
    store = ArtifactStore((tmp_path / "store").as_posix())
    (tmp_path / "datasets.yaml").write_text(
        "suite: 99\ndatasets:\n"
        "- {id: 3, name: kr-vs-kp, target: class}\n"
        "- {id: 11, name: balance-scale, target: class}\n"
        "- {id: 15, name: breast-w, target: Class}\n"
        "- {id: 18, name: mfeat-morph, target: class}\n"
        "- {id: 22, name: no-run, target: class}\n", encoding="utf-8")
    # cc18_3: copula wins; train_copy would score higher but must never be a candidate
    _write_features(store, "cc18_3", {"nr_inst": 100.0, "nr_attr": 5.0, "sd_ratio": None})
    _write_run(store, "cc18_3", {"gaussian_copula": (0.8, 0.5), "ctgan": (0.6, 0.5),
                                 "tvae": (0.95, 0.9), "train_copy": (1.0, 0.5)})
    # cc18_11: every generator fails the gate -> label "none", kept
    _write_features(store, "cc18_11", {"nr_inst": 50.0, "nr_attr": 3.0})
    _write_run(store, "cc18_11", {"gaussian_copula": (0.8, 0.9), "ctgan": (0.6, 0.95)})
    # cc18_15: metafeature extraction failed -> excluded
    store.write_json(keys.dataset_metafeatures("cc18_15"), {"error": "pymfe exploded"})
    _write_run(store, "cc18_15", {"gaussian_copula": (0.8, 0.5)})
    # cc18_18: no metafeatures.json -> excluded
    _write_run(store, "cc18_18", {"gaussian_copula": (0.8, 0.5)})
    # cc18_22: metafeatures but no run -> excluded
    _write_features(store, "cc18_22", {"nr_inst": 10.0, "nr_attr": 1.0})
    return store, tmp_path / "datasets.yaml"


def test_build_meta_dataset(meta_store):
    store, yaml_path = meta_store
    df = build_meta_dataset(store, yaml_path)
    assert df["dataset_id"].tolist() == ["cc18_3", "cc18_11"]
    by = df.set_index("dataset_id")
    assert by.loc["cc18_3", "label"] == "gaussian_copula"
    assert by.loc["cc18_11", "label"] == "none"
    assert by.loc["cc18_3", "nr_inst"] == 100.0
    assert np.isnan(by.loc["cc18_3", "sd_ratio"])
    assert by.loc["cc18_3", "openml_id"] == 3 and by.loc["cc18_3", "name"] == "kr-vs-kp"
    assert by.loc["cc18_3", "n_reports"] == 6  # train_copy reports dropped
    # balanced scores for every candidate, gate-failing ones kept with pass__ = False
    assert by.loc["cc18_3", "score__gaussian_copula"] > by.loc["cc18_3", "score__ctgan"]
    assert by.loc["cc18_3", "score__tvae"] > by.loc["cc18_3", "score__gaussian_copula"]
    assert bool(by.loc["cc18_3", "pass__tvae"]) is False
    assert bool(by.loc["cc18_3", "pass__gaussian_copula"]) is True
    assert not any("train_copy" in c for c in df.columns)
    assert not by.loc["cc18_11", ["pass__gaussian_copula", "pass__ctgan"]].astype(bool).any()
    excluded = {e["dataset_id"]: e["reason"] for e in df.attrs["excluded"]}
    assert set(excluded) == {"cc18_15", "cc18_18", "cc18_22"}
    assert "pymfe exploded" in excluded["cc18_15"]
    assert "metafeatures" in excluded["cc18_18"]
    assert "run" in excluded["cc18_22"]


def test_build_meta_dataset_failed_generator_is_not_a_candidate(tmp_path):
    """mst jobs that failed entirely (failed reports are terminal) are never the label and
    do not crash the build."""
    store = ArtifactStore((tmp_path / "store").as_posix())
    yaml_path = tmp_path / "d.yaml"
    yaml_path.write_text("datasets:\n- {id: 7, name: x, target: y}\n", encoding="utf-8")
    _write_features(store, "cc18_7", {"f": 1.0})
    _write_run(store, "cc18_7", {"gaussian_copula": (0.6, 0.5), "mst_eps1": (0.99, 0.5)})
    rid = "meta-cc18_7"
    for s in (0, 1):  # every mst seed failed
        store.write_json(keys.report(rid, f"mst_eps1__s{s}"),
                         MetricReport(job_id=f"mst_eps1__s{s}", run_id=rid, dataset_id="cc18_7",
                                      generator="mst_eps1", seed=s, status="failed",
                                      error="bounds"))
    df = build_meta_dataset(store, yaml_path)
    assert df.attrs["excluded"] == []
    row = df.iloc[0]
    assert row["label"] == "gaussian_copula"
    assert bool(row["pass__mst_eps1"]) is False and np.isnan(row["score__mst_eps1"])
    assert regret(row, "mst_eps1") == pytest.approx(0.0)  # only one gate-passer


@pytest.mark.parametrize("missing_gen", ["ctgan", "train_copy"])
def test_build_meta_dataset_incomplete_run_is_excluded(tmp_path, missing_gen):
    """A planned job without a report (still running) excludes the dataset; it is never
    labelled none, even when the reported candidates all fail the gate."""
    store = ArtifactStore((tmp_path / "store").as_posix())
    yaml_path = tmp_path / "d.yaml"
    yaml_path.write_text("datasets:\n- {id: 7, name: x, target: y}\n", encoding="utf-8")
    _write_features(store, "cc18_7", {"f": 1.0})
    _write_run(store, "cc18_7", {"gaussian_copula": (0.6, 0.9)})  # fails the gate
    rid = "meta-cc18_7"
    manifest = store.read_json(keys.run_manifest(rid))
    manifest["jobs"].append({"job_id": f"{missing_gen}__s0", "run_id": rid,
                             "dataset_id": "cc18_7", "generator": missing_gen, "seed": 0})
    store.write_json(keys.run_manifest(rid), manifest)
    df = build_meta_dataset(store, yaml_path)
    assert df.empty
    assert df.attrs["excluded"] == [{"dataset_id": "cc18_7",
                                     "reason": "incomplete run (2/3 jobs reported)"}]


def test_build_meta_dataset_score_matches_recommend(meta_store):
    from stl_core.aggregate import aggregate
    from stl_core.decision import recommend

    store, yaml_path = meta_store
    df = build_meta_dataset(store, yaml_path).set_index("dataset_id")
    reps = [MetricReport.model_validate(store.read_json(k))
            for k in store.list("runs/meta-cc18_3") if k.endswith("report.json")]
    rec = recommend(aggregate([r for r in reps if r.generator != "train_copy"]), "balanced")
    for c in rec.ranking:
        assert df.loc["cc18_3", f"score__{c.generator}"] == pytest.approx(c.score)


# ---------------------------------------------------------------- artifact


def test_train_final_save_load_predict_round_trip(tmp_path):
    store = ArtifactStore((tmp_path / "store").as_posix())
    df = _signal_df()
    df.loc[0, "noise"] = np.nan
    df = pd.concat([df, pd.DataFrame([_row(99, "none", {"signal": 0.0, "noise": 0.0},
                                           {"gen_a": 0.1}, passed=set())])],
                   ignore_index=True)
    artifact = train_final(df)
    assert artifact["features"] == ["noise", "signal"]
    assert artifact["labels"] == ["gen_a", "gen_b"]
    assert set(artifact["medians"]) == {"noise", "signal"}
    md = artifact["metadata"]
    assert md["n_datasets"] == 35 and md["n_none"] == 1
    assert md["sklearn_version"] and md["pymfe_version"] == "0.4.4" and md["trained_at"]

    save_artifact(store, artifact)
    assert store.exists("meta/predictor.joblib")
    summary = store.read_json("meta/predictor.json")
    assert "model" not in summary
    assert summary["features"] == ["noise", "signal"] and summary["n_datasets"] == 35

    loaded = load_artifact(store)
    got = predict(loaded, {"signal": 2.5, "noise": 0.1})
    assert got["generator"] == "gen_b"
    assert set(got["probabilities"]) == {"gen_a", "gen_b"}
    assert sum(got["probabilities"].values()) == pytest.approx(1.0)
    assert predict(loaded, {"signal": -2.5})["generator"] == "gen_a"  # missing -> median
    assert predict(loaded, {"signal": -2.5, "noise": None, "extra": 1})["generator"] == "gen_a"
    assert predict(loaded, {"signal": -2.5}) == predict(artifact, {"signal": -2.5})


def test_load_artifact_missing_returns_none(tmp_path):
    assert load_artifact(ArtifactStore((tmp_path / "s").as_posix())) is None


def test_train_final_requires_labelled_rows():
    df = pd.DataFrame([_row(0, "none", {"f": 1.0}, {"a": 0.5}, passed=set())])
    with pytest.raises(ValueError, match="labelled"):
        train_final(df)


# ---------------------------------------------------------------- CLI stage


def test_meta_eval_stage(meta_store, monkeypatch, capsys):
    store, yaml_path = meta_store
    monkeypatch.setenv("STL_STORAGE_ROOT", store.root)
    rc = run.main(["meta-eval", "--datasets-yaml", str(yaml_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "insufficient data" in out
    assert "cc18_15" in out  # exclusions are printed
    lodo = store.read_json("meta/lodo.json")
    assert lodo["insufficient_data"] is True and lodo["n_none"] == 1
    assert "max(P) - min(P)" in lodo["regret_definition"]
    assert {e["dataset_id"] for e in lodo["excluded"]} == {"cc18_15", "cc18_18", "cc18_22"}
    assert not store.exists("meta/predictor.joblib")


def test_meta_eval_stage_train(tmp_path, monkeypatch, capsys):
    store = ArtifactStore((tmp_path / "store").as_posix())
    lines = ["suite: 99", "datasets:"]
    for i in range(6):
        did = f"cc18_{i}"
        lines.append(f"- {{id: {i}, name: ds{i}, target: y}}")
        _write_features(store, did, {"f": float(i), "g": 1.0})
        winner, loser = ("gaussian_copula", "ctgan") if i < 3 else ("ctgan", "gaussian_copula")
        _write_run(store, did, {winner: (0.9, 0.5), loser: (0.5, 0.5)})
    yaml_path = tmp_path / "d.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("STL_STORAGE_ROOT", store.root)
    assert run.main(["meta-eval", "--datasets-yaml", str(yaml_path), "--train"]) == 0
    out = capsys.readouterr().out
    assert "cc18_5" in out and "accuracy" in out
    lodo = store.read_json("meta/lodo.json")
    assert lodo["insufficient_data"] is False and len(lodo["predictions"]) == 6
    art = load_artifact(store)
    assert art["metadata"]["n_datasets"] == 6
    assert predict(art, {"f": 5.0, "g": 1.0})["generator"] == "ctgan"
