import numpy as np
import pandas as pd
import pytest
from stl_core.schemas import MetricReport
from stl_meta.meta import (
    build_meta_dataset,
    load_predictor,
    lodo_evaluate,
    predict_features,
    save_predictor,
    train_final,
)
from stl_runner import keys
from stl_runner.storage import ArtifactStore


def _df():
    rows = []
    for i, x in enumerate([-3, -2, -1, 1, 2, 3]):
        label = "gaussian_copula" if x < 0 else "mst_eps1"
        rows.append({
            "dataset_id": f"d{i}",
            "name": f"d{i}",
            "label": label,
            "signal": float(x),
            "noise": float(i % 2),
            "score_gaussian_copula": 0.9 if label == "gaussian_copula" else 0.4,
            "score_mst_eps1": 0.9 if label == "mst_eps1" else 0.4,
        })
    rows.append({
        "dataset_id": "none",
        "name": "none",
        "label": "none",
        "signal": 0.0,
        "noise": 0.0,
        "score_gaussian_copula": None,
        "score_mst_eps1": None,
    })
    return pd.DataFrame(rows)


def test_lodo_model_beats_baseline_on_signal():
    out = lodo_evaluate(_df())
    assert out["n_none"] == 1
    assert out["accuracy"] > out["baseline_accuracy"]
    assert out["mean_regret"] < out["baseline_mean_regret"]
    assert {p["true"] for p in out["predictions"]} == {"gaussian_copula", "mst_eps1"}


def test_lodo_random_label_runs_cleanly():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(12):
        label = "gen_a" if i % 2 == 0 else "gen_b"
        rows.append({
            "dataset_id": f"r{i}",
            "name": f"r{i}",
            "label": label,
            "feat1": float(rng.normal()),
            "feat2": float(rng.normal()),
            "score_gen_a": 0.5,
            "score_gen_b": 0.5,
        })
    out = lodo_evaluate(pd.DataFrame(rows))
    assert out["n_none"] == 0
    assert len(out["predictions"]) == 12
    assert out["accuracy"] is not None
    assert out["baseline_accuracy"] is not None


def test_lodo_no_leakage_in_imputation_and_baseline():
    """Verify that test fold values do not contaminate training fold medians or mode baseline."""
    # Construct a dataset where test fold has an extreme outlier in feat1 and a unique label
    rows = [
        {"dataset_id": "d0", "name": "d0", "label": "A", "feat1": 10.0, "feat2": 1.0,
         "score_A": 0.8},
        {"dataset_id": "d1", "name": "d1", "label": "A", "feat1": 10.0, "feat2": 2.0,
         "score_A": 0.8},
        {"dataset_id": "d2", "name": "d2", "label": "B", "feat1": 10.0, "feat2": 3.0,
         "score_B": 0.8},
        {"dataset_id": "d3", "name": "d3", "label": "B", "feat1": np.nan, "feat2": 4.0,
         "score_B": 0.8},
    ]
    df = pd.DataFrame(rows)
    out = lodo_evaluate(df)
    assert len(out["predictions"]) == 4
    # Each fold's baseline prediction must be the mode of the OTHER 3 rows
    # When fold d0 (A) is held out, remaining are A, B, B -> mode is B
    p0 = next(p for p in out["predictions"] if p["dataset_id"] == "d0")
    assert p0["baseline"] == "B"
    # When fold d2 (B) is held out, remaining are A, A, B -> mode is A
    p2 = next(p for p in out["predictions"] if p["dataset_id"] == "d2")
    assert p2["baseline"] == "A"


def test_lodo_handles_all_none():
    df = pd.DataFrame([
        {"dataset_id": "d1", "name": "d1", "label": "none", "f1": 1.0},
        {"dataset_id": "d2", "name": "d2", "label": None, "f1": 2.0},
    ])
    out = lodo_evaluate(df)
    assert out["predictions"] == []
    assert out["accuracy"] is None
    assert out["n_none"] == 1


def test_train_final_requires_labelled_data():
    empty_df = pd.DataFrame([{"dataset_id": "d1", "label": "none", "f1": 1.0}])
    with pytest.raises(ValueError, match="no labelled meta-dataset rows"):
        train_final(empty_df)


def test_train_final_predicts_probabilities():
    artifact = train_final(_df())
    got = predict_features(artifact, {"signal": 2.5, "noise": 0.0})
    assert got["available"] is True
    assert got["generator"] == "mst_eps1"
    assert set(got["probabilities"]) == {"gaussian_copula", "mst_eps1"}
    assert got["model"]["n_datasets"] == 6


def test_save_and_load_predictor_round_trip(tmp_path):
    store = ArtifactStore(str(tmp_path / "store"))
    artifact = train_final(_df())
    summary = lodo_evaluate(_df())
    save_predictor(store, artifact, summary)

    loaded = load_predictor(store)
    assert loaded is not None
    assert loaded["features"] == artifact["features"]
    assert loaded["labels"] == artifact["labels"]

    # Predict with loaded artifact
    res = predict_features(loaded, {"signal": -2.0, "noise": 1.0})
    assert res["available"] is True
    assert res["generator"] == "gaussian_copula"

    # Verify summary json
    summary_data = store.read_json(keys.meta_predictor_summary())
    assert summary_data["n_datasets"] == 6
    assert "lodo" in summary_data


def test_build_meta_dataset_with_reports(tmp_path):
    store = ArtifactStore(str(tmp_path / "store"))
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(
        "suite: 99\n"
        "datasets:\n"
        "  - id: 3\n"
        "    name: test_ds\n"
        "    target: y\n"
    )

    # Store metafeatures
    store.write_json(keys.dataset_metafeatures("cc18_3"), {
        "features": {"nr_inst": 100.0, "nr_attr": 5.0},
        "seed": 0,
    })

    # Store run manifest and report
    rid = "meta-cc18_3"
    jid = "copula__s0"
    store.write_json(keys.run_manifest(rid), {
        "run_id": rid,
        "dataset_id": "cc18_3",
        "jobs": [{"job_id": jid, "generator": "gaussian_copula", "seed": 0, "run_id": rid,
                  "dataset_id": "cc18_3"}],
    })
    report = MetricReport(
        job_id=jid,
        run_id=rid,
        dataset_id="cc18_3",
        generator="gaussian_copula",
        seed=0,
        status="done",
        fidelity={"quality_score": 0.8},
        utility={"importance_spearman": 0.7},
        privacy={"dcr_share": 0.5, "singling_out_risk": 0.0, "linkability_risk": 0.0,
                 "inference_risk": 0.0, "mia_advantage": 0.0},
    )
    store.write_json(keys.report(rid, jid), report.model_dump(mode="json"))

    meta_df = build_meta_dataset(store, yaml_path)
    assert len(meta_df) == 1
    assert meta_df.iloc[0]["dataset_id"] == "cc18_3"
    assert meta_df.iloc[0]["label"] == "gaussian_copula"
    assert meta_df.iloc[0]["nr_inst"] == 100.0
    assert meta_df.iloc[0]["score_gaussian_copula"] is not None

