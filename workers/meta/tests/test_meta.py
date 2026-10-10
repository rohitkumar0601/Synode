import pandas as pd

from stl_meta.meta import lodo_evaluate, predict_features, train_final


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


def test_train_final_predicts_probabilities():
    artifact = train_final(_df())
    got = predict_features(artifact, {"signal": 2.5, "noise": 0.0})
    assert got["available"] is True
    assert got["generator"] == "mst_eps1"
    assert set(got["probabilities"]) == {"gaussian_copula", "mst_eps1"}
    assert got["model"]["n_datasets"] == 6
