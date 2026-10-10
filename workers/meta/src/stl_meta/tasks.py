"""Celery task for dataset meta-features and best-generator suggestions."""

from __future__ import annotations

import traceback

from stl_runner import keys
from stl_runner.celery_app import app
from stl_runner.storage import store_from_env

from stl_meta.features import MetaFeatureError, compute_for_dataset
from stl_meta.meta import load_predictor, predict_features, write_json


@app.task(name="stl.metafeatures")
def metafeatures(dataset_id: str, max_rows: int = 5000, timeout_s: float = 300.0) -> str:
    store = store_from_env()
    out_key = keys.dataset_suggestion(dataset_id)
    try:
        rec = compute_for_dataset(store, dataset_id, seed=0, max_rows=max_rows,
                                  timeout_s=timeout_s)
        artifact = load_predictor(store)
        if artifact is None:
            write_json(store, out_key, {
                "available": False,
                "reason": "no predictor artifact has been trained yet",
            })
        else:
            write_json(store, out_key, predict_features(artifact, rec["features"]))
    except MetaFeatureError as exc:
        write_json(store, out_key, {"available": False, "reason": str(exc)})
    except Exception:
        write_json(store, out_key, {"available": False, "reason": traceback.format_exc()})
    return dataset_id
