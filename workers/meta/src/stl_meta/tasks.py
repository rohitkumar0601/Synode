"""Celery task for dataset meta-features and best-generator suggestions."""

from __future__ import annotations

import logging

from stl_runner import keys
from stl_runner.celery_app import app
from stl_runner.storage import store_from_env

from stl_meta.features import compute_for_dataset
from stl_meta.meta import load_predictor, predict_features, write_json

log = logging.getLogger(__name__)


@app.task(name="stl.metafeatures")
def metafeatures(dataset_id: str, max_rows: int = 5000, timeout_s: float = 300.0) -> str:
    """Meta-features of a registered dataset (sampled to ``max_rows``, pymfe given up after
    ``timeout_s``) -> ``datasets/<id>/metafeatures.json`` (an ``error`` record if
    extraction fails). Only when ``meta/predictor.joblib`` exists is
    ``datasets/<id>/suggestion.json`` written (``available: true``); otherwise any stale
    suggestion is removed and the API reports why none is available. Never raises: the
    task is fire-and-forget after registration."""
    try:
        out_key = keys.dataset_suggestion(keys.validate_component(dataset_id, "dataset id"))
    except Exception:
        log.exception("stl.metafeatures: invalid dataset id %r", dataset_id)
        return dataset_id
    try:
        store = store_from_env()
        store.delete(out_key)  # never leave a suggestion computed from older data
        rec = compute_for_dataset(store, dataset_id, seed=0, max_rows=max_rows,
                                  timeout_s=timeout_s)
        artifact = load_predictor(store)
        if artifact is None:
            log.info("stl.metafeatures %s: no predictor trained yet; no suggestion", dataset_id)
        else:
            write_json(store, out_key, predict_features(artifact, rec["features"]))
    except Exception:
        log.exception("stl.metafeatures failed for dataset %s", dataset_id)
    return dataset_id
