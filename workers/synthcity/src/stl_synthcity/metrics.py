"""synthcity's evaluation metrics for one generated table (runs in the synthcity env).

- AlphaPrecision (Alaa et al. 2022, one-class embedding "OC" variants): alpha_precision,
  beta_recall, authenticity.
- SyntheticDetectionXGB: detection_auc (mean cross-validated AUC, real vs synthetic).
- DomiasMIAKDE (van Breugel et al. 2023): membership inference with members = a TRAIN
  sample and non-members / reference = disjoint halves of a CONTROL sample;
  mia_auc and mia_domias_advantage = max(0, 2 * (auc - 0.5)). (The eval worker adds a
  nearest-neighbour distance attack; the reported mia_advantage is the max of both.)

Data are encoded the way synthcity's own ``Metrics.evaluate`` does it: one set of
encoders (label encoders for object / low-cardinality columns, datetime encoders) is fitted
on the union of all tables, then applied to each table. The codes are then min-max scaled
with bounds from the REAL tables only (synthetic values may fall outside [0, 1], as in
synthcity's own covariate normalisation): AlphaPrecision's one-class embedding needs scaled
inputs, and real rows must not move when a synthetic table has outliers.
"""

from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from stl_core.metadata import TableMetadata

logger.disable("synthcity")

# 2: real-only scaling bounds, AlphaPrecision/detection cap 20,000, mia_domias_advantage key.
METRICS_VERSION = 2
METRIC_KEYS = ("alpha_precision", "beta_recall", "authenticity", "detection_auc", "mia_auc",
               "mia_domias_advantage")


def _seed_all(seed: int) -> None:
    # The one-class embedding behind AlphaPrecision trains an unseeded torch network.
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def encode_scaled(real: list[pd.DataFrame], synth: list[pd.DataFrame],
                  cols: list[str]) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Encode all tables with one set of synthcity encoders (fitted on every table, so a
    category gets the same code everywhere), then min-max scale every column with bounds
    from the encoded REAL tables only."""
    from synthcity.plugins.core.dataloader import GenericDataLoader

    def objects(df: pd.DataFrame) -> pd.DataFrame:
        return df[cols].astype({c: object for c in cols if df[c].dtype == "category"})

    real, synth = [objects(d) for d in real], [objects(d) for d in synth]
    _, encoders = GenericDataLoader(pd.concat(real + synth, ignore_index=True)).encode()

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        return GenericDataLoader(df).encode(encoders)[0].dataframe().astype(float)

    real_enc, synth_enc = [encode(d) for d in real], [encode(d) for d in synth]
    stacked = pd.concat(real_enc, ignore_index=True)
    lo = stacked.min()
    span = (stacked.max() - lo).replace(0.0, 1.0)
    return ([(d - lo) / span for d in real_enc], [(d - lo) / span for d in synth_enc])


def compute_synthcity_metrics(train: pd.DataFrame, control: pd.DataFrame,
                              synth: pd.DataFrame, metadata: TableMetadata, seed: int,
                              max_rows: int = 20_000,
                              max_mia_synth_rows: int = 20_000) -> dict:
    """synthcity fidelity and membership-inference metrics on seeded row samples.

    AlphaPrecision and detection compare equal-size TRAIN and synthetic samples of
    n = min(len(train), len(synth), max_rows) rows. DOMIAS uses m = min(len(train),
    len(control) // 2, max_rows) members from TRAIN and m non-members plus m reference rows
    from CONTROL; its synthetic density is fitted on up to ``max_mia_synth_rows`` synthetic
    rows (the whole released table on the benchmark datasets), so that a copied TRAIN row
    is actually in it.
    """
    from synthcity.metrics.eval_detection import SyntheticDetectionXGB
    from synthcity.metrics.eval_privacy import DomiasMIAKDE
    from synthcity.metrics.eval_statistical import AlphaPrecision
    from synthcity.plugins.core.dataloader import GenericDataLoader

    cols = metadata.model_columns
    n = int(min(len(train), len(synth), max_rows))
    m = int(min(len(train), len(control) // 2, max_rows))
    if n < 2 or m < 2:
        raise ValueError(f"too few rows for synthcity metrics (n={n}, m={m})")
    real_s = train[cols].sample(n, random_state=seed).reset_index(drop=True)
    syn_s = synth[cols].sample(n, random_state=seed).reset_index(drop=True)
    members = train[cols].sample(m, random_state=seed + 1).reset_index(drop=True)
    gt = control[cols].sample(2 * m, random_state=seed).reset_index(drop=True)
    syn_mia = synth[cols].sample(min(len(synth), max_mia_synth_rows),
                                 random_state=seed).reset_index(drop=True)

    # Scaling matters: on raw codes AlphaPrecision's one-class embedding collapses on Adult
    # (fnlwgt ~1e5) and every table, even a copy of TRAIN, scores alpha = beta = 0. Detection
    # (trees) and DOMIAS-KDE (full-covariance kernel) are invariant to the affine map.
    (real_e, mem_e, gt_e), (syn_e, syn_mia_e) = encode_scaled(
        [real_s, members, gt], [syn_s, syn_mia], cols)
    real_dl, syn_dl, mem_dl, gt_dl, syn_mia_dl = (
        GenericDataLoader(d) for d in (real_e, syn_e, mem_e, gt_e, syn_mia_e))
    out: dict = {"metrics_version": METRICS_VERSION, "n_rows": n, "n_mia_members": m,
                 **{k: None for k in METRIC_KEYS}}
    errors: dict[str, str] = {}
    workspace = Path(tempfile.mkdtemp(prefix="stl_sc_metrics_"))
    common = dict(random_state=seed, workspace=workspace, use_cache=False)

    def attempt(name: str, fn) -> None:
        # Each metric fails independently; the failure is recorded, the others still run.
        _seed_all(seed)
        try:
            fn()
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    def alpha_precision() -> None:
        ap = AlphaPrecision(**common).evaluate(real_dl, syn_dl)
        out["alpha_precision"] = float(ap["delta_precision_alpha_OC"])
        out["beta_recall"] = float(ap["delta_coverage_beta_OC"])
        out["authenticity"] = float(ap["authenticity_OC"])

    def detection() -> None:
        out["detection_auc"] = float(SyntheticDetectionXGB(**common).evaluate(
            real_dl, syn_dl)["mean"])

    def domias() -> None:
        auc = float(DomiasMIAKDE(**common).evaluate(gt_dl, syn_mia_dl, mem_dl, syn_mia_dl,
                                                    reference_size=m)["aucroc"])
        out["mia_auc"] = auc
        out["mia_domias_advantage"] = float(max(0.0, 2 * (auc - 0.5)))

    try:
        attempt("alpha_precision", alpha_precision)
        attempt("detection", detection)
        attempt("domias", domias)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    if errors:
        out["errors"] = errors
    return out


def compute_stored_job(store, job, max_rows: int = 20_000) -> dict:
    """Compute a generated job's synthcity metrics and write them to
    keys.synthcity_metrics; any failure is written as ``{"error": "..."}``."""
    from stl_runner import keys
    from stl_runner.pipeline import ensure_splits, load_metadata

    try:
        metadata = load_metadata(store, job.dataset_id)
        splits = ensure_splits(store, job.dataset_id, job.seed)
        synth = store.read_df(keys.synthetic(job.run_id, job.job_id))
        result = compute_synthcity_metrics(splits.train, splits.control, synth, metadata,
                                           job.seed, max_rows=max_rows)
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
    store.write_json(keys.synthcity_metrics(job.run_id, job.job_id), result)
    return result
