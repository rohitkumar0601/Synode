"""Privacy computed in the eval env: DCR share, Anonymeter attacks, exact-match and QI-uniqueness
safeguards, and a nearest-neighbour distance membership attack (the second MIA check; the
primary one, synthcity DOMIAS, comes from the `metrics` stage in the synthcity env and is
merged by stl_eval.evaluate)."""

from __future__ import annotations

import itertools
import math
import warnings as _pywarnings

import numpy as np
import pandas as pd
from anonymeter.evaluators import (
    InferenceEvaluator,
    LinkabilityEvaluator,
    SinglingOutEvaluator,
)
from anonymeter.evaluators import singling_out_evaluator as _so_module
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from stl_core.metadata import ColumnType, TableMetadata

from stl_eval.preprocessing import DistanceEncoder


def _nn_distance(refs: np.ndarray, queries: np.ndarray) -> np.ndarray:
    return NearestNeighbors(n_neighbors=1).fit(refs).kneighbors(queries)[0][:, 0]


def dcr_share(train: pd.DataFrame, control: pd.DataFrame, synth: pd.DataFrame,
              metadata: TableMetadata, seed: int, max_rows: int = 5000) -> float:
    enc = DistanceEncoder(metadata).fit(train)
    n = min(len(control), len(train), max_rows)
    C = enc.transform(control.sample(n, random_state=seed))
    S = enc.transform(synth.sample(min(len(synth), max_rows), random_state=seed))
    shuffled = train.sample(frac=1.0, random_state=seed)
    chunks = [shuffled.iloc[i : i + n] for i in range(0, len(shuffled) - n + 1, n)]
    d_control = _nn_distance(C, S)
    shares = []
    for chunk in chunks:
        d_train = _nn_distance(enc.transform(chunk), S)
        shares.append(np.mean(d_train < d_control) + 0.5 * np.mean(d_train == d_control))
    return float(np.mean(shares))


def _row_keys(df: pd.DataFrame, metadata: TableMetadata, cols: list[str]) -> pd.Series:
    """One hash per row over ``cols``, normalised so that dtype differences (int vs float,
    category vs object) do not hide equal values."""
    norm = pd.DataFrame(index=range(len(df)))
    for c in cols:
        s = df[c].reset_index(drop=True)
        ct = metadata.columns[c]
        if ct == ColumnType.numerical:
            norm[c] = pd.to_numeric(s, errors="coerce").astype(float)
        elif ct == ColumnType.datetime:
            norm[c] = pd.to_datetime(s, errors="coerce").astype("datetime64[ns]").astype("int64")
        else:
            norm[c] = s.astype(object).where(s.notna(), None).astype(str)
    return pd.util.hash_pandas_object(norm, index=False)


def exact_match_rates(train: pd.DataFrame, control: pd.DataFrame, synth: pd.DataFrame,
                      metadata: TableMetadata, seed: int) -> dict[str, float]:
    """Exact-match safeguards of the "not attackable" singling-out rule.

    exact_match_train / exact_match_control: share of synthetic rows (model columns) equal
    to some row of a TRAIN / CONTROL reference of the same size, n = min(len(train),
    len(control)) seeded rows each, so a larger reference cannot inflate either rate.
    exact_match_excess = train - control (per seed). exact_match_excess_scaled = that excess
    times len(train) / n (the gated value): the TRAIN reference holds only n / len(train) of
    TRAIN, so a generator copying a share s of TRAIN shows an excess of only s * n / len(train).
    exact_match_train_full: the rate against the whole TRAIN split (reported, not gated).
    """
    cols = metadata.model_columns
    n = min(len(train), len(control))
    syn = _row_keys(synth, metadata, cols)

    def rate(ref: pd.DataFrame) -> float:
        return float(syn.isin(set(_row_keys(ref, metadata, cols))).mean())

    em_train = rate(train.sample(n, random_state=seed))
    em_control = rate(control.sample(n, random_state=seed))
    return {"exact_match_train": em_train, "exact_match_control": em_control,
            "exact_match_excess": em_train - em_control,
            "exact_match_excess_scaled": (em_train - em_control) * len(train) / n,
            "exact_match_train_full": rate(train)}


QI_SUBSET_SIZE = 3
QI_MAX_SUBSETS = 50


def qi_subsets(features: list[str], seed: int) -> list[tuple[str, ...]]:
    """All 3-column subsets of the features, or QI_MAX_SUBSETS distinct seeded random ones
    when there are more (a single subset of all features when there are fewer than 3)."""
    k = min(QI_SUBSET_SIZE, len(features))
    if math.comb(len(features), k) <= QI_MAX_SUBSETS:
        return list(itertools.combinations(features, k))
    rng = np.random.default_rng(seed)
    chosen: set[tuple[str, ...]] = set()
    out: list[tuple[str, ...]] = []
    while len(out) < QI_MAX_SUBSETS:
        idx = tuple(sorted(rng.choice(len(features), size=k, replace=False)))
        subset = tuple(features[i] for i in idx)
        if subset not in chosen:
            chosen.add(subset)
            out.append(subset)
    return out


def qi_unique_share(synth: pd.DataFrame, metadata: TableMetadata, seed: int) -> float:
    """Mean over qi_subsets() of the share of synthetic rows whose values on that subset
    occur once in the synthetic table (an attacker's view with few quasi-identifiers)."""
    shares = [float((~_row_keys(synth, metadata, list(s)).duplicated(keep=False)).mean())
              for s in qi_subsets(metadata.feature_columns, seed)]
    return float(np.mean(shares))


ANONYMETER_KEYS = ("singling_out_risk", "linkability_risk", "inference_risk")
SO_MAX_ROUNDS = 5
SO_OTHER_MODE = {"multivariate": "univariate", "univariate": "multivariate"}


def _clip(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


class _Inconclusive(Exception):
    """Singling-out could not run enough attacks to be conclusive."""

    def __init__(self, message: str, generated: int) -> None:
        super().__init__(message)
        self.generated = generated


class _Unstable(_Inconclusive):
    """Singling-out query count never stabilised: an attack error, not inconclusive."""


def resolve_secrets(inference_secrets: str | list[str], metadata: TableMetadata) -> list[str]:
    """Secret columns for the inference attack; ValueError for unknown columns."""
    cols = metadata.model_columns
    if inference_secrets == "all":
        return list(cols)
    if inference_secrets == "target":
        return [metadata.target]
    secrets = [inference_secrets] if isinstance(inference_secrets, str) else list(
        inference_secrets)
    unknown = [c for c in secrets if c not in cols]
    if unknown or not secrets:
        raise ValueError(f"inference_secrets must be 'all', 'target' or model columns; "
                         f"unknown: {unknown}")
    return secrets


def anonymeter_risks(train: pd.DataFrame, control: pd.DataFrame, synth: pd.DataFrame,
                     metadata: TableMetadata, seed: int, n_attacks: int = 500,
                     singling_out_mode: str = "multivariate",
                     inference_secrets: str | list[str] = "all",
                     min_attacks: int = 100, family_alpha: float = 0.05,
                     details: dict | None = None,
                     ) -> tuple[dict[str, float | None], list[str], dict[str, float | None]]:
    """Anonymeter singling-out, linkability and inference risks with lower confidence bounds.

    Inference is attacked once per secret column ("all": every model column; "target": the
    target only; a column name; or an explicit list), using the other model columns as
    auxiliary knowledge; inference_risk is the max over secrets whose attack succeeded
    (None if all failed).

    Each risk also gets a ``<key>_lower`` bound: the lower end of Anonymeter's two-sided
    interval at level 1 - 2*family_alpha/m with m = 2 + len(secrets), i.e. a one-sided
    Bonferroni-corrected 1 - family_alpha/m bound over all m attacks.
    inference_risk_lower is the max over secrets of their lower bounds. Fewer than
    ``min_attacks`` effective attacks make a risk and its bound None (inconclusive).

    Singling-out runs ``singling_out_mode`` first; if it is inconclusive (fewer than
    ``min_attacks`` queries generated) the other mode runs; the risk and its bound come from
    the conclusive mode ("measured"). If both modes are inconclusive the outcome is
    "not_attackable" (risk None; the gate applies its safeguards). A crash, or a query
    count that never stabilises, is "error" (fails closed).
    When ``details`` is a dict it receives singling_out_status, singling_out_mode_used and
    singling_out_queries (generated query count per mode run).
    Returns (risks, warnings, per-secret inference risks).
    """
    if singling_out_mode not in SO_OTHER_MODE:
        raise ValueError(f"singling_out_mode must be one of {sorted(SO_OTHER_MODE)}; "
                         f"got {singling_out_mode!r}")
    cols = metadata.model_columns
    secrets = resolve_secrets(inference_secrets, metadata)
    ori, syn, ctl = train[cols], synth[cols], control[cols]
    n = int(min(n_attacks, len(ctl), len(syn)))
    feats = metadata.feature_columns
    half = max(1, len(feats) // 2)
    risks: dict[str, float | None] = {}
    warnings: list[str] = []
    if n < min_attacks:
        warnings.append(f"anonymeter: only {n} attacks possible (< min_attacks={min_attacks}); "
                        "privacy inconclusive")
        for key in ANONYMETER_KEYS:
            risks[key] = risks[f"{key}_lower"] = None
        return risks, warnings, {s: None for s in secrets}

    m = 2 + len(secrets)
    level = 1 - 2 * family_alpha / m  # two-sided level whose lower bound is one-sided 1 - alpha/m

    def _bounds(evaluator) -> tuple[float, float]:
        point = evaluator.risk().value
        with _pywarnings.catch_warnings():  # same sanity warnings as the call above
            _pywarnings.simplefilter("ignore")
            lower = evaluator.risk(confidence_level=level).ci[0]
        return _clip(point), _clip(lower)

    def record(key: str, run) -> None:
        np.random.seed(seed)
        try:
            risks[key], risks[f"{key}_lower"] = _bounds(run())
        except Exception as exc:  # each attack fails independently
            risks[key] = risks[f"{key}_lower"] = None
            warnings.append(f"{key}: {type(exc).__name__}: {exc}")

    def singling_out(n_att: int, max_attempts: int, mode: str):
        # Relies on anonymeter==1.0.0 internals: its query generator draws from a
        # module-level rng (np.random.seed does not reach it), and the number of
        # generated queries is not stored on the evaluator, so we record it.
        _so_module.rng = np.random.default_rng(seed)
        generated: list[int] = []
        original = _so_module._generate_singling_out_queries

        def recording(*args, **kwargs):
            out = original(*args, **kwargs)
            generated.append(len(out))
            return out

        _so_module._generate_singling_out_queries = recording
        try:
            evaluator = SinglingOutEvaluator(
                ori=ori, syn=syn, control=ctl, n_attacks=n_att,
                n_cols=min(3, len(cols)),
                # Anonymeter's default (10M) can run for hours when few unique queries exist.
                max_attempts=max_attempts,
            ).evaluate(mode=mode)
        finally:
            _so_module._generate_singling_out_queries = original
        return evaluator, generated[-1]

    cap = max(5_000, 50 * n)

    def singling_out_fixed_point(mode: str):
        # Anonymeter divides by the requested n_attacks, so a shortfall in generated
        # queries would understate risk: re-run with the achieved count until all requested
        # queries are generated (the count is then stable).
        requested = n
        for _ in range(SO_MAX_ROUNDS):
            evaluator, k = singling_out(requested, cap, mode)
            if k >= requested:
                return evaluator, k
            if k < max(1, min_attacks):
                raise _Inconclusive(f"singling_out_risk[{mode}]: query generation fell short "
                                    "of min_attacks; inconclusive", k)
            warnings.append(f"singling_out_risk[{mode}]: only {k} of {requested} queries "
                            f"generated; re-ran with n_attacks={k}")
            requested = k
        # Enough queries every round but no stable count: the attack misbehaved. This is an
        # error (fails closed), never evidence that singling-out is impossible.
        raise _Unstable(f"singling_out_risk[{mode}]: query count did not stabilise in "
                        f"{SO_MAX_ROUNDS} rounds (last {k}); error", k)

    # Configured mode first; the other mode only if the first is inconclusive.
    modes = [singling_out_mode, SO_OTHER_MODE[singling_out_mode]]
    queries: dict[str, int | None] = {}
    measured: tuple[str, float, float] | None = None
    status = None
    for mode in modes:
        np.random.seed(seed)
        try:
            evaluator, queries[mode] = singling_out_fixed_point(mode)
            measured = (mode, *_bounds(evaluator))
            break
        except _Unstable as exc:
            queries[mode] = exc.generated
            warnings.append(str(exc))
            status = "error"
            break
        except _Inconclusive as exc:
            queries[mode] = exc.generated
            warnings.append(str(exc))
        except Exception as exc:  # a crash is an error, never "not attackable"
            queries.setdefault(mode, None)
            warnings.append(f"singling_out_risk[{mode}]: {type(exc).__name__}: {exc}")
            status = "error"
            break
    risks["singling_out_risk"] = risks["singling_out_risk_lower"] = None
    mode_used = None
    if status is None and measured is not None:
        status = "measured"
        mode_used, risks["singling_out_risk"], risks["singling_out_risk_lower"] = measured
    elif status is None:
        status, mode_used = "not_attackable", "both"
    counts = ", ".join(f"{m} {q} queries" for m, q in queries.items())
    warnings.append(f"singling_out_risk: status {status}, mode {mode_used} ({counts}; "
                    f"min_attacks={min_attacks})")
    if details is not None:
        details.update(singling_out_status=status, singling_out_mode_used=mode_used,
                       singling_out_queries=queries)
    record("linkability_risk", lambda: LinkabilityEvaluator(
        ori=ori, syn=syn, control=ctl, n_attacks=n, n_neighbors=10,
        aux_cols=(feats[:half], feats[half:]),
    ).evaluate(n_jobs=-2))
    by_secret: dict[str, float | None] = {}
    lowers: list[float] = []
    for secret in secrets:
        np.random.seed(seed)
        try:
            point, lower = _bounds(InferenceEvaluator(
                ori=ori, syn=syn, control=ctl, n_attacks=n,
                aux_cols=[c for c in cols if c != secret], secret=secret,
                regression=metadata.columns[secret] == ColumnType.numerical,
            ).evaluate(n_jobs=-2))
        except Exception as exc:  # each secret's attack fails independently
            by_secret[secret] = None
            warnings.append(f"inference_risk[{secret}]: {type(exc).__name__}: {exc}")
            continue
        by_secret[secret] = point
        lowers.append(lower)
    succeeded = [v for v in by_secret.values() if v is not None]
    risks["inference_risk"] = max(succeeded) if succeeded else None
    # "Is there evidence that any attribute leaks?": the largest per-secret lower bound.
    risks["inference_risk_lower"] = max(lowers) if lowers else None
    return risks, warnings, by_secret



def mia_distance_advantage(train: pd.DataFrame, control: pd.DataFrame, synth: pd.DataFrame,
                           metadata: TableMetadata, seed: int, max_rows: int = 2000) -> float:
    """Nearest-neighbour membership attack: members = n TRAIN rows, non-members = n CONTROL
    rows (n = min(len(train), len(control), max_rows)); score = minus the distance to the
    nearest synthetic row (up to 20,000 rows) on the shared DistanceEncoder encoding.
    Advantage = max(0, 2 * (AUC - 0.5)). Works on all-categorical tables."""
    enc = DistanceEncoder(metadata).fit(train)
    n = min(len(train), len(control), max_rows)
    members = enc.transform(train.sample(n, random_state=seed))
    non_members = enc.transform(control.sample(n, random_state=seed))
    S = enc.transform(synth.sample(min(len(synth), 20_000), random_state=seed))
    scores = -np.r_[_nn_distance(S, members), _nn_distance(S, non_members)]
    labels = np.r_[np.ones(n), np.zeros(n)]
    return float(max(0.0, 2 * (roc_auc_score(labels, scores) - 0.5)))
