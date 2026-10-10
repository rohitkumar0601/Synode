"""Artifact key layout. Every stage reads and writes through these keys."""

import re


def dataset_raw(dataset_id: str) -> str:
    return f"datasets/{dataset_id}/raw.parquet"


def dataset_metadata(dataset_id: str) -> str:
    return f"datasets/{dataset_id}/metadata.json"


def dataset_fingerprint(dataset_id: str) -> str:
    return f"datasets/{dataset_id}/fingerprint.json"


def dataset_public(dataset_id: str) -> str:
    """Public (codebook) information, e.g. numeric bounds for DP generators; see
    pipeline.register_dataset."""
    return f"datasets/{dataset_id}/public.json"


def dataset_metafeatures(dataset_id: str) -> str:
    return f"datasets/{dataset_id}/metafeatures.json"


def dataset_suggestion(dataset_id: str) -> str:
    return f"datasets/{dataset_id}/suggestion.json"


def meta_predictor() -> str:
    return "meta/predictor.joblib"


def meta_predictor_summary() -> str:
    return "meta/predictor.json"


def meta_lodo() -> str:
    return "meta/lodo.json"


def split_source(dataset_id: str, seed: int) -> str:
    return f"datasets/{dataset_id}/splits/{seed}/source.json"


def split_part(dataset_id: str, seed: int, part: str) -> str:
    return f"datasets/{dataset_id}/splits/{seed}/{part}.parquet"


def run_manifest(run_id: str) -> str:
    return f"runs/{run_id}/manifest.json"


def synthetic(run_id: str, job_id: str) -> str:
    return f"runs/{run_id}/{job_id}/synthetic.parquet"


def cost(run_id: str, job_id: str) -> str:
    return f"runs/{run_id}/{job_id}/cost.json"


def status(run_id: str, job_id: str) -> str:
    return f"runs/{run_id}/{job_id}/status.json"


def privacy(run_id: str, job_id: str) -> str:
    """A DP generator's privacy_spent record (epsilon split and bounds path)."""
    return f"runs/{run_id}/{job_id}/privacy.json"


def report(run_id: str, job_id: str) -> str:
    return f"runs/{run_id}/{job_id}/report.json"


def synthcity_metrics(run_id: str, job_id: str) -> str:
    """synthcity metrics of a generated job, written by the `metrics` stage (synthcity env)
    and merged by evaluate; ``{"error": "..."}`` when they could not be computed."""
    return f"runs/{run_id}/{job_id}/synthcity_metrics.json"


_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_component(value: str, what: str) -> str:
    """Reject ids that could escape their key prefix (separators, '..', odd characters).

    fullmatch, not match with '$': '$' also matches before a trailing newline."""
    if not isinstance(value, str) or not _COMPONENT.fullmatch(value) or ".." in value:
        raise ValueError(f"invalid {what} {value!r}: must match [A-Za-z0-9][A-Za-z0-9._-]* "
                         "and not contain '..'")
    return value
