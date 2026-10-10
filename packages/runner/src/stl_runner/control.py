"""Run cancellation through the artifact store.

POST /runs/{id}/cancel writes ``runs/<run>/cancel.json``; every task checks it at start
(``cancel_if_requested``), the generate stage polls it while the child process trains, and
evaluate/metrics check it before writing their output. Retrying a run clears the flag.
"""

from __future__ import annotations

from datetime import UTC, datetime

from stl_core.schemas import JobSpec

from stl_runner import keys
from stl_runner.storage import ArtifactStore

CANCELLED = "cancelled"
CANCEL_ERROR = "cancelled by user"
# Job states that never change again without a retry.
TERMINAL_STATES = frozenset({"done", "failed", CANCELLED})


def request_cancel(store: ArtifactStore, run_id: str) -> None:
    store.write_json(keys.cancel(run_id), {"requested_at": datetime.now(UTC).isoformat()})


def is_cancelled(store: ArtifactStore, run_id: str) -> bool:
    return store.exists(keys.cancel(run_id))


def clear_cancel(store: ArtifactStore, run_id: str) -> None:
    store.delete(keys.cancel(run_id))


def mark_cancelled(store: ArtifactStore, job: JobSpec) -> None:
    """Set the job's state to cancelled unless it already finished (done/failed results stay)."""
    key = keys.status(job.run_id, job.job_id)
    current = store.read_json(key).get("state") if store.exists(key) else "queued"
    if current not in ("done", "failed"):
        store.write_json(key, {"state": CANCELLED, "error": CANCEL_ERROR})


def cancel_if_requested(store: ArtifactStore, job: JobSpec) -> bool:
    """True (and the job marked cancelled) if the job's run was cancelled: skip the work."""
    if not is_cancelled(store, job.run_id):
        return False
    mark_cancelled(store, job)
    return True
