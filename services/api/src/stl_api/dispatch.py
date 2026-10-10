"""Job dispatch: the API builds Celery chains by task name only (no worker imports)."""
from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from typing import Protocol

from celery import Celery, chain, signature
from stl_core.catalog import CATALOG
from stl_core.schemas import JobSpec

# synthcity metrics get their own queue (a worker in the synthcity image), so they never wait
# behind hours-long training on q.synthcity.
METRICS_QUEUE = "q.metrics"
EVAL_QUEUE = "q.eval"
# Every queue a job's chain can sit in (q.synthcity also holds stl.metrics messages enqueued
# before METRICS_QUEUE existed).
JOB_QUEUES = sorted({g.queue for g in CATALOG.values()} | {"q.synthcity", METRICS_QUEUE,
                                                           EVAL_QUEUE})
JOB_TASKS = ("stl.generate", "stl.metrics", "stl.evaluate")

# (run_id, job_id) -> {(task name, queue)} of messages waiting in (or reserved from) the broker
Pending = dict[tuple[str, str], set[tuple[str, str]]]


def _eval_signatures(job_json: str) -> list:
    return [
        signature("stl.metrics", args=[job_json], queue=METRICS_QUEUE, immutable=True),
        signature("stl.evaluate", args=[job_json], queue=EVAL_QUEUE, immutable=True),
    ]


def job_chain(job: JobSpec, allow_gpu_only: bool = False):
    """generate (generator's queue) -> metrics (q.metrics) -> evaluate (q.eval)."""
    job_json = job.model_dump_json()
    return chain(
        signature("stl.generate", args=[job_json], kwargs={"allow_gpu_only": allow_gpu_only},
                  queue=CATALOG[job.generator].queue, immutable=True),
        signature("stl.metrics", args=[job_json], queue=METRICS_QUEUE, immutable=True),
        signature("stl.evaluate", args=[job_json], queue=EVAL_QUEUE, immutable=True),
    )


def eval_chain(job: JobSpec):
    """metrics -> evaluate only: resumes a job whose synthetic table already exists."""
    return chain(*_eval_signatures(job.model_dump_json()))


def parse_pending(messages: Iterable[tuple[str, str | bytes]]) -> Pending:
    """Index raw Celery (protocol 2, kombu redis) message envelopes by job.

    ``messages`` are (queue, envelope JSON) pairs; envelopes that are not job tasks or cannot
    be decoded are skipped."""
    out: Pending = {}
    for queue, raw in messages:
        try:
            env = json.loads(raw)
            task = env["headers"]["task"]
            if task not in JOB_TASKS:
                continue
            body = env["body"]
            if env.get("properties", {}).get("body_encoding") == "base64":
                body = base64.b64decode(body)
            args = json.loads(body)[0]
            job = json.loads(args[0])
            key = (job["run_id"], job["job_id"])
        except Exception:
            continue
        out.setdefault(key, set()).add((task, queue))
    return out


class Dispatcher(Protocol):
    def enqueue(self, job: JobSpec, allow_gpu_only: bool) -> None: ...
    def enqueue_eval(self, job: JobSpec) -> None: ...
    def enqueue_metafeatures(self, dataset_id: str) -> None: ...
    def pending_jobs(self) -> Pending | None: ...


class NullDispatcher:
    """Records calls in memory (STL_DISPATCH=none, tests). ``pending`` is what
    pending_jobs reports (tests set it)."""

    def __init__(self) -> None:
        self.calls: list[tuple[JobSpec, bool]] = []
        self.eval_calls: list[JobSpec] = []
        self.meta_calls: list[str] = []
        self.pending: Pending = {}

    def enqueue(self, job: JobSpec, allow_gpu_only: bool) -> None:
        self.calls.append((job, allow_gpu_only))

    def enqueue_eval(self, job: JobSpec) -> None:
        self.eval_calls.append(job)

    def enqueue_metafeatures(self, dataset_id: str) -> None:
        self.meta_calls.append(dataset_id)

    def pending_jobs(self) -> Pending | None:
        return self.pending


class CeleryDispatcher:
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._app: Celery | None = None

    @property
    def app(self) -> Celery:
        if self._app is None:
            # No result backend: the API never reads task results (status lives in the store).
            self._app = Celery("stl_api", broker=self.redis_url)
            self._app.conf.update(task_serializer="json", result_serializer="json",
                                  task_ignore_result=True,
                                  accept_content=["json"], task_default_queue="q.default",
                                  broker_connection_retry_on_startup=True)
        return self._app

    def enqueue(self, job: JobSpec, allow_gpu_only: bool) -> None:
        # chain.apply_async(app=...) collides with chain.run's own `app` kwarg, so bind the
        # app as current instead.
        self.app.set_current()
        job_chain(job, allow_gpu_only).apply_async()

    def enqueue_eval(self, job: JobSpec) -> None:
        self.app.set_current()
        eval_chain(job).apply_async()

    def enqueue_metafeatures(self, dataset_id: str) -> None:
        self.app.set_current()
        signature("stl.metafeatures", args=[dataset_id], queue="q.meta",
                  immutable=True).apply_async()

    def pending_jobs(self) -> Pending | None:
        """Job messages waiting in the redis queues or reserved by a worker (kombu's
        ``unacked`` hash). None when the broker cannot be inspected (not redis)."""
        with self.app.connection_for_read() as conn:
            client = getattr(conn.default_channel, "client", None)
            if client is None or not hasattr(client, "lrange"):
                return None
            raw: list[tuple[str, str | bytes]] = []
            for queue in JOB_QUEUES:
                raw.extend((queue, m) for m in client.lrange(queue, 0, -1))
            for value in client.hvals("unacked"):
                try:
                    payload, _exchange, routing_key = json.loads(value)
                    raw.append((routing_key, json.dumps(payload)))
                except Exception:
                    continue
        return parse_pending(raw)


def make_dispatcher(dispatch: str, redis_url: str) -> Dispatcher:
    return NullDispatcher() if dispatch == "none" else CeleryDispatcher(redis_url)
