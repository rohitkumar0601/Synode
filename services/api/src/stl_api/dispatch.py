"""Job dispatch: the API builds Celery chains by task name only (no worker imports)."""
from __future__ import annotations

from typing import Protocol

from celery import Celery, chain, signature
from stl_core.catalog import CATALOG
from stl_core.schemas import JobSpec


def job_chain(job: JobSpec, allow_gpu_only: bool = False):
    """generate (generator's queue) -> metrics (q.synthcity) -> evaluate (q.eval)."""
    job_json = job.model_dump_json()
    return chain(
        signature("stl.generate", args=[job_json], kwargs={"allow_gpu_only": allow_gpu_only},
                  queue=CATALOG[job.generator].queue, immutable=True),
        signature("stl.metrics", args=[job_json], queue="q.synthcity", immutable=True),
        signature("stl.evaluate", args=[job_json], queue="q.eval", immutable=True),
    )


class Dispatcher(Protocol):
    def enqueue(self, job: JobSpec, allow_gpu_only: bool) -> None: ...
    def enqueue_metafeatures(self, dataset_id: str) -> None: ...


class NullDispatcher:
    """Records calls in memory (STL_DISPATCH=none, tests)."""

    def __init__(self) -> None:
        self.calls: list[tuple[JobSpec, bool]] = []
        self.meta_calls: list[str] = []

    def enqueue(self, job: JobSpec, allow_gpu_only: bool) -> None:
        self.calls.append((job, allow_gpu_only))

    def enqueue_metafeatures(self, dataset_id: str) -> None:
        self.meta_calls.append(dataset_id)


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

    def enqueue_metafeatures(self, dataset_id: str) -> None:
        self.app.set_current()
        signature("stl.metafeatures", args=[dataset_id], queue="q.meta",
                  immutable=True).apply_async()


def make_dispatcher(dispatch: str, redis_url: str) -> Dispatcher:
    return NullDispatcher() if dispatch == "none" else CeleryDispatcher(redis_url)
