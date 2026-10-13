"""Celery app shared by every worker. Task modules are listed in STL_CELERY_INCLUDE."""

from __future__ import annotations

import os

from celery import Celery


def make_app(**overrides) -> Celery:
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    include = [m.strip() for m in
               os.environ.get("STL_CELERY_INCLUDE", "stl_runner.tasks").split(",") if m.strip()]
    app = Celery("stl", broker=redis_url, backend=redis_url, include=include)
    app.conf.update(
        task_acks_late=True, worker_prefetch_multiplier=1,
        task_serializer="json", result_serializer="json", accept_content=["json"],
        task_default_queue="q.default", broker_connection_retry_on_startup=True,
        # acks_late + redis: an unacked task is redelivered after visibility_timeout (default
        # 1 h). Keep it above the longest job time cap (library profile: 24 h) so a long fit is
        # never handed to a second worker.
        broker_transport_options={"visibility_timeout": 90_000},
    )
    app.conf.update(overrides)
    return app


app = make_app()
