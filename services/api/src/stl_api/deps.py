"""FastAPI dependencies: DB session, artifact store, settings."""
from __future__ import annotations

from stl_runner.storage import ArtifactStore, store_from_env

from stl_api.db import get_session
from stl_api.settings import Settings, get_settings

__all__ = ["Settings", "get_session", "get_settings", "get_store"]


def get_store() -> ArtifactStore:
    return store_from_env()
