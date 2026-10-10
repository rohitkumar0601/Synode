from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from stl_api import models  # noqa: F401  (register tables)
from stl_api.settings import get_settings


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    url = url or get_settings().database_url
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    return create_engine(url, **kwargs)


def init_db(engine: Engine) -> None:
    """Create tables for SQLite (tests/dev). Postgres is managed by Alembic."""
    if engine.dialect.name == "sqlite":
        SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
