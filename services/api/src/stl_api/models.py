from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(UTC)


class Dataset(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    source: str  # "upload" | "builtin"
    target: str
    n_rows: int | None = None
    metadata_json: str = "{}"
    fingerprint: str | None = None
    registered: bool = False
    created_at: datetime = Field(
        default_factory=_now, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class Run(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset_id: str = Field(foreign_key="dataset.id", index=True)
    generators_json: str
    seeds_json: str
    preset: str
    weights_json: str | None = None
    allow_gpu_only: bool = False
    created_at: datetime = Field(
        default_factory=_now, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)  # f"{run_id}/{job_id}"
    run_id: str = Field(foreign_key="run.id", index=True)
    job_id: str
    generator: str
    seed: int
    state: str = "queued"
    error: str | None = None
    updated_at: datetime = Field(
        default_factory=_now, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
