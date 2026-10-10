from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    max_upload_mb: int
    cors_origins: list[str]
    dispatch: str  # "celery" | "none"
    time_cap_seconds: float = 7200.0
    # POST /runs/{id}/resume treats a running job whose status has not changed for this long
    # as lost (default: the time cap + 600 s).
    stall_seconds: float = 7800.0

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        dispatch = env("STL_DISPATCH", "celery")
        if dispatch not in ("celery", "none"):
            raise ValueError(f"STL_DISPATCH must be 'celery' or 'none', got {dispatch!r}")
        try:
            cap = float(env("STL_TIME_CAP_SECONDS", "7200"))
        except ValueError as e:
            raise ValueError(f"STL_TIME_CAP_SECONDS must be a number: {e}") from e
        if not cap > 0 or cap == float("inf"):
            raise ValueError(f"STL_TIME_CAP_SECONDS must be positive and finite, got {cap}")
        try:
            stall = float(env("STL_STALL_SECONDS", str(cap + 600)))
        except ValueError as e:
            raise ValueError(f"STL_STALL_SECONDS must be a number: {e}") from e
        if not stall > 0 or stall == float("inf"):
            raise ValueError(f"STL_STALL_SECONDS must be positive and finite, got {stall}")
        origins = env("STL_CORS_ORIGINS", "http://localhost:3010")
        return cls(
            database_url=env("DATABASE_URL", "sqlite:///./outputs/api.db"),
            redis_url=env("REDIS_URL", "redis://localhost:6379/0"),
            max_upload_mb=int(env("STL_MAX_UPLOAD_MB", "200")),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
            dispatch=dispatch,
            time_cap_seconds=cap,
            stall_seconds=stall,
        )


def get_settings() -> Settings:
    return Settings.from_env()
