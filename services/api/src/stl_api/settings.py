from __future__ import annotations

import os
from dataclasses import dataclass

from stl_core.budget import DEFAULT_PROFILE, check_profile


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    max_upload_mb: int
    cors_origins: list[str]
    dispatch: str  # "celery" | "none"
    # Default training-budget profile for POST /runs (stl_core.budget; a run may override it).
    budget_profile: str = DEFAULT_PROFILE
    # Per-job time cap. None (STL_TIME_CAP_SECONDS unset) means the run's budget profile cap
    # (library 86400, equal50 7200, smoke 1800); an explicit value overrides every profile's.
    time_cap_seconds: float | None = None
    # POST /runs/{id}/resume treats a running job whose status has not changed for this long
    # as lost. None (STL_STALL_SECONDS unset) means the job's own time cap + 600 s.
    stall_seconds: float | None = None

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        dispatch = env("STL_DISPATCH", "celery")
        if dispatch not in ("celery", "none"):
            raise ValueError(f"STL_DISPATCH must be 'celery' or 'none', got {dispatch!r}")
        profile = env("STL_BUDGET_PROFILE", DEFAULT_PROFILE)
        try:
            check_profile(profile)
        except ValueError as e:
            raise ValueError(f"STL_BUDGET_PROFILE: {e}") from e
        cap = _seconds("STL_TIME_CAP_SECONDS")
        stall = _seconds("STL_STALL_SECONDS")
        if stall is None and cap is not None:
            stall = cap + 600
        origins = env("STL_CORS_ORIGINS", "http://localhost:3010")
        return cls(
            database_url=env("DATABASE_URL", "sqlite:///./outputs/api.db"),
            redis_url=env("REDIS_URL", "redis://localhost:6379/0"),
            max_upload_mb=int(env("STL_MAX_UPLOAD_MB", "200")),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
            dispatch=dispatch,
            budget_profile=profile,
            time_cap_seconds=cap,
            stall_seconds=stall,
        )


def _seconds(name: str) -> float | None:
    """A positive, finite number of seconds from the environment; None when unset."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a number: {e}") from e
    if not value > 0 or value == float("inf"):
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


def get_settings() -> Settings:
    return Settings.from_env()
