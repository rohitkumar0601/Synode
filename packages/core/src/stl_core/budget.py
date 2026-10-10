"""Training-budget profiles (see experiments/conf/budget/*.yaml, which must agree).

* ``library``: no epoch override, so every library trains with its own defaults (e.g.
  synthcity ctgan n_iter=2000, tvae 1000, be-great epochs 100); the benchmark default.
* ``equal50``: the old equal-budget rule, 50 epochs for every deep generator (CPU benchmark).
* ``smoke``: 5 epochs, for quick end-to-end checks.

``epochs`` None means "do not pass an epoch count to the library".
"""

from __future__ import annotations

DEFAULT_PROFILE = "library"
BUDGET_PROFILES: dict[str, dict] = {
    "library": {"epochs": None, "time_cap_seconds": 86400},
    "equal50": {"epochs": 50, "time_cap_seconds": 7200},
    "smoke": {"epochs": 5, "time_cap_seconds": 1800},
}


def check_profile(profile: str) -> str:
    if profile not in BUDGET_PROFILES:
        raise ValueError(f"unknown budget profile {profile!r}; known: {sorted(BUDGET_PROFILES)}")
    return profile
