"""Validation of public (codebook) bounds for the DP generators, shared by the CLI and MST."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def bound_value_problems(public_bounds: Mapping, log_scale: Sequence) -> list[str]:
    """Problems with public bound values, one message per bad column (empty if valid).

    Each bound must be a list/tuple of exactly two finite numbers [lower, upper] with
    lower < upper; a log-scale column additionally needs lower >= 0 (nonnegative).
    """
    problems = []
    for col, bound in public_bounds.items():
        if isinstance(bound, (str, bytes)) or not isinstance(bound, Sequence) or len(bound) != 2:
            problems.append(f"{col!r}: {bound!r} is not a [lower, upper] pair")
            continue
        try:
            lower, upper = (float(v) for v in bound)
        except (TypeError, ValueError):
            problems.append(f"{col!r}: {list(bound)!r} is not numeric")
            continue
        if not (math.isfinite(lower) and math.isfinite(upper)):
            problems.append(f"{col!r}: bounds must be finite (got [{lower:g}, {upper:g}])")
        elif not lower < upper:
            problems.append(f"{col!r}: need lower < upper (got [{lower:g}, {upper:g}])")
        elif col in log_scale and lower < 0:
            problems.append(f"{col!r}: public_log_scale column must be nonnegative "
                            f"(lower {lower:g})")
    return problems
