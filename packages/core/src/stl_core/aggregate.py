"""Aggregate per-seed reports into mean ± std (and worst gate values) per generator."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from stl_core.schemas import WORST_KEYS, AggregatedReport, MetricReport


def aggregate(reports: list[MetricReport]) -> list[AggregatedReport]:
    by_gen: dict[str, list[MetricReport]] = defaultdict(list)
    for r in reports:
        by_gen[r.generator].append(r)
    out = []
    for gen in sorted(by_gen):
        rs = by_gen[gen]
        done = [r.flat() for r in rs if r.status == "done"]
        keys = sorted({k for f in done for k in f})
        mean: dict[str, float | None] = {}
        std: dict[str, float | None] = {}
        missing: list[str] = []
        for k in keys:
            vals = [f.get(k) for f in done]
            present = [float(v) for v in vals if v is not None and math.isfinite(v)]
            if len(present) < len(vals):
                missing.append(k)
            if present:
                mean[k] = float(np.mean(present))
                std[k] = float(np.std(present, ddof=1)) if len(present) > 1 else 0.0
            else:
                mean[k], std[k] = None, None
        worst: dict[str, float | None] = {}
        for k in WORST_KEYS:
            present = [float(v) for f in done
                       if (v := f.get(k)) is not None and math.isfinite(v)]
            worst[k] = max(present) if present else None
        out.append(AggregatedReport(generator=gen, n_seeds=len(rs),
                                    n_failed=len(rs) - len(done), mean=mean, std=std,
                                    missing=missing, worst=worst,
                                    singling_out_status=[r.privacy.singling_out_status
                                                         for r in rs if r.status == "done"]))
    return out
