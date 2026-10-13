"""Cost: fit / sample seconds and peak RSS per (dataset, generator), plus job outcomes.

Per panel (runs selected exactly like RQ1: ``rq1.select_panel``), one row per (dataset,
generator) of the panel run:

* outcome counts over every planned job: ``done``, ``failed`` (any other error),
  ``time_cap`` (failed with the runner's "time cap of …s exceeded" message), ``cancelled``
  and ``pending`` (queued / still running). The counts are disjoint and sum to ``n_jobs``.
* median, 25th / 75th percentile and IQR (q75 - q25, linear interpolation) of
  ``cost.fit_seconds``, ``cost.sample_seconds`` and ``cost.peak_memory_mb`` (peak process
  RSS) over the **done** jobs only (seeds); a failed / time-capped / cancelled job's partial
  ``cost.json`` would mix interrupted fits into the distribution, so it is only counted.

Nothing is recomputed: the values are the stored cost metrics gathered by ``collect``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stl_analysis.rq1 import (  # noqa: E402
    SMOKE_STAMP,
    UNSPECIFIED,
    Panel,
    _write,
    panel_provenance,
    to_markdown,
)

OUTCOMES = ("done", "failed", "time_cap", "cancelled", "pending")
TIME_CAP_TEXT = "time cap"
MEASURES = (("fit_s", "cost.fit_seconds"), ("sample_s", "cost.sample_seconds"),
            ("peak_rss_mb", "cost.peak_memory_mb"))
COLS = ["panel", "dataset_id", "run_id", "generator", "n_jobs",
        *(f"n_{o}" for o in OUTCOMES), "n_cost",
        *(f"{m}_{s}" for m, _ in MEASURES for s in ("median", "q25", "q75", "iqr"))]
MD_COLS = ["dataset_id", "generator", "n_jobs", "n_done", "n_failed", "n_time_cap",
           "n_cancelled", "n_pending", "fit_s_median", "fit_s_iqr", "sample_s_median",
           "sample_s_iqr", "peak_rss_mb_median", "peak_rss_mb_iqr"]
FIT_COLOR = "#2a78d6"
SAMPLE_COLOR = "#eb6834"
INK = "#333333"
GRID = "#dddddd"


def _text(v: Any) -> str:
    if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v)


def job_outcome(row: pd.Series) -> str:
    """One of OUTCOMES for a collect row (a time-cap failure is not a plain failure)."""
    err, state, report = (_text(row.get(k)) for k in ("error_last_line", "state",
                                                       "report_status"))
    if TIME_CAP_TEXT in err.lower():
        return "time_cap"
    if state == "cancelled":
        return "cancelled"
    if state == "failed" or report == "failed":
        return "failed"
    if state == "done":
        return "done"
    return "pending"


def _stats(values: pd.Series) -> dict[str, float]:
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if v.empty:
        return {"median": math.nan, "q25": math.nan, "q75": math.nan, "iqr": math.nan}
    q25, q75 = float(v.quantile(0.25)), float(v.quantile(0.75))
    return {"median": float(v.median()), "q25": q25, "q75": q75, "iqr": q75 - q25}


def cost_table(df: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    """One row per (dataset, generator) of the panel's runs; see the module docstring."""
    rows: list[dict[str, Any]] = []
    for run in panel.runs:
        jobs = df[(df.store == run["store"]) & (df.run_id == run["run_id"])].copy()
        if jobs.empty:
            continue
        jobs["outcome"] = jobs.apply(job_outcome, axis=1)
        for gen, g in jobs.groupby("generator", sort=True):
            done = g[g.outcome == "done"]
            row: dict[str, Any] = {
                "panel": panel.name, "dataset_id": run["dataset_id"],
                "run_id": run["run_id"], "generator": gen, "n_jobs": int(len(g)),
                **{f"n_{o}": int((g.outcome == o).sum()) for o in OUTCOMES},
                "n_cost": int(done["cost.fit_seconds"].notna().sum())
                if "cost.fit_seconds" in done.columns else 0,
            }
            for name, col in MEASURES:
                vals = done[col] if col in done.columns else pd.Series(dtype=float)
                row.update({f"{name}_{k}": v for k, v in _stats(vals).items()})
            rows.append(row)
    out = pd.DataFrame(rows, columns=COLS)
    for c in COLS[COLS.index("n_cost") + 1:]:
        out[c] = out[c].astype("float64")
    return out.sort_values(["dataset_id", "generator"]).reset_index(drop=True)


_TEX = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
        "$": r"\$", "{": r"\{", "}": r"\}"}


def _tex_escape(v: Any) -> str:
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.3g}"
    s = "" if v is None else str(v)
    return "".join(_TEX.get(ch, ch) for ch in s)


def to_latex(df: pd.DataFrame, caption: str, label: str, stamp: str | None) -> str:
    """A plain booktabs tabular (no jinja2 dependency)."""
    lines = [f"% {stamp}"] if stamp else []
    lines += [r"\begin{table}[ht]", r"\centering", r"\small",
              r"\begin{tabular}{" + "l" * 2 + "r" * (len(df.columns) - 2) + "}", r"\toprule",
              " & ".join(_tex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(_tex_escape(v) for v in r) + r" \\"
              for r in df.itertuples(index=False)]
    cap = f"{stamp}. {caption}" if stamp else caption
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{" + _tex_escape(cap) + "}",
              r"\label{" + label + "}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def cost_figure(table: pd.DataFrame, panel: Panel, out: Path) -> list[str]:
    """Horizontal bars per dataset (small multiples, shared log-scale seconds axis): median
    fit and sample seconds per generator, whiskers = IQR (q25–q75)."""
    datasets = sorted(table.dataset_id.unique())
    if not datasets:
        return []
    n_gen = max(len(table[table.dataset_id == d]) for d in datasets)
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets),
                                                        1.2 + 0.42 * n_gen),
                             squeeze=False, sharex=True)
    positive = table[["fit_s_q25", "sample_s_q25", "fit_s_median", "sample_s_median"]] \
        .to_numpy(float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    floor = float(positive.min()) / 2 if positive.size else 0.01
    for ax, ds in zip(axes[0], datasets, strict=True):
        t = table[table.dataset_id == ds].sort_values("fit_s_median", na_position="first")
        y = np.arange(len(t))
        h = 0.36
        for off, key, color, label in ((h / 2, "fit_s", FIT_COLOR, "fit"),
                                       (-h / 2, "sample_s", SAMPLE_COLOR, "sample")):
            med = t[f"{key}_median"].to_numpy(float)
            lo = t[f"{key}_q25"].to_numpy(float)
            hi = t[f"{key}_q75"].to_numpy(float)
            shown = np.where(np.isfinite(med), np.maximum(med, floor), np.nan)
            ax.barh(y + off, shown - floor, left=floor, height=h - 0.04, color=color,
                    label=f"{label} seconds (median)", zorder=2)
            err = np.vstack([np.clip(shown - np.maximum(lo, floor), 0, None),
                             np.clip(np.maximum(hi, floor) - shown, 0, None)])
            ok = np.isfinite(shown)
            ax.errorbar(shown[ok], (y + off)[ok], xerr=err[:, ok], fmt="none",
                        ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        labels = [f"{g}" + ("" if r.n_done else "  (no done job)")
                  + (f"  [{int(r.n_time_cap)} time cap]" if r.n_time_cap else "")
                  for g, r in zip(t.generator, t.itertuples(), strict=True)]
        ax.set_yticks(y, labels, fontsize=7.5)
        ax.set_xscale("log")
        ax.set_xlabel("seconds (log scale)", fontsize=8, color=INK)
        ax.set_title(ds, fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=INK)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower right")
    title = (f"Cost {panel.name}: fit / sample seconds per generator, done jobs "
             f"(median, whiskers IQR), profile {panel.budget_profile or UNSPECIFIED}")
    if panel.is_smoke:
        title += f"\n{SMOKE_STAMP}"
    fig.suptitle(title, fontsize=9.5, color=INK)
    fig.tight_layout()
    files = []
    for ext in ("png", "pdf"):
        p = out / "figures" / f"cost_{panel.name}_seconds.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        files.append(str(p))
    plt.close(fig)
    return files


def run_cost(df: pd.DataFrame, panel: Panel, out: str | Path) -> dict[str, Any]:
    """Every cost artefact of one panel under ``out``/{tables,figures}."""
    out = Path(out)
    tables = out / "tables"
    stamp = SMOKE_STAMP if panel.is_smoke else None
    tail = f" — {SMOKE_STAMP}" if panel.is_smoke else ""
    caption = (f"Cost {panel.name}: job outcomes and fit / sample seconds and peak RSS (MB) "
               f"over done jobs (median, IQR = q75 - q25); profile "
               f"{panel.budget_profile or UNSPECIFIED}")
    table = cost_table(df, panel)
    files = [str(_write(tables / f"cost_{panel.name}.csv", table.to_csv(index=False))),
             str(_write(tables / f"cost_{panel.name}.md",
                        to_markdown(table[MD_COLS], caption + tail, index=False))),
             str(_write(tables / f"cost_{panel.name}.tex",
                        to_latex(table[MD_COLS], caption, f"tbl:cost_{panel.name}",
                                 stamp)))]
    files += cost_figure(table, panel, out)
    prov_path = tables / f"cost_{panel.name}.provenance.json"
    files.append(str(prov_path))
    prov = {**panel_provenance(df, panel, files),
            "statistics": "median, q25, q75, IQR (linear interpolation) over done jobs",
            "outcomes": {"time_cap": f"error text contains {TIME_CAP_TEXT!r}",
                         "failed": "state failed or report status failed (not time cap)",
                         "pending": "queued / generated / fitting / other unfinished"}}
    _write(prov_path, json.dumps(prov, indent=2, default=str))
    return {"table": table, "files": files}
