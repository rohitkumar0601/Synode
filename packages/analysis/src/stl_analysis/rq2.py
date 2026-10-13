"""RQ2: how differential privacy (smartnoise-synth MST, ε ∈ {0.1, 1, 10}) moves results.

Per panel dataset (one run per dataset, selected exactly like RQ1: ``rq1.select_panel``):

* ε curves: per-seed metrics of each MST ε against log ε, with the seed spread drawn as a
  95% t-interval (>= 3 seeds) or a min–max band (fewer); gate pass/fail markers per ε from
  core's seed-aggregated gate; a horizontal reference line at the value of the best
  gate-passing non-DP generator (highest core balanced score). A missing ε or an ε whose
  jobs all failed leaves a gap and an annotation, never a crash.
* Pareto shift: core's ``recommend`` / ``pareto_front`` over the ranked candidates without
  and with the MST candidates (train_copy and nominal-DP PrivBayes are never in a front;
  gate-failing candidates are shown but never on a front).
* ε budget: privacy.json bounds path (public / private / mixed / none) and ε spent on
  bounds per dataset and ε.
* PrivBayes (``nominal_dp_audit``, not ε-DP) gets its own figure and table, never the MST
  figure.

Nothing is recomputed: metrics are the stored per-job values gathered by ``collect``;
gates, axis scores, balanced scores and fronts come from stl_core.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402
from stl_core.catalog import CATALOG  # noqa: E402
from stl_core.decision import AXES, pareto_front, recommend  # noqa: E402

from stl_analysis.collect import PRESET  # noqa: E402
from stl_analysis.rq1 import (  # noqa: E402
    SMOKE_STAMP,
    UNSPECIFIED,
    Panel,
    _write,
    panel_provenance,
    role_of,
    run_aggregates,
    to_markdown,
)

EPS_GRID = (0.1, 1.0, 10.0)
DP_FAMILY = "differential_privacy"
AUDIT_FAMILY = "nominal_dp_audit"
KINDS = {"mst": DP_FAMILY, "privbayes": AUDIT_FAMILY}
PRIVBAYES_TITLE = "PrivBayes — nominal DP audit (not ε-DP)"
TSTR_MEAN = "utility.tstr_mean"
METRICS: tuple[tuple[str, str], ...] = (
    ("fidelity.quality_score", "Fidelity: quality score ↑"),
    (TSTR_MEAN, "Utility: TSTR mean over models ↑"),
    ("score", "Balanced score (core, per seed) ↑"),
    ("privacy.dcr_share", "DCR share ↓"),
    ("privacy.singling_out_risk_lower", "Singling-out risk, lower bound ↓"),
    ("privacy.linkability_risk_lower", "Linkability risk, lower bound ↓"),
    ("privacy.inference_risk_lower", "Inference risk, lower bound ↓"),
    ("privacy.mia_advantage", "MIA advantage ↓"),
)
T_MIN_SEEDS = 3
BAND_T = "95% t-interval"
BAND_MINMAX = "min–max"
FRONT_AXES = ("fidelity_utility", "privacy")
FRONT_CHOICE = (
    "Pareto axes: x = core privacy axis score, y = fidelity_utility = mean of core's "
    "fidelity and utility axis scores (their balanced-preset weight). front_non_dp / "
    "front_with_dp: core.decision.pareto_front on these two axes over the gate-passing "
    "candidates with complete scores, without / with the MST candidates; front3_*: core "
    "recommend()'s own on_pareto_front over (fidelity, utility, privacy). train_copy "
    "(sanity) and PrivBayes (nominal-DP audit) are never candidates; gate-failing "
    "candidates are listed but never on a front."
)
CURVE_COLS = ["dataset_id", "run_id", "kind", "generator", "epsilon", "metric", "n_jobs",
              "n_ok", "n_failed", "mean", "lo", "hi", "band", "gate_passed", "annotation",
              "metric_note", "ref_generator", "ref_value"]
DP_COLOR = "#2a78d6"
NONDP_COLOR = "#eb6834"
REF_COLOR = "#52514e"
INK = "#333333"
GRID = "#dddddd"


# ---------------------------------------------------------------- helpers


def parse_epsilon(generator: str) -> float | None:
    """ε of a DP / nominal-DP generator: the catalog params' epsilon, else ``_eps<x>`` in
    the name; None for a generator without ε."""
    info = CATALOG.get(generator)
    if info is not None and "epsilon" in info.params:
        return float(info.params["epsilon"])
    m = re.search(r"_eps(\d+(?:\.\d+)?)$", generator)
    return float(m.group(1)) if m else None


def _kind_of(generator: str) -> str | None:
    info = CATALOG.get(generator)
    family = info.family if info else None
    if family is None and parse_epsilon(generator) is not None:
        family = AUDIT_FAMILY if generator.startswith("privbayes") else DP_FAMILY
    return next((k for k, f in KINDS.items() if f == family), None)


def _run_rows(df: pd.DataFrame, run: dict[str, Any]) -> pd.DataFrame:
    rows = df[(df.store == run["store"]) & (df.run_id == run["run_id"])].copy()
    tstr = [c for c in rows.columns if c.startswith("utility.tstr.")]
    # The utility axis' TSTR term: mean over models of the stored ratios clipped to [0, 1].
    rows[TSTR_MEAN] = rows[tstr].clip(0, 1).mean(axis=1) if tstr else np.nan
    return rows


def _done(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[(rows.report_status == "done") & rows.report_error.isna()]


def _failure_reason(jobs: pd.DataFrame) -> str:
    errs = jobs.error_last_line.dropna().astype(str).tolist()
    if any("could not privately infer bounds" in e for e in errs):
        return "private-bounds fallback"
    if errs:
        e = errs[0]
        return e if len(e) <= 70 else e[:67] + "..."
    states = sorted(set(jobs.state.fillna("unknown").astype(str)))
    return f"no report (state {', '.join(states)})"


def _spread(values: pd.Series) -> tuple[float, float, float, str]:
    v = values.dropna().astype(float).to_numpy()
    if v.size == 0:
        return math.nan, math.nan, math.nan, ""
    mean = float(v.mean())
    if v.size >= T_MIN_SEEDS:
        half = float(stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size))
        return mean, mean - half, mean + half, BAND_T
    return mean, float(v.min()), float(v.max()), BAND_MINMAX


def _decisions(df: pd.DataFrame, run: dict[str, Any]):
    aggs = run_aggregates(df, run)
    return aggs, {c.generator: c for c in recommend(aggs, preset=PRESET).ranking}


def _reference(cands: dict) -> str | None:
    """Best gate-passing non-DP ranked generator by core's balanced score."""
    ok = [c for g, c in cands.items() if role_of(g) == "ranked" and _kind_of(g) is None
          and c.gate_passed and c.score is not None]
    return max(ok, key=lambda c: (c.score, c.generator)).generator if ok else None


def _fmt_eps(eps: float) -> str:
    return f"{eps:g}"


# ---------------------------------------------------------------- tables


def curve_table(df: pd.DataFrame, panel: Panel, kind: str = "mst") -> pd.DataFrame:
    """Long table: one row per (dataset, ε, metric) of the ``kind`` ("mst" | "privbayes")
    generators, with the seed mean, band (lo, hi, kind), core's seed-aggregated gate, an
    annotation for a missing / failed ε, and the non-DP reference generator and value.
    Datasets without any generator of ``kind`` have no rows."""
    rows: list[dict[str, Any]] = []
    for run in panel.runs:
        jobs = _run_rows(df, run)
        gens = {parse_epsilon(g): g for g in sorted(jobs.generator.dropna().unique())
                if _kind_of(g) == kind}
        if not gens:
            continue
        _, cands = _decisions(df, run)
        ref = _reference(cands)
        ref_rows = _done(jobs[jobs.generator == ref]) if ref else jobs.iloc[0:0]
        for eps in sorted(set(EPS_GRID) | set(gens)):
            gen = gens.get(eps)
            g_jobs = jobs[jobs.generator == gen] if gen else jobs.iloc[0:0]
            ok = _done(g_jobs)
            n_failed = int(((g_jobs.state == "failed") | (g_jobs.report_status == "failed"))
                           .sum())
            if gen is None:
                note = f"ε={_fmt_eps(eps)} not in run"
            elif ok.empty:
                note = f"ε={_fmt_eps(eps)} failed: {_failure_reason(g_jobs)}"
            elif len(ok) < len(g_jobs):
                bad = g_jobs[~g_jobs.index.isin(ok.index)]
                note = (f"ε={_fmt_eps(eps)}: {len(bad)}/{len(g_jobs)} seeds without a "
                        f"report ({_failure_reason(bad)})")
            else:
                note = ""
            cand = cands.get(gen) if gen else None
            for metric, _ in METRICS:
                vals = ok[metric] if metric in ok.columns else pd.Series(dtype=float)
                mean, lo, hi, band = _spread(vals)
                metric_note = ""
                if len(ok) and vals.notna().sum() == 0:
                    status = set(ok["privacy.singling_out_status"].dropna().astype(str))
                    metric_note = ("not attackable" if metric.startswith(
                        "privacy.singling_out") and status == {"not_attackable"}
                        else "not measured")
                ref_val = (float(ref_rows[metric].astype(float).mean())
                           if ref and metric in ref_rows.columns and len(ref_rows) else math.nan)
                rows.append({
                    "dataset_id": run["dataset_id"], "run_id": run["run_id"], "kind": kind,
                    "generator": gen, "epsilon": eps, "metric": metric,
                    "n_jobs": int(len(g_jobs)), "n_ok": int(vals.notna().sum()),
                    "n_failed": n_failed, "mean": mean, "lo": lo, "hi": hi, "band": band,
                    "gate_passed": None if cand is None else bool(cand.gate_passed),
                    "annotation": note, "metric_note": metric_note,
                    "ref_generator": ref, "ref_value": ref_val,
                })
    out = pd.DataFrame(rows, columns=CURVE_COLS)
    for c in ("epsilon", "mean", "lo", "hi", "ref_value"):
        out[c] = out[c].astype("float64")
    return out


def pareto_shift(df: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    """One row per (dataset, ranked candidate incl. MST): core axis scores, gate, and front
    membership without (non-DP only) and with the MST candidates; see FRONT_CHOICE."""
    rows: list[dict[str, Any]] = []
    for run in panel.runs:
        aggs = [a for a in run_aggregates(df, run) if role_of(a.generator) == "ranked"]
        if not aggs:
            continue
        non_dp = [a for a in aggs if _kind_of(a.generator) is None]
        rec_all = {c.generator: c for c in recommend(aggs, preset=PRESET).ranking}
        rec_non = ({c.generator: c for c in recommend(non_dp, preset=PRESET).ranking}
                   if non_dp else {})

        def uf(c) -> float | None:
            f, u = c.scores.fidelity, c.scores.utility
            return None if f is None or u is None else (f + u) / 2

        eligible = {g: (uf(c), c.scores.privacy) for g, c in rec_all.items()
                    if c.gate_passed and uf(c) is not None and c.scores.privacy is not None}
        front_all = pareto_front(eligible)
        front_non = pareto_front({g: p for g, p in eligible.items() if g in rec_non})
        for g, c in sorted(rec_all.items()):
            rows.append({
                "dataset_id": run["dataset_id"], "run_id": run["run_id"], "generator": g,
                "family": CATALOG[g].family if g in CATALOG else None,
                "is_dp": _kind_of(g) == "mst", "epsilon": parse_epsilon(g),
                "gate_passed": bool(c.gate_passed), "gate_reasons": "; ".join(c.gate_failures),
                **{a: getattr(c.scores, a) for a in AXES},
                "fidelity_utility": uf(c), "score": c.score,
                "front_non_dp": g in front_non, "front_with_dp": g in front_all,
                "front3_non_dp": bool(rec_non[g].on_pareto_front) if g in rec_non else False,
                "front3_with_dp": bool(c.on_pareto_front),
            })
    cols = ["dataset_id", "run_id", "generator", "family", "is_dp", "epsilon", "gate_passed",
            "gate_reasons", *AXES, "fidelity_utility", "score", "front_non_dp",
            "front_with_dp", "front3_non_dp", "front3_with_dp"]
    out = pd.DataFrame(rows, columns=cols)
    for c in (*AXES, "fidelity_utility", "score", "epsilon"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


def budget_table(df: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    """Per dataset and MST ε: privacy.json bounds path(s), ε spent on bounds
    (``preprocessor``; max over seeds) and ε left to MST (``synthesizer``; min over seeds)."""
    rows: list[dict[str, Any]] = []
    for run in panel.runs:
        jobs = _run_rows(df, run)
        for gen, g in jobs[jobs.generator.map(_kind_of) == "mst"].groupby("generator"):
            paths = sorted(set(g.dp_bounds.dropna().astype(str)))
            pre, syn = g.dp_epsilon_preprocessor.dropna(), g.dp_epsilon_synthesizer.dropna()
            rows.append({
                "dataset_id": run["dataset_id"], "run_id": run["run_id"], "generator": gen,
                "epsilon": parse_epsilon(gen), "n_jobs": int(len(g)),
                "n_privacy_json": int(g.dp_bounds.notna().sum()),
                "n_failed": int(((g.state == "failed") | (g.report_status == "failed")).sum()),
                "bounds_path": "/".join(paths) if paths else "",
                "eps_bounds": float(pre.max()) if len(pre) else math.nan,
                "eps_synthesizer": float(syn.min()) if len(syn) else math.nan,
                "eps_bounds_varies": bool(pre.nunique() > 1),
            })
    cols = ["dataset_id", "run_id", "generator", "epsilon", "n_jobs", "n_privacy_json",
            "n_failed", "bounds_path", "eps_bounds", "eps_synthesizer", "eps_bounds_varies"]
    return pd.DataFrame(rows, columns=cols).sort_values(["dataset_id", "epsilon"]) \
        .reset_index(drop=True)


# ---------------------------------------------------------------- figures


def _save(fig, out: Path, stem: str) -> list[str]:
    files = []
    for ext in ("png", "pdf"):
        p = out / "figures" / f"{stem}.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        files.append(str(p))
    plt.close(fig)
    return files


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK)


def eps_figure(cur: pd.DataFrame, title: str, color: str, out: Path, stem: str) -> list[str]:
    """2 x 4 small multiples (one per metric) of one dataset's curves vs log ε."""
    eps = sorted(cur.epsilon.unique())
    bands = sorted(set(cur.band) - {""})
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), squeeze=False)
    ref_gen = next((g for g in cur.ref_generator.dropna()), None)
    for ax, (metric, label) in zip(axes.ravel(), METRICS, strict=True):
        m = cur[cur.metric == metric].set_index("epsilon").reindex(eps)
        x = np.array(eps, float)
        ax.fill_between(x, m.lo.to_numpy(float), m.hi.to_numpy(float), color=color,
                        alpha=0.18, linewidth=0, zorder=1)
        ax.plot(x, m["mean"].to_numpy(float), color=color, linewidth=2, zorder=2)
        for e, r in m.iterrows():
            if math.isnan(r["mean"]):
                if r.metric_note:
                    gap = r.metric_note
                elif "failed" in str(r.annotation):
                    gap = "failed"
                elif "not in run" in str(r.annotation):
                    gap = "not in run"
                else:
                    gap = "no data"
                ax.annotate(gap,
                            xy=(e, 0.04), xycoords=("data", "axes fraction"), ha="center",
                            fontsize=6.5, color=REF_COLOR, style="italic")
                continue
            passed = r.gate_passed is True or r.gate_passed == 1.0
            ax.plot([e], [r["mean"]], marker="o" if passed else "X", markersize=8,
                    color=color if passed else "white", markeredgecolor=color,
                    markeredgewidth=1.5, linestyle="none", zorder=3)
        ref = m.ref_value.dropna()
        if len(ref):
            ax.axhline(float(ref.iloc[0]), color=REF_COLOR, linestyle="--", linewidth=1,
                       zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(x, [_fmt_eps(e) for e in eps])
        ax.minorticks_off()
        ax.set_xlim(x.min() / 2, x.max() * 2)
        ax.set_title(label, fontsize=8.5, color=INK)
        ax.set_xlabel("ε (log scale)", fontsize=7, color=INK)
        _style(ax)
    band_label = " / ".join(f"{b}{' (n ≥ 3 seeds)' if b == BAND_T else ' over seeds'}"
                            for b in bands) or "no seeds"
    handles = [Line2D([], [], color=color, linewidth=2, label="seed mean"),
               Patch(color=color, alpha=0.18, label=f"seed spread: {band_label}"),
               Line2D([], [], marker="o", color=color, linestyle="none", markersize=7,
                      label="privacy gate passed (core, all seeds)"),
               Line2D([], [], marker="X", color="white", markeredgecolor=color,
                      linestyle="none", markersize=7, label="privacy gate failed")]
    if ref_gen:
        handles.append(Line2D([], [], color=REF_COLOR, linestyle="--", linewidth=1,
                              label=f"reference: best gate-passing non-DP ({ref_gen})"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    notes = sorted({a for a in cur.annotation if a})
    if notes:
        fig.text(0.5, -0.06, "Gaps: " + "; ".join(notes), ha="center", fontsize=7,
                 color=REF_COLOR)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, stem)


def pareto_figure(par: pd.DataFrame, title: str, out: Path, stem: str) -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 5.2))
    pts = par.dropna(subset=["privacy", "fidelity_utility"])
    for r in pts.itertuples():
        color = DP_COLOR if r.is_dp else NONDP_COLOR
        ax.plot([r.privacy], [r.fidelity_utility], marker="s" if r.is_dp else "o",
                markersize=8, color=color if r.gate_passed else "white",
                markeredgecolor=color, markeredgewidth=1.5, linestyle="none", zorder=3)
        ax.annotate(r.generator, (r.privacy, r.fidelity_utility), xytext=(5, 4),
                    textcoords="offset points", fontsize=7, color=INK)
    for col, color, style, label in (("front_non_dp", NONDP_COLOR, "--", "front without DP"),
                                     ("front_with_dp", DP_COLOR, "-", "front with MST")):
        f = pts[pts[col]].sort_values("privacy")
        if len(f):
            ax.plot(f.privacy, f.fidelity_utility, color=color, linestyle=style,
                    linewidth=2 if style == "-" else 1.5, zorder=2, label=label)
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", color=NONDP_COLOR, linestyle="none",
                       label="non-DP candidate"),
                Line2D([], [], marker="s", color=DP_COLOR, linestyle="none",
                       label="MST (ε-DP) candidate"),
                Line2D([], [], marker="o", color="white", markeredgecolor=REF_COLOR,
                       linestyle="none", label="hollow: gate failed (never on a front)")]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="best")
    ax.set_xlabel("privacy axis score (core) →", fontsize=8, color=INK)
    ax.set_ylabel("mean(fidelity, utility) axis scores (core) →", fontsize=8, color=INK)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    missing = sorted(set(par.generator) - set(pts.generator))
    if missing:
        fig.text(0.5, -0.02, f"Not plotted (incomplete scores): {', '.join(missing)}",
                 ha="center", fontsize=7, color=REF_COLOR)
    ax.set_title(title, fontsize=9, color=INK)
    fig.tight_layout()
    return _save(fig, out, stem)


# ---------------------------------------------------------------- artefacts


def _curves_md(cur: pd.DataFrame, caption: str) -> str:
    cols = ["dataset_id", "generator", "epsilon", "metric", "n_ok", "mean", "lo", "hi", "band",
            "gate_passed", "annotation", "metric_note", "ref_generator", "ref_value"]
    return to_markdown(cur[cols], caption, index=False)


def run_rq2(df: pd.DataFrame, panel: Panel, out: str | Path) -> dict[str, Any]:
    """Every RQ2 artefact of one panel under ``out``/{tables,figures}."""
    out = Path(out)
    tables = out / "tables"
    (tables / f"rq2_{panel.name}_refused.md").unlink(missing_ok=True)
    stamp = f" — {SMOKE_STAMP}" if panel.is_smoke else ""
    files: list[str] = []
    res: dict[str, Any] = {"annotations": {}, "titles": {}, "plotted": {}}
    profile = panel.budget_profile or UNSPECIFIED
    budget = budget_table(df, panel)
    files.append(str(_write(tables / f"rq2_{panel.name}_eps_budget.csv",
                            budget.to_csv(index=False))))
    files.append(str(_write(tables / f"rq2_{panel.name}_eps_budget.md", to_markdown(
        budget, f"RQ2 {panel.name}: MST ε budget from privacy.json — bounds path "
                f"(public = codebook bounds, 0 ε; private / mixed = 20% of ε spent on "
                f"private bounds) and ε spent on bounds / left to MST{stamp}", index=False))))
    curves = {k: curve_table(df, panel, k) for k in KINDS}
    for kind, cur in curves.items():
        if kind == "privbayes" and cur.empty:
            continue
        what = ("MST (ε-DP)" if kind == "mst" else PRIVBAYES_TITLE)
        files.append(str(_write(tables / f"rq2_{panel.name}_{kind}_curves.csv",
                                cur.to_csv(index=False))))
        files.append(str(_write(tables / f"rq2_{panel.name}_{kind}_curves.md", _curves_md(
            cur, f"RQ2 {panel.name}: {what} metrics per ε (seed mean and spread; gate from "
                 f"core; ref = best gate-passing non-DP generator){stamp}"))))
    par = pareto_shift(df, panel)
    files.append(str(_write(tables / f"rq2_{panel.name}_pareto.csv", par.to_csv(index=False))))
    files.append(str(_write(tables / f"rq2_{panel.name}_pareto.md", to_markdown(
        par[["dataset_id", "generator", "is_dp", "gate_passed", "fidelity", "utility",
             "privacy", "fidelity_utility", "front_non_dp", "front_with_dp", "front3_non_dp",
             "front3_with_dp"]],
        f"RQ2 {panel.name}: Pareto-front membership without / with MST{stamp}",
        index=False) + "\n" + FRONT_CHOICE + "\n")))
    for run in panel.runs:
        ds = run["dataset_id"]
        n_seeds = run["n_seeds"]
        head = f"run {run['run_id']}, {n_seeds} seed{'s' if n_seeds != 1 else ''}, " \
               f"profile {profile}"
        res["titles"][ds], res["plotted"][ds] = {}, {}
        b = budget[budget.dataset_id == ds]
        spent = " · ".join(f"ε={_fmt_eps(r.epsilon)}: {r.bounds_path or 'no privacy.json'}"
                           + ("" if math.isnan(r.eps_bounds) else f", bounds ε {r.eps_bounds:g}")
                           for r in b.itertuples())
        for kind, cur in curves.items():
            c = cur[cur.dataset_id == ds]
            if c.empty:
                continue
            if kind == "mst":
                title = f"RQ2 {ds}: MST (ε-DP) metrics vs ε — {head}"
                if spent:
                    title += f"\nprivacy.json bounds path: {spent}"
                stem, color = f"rq2_{panel.name}_{ds}_mst_eps", DP_COLOR
            else:
                title = f"{PRIVBAYES_TITLE} — {ds}, {head}"
                stem, color = f"rq2_{panel.name}_{ds}_privbayes_audit", REF_COLOR
            if panel.is_smoke:
                title += f"\n{SMOKE_STAMP}"
            res["titles"][ds][kind] = title
            res["plotted"][ds][kind] = sorted(c.generator.dropna().unique(),
                                              key=lambda g: parse_epsilon(g) or 0)
            files += eps_figure(c, title, color, out, stem)
            if kind == "mst":
                res["annotations"][ds] = sorted({a for a in c.annotation if a})
        p = par[par.dataset_id == ds]
        if not p.empty:
            title = f"RQ2 {ds}: Pareto front without vs with MST — {head}"
            if panel.is_smoke:
                title += f"\n{SMOKE_STAMP}"
            files += pareto_figure(p, title, out, f"rq2_{panel.name}_{ds}_pareto")
    prov_path = tables / f"rq2_{panel.name}.provenance.json"
    files.append(str(prov_path))
    prov = {**panel_provenance(df, panel, files), "eps_grid": list(EPS_GRID),
            "band": f"{BAND_T} with >= {T_MIN_SEEDS} seeds, else {BAND_MINMAX}",
            "reference": "best gate-passing non-DP ranked generator by core balanced score",
            "front_choice": FRONT_CHOICE, "annotations": res["annotations"],
            "datasets_without_mst": sorted({r["dataset_id"] for r in panel.runs}
                                           - set(curves["mst"].dataset_id))}
    _write(prov_path, json.dumps(prov, indent=2, default=str))
    res["files"] = files
    return res


def refused(panel: str, out: str | Path, error: Exception) -> Path:
    return _write(Path(out) / "tables" / f"rq2_{panel}_refused.md",
                  f"# RQ2 panel `{panel}` refused\n\n{error}\n")
