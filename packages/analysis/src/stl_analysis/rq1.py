"""RQ1: per-dataset rankings, autorank critical-difference diagrams, dataset characteristics.

Panels (budget profiles are never mixed inside one panel):

* ``core``: the non-meta benchmark runs (adult / credit / diabetes). One run per dataset:
  by default the most recent run (manifest modification time) that has at least one
  "done" report, preferring non-smoke runs; or exactly the runs passed with ``run_ids``.
* ``cc18``: the RQ3 meta-benchmark runs ``meta-cc18_*`` (planned with ``equal50``).

Smoke runs (see collect.smoke_reasons) are used only with ``allow_smoke``, and everything
made from them carries ``SMOKE_STAMP``.

Scores are never recomputed: per run, the stored ``report.json`` files go through
``stl_core.aggregate.aggregate`` and ``stl_core.decision.recommend(preset="balanced")``,
exactly like the API report route, so the score and the gate are seed-aggregated by core.
Roles: ``train_copy`` (family sanity_ceiling) is a sanity row only; PrivBayes
(``nominal_dp_audit``) is reported separately; every other generator is ``ranked``, and is
rankable only if it passed the gate with a complete score.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from autorank import autorank, create_report, latex_table, plot_stats  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from stl_core.aggregate import aggregate  # noqa: E402
from stl_core.catalog import CATALOG  # noqa: E402
from stl_core.decision import recommend  # noqa: E402
from stl_core.metadata import ColumnType, TableMetadata  # noqa: E402
from stl_core.schemas import AggregatedReport, MetricReport  # noqa: E402
from stl_runner import keys  # noqa: E402

from stl_analysis.collect import PRESET, open_store  # noqa: E402

PANELS = ("core", "cc18")
SMOKE_STAMP = "SMOKE — not for the thesis"
ALPHA = 0.05
# autorank refuses fewer than five paired samples ("requires at least five performance
# estimations"), so a CD diagram needs five datasets, not the three of the plan.
MIN_DATASETS = 5
MIN_GENERATORS = 2
UNSPECIFIED = "unspecified"
CD_CHOICE = (
    "CD-diagram generators: the ranked generators (train_copy and nominal-DP PrivBayes "
    "excluded) that passed the privacy gate with a complete balanced score on every "
    "dataset of the panel. autorank needs a complete datasets x generators matrix, and a "
    "gate-failing candidate must never be ranked, so a generator failing (or missing) on "
    "any dataset is dropped rather than imputed."
)
CHARACTERISTICS = ("n_rows", "n_features", "max_cat_cardinality", "minority_class_share",
                   "mean_abs_corr_numeric")
MF_KEYS = ("nr_inst", "nr_attr", "nr_num", "nr_cat", "nr_class", "class_ent", "cor.mean")
MARK = "#3b6fb6"
INK = "#333333"


class PanelError(ValueError):
    """The requested panel cannot be built honestly (mixed profiles, smoke, duplicates)."""


@dataclass
class Panel:
    name: str
    runs: list[dict[str, Any]]
    budget_profile: str | None
    is_smoke: bool
    allow_smoke: bool
    notes: list[str] = field(default_factory=list)

    @property
    def stamp(self) -> str:
        return f" [{SMOKE_STAMP}]" if self.is_smoke else ""


# ---------------------------------------------------------------- panel selection


def _runs_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (store, run_id), g in df.groupby(["store", "run_id"], sort=True):
        profiles = g.budget_profile.dropna().unique().tolist()
        reasons = sorted({r for rs in g.smoke_reasons.dropna() for r in str(rs).split("; ")
                          if r})
        rows.append({
            "store": store, "run_id": run_id,
            "dataset_id": g.dataset_id.dropna().iloc[0] if g.dataset_id.notna().any() else None,
            "is_meta": bool(g.is_meta.fillna(False).any()),
            "is_smoke": bool(g.is_smoke.fillna(False).any()), "smoke_reasons": reasons,
            "budget_profile": profiles[0] if profiles else None,
            "n_seeds": int(g.seed.nunique()),
            "n_done": int((g.report_status == "done").sum()),
            "n_jobs": int(len(g)), "n_reported": int(g.report_status.notna().sum()),
        })
    return pd.DataFrame(rows, columns=["store", "run_id", "dataset_id", "is_meta", "is_smoke",
                                       "smoke_reasons", "budget_profile", "n_seeds", "n_done",
                                       "n_jobs", "n_reported"])


def _mtime(store: str, run_id: str) -> float:
    return open_store(store).modified(keys.run_manifest(run_id))


def _run_record(r: pd.Series) -> dict[str, Any]:
    return {"store": r.store, "run_id": r.run_id, "dataset_id": r.dataset_id,
            "budget_profile": r.budget_profile, "n_seeds": int(r.n_seeds),
            "is_smoke": bool(r.is_smoke), "smoke_reasons": list(r.smoke_reasons),
            "n_jobs": int(r.n_jobs), "n_reported": int(r.n_reported)}


def select_panel(df: pd.DataFrame, name: str, run_ids: list[str] | None = None,
                 allow_smoke: bool = False) -> Panel:
    """The runs of panel ``name`` (one per dataset); see the module docstring."""
    if name not in PANELS:
        raise KeyError(f"unknown panel {name!r}; choose from {PANELS}")
    runs = _runs_table(df)
    runs = runs[runs.is_meta == (name == "cc18")]
    notes: list[str] = []
    if run_ids is not None:
        chosen = runs[runs.run_id.isin(run_ids)]
        dup = chosen.dataset_id[chosen.dataset_id.duplicated()].unique().tolist()
        if dup:
            raise PanelError(f"panel {name}: one run per dataset; several runs given for "
                             f"{dup}: {chosen[chosen.dataset_id.isin(dup)].run_id.tolist()}")
        smoke = chosen[chosen.is_smoke]
        if len(smoke) and not allow_smoke:
            raise PanelError(f"panel {name}: smoke runs {smoke.run_id.tolist()} need "
                             "--allow-smoke (they are stamped 'not for the thesis')")
    else:
        cand = runs[runs.n_done > 0]
        skipped = cand[cand.is_smoke] if not allow_smoke else cand.iloc[0:0]
        if not allow_smoke:
            cand = cand[~cand.is_smoke]
        if len(skipped):
            notes.append(f"smoke runs skipped (no --allow-smoke): {skipped.run_id.tolist()}")
            if cand.empty:
                raise PanelError(f"panel {name}: only smoke runs have results "
                                 f"({skipped.run_id.tolist()}); pass --allow-smoke to use "
                                 "them (stamped 'not for the thesis')")
        no_results = runs[runs.n_done == 0].run_id.tolist()
        if no_results:
            notes.append(f"runs without any done report ignored: {no_results}")
        cand = cand.assign(mtime=[_mtime(s, r) for s, r in zip(cand.store, cand.run_id,
                                                                  strict=True)])
        cand = cand.sort_values(["dataset_id", "is_smoke", "mtime"],
                                ascending=[True, True, False])
        chosen = cand.groupby("dataset_id", sort=True).head(1)
        for ds, g in cand.groupby("dataset_id", sort=True):
            if len(g) > 1:
                notes.append(f"{ds}: using {g.run_id.iloc[0]} (most recent"
                             f"{'' if allow_smoke else ' non-smoke'}, non-smoke preferred); "
                             f"not used: {g.run_id.iloc[1:].tolist()}")
    profiles = sorted({p if isinstance(p, str) else UNSPECIFIED for p in chosen.budget_profile})
    if len(profiles) > 1:
        by = {p: chosen[chosen.budget_profile.map(
            lambda v: v if isinstance(v, str) else UNSPECIFIED) == p].run_id.tolist()
              for p in profiles}
        raise PanelError(f"panel {name} mixes budget profiles {by}; one CD diagram never "
                         "mixes profiles: pass --run to choose runs of a single profile")
    chosen = chosen.sort_values(["dataset_id", "run_id"])
    for r in chosen[chosen.n_reported < chosen.n_jobs].itertuples():
        notes.append(f"{r.run_id} is incomplete: {r.n_reported}/{r.n_jobs} jobs reported "
                     "(still running, failed or cancelled); its scores cover the reports only")
    return Panel(name=name, runs=[_run_record(r) for _, r in chosen.iterrows()],
                 budget_profile=profiles[0] if profiles and profiles[0] != UNSPECIFIED
                 else None,
                 is_smoke=bool(chosen.is_smoke.any()), allow_smoke=allow_smoke, notes=notes)


# ---------------------------------------------------------------- scores and ranks


def role_of(generator: str) -> str:
    family = CATALOG[generator].family if generator in CATALOG else None
    if family == "sanity_ceiling":
        return "sanity"
    if family == "nominal_dp_audit":
        return "nominal_dp_audit"
    return "ranked"


def run_aggregates(df: pd.DataFrame, run: dict[str, Any]) -> list[AggregatedReport]:
    """core's ``aggregate`` over the readable stored reports of one panel run (all seeds)."""
    jobs = df[(df.store == run["store"]) & (df.run_id == run["run_id"])
              & df.report_status.notna() & df.report_error.isna()]
    if jobs.empty:
        return []
    store = open_store(run["store"])
    return aggregate([MetricReport.model_validate(store.read_json(keys.report(run["run_id"], j)))
                      for j in jobs.job_id])


def dataset_scores(df: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    """One row per (dataset, generator) of the panel: core's seed-aggregated gate and
    balanced score over the run's stored reports, the role and, among rankable candidates
    of the dataset, the rank (1 = best, ties averaged) and the winner."""
    rows: list[dict[str, Any]] = []
    for run in panel.runs:
        aggs = {a.generator: a for a in run_aggregates(df, run)}
        if not aggs:
            continue
        for cand in recommend(list(aggs.values()), preset=PRESET).ranking:
            agg = aggs[cand.generator]
            role = role_of(cand.generator)
            rows.append({
                "panel": panel.name, "store": run["store"], "run_id": run["run_id"],
                "dataset_id": run["dataset_id"], "generator": cand.generator,
                "family": CATALOG[cand.generator].family if cand.generator in CATALOG
                else None,
                "role": role, "n_seeds": agg.n_seeds, "n_failed": agg.n_failed,
                "gate_passed": cand.gate_passed,
                "gate_reasons": "; ".join(cand.gate_failures),
                "score": cand.score, "score_fidelity": cand.scores.fidelity,
                "score_utility": cand.scores.utility, "score_privacy": cand.scores.privacy,
                "rankable": role == "ranked" and cand.gate_passed and cand.score is not None,
            })
    cols = ["panel", "store", "run_id", "dataset_id", "generator", "family", "role",
            "n_seeds", "n_failed", "gate_passed", "gate_reasons", "score", "score_fidelity",
            "score_utility", "score_privacy", "rankable"]
    out = pd.DataFrame(rows, columns=cols)
    for c in ("score", "score_fidelity", "score_utility", "score_privacy"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["rankable"] = out.rankable.astype(bool)
    out["gate_passed"] = out.gate_passed.astype(bool)
    out["rank"] = np.nan
    out["is_winner"] = False
    for _, g in out[out.rankable].groupby("dataset_id"):
        out.loc[g.index, "rank"] = g.score.rank(ascending=False, method="average")
        best = g.sort_values(["score", "generator"], ascending=[False, True]).index[0]
        out.loc[best, "is_winner"] = True
    return out


def rank_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Datasets x ranked generators: the rank among gate-passing candidates ("1", "2.5"),
    "gate✗" for a gate failure, "n/a" for a gate pass without a complete score, "—" when the
    generator has no report on that dataset. train_copy / PrivBayes are not columns."""
    ranked = scores[scores.role == "ranked"]
    datasets = sorted(scores.dataset_id.unique())
    order = (ranked.groupby("generator")["rank"].mean().sort_values(na_position="last")
             .index.tolist())
    table = pd.DataFrame("—", index=pd.Index(datasets, name="dataset_id"), columns=order)
    for r in ranked.itertuples():
        if r.rankable:
            cell = f"{r.rank:g}"
        elif not r.gate_passed:
            cell = "gate✗"
        else:
            cell = "n/a"
        table.loc[r.dataset_id, r.generator] = cell
    return table


# ---------------------------------------------------------------- CD diagram


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _insufficient(panel: Panel, out: Path, reason: str, extra: dict) -> dict[str, Any]:
    lines = [f"# RQ1 panel `{panel.name}`: no CD diagram{panel.stamp}", "", reason, "",
             f"- datasets in panel: {extra.get('n_datasets', 0)} "
             f"({', '.join(extra.get('datasets', [])) or 'none'})",
             f"- generators passing the gate on every dataset: "
             f"{', '.join(extra.get('generators', [])) or 'none'}",
             f"- dropped generators: {', '.join(extra.get('dropped', [])) or 'none'}",
             f"- requirement: >= {MIN_DATASETS} datasets (autorank's minimum of five paired "
             f"samples) and >= {MIN_GENERATORS} common gate-passing generators", "",
             CD_CHOICE]
    lines += [f"- note: {n}" for n in panel.notes]
    path = _write(out / "tables" / f"rq1_{panel.name}_insufficient.md", "\n".join(lines) + "\n")
    return {"status": "insufficient", "reason": reason, "files": [str(path)], **extra}


def cd_diagram(scores: pd.DataFrame, panel: Panel, out: str | Path) -> dict[str, Any]:
    """autorank (alpha 0.05; it chooses the tests) over the datasets x common gate-passing
    generators score matrix: ``plot_stats`` figure (PNG+PDF), ``create_report`` text and
    ``latex_table``; or ``rq1_<panel>_insufficient.md`` explaining why not."""
    out = Path(out)
    datasets = sorted({r["dataset_id"] for r in panel.runs} | set(scores.dataset_id))
    ranked = scores[scores.role == "ranked"]
    matrix = (ranked[ranked.rankable].pivot(index="dataset_id", columns="generator",
                                            values="score")
              .reindex(index=datasets))
    keep = sorted(c for c in matrix.columns if matrix[c].notna().all())
    dropped = sorted(set(ranked.generator) - set(keep))
    matrix = matrix[keep]
    extra = {"n_datasets": len(datasets), "datasets": datasets, "generators": keep,
             "dropped": dropped}
    if len(datasets) < MIN_DATASETS:
        return _insufficient(panel, out, f"Only {len(datasets)} datasets in the panel; "
                             f"autorank needs at least {MIN_DATASETS}.", extra)
    if len(keep) < MIN_GENERATORS:
        return _insufficient(panel, out, f"Only {len(keep)} generators passed the gate on "
                             f"all {len(datasets)} datasets; at least {MIN_GENERATORS} are "
                             "needed.", extra)
    matrix.columns.name = None
    _write(out / "tables" / f"rq1_{panel.name}_cd_matrix.csv", matrix.to_csv())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = autorank(matrix.reset_index(drop=True), alpha=ALPHA, verbose=False)
        except ValueError as e:
            return _insufficient(panel, out, f"autorank refused the matrix: {e}", extra)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            create_report(res)
        report_text = buf.getvalue()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            latex_table(res, label=f"tbl:rq1_{panel.name}")
        tex = buf.getvalue()
        ax = plot_stats(res, allow_insignificant=True)
    files: list[str] = []
    header = (f"RQ1 panel {panel.name}: {len(datasets)} datasets, {len(keep)} generators, "
              f"budget profile {panel.budget_profile or UNSPECIFIED}, alpha {ALPHA}")
    if panel.is_smoke:
        header = f"{SMOKE_STAMP}\n{header}"
        tex = tex.replace(r"\caption{Summary of populations}",
                          r"\caption{" + SMOKE_STAMP + ". Summary of populations}")
        tex = f"% {SMOKE_STAMP}\n{tex}"
    files.append(str(_write(out / "tables" / f"rq1_{panel.name}_autorank.txt",
                            f"{header}\n{CD_CHOICE}\nDropped: {dropped or 'none'}\n\n"
                            f"{report_text}")))
    files.append(str(_write(out / "tables" / f"rq1_{panel.name}_autorank.tex", tex)))
    significant = res.pvalue is not None and res.pvalue < ALPHA
    if ax is None:
        files.append(str(_write(out / "tables" / f"rq1_{panel.name}_cd_noplot.md",
                                f"autorank has no plot for the {res.omnibus} test "
                                f"(two populations, non-normal){panel.stamp}.\n")))
    else:
        fig = ax.get_figure()
        title = (f"RQ1 {panel.name}: {res.omnibus} + {res.posthoc}, p={res.pvalue:.3g}"
                 f"{'' if significant else ' (not significant)'}, n={len(datasets)} datasets")
        if panel.is_smoke:
            title += f"\n{SMOKE_STAMP}"
        fig.suptitle(title, fontsize=9, color=INK)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = out / "figures" / f"rq1_{panel.name}_cd.{ext}"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=200, bbox_inches="tight")
            files.append(str(p))
        plt.close(fig)
    # rankdf is sorted by the central tendency (ascending for ANOVA/Tukey), so the ranking
    # is read from the mean ranks (1 = best, order "descending": higher score is better).
    return {"status": "ok", "ranking": res.rankdf.meanrank.astype(float).sort_values()
            .index.tolist(),
            "mean_rank": {str(k): float(v) for k, v in res.rankdf.meanrank.items()},
            "omnibus": res.omnibus, "posthoc": res.posthoc,
            "pvalue": None if res.pvalue is None else float(res.pvalue),
            "cd": None if res.cd is None else float(res.cd), "significant": significant,
            "files": files, **extra}


# ---------------------------------------------------------------- dataset characteristics


def _table_for(store, dataset_id: str) -> pd.DataFrame | None:
    for key in (keys.dataset_raw(dataset_id), keys.split_part(dataset_id, 0, "train")):
        if store.exists(key):
            return store.read_df(key)
    return None


def _characteristics(df: pd.DataFrame, md: TableMetadata) -> dict[str, float]:
    feats = [c for c in md.feature_columns if c in df.columns]
    cats = [c for c in feats if md.columns[c] == ColumnType.categorical]
    nums = [c for c in feats if md.columns[c] == ColumnType.numerical]
    minority = math.nan
    if md.task == "classification" and md.target in df.columns:
        shares = df[md.target].value_counts(normalize=True, dropna=True)
        minority = float(shares.min()) if len(shares) else math.nan
    corr = math.nan
    if len(nums) >= 2:
        c = df[nums].apply(pd.to_numeric, errors="coerce").corr().abs().to_numpy()
        upper = c[np.triu_indices_from(c, k=1)]
        upper = upper[np.isfinite(upper)]
        corr = float(upper.mean()) if upper.size else math.nan
    return {"n_rows": len(df), "n_features": len(feats),
            "max_cat_cardinality": max((int(df[c].nunique()) for c in cats), default=0),
            "minority_class_share": minority, "mean_abs_corr_numeric": corr}


def dataset_characteristics(panel: Panel) -> pd.DataFrame:
    """One row per panel dataset from the stored registration (``raw.parquet``, or the
    seed-0 TRAIN split, plus ``metadata.json``) and, when present, selected pymfe
    meta-features of ``metafeatures.json`` as ``mf.<name>``."""
    rows = []
    for run in panel.runs:
        ds = run["dataset_id"]
        store = open_store(run["store"])
        row: dict[str, Any] = {"dataset_id": ds, "store": run["store"],
                               **{c: math.nan for c in CHARACTERISTICS}}
        md_key = keys.dataset_metadata(ds)
        table = _table_for(store, ds)
        if store.exists(md_key) and table is not None:
            row.update(_characteristics(table, TableMetadata.model_validate(
                store.read_json(md_key))))
        mf_key = keys.dataset_metafeatures(ds)
        feats = (store.read_json(mf_key).get("features") or {}) if store.exists(mf_key) else {}
        for k in MF_KEYS:
            v = feats.get(k)
            row[f"mf.{k}"] = math.nan if v is None else float(v)
        rows.append(row)
    cols = ["dataset_id", "store", *CHARACTERISTICS, *(f"mf.{k}" for k in MF_KEYS)]
    return pd.DataFrame(rows, columns=cols)


def winner_plot(chars: pd.DataFrame, panel: Panel, out: Path) -> list[str]:
    """Strip plot per characteristic: one point per dataset, x = its winning generator."""
    if chars.empty:
        return []
    winners = sorted(chars.winner.unique())
    pos = {w: i for i, w in enumerate(winners)}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(CHARACTERISTICS), figsize=(3 * len(CHARACTERISTICS), 3.4),
                             squeeze=False)
    for ax, ch in zip(axes[0], CHARACTERISTICS, strict=True):
        x = chars.winner.map(pos).to_numpy(float) + rng.uniform(-0.12, 0.12, len(chars))
        ax.scatter(x, chars[ch], s=36, color=MARK, edgecolor="white", linewidth=1, zorder=3)
        ax.set_xticks(range(len(winners)), winners, rotation=30, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(winners) - 0.4)
        ax.set_title(ch.replace("_", " "), fontsize=9, color=INK)
        if ch in ("n_rows", "n_features", "max_cat_cardinality"):
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK)
    title = (f"RQ1 {panel.name}: dataset characteristics by winning generator "
             f"(n={len(chars)} datasets)")
    if panel.is_smoke:
        title += f"\n{SMOKE_STAMP}"
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    files = []
    for ext in ("png", "pdf"):
        p = out / "figures" / f"rq1_{panel.name}_winner_vs_characteristics.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        files.append(str(p))
    plt.close(fig)
    return files


# ---------------------------------------------------------------- artefacts


def to_markdown(df: pd.DataFrame, caption: str, index: bool = True) -> str:
    """A GitHub markdown table (no tabulate dependency)."""
    d = df.reset_index() if index else df
    def cell(v: Any) -> str:
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.3g}"
        return "" if v is None or v is pd.NA else str(v)
    lines = [f"**{caption}**", "", "| " + " | ".join(map(str, d.columns)) + " |",
             "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in d.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def panel_provenance(df: pd.DataFrame, panel: Panel, files: list[str]) -> dict:
    """Provenance shared by every panel stage: runs, smoke stamp, profile, seeds, metrics
    versions and generator library versions of the panel's jobs, and the output files."""
    sel = pd.concat([df[(df.store == r["store"]) & (df.run_id == r["run_id"])]
                     for r in panel.runs]) if panel.runs else df.iloc[0:0]
    libs: dict[str, list[str]] = {}
    for lib, ver in sel[["generator_library", "generator_library_version"]].dropna() \
            .drop_duplicates().itertuples(index=False):
        libs.setdefault(lib, []).append(ver)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "panel": panel.name, "preset": PRESET,
        "smoke": panel.is_smoke, "stamp": SMOKE_STAMP if panel.is_smoke else None,
        "allow_smoke": panel.allow_smoke, "budget_profile": panel.budget_profile,
        "n_datasets": len(panel.runs), "runs": panel.runs, "notes": panel.notes,
        "n_seeds": sorted({r["n_seeds"] for r in panel.runs}),
        "metrics_versions": sorted({int(v) for v in sel.metrics_version.dropna()}),
        "reports_without_metrics_version": int(
            (sel.report_status.notna() & sel.metrics_version.isna()).sum()),
        "generator_libraries": {k: sorted(v) for k, v in sorted(libs.items())},
        "outputs": files,
    }


def _provenance(df: pd.DataFrame, panel: Panel, cd: dict, files: list[str]) -> dict:
    return {**panel_provenance(df, panel, files), "alpha": ALPHA,
            "cd_generator_choice": CD_CHOICE,
            "cd": {k: v for k, v in cd.items() if k != "files"}}


def run_panel(df: pd.DataFrame, panel: Panel, out: str | Path) -> dict[str, Any]:
    """Every RQ1 artefact of one panel under ``out``/{tables,figures}."""
    out = Path(out)
    tables = out / "tables"
    for stale in (f"rq1_{panel.name}_refused.md", f"rq1_{panel.name}_insufficient.md",
                  f"rq1_{panel.name}_cd_noplot.md"):
        (tables / stale).unlink(missing_ok=True)
    caption_tail = f" — {SMOKE_STAMP}" if panel.is_smoke else ""
    files: list[str] = []
    scores = dataset_scores(df, panel)
    if not scores.empty:
        files.append(str(_write(tables / f"rq1_{panel.name}_scores.csv",
                                scores.to_csv(index=False))))
        ranks = rank_table(scores)
        files.append(str(_write(tables / f"rq1_{panel.name}_ranks.csv", ranks.to_csv())))
        files.append(str(_write(tables / f"rq1_{panel.name}_ranks.md", to_markdown(
            ranks, f"RQ1 {panel.name}: rank by balanced score among gate-passing generators "
                   f"(gate✗ = failed the privacy gate, never ranked; profile "
                   f"{panel.budget_profile or UNSPECIFIED}){caption_tail}"))))
        side = scores[scores.role != "ranked"][
            ["dataset_id", "generator", "role", "n_seeds", "gate_passed", "score",
             "gate_reasons"]]
        if not side.empty:
            files.append(str(_write(tables / f"rq1_{panel.name}_sanity_audit.md", to_markdown(
                side, f"RQ1 {panel.name}: train_copy sanity row and nominal-DP PrivBayes "
                      f"audit (not ranked){caption_tail}", index=False))))
    cd = cd_diagram(scores, panel, out)
    files += cd["files"]
    chars = dataset_characteristics(panel)
    if not chars.empty:
        win = scores[scores.is_winner].set_index("dataset_id").generator
        chars["winner"] = chars.dataset_id.map(win).fillna("none")
        files.append(str(_write(tables / f"rq1_{panel.name}_characteristics.csv",
                                chars.to_csv(index=False))))
        files.append(str(_write(tables / f"rq1_{panel.name}_characteristics.md", to_markdown(
            chars.drop(columns=["store"]), f"RQ1 {panel.name}: dataset characteristics and "
            f"winning generator{caption_tail}", index=False))))
        files += winner_plot(chars, panel, out)
    prov_path = tables / f"rq1_{panel.name}.provenance.json"
    files.append(str(prov_path))
    _write(prov_path, json.dumps(_provenance(df, panel, cd, files), indent=2, default=str))
    return {"panel": panel.name, "cd": cd, "files": files}


def refused(panel: str, out: str | Path, error: Exception) -> Path:
    return _write(Path(out) / "tables" / f"rq1_{panel}_refused.md",
                  f"# RQ1 panel `{panel}` refused\n\n{error}\n")
