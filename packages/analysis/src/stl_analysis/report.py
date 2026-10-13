"""Report assembly: collect -> RQ1 (all panels) -> RQ2 (core; cc18 optional) -> cost ->
``reports/results.md``.

Every number in results.md is rendered from a table written under ``<out>/tables`` (the
generated markdown tables are embedded verbatim; the few inline numbers are read from those
tables' DataFrames or from the stages' provenance), never typed by hand. A panel that cannot
be built honestly (smoke runs without ``allow_smoke``, mixed profiles, duplicate runs) is
refused, noted in results.md and in ``<stage>_<panel>_refused.md``, and is not an error;
only hard errors (a missing store, an unreadable manifest, a crash) raise.

Run selection: ``run_ids`` are split by panel (``meta-*`` ids -> cc18, the others -> core);
a panel none of whose runs is named falls back to rq1's default selection (one run per
dataset, most recent, non-smoke preferred).
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from stl_runner import keys

from stl_analysis.collect import META_PREFIX, collect, open_store, provenance
from stl_analysis.cost import run_cost
from stl_analysis.rq1 import (
    PANELS,
    SMOKE_STAMP,
    UNSPECIFIED,
    Panel,
    PanelError,
    _write,
    run_panel,
    select_panel,
    to_markdown,
)
from stl_analysis.rq1 import refused as rq1_refused
from stl_analysis.rq2 import EPS_GRID, run_rq2
from stl_analysis.rq2 import refused as rq2_refused

RQ3_NOTEBOOK = "notebooks/rq3_meta.ipynb"
RQ1_NOTEBOOK = "notebooks/rq1_benchmark.ipynb"
RQ2_NOTEBOOK = "notebooks/rq2_dp.ipynb"
TIME_CAP_ITEM = "time cap exceeded"
LODO_METRICS = (("accuracy", "accuracy", "baseline_accuracy"),
                ("balanced_accuracy", "balanced_accuracy", "baseline_balanced_accuracy"),
                ("mean_regret", "mean_regret", "baseline_mean_regret"),
                ("n_gate_failing", "n_gate_failing", "baseline_n_gate_failing"))


# ---------------------------------------------------------------- helpers


def _rel(path: str | Path, out: Path) -> str:
    return os.path.relpath(Path(path), out).replace("\\", "/")


def _embed(path: str | Path) -> str:
    """A generated markdown table, verbatim."""
    return Path(path).read_text(encoding="utf-8").rstrip() + "\n"


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _ids_for(name: str, run_ids: list[str] | None) -> list[str] | None:
    if not run_ids:
        return None
    sel = [r for r in run_ids if r.startswith(META_PREFIX) == (name == "cc18")]
    return sel or None


def _eps_label(eps: float) -> str:
    return f"ε={eps:g}"


def eps_summary(cur: pd.DataFrame) -> pd.DataFrame:
    """Compact view of an RQ2 curve table: one row per (dataset, metric), one column per ε
    with the seed mean (3 significant digits) or the gap's reason, plus a ``gate (core)``
    row per dataset and the non-DP reference."""
    eps = sorted(set(EPS_GRID) | set(cur.epsilon.dropna()))
    cols = ["dataset_id", "metric", *(_eps_label(e) for e in eps), "ref_generator",
            "ref_value"]
    rows: list[dict[str, Any]] = []
    for ds, d in cur.groupby("dataset_id", sort=True):
        gate = {"dataset_id": ds, "metric": "gate (core)", "ref_generator": "",
                "ref_value": ""}
        for metric in dict.fromkeys(d.metric):
            m = d[d.metric == metric].set_index("epsilon")
            row = {"dataset_id": ds, "metric": metric,
                   "ref_generator": next(iter(m.ref_generator.dropna()), "") or "",
                   "ref_value": _fmt(next(iter(m.ref_value.dropna()), math.nan))}
            for e in eps:
                if e not in m.index:
                    row[_eps_label(e)] = ""
                    continue
                r = m.loc[e]
                if not math.isnan(r["mean"]):
                    cell = f"{r['mean']:.3g}"
                elif r.metric_note:
                    cell = r.metric_note
                elif "failed" in str(r.annotation):
                    cell = "failed"
                elif "not in run" in str(r.annotation):
                    cell = "not in run"
                else:
                    cell = "no data"
                row[_eps_label(e)] = cell
                g = r.gate_passed
                gate[_eps_label(e)] = ("" if g is None or (isinstance(g, float) and math.isnan(g))
                                       else ("✓" if bool(g) else "✗"))
            rows.append(row)
        rows.append(gate)
    return pd.DataFrame(rows, columns=cols).fillna("")


# ---------------------------------------------------------------- RQ3 pointer


def _find_lodo(stores: list[str]) -> tuple[str, dict] | None:
    for root in stores:
        try:
            store = open_store(root)
        except FileNotFoundError:
            continue
        if store.exists(keys.meta_lodo()):
            return root, store.read_json(keys.meta_lodo())
    return None


def rq3_section(stores: list[str], out: str | Path) -> tuple[str, list[str]]:
    """The RQ3 pointer section (markdown) and the files it wrote: a summary of the first
    ``meta/lodo.json`` found in ``stores`` (meta-model vs majority baseline) or a note."""
    out = Path(out)
    found = _find_lodo(stores)
    lines = ["RQ3 (can a meta-model predict the best generator from meta-features?) is "
             f"analysed in `{RQ3_NOTEBOOK}`; this section only summarises the stored "
             "leave-one-dataset-out (LODO) evaluation.", ""]
    if found is None:
        lines.append(f"There is no `meta/lodo.json` in the stores ({', '.join(stores)}): run "
                     "the meta-benchmark and `stl_meta` LODO evaluation first "
                     "(`scripts/run_meta_benchmark.sh`; README \"RQ3\").")
        return "\n".join(lines) + "\n", []
    root, lodo = found
    tables = out / "tables"
    summary = pd.DataFrame(
        [{"quantity": q, "meta_model": lodo.get(a), "majority_baseline": lodo.get(b)}
         for q, a, b in LODO_METRICS], columns=["quantity", "meta_model", "majority_baseline"]
    ).astype(object)
    counts = pd.DataFrame([
        {"quantity": "n_datasets", "value": lodo.get("n_datasets")},
        {"quantity": "n_labelled", "value": lodo.get("n_labelled")},
        {"quantity": "n_none (no candidate passed the gate)", "value": lodo.get("n_none")},
        {"quantity": "n_excluded (missing meta-features / run, incomplete)",
         "value": len(lodo.get("excluded") or [])},
        {"quantity": "n_labels", "value": len(lodo.get("labels") or [])},
    ]).astype(object)
    files = [str(_write(tables / "rq3_lodo_summary.csv", summary.to_csv(index=False))),
             str(_write(tables / "rq3_lodo_summary.md", to_markdown(
                 summary, "RQ3 LODO: meta-model vs majority baseline (from meta/lodo.json; "
                          "regret as defined in lodo.json regret_definition)", index=False))),
             str(_write(tables / "rq3_lodo_counts.csv", counts.to_csv(index=False))),
             str(_write(tables / "rq3_lodo_counts.md", to_markdown(
                 counts, "RQ3 LODO: meta-dataset size", index=False)))]
    lines.append(f"Source: `meta/lodo.json` in `{root}`.")
    lines.append("")
    if lodo.get("insufficient_data"):
        lines += [f"**Insufficient data**: {lodo.get('reason')}. No accuracy or regret is "
                  "reported.", ""]
    lines += [_embed(files[1]), _embed(files[3])]
    return "\n".join(lines) + "\n", files


# ---------------------------------------------------------------- tables of the report


def _provenance_table(panels: dict[str, Panel]) -> pd.DataFrame:
    rows = [{"panel": name, "dataset_id": r["dataset_id"], "run_id": r["run_id"],
             "store": r["store"], "n_seeds": r["n_seeds"],
             "budget_profile": r["budget_profile"] or UNSPECIFIED,
             "n_jobs": r["n_jobs"], "n_reported": r["n_reported"],
             "smoke": "; ".join(r["smoke_reasons"]) if r["is_smoke"] else ""}
            for name, p in panels.items() for r in p.runs]
    return pd.DataFrame(rows, columns=["panel", "dataset_id", "run_id", "store", "n_seeds",
                                       "budget_profile", "n_jobs", "n_reported", "smoke"])


def _limitations(df: pd.DataFrame, panels: dict[str, Panel], costs: dict[str, pd.DataFrame],
                 rq2: dict[str, dict], refusals: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, p in panels.items():
        smoke = [r for r in p.runs if r["is_smoke"]]
        rows.append({"item": "smoke runs", "panel": name, "count": len(smoke),
                     "detail": "; ".join(f"{r['run_id']} ({', '.join(r['smoke_reasons'])})"
                                         for r in smoke)})
        sel = pd.concat([df[(df.store == r["store"]) & (df.run_id == r["run_id"])]
                         for r in p.runs])
        done = sel[(sel.report_status == "done") & sel.report_error.isna()]
        col = "privacy.mia_domias_advantage"
        na = done[done[col].isna()] if col in done.columns else done
        rows.append({"item": "DOMIAS n/a", "panel": name, "count": int(len(na)),
                     "detail": f"of {len(done)} done reports; "
                               + ", ".join(f"{g} x{n}" for g, n in
                                           na.generator.value_counts().sort_index().items())})
        c = costs.get(name)
        for item, col in (("failed jobs", "n_failed"), (TIME_CAP_ITEM, "n_time_cap"),
                          ("cancelled jobs", "n_cancelled"), ("pending jobs", "n_pending")):
            if c is None or c.empty:
                rows.append({"item": item, "panel": name, "count": 0, "detail": ""})
                continue
            hit = c[c[col] > 0]
            rows.append({"item": item, "panel": name, "count": int(c[col].sum()),
                         "detail": "; ".join(f"{r.dataset_id}/{r.generator} x{getattr(r, col)}"
                                             for r in hit.itertuples())})
        if name in rq2:
            cur, budget = rq2[name]["curves"], rq2[name]["budget"]
            fb = cur[cur.annotation.str.contains("private-bounds fallback", na=False)] \
                [["dataset_id", "annotation"]].drop_duplicates()
            rows.append({"item": "MST private-bounds fallback failures", "panel": name,
                         "count": int(len(fb)),
                         "detail": "; ".join(f"{r.dataset_id}: {r.annotation}"
                                             for r in fb.itertuples())})
            priv = budget[budget.bounds_path.str.contains("private|mixed", na=False)]
            rows.append({"item": "MST ε spent on private bounds", "panel": name,
                         "count": int(len(priv)),
                         "detail": "; ".join(f"{r.dataset_id} ε={r.epsilon:g}: "
                                             f"{r.eps_bounds:g} on bounds ({r.bounds_path})"
                                             for r in priv.itertuples())})
    for name, why in refusals.items():
        rows.append({"item": "refused panel", "panel": name, "count": 1, "detail": why})
    return pd.DataFrame(rows, columns=["item", "panel", "count", "detail"])


# ---------------------------------------------------------------- results.md sections


def _rq1_md(name: str, panel: Panel, res: dict, out: Path) -> list[str]:
    tables = out / "tables"
    lines = [f"### Panel `{name}`{panel.stamp}", "",
             f"{len(panel.runs)} dataset(s): "
             + ", ".join(f"{r['dataset_id']} (`{r['run_id']}`)" for r in panel.runs), ""]
    ranks = tables / f"rq1_{name}_ranks.md"
    if ranks.exists():
        lines += [_embed(ranks)]
    scores_csv = tables / f"rq1_{name}_scores.csv"
    if scores_csv.exists():
        sc = pd.read_csv(scores_csv)
        win = sc[sc.is_winner.astype(bool)].set_index("dataset_id")
        lines.append("Winners (best gate-passing ranked generator, core balanced score; from "
                     f"`{_rel(scores_csv, out)}`):")
        for ds in sorted(sc.dataset_id.unique()):
            lines.append(f"- {ds}: " + (f"{win.loc[ds, 'generator']} (score "
                                        f"{_fmt(float(win.loc[ds, 'score']))})"
                                        if ds in win.index else "none (no rankable candidate)"))
        lines.append("")
    cd = res["cd"]
    if cd["status"] == "ok":
        fig = next((f for f in cd["files"] if f.endswith(".png")), None)
        lines.append(f"autorank: {cd['omnibus']} + {cd['posthoc']}, p = {_fmt(cd['pvalue'])}"
                     f"{'' if cd['significant'] else ' (not significant)'}; mean-rank order "
                     f"{', '.join(cd['ranking'])}."
                     + (f" Figure: ![CD]({_rel(fig, out)})" if fig else ""))
    else:
        note = tables / f"rq1_{name}_insufficient.md"
        lines.append(f"No CD diagram: {cd['reason']} (see `{_rel(note, out)}`).")
    lines.append("")
    side = tables / f"rq1_{name}_sanity_audit.md"
    if side.exists():
        lines += [f"train_copy sanity row / PrivBayes audit (not ranked): `{_rel(side, out)}`.",
                  ""]
    chars = tables / f"rq1_{name}_characteristics.md"
    if chars.exists():
        lines += [_embed(chars)]
        fig = out / "figures" / f"rq1_{name}_winner_vs_characteristics.png"
        if fig.exists():
            lines += [f"![winner vs characteristics]({_rel(fig, out)})", ""]
    if panel.notes:
        lines += ["Selection notes:", *(f"- {n}" for n in panel.notes), ""]
    return lines


def _rq2_md(name: str, panel: Panel, res: dict, out: Path) -> list[str]:
    tables = out / "tables"
    lines = [f"### Panel `{name}`{panel.stamp}", ""]
    summ = tables / f"rq2_{name}_eps_summary.md"
    if res["curves"].empty:
        lines += ["No MST (ε-DP) generator in the panel runs.", ""]
    else:
        lines += [_embed(summ), f"Full curve table with bands: "
                  f"`{_rel(tables / f'rq2_{name}_mst_curves.md', out)}`.", ""]
    for ds in sorted({r["dataset_id"] for r in panel.runs}):
        for stem, what in ((f"rq2_{name}_{ds}_mst_eps", "MST ε curves"),
                           (f"rq2_{name}_{ds}_pareto", "Pareto front without / with MST")):
            f = out / "figures" / f"{stem}.png"
            if f.exists():
                lines.append(f"- {ds}, {what}: ![{what}]({_rel(f, out)})")
    lines.append("")
    par = res["pareto"]
    if not par.empty:
        lines.append("Front shift (axes: core privacy vs mean(fidelity, utility); from "
                     f"`{_rel(tables / f'rq2_{name}_pareto.csv', out)}`):")
        for ds, p in par.groupby("dataset_id", sort=True):
            non = sorted(p.generator[p.front_non_dp])
            both = sorted(p.generator[p.front_with_dp])
            joined = sorted(set(both) - set(non))
            left = sorted(set(non) - set(both))
            lines.append(f"- {ds}: front without DP {non or 'empty'}; with MST {both or 'empty'}"
                         f"; MST joining: {joined or 'none'}; pushed off: {left or 'none'}")
        lines += ["", _embed(tables / f"rq2_{name}_pareto.md")]
    lines += [_embed(tables / f"rq2_{name}_eps_budget.md")]
    pb = tables / f"rq2_{name}_privbayes_curves.md"
    if res["has_privbayes"]:
        figs = sorted((out / "figures").glob(f"rq2_{name}_*_privbayes_audit.png"))
        lines += ["PrivBayes is a nominal DP audit (not ε-DP): it is never on the MST curves "
                  f"or fronts; its own table is `{_rel(pb, out)}`."]
        lines += [f"- ![PrivBayes audit]({_rel(f, out)})" for f in figs]
    else:
        lines.append("No PrivBayes (nominal DP audit, not ε-DP) jobs in the panel runs.")
    lines.append("")
    return lines


def _cost_md(name: str, panel: Panel, out: Path) -> list[str]:
    tables = out / "tables"
    fig = out / "figures" / f"cost_{name}_seconds.png"
    lines = [f"### Panel `{name}`{panel.stamp}", "", _embed(tables / f"cost_{name}.md")]
    if fig.exists():
        lines.append(f"![cost]({_rel(fig, out)})")
    lines += [f"LaTeX: `{_rel(tables / f'cost_{name}.tex', out)}`.", ""]
    return lines


# ---------------------------------------------------------------- driver


def build_report(stores: list[str], run_ids: list[str] | None = None,
                 allow_smoke: bool = False, out: str | Path = "reports",
                 rq2_cc18: bool = False, log=print) -> dict[str, Any]:
    out = Path(out)
    tables = out / "tables"
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    files: list[str] = []
    # collect
    df = collect(stores, include_meta=True)
    res_path = tables / "results.parquet"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(res_path, index=False)
    df.to_csv(res_path.with_suffix(".csv"), index=False)
    coll_prov = provenance(df, stores, run_ids, True)
    files += [str(res_path), str(res_path.with_suffix(".csv")),
              str(_write(tables / "results.provenance.json",
                         json.dumps(coll_prov, indent=2)))]
    unknown = sorted(set(run_ids or []) - set(df.run_id.dropna()))
    log(f"collect: {len(df)} jobs; {coll_prov['n_reports']} with reports")
    # panels
    panels: dict[str, Panel] = {}
    refusals: dict[str, str] = {}
    empty: list[str] = []
    for name in PANELS:
        try:
            p = select_panel(df, name, run_ids=_ids_for(name, run_ids),
                             allow_smoke=allow_smoke)
        except PanelError as e:
            refusals[name] = str(e)
            files.append(str(rq1_refused(name, out, e)))
            if name == "core" or rq2_cc18:
                files.append(str(rq2_refused(name, out, e)))
            log(f"panel {name}: refused")
            continue
        if not p.runs:
            empty.append(name)
            log(f"panel {name}: no runs")
            continue
        panels[name] = p
    # stages
    rq1_res, rq2_res, costs = {}, {}, {}
    for name, p in panels.items():
        rq1_res[name] = run_panel(df, p, out)
        files += rq1_res[name]["files"]
        log(f"rq1 {name}: {len(p.runs)} datasets{' [SMOKE]' if p.is_smoke else ''}")
        if name == "core" or rq2_cc18:
            r = run_rq2(df, p, out)
            cur = pd.read_csv(tables / f"rq2_{name}_mst_curves.csv",
                              keep_default_na=True).fillna({"annotation": "",
                                                            "metric_note": ""})
            summ = eps_summary(cur)
            files += r["files"] + [
                str(_write(tables / f"rq2_{name}_eps_summary.csv", summ.to_csv(index=False))),
                str(_write(tables / f"rq2_{name}_eps_summary.md", to_markdown(
                    summ, f"RQ2 {name}: MST seed-mean metric per ε (gap reason when "
                          f"missing; gate (core) per ε; ref = best gate-passing non-DP)"
                          f"{' — ' + SMOKE_STAMP if p.is_smoke else ''}", index=False)))]
            rq2_res[name] = {
                "curves": cur,
                "pareto": pd.read_csv(tables / f"rq2_{name}_pareto.csv"),
                "budget": pd.read_csv(tables / f"rq2_{name}_eps_budget.csv"),
                "has_privbayes": any("privbayes_curves" in f for f in r["files"]),
            }
            log(f"rq2 {name}: done")
        c = run_cost(df, p, out)
        costs[name] = c["table"]
        files += c["files"]
        log(f"cost {name}: {len(c['table'])} rows")
    rq3_text, rq3_files = rq3_section(stores, out)
    files += rq3_files
    # report tables
    prov_t = _provenance_table(panels)
    files += [str(_write(tables / "report_provenance.csv", prov_t.to_csv(index=False))),
              str(_write(tables / "report_provenance.md", to_markdown(
                  prov_t, "Runs used by this report (one per dataset and panel)",
                  index=False)))]
    lim = _limitations(df, panels, costs, rq2_res, refusals)
    files += [str(_write(tables / "report_limitations.csv", lim.to_csv(index=False))),
              str(_write(tables / "report_limitations.md", to_markdown(
                  lim, "Limitations (auto-filled from the tables above)", index=False)))]
    smoke = any(p.is_smoke for p in panels.values())
    libs: dict[str, set[str]] = {}
    mversions: set[int] = set()
    for name in panels:
        pp = json.loads((tables / f"rq1_{name}.provenance.json").read_text(encoding="utf-8"))
        for lib, vers in pp["generator_libraries"].items():
            libs.setdefault(lib, set()).update(vers)
        mversions.update(pp["metrics_versions"])
    profiles = sorted({p.budget_profile or UNSPECIFIED for p in panels.values()})
    # results.md
    md: list[str] = ["# Synthetic Data Trust Lab — results", ""]
    if smoke:
        smoke_runs = [r["run_id"] for p in panels.values() for r in p.runs if r["is_smoke"]]
        md += [f"> **{SMOKE_STAMP}.** Smoke inputs (budget profile smoke, fewer epochs than "
               f"the profile, or fewer than 3 seeds): {', '.join(smoke_runs)}. Every "
               "artefact made from them carries this stamp.", ""]
    md += ["_Auto-generated by `python -m stl_analysis report` "
           "(`scripts/make_report.sh`). Every number below is rendered from a table under "
           "`tables/`; do not edit by hand._", "", "## Provenance", "",
           f"- generated_at: {generated_at}",
           f"- stores: {', '.join(f'`{s}`' for s in stores)}",
           f"- run filter: {', '.join(run_ids) if run_ids else 'none (default selection)'}"
           + (f" — not found: {', '.join(unknown)}" if unknown else ""),
           f"- allow_smoke: {allow_smoke}; RQ2 cc18 panel: {rq2_cc18}",
           f"- panels: {', '.join(panels) or 'none'}"
           + (f"; refused: {', '.join(refusals)}" if refusals else "")
           + (f"; without runs: {', '.join(empty)}" if empty else ""),
           f"- budget profiles: {', '.join(profiles) or 'n/a'}",
           f"- metrics versions: {', '.join(map(str, sorted(mversions))) or 'n/a'}",
           "- generator libraries: " + (", ".join(f"{k} {'/'.join(sorted(v))}"
                                                  for k, v in sorted(libs.items())) or "n/a"),
           "- analysis versions: " + ", ".join(f"{k} {v}" for k, v in
                                               coll_prov["versions"].items() if v),
           f"- results table: `{_rel(res_path, out)}` (all {len(df)} jobs of the stores)", "",
           _embed(tables / "report_provenance.md")]
    md += ["## RQ1", "", f"Notebook: `{RQ1_NOTEBOOK}`.", ""]
    for name in PANELS:
        if name in panels:
            md += _rq1_md(name, panels[name], rq1_res[name], out)
        elif name in refusals:
            md += [f"### Panel `{name}`", "", f"Panel refused: {refusals[name]}", ""]
        else:
            md += [f"### Panel `{name}`", "", "No runs in the stores.", ""]
    md += ["## RQ2", "", f"Notebook: `{RQ2_NOTEBOOK}`.", ""]
    for name in (PANELS if rq2_cc18 else ("core",)):
        if name in rq2_res:
            md += _rq2_md(name, panels[name], rq2_res[name], out)
        elif name in refusals:
            md += [f"### Panel `{name}`", "", f"Panel refused: {refusals[name]}", ""]
        else:
            md += [f"### Panel `{name}`", "", "No runs in the stores.", ""]
    md += ["## Cost", "",
           "Fit / sample seconds and peak RSS over done jobs (median, IQR); outcome counts "
           "over every planned job (time cap = failed with the runner's time-cap message).",
           ""]
    for name in panels:
        md += _cost_md(name, panels[name], out)
    md += ["## RQ3", "", rq3_text, "## Limitations", "", _embed(
        tables / "report_limitations.md")]
    results = _write(out / "results.md", "\n".join(md).rstrip() + "\n")
    files.append(str(results))
    prov = {"generated_at": generated_at, "stores": list(stores), "run_ids": run_ids,
            "unknown_run_ids": unknown, "allow_smoke": allow_smoke, "rq2_cc18": rq2_cc18,
            "smoke": smoke, "stamp": SMOKE_STAMP if smoke else None,
            "panels": {n: {"runs": p.runs, "budget_profile": p.budget_profile,
                           "is_smoke": p.is_smoke, "notes": p.notes}
                       for n, p in panels.items()},
            "refused": refusals, "empty_panels": empty,
            "metrics_versions": sorted(mversions),
            "generator_libraries": {k: sorted(v) for k, v in sorted(libs.items())},
            "versions": coll_prov["versions"]}
    prov_path = tables / "report.provenance.json"
    files.append(str(prov_path))
    prov["outputs"] = files
    _write(prov_path, json.dumps(prov, indent=2, default=str))
    return {"results_md": str(results), "files": files, "smoke": smoke,
            "refused": refusals, "panels": list(panels)}
