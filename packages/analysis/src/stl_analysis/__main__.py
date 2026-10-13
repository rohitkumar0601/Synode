"""CLI: ``python -m stl_analysis <command>`` (run via ``scripts/envrun.sh analysis``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stl_analysis.collect import collect, provenance


def _cmd_collect(args: argparse.Namespace) -> int:
    df = collect(args.store, run_ids=args.run, include_meta=not args.no_meta)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)
    prov = provenance(df, args.store, args.run, not args.no_meta)
    prov["outputs"] = [str(out), str(out.with_suffix(".csv"))]
    prov_path = out.with_name(f"{out.stem}.provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    n_runs = sum(len(r) for r in prov["runs"].values())
    print(f"{len(df)} jobs from {n_runs} runs ({prov['n_reports']} with reports) -> {out}, "
          f"{out.with_suffix('.csv').name}, {prov_path.name}")
    return 0


def _cmd_rq1(args: argparse.Namespace) -> int:
    from stl_analysis.rq1 import PANELS, PanelError, refused, run_panel, select_panel

    df = collect(args.store, include_meta=args.panel in ("cc18", "all"))
    panels = PANELS if args.panel == "all" else (args.panel,)
    rc = 0
    for name in panels:
        try:
            panel = select_panel(df, name, run_ids=args.run, allow_smoke=args.allow_smoke)
        except PanelError as e:
            print(f"rq1 {name}: refused: {e} -> {refused(name, args.output, e)}")
            rc = 1
            continue
        res = run_panel(df, panel, args.output)
        cd = res["cd"]
        what = (f"CD ok ({cd['omnibus']}/{cd['posthoc']}, p={cd['pvalue']:.3g}, ranking "
                f"{cd['ranking']})" if cd["status"] == "ok" else f"no CD: {cd['reason']}")
        print(f"rq1 {name}: {len(panel.runs)} datasets"
              f"{' [SMOKE]' if panel.is_smoke else ''}; {what}")
        for f in res["files"]:
            print(f"  {f}")
    return rc


def _cmd_rq2(args: argparse.Namespace) -> int:
    from stl_analysis.rq1 import PanelError, select_panel
    from stl_analysis.rq2 import refused, run_rq2

    df = collect(args.store, include_meta=args.panel == "cc18")
    try:
        panel = select_panel(df, args.panel, run_ids=args.run, allow_smoke=args.allow_smoke)
    except PanelError as e:
        print(f"rq2 {args.panel}: refused: {e} -> {refused(args.panel, args.output, e)}")
        return 1
    res = run_rq2(df, panel, args.output)
    print(f"rq2 {args.panel}: {len(panel.runs)} datasets{' [SMOKE]' if panel.is_smoke else ''}")
    for ds, notes in res["annotations"].items():
        for n in notes:
            print(f"  {ds}: {n}".replace("ε", "eps"))  # Windows consoles may be cp1252
    for f in res["files"]:
        print(f"  {f}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from stl_analysis.report import build_report

    try:
        res = build_report(args.store, run_ids=args.run, allow_smoke=args.allow_smoke,
                           out=args.output, rq2_cc18=args.rq2_cc18)
    except FileNotFoundError as e:
        print(f"report: {e}", file=sys.stderr)
        return 2
    print(f"report: {len(res['files'])} files; panels {res['panels'] or 'none'}"
          f"{'; refused ' + ', '.join(res['refused']) if res['refused'] else ''}"
          f"{' [SMOKE]' if res['smoke'] else ''} -> {res['results_md']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stl_analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect", help="tidy results table (one row per job) from stores")
    p.add_argument("--store", action="append", required=True,
                   help="artifact store root (repeatable)")
    p.add_argument("--run", action="append", default=None,
                   help="only this run id (repeatable)")
    p.add_argument("--no-meta", action="store_true", help="drop meta-* (RQ3) runs")
    p.add_argument("-o", "--output", default="reports/tables/results.parquet",
                   help="parquet path; .csv and .provenance.json are written next to it")
    p.set_defaults(func=_cmd_collect)
    p = sub.add_parser("rq1", help="RQ1 rank tables, autorank CD diagrams, characteristics")
    p.add_argument("--store", action="append", required=True,
                   help="artifact store root (repeatable)")
    p.add_argument("--panel", choices=["core", "cc18", "all"], default="all")
    p.add_argument("--run", action="append", default=None,
                   help="use exactly these run ids (repeatable; one per dataset)")
    p.add_argument("--allow-smoke", action="store_true",
                   help="allow smoke runs (artefacts stamped 'SMOKE — not for the thesis')")
    p.add_argument("-o", "--output", default="reports",
                   help="output root: tables/ and figures/ are written below it")
    p.set_defaults(func=_cmd_rq1)
    p = sub.add_parser("rq2", help="RQ2 MST epsilon curves, Pareto-front shift, epsilon budget")
    p.add_argument("--store", action="append", required=True,
                   help="artifact store root (repeatable)")
    p.add_argument("--panel", choices=["core", "cc18"], default="core")
    p.add_argument("--run", action="append", default=None,
                   help="use exactly these run ids (repeatable; one per dataset)")
    p.add_argument("--allow-smoke", action="store_true",
                   help="allow smoke runs (artefacts stamped 'SMOKE — not for the thesis')")
    p.add_argument("-o", "--output", default="reports",
                   help="output root: tables/ and figures/ are written below it")
    p.set_defaults(func=_cmd_rq2)
    p = sub.add_parser("report", help="collect -> rq1 -> rq2 -> cost -> results.md")
    p.add_argument("--store", action="append", required=True,
                   help="artifact store root (repeatable)")
    p.add_argument("--run", action="append", default=None,
                   help="use exactly these run ids (repeatable; meta-* ids go to the cc18 "
                        "panel, the others to core; a panel with none named uses the "
                        "default selection)")
    p.add_argument("--allow-smoke", action="store_true",
                   help="allow smoke runs (artefacts stamped 'SMOKE — not for the thesis')")
    p.add_argument("--rq2-cc18", action="store_true",
                   help="also run RQ2 on the cc18 meta-benchmark panel")
    p.add_argument("-o", "--output", default="reports",
                   help="output root: results.md, tables/ and figures/ are written below it")
    p.set_defaults(func=_cmd_report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
