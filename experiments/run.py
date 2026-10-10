"""Benchmark CLI. Each stage runs in the environment that has its libraries:

  bash scripts/envrun.sh eval python experiments/run.py prepare
  bash scripts/envrun.sh sdv  python experiments/run.py generate     # gaussian_copula + baselines
  bash scripts/envrun.sh eval python experiments/run.py generate     # whatever eval can run
  bash scripts/envrun.sh synthcity python experiments/run.py metrics  # synthcity metrics
  bash scripts/envrun.sh eval python experiments/run.py evaluate
  bash scripts/envrun.sh eval python experiments/run.py report --preset balanced
  bash scripts/envrun.sh meta python experiments/run.py metafeatures  # RQ3 meta-features
  bash scripts/envrun.sh meta python experiments/run.py meta-eval      # RQ3 LODO [--train]

Hydra overrides go before the subcommand:  -o datasets=adult -o seeds=[0] -o max_rows=2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from stl_core.aggregate import aggregate
from stl_core.bounds import bound_value_problems
from stl_core.catalog import is_available
from stl_core.decision import recommend
from stl_core.metadata import ColumnType, infer_metadata
from stl_core.schemas import MetricReport
from stl_runner import keys
from stl_runner.datasets import load_dataset
from stl_runner.pipeline import (
    ensure_splits,
    gpu_only_blocked,
    job_state,
    load_jobs,
    load_metadata,
    plan_run,
    register_dataset,
    run_generate_with_cap,
    set_state,
)
from stl_runner.storage import ArtifactStore

CONF_DIR = Path(__file__).resolve().parent / "conf"


def load_config(overrides: list[str]) -> dict:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def _store(cfg: dict) -> ArtifactStore:
    return ArtifactStore(os.environ.get("STL_STORAGE_ROOT") or cfg["storage_root"])


def _run_id(cfg: dict, args) -> str:
    return args.run or cfg["run_id"] or f"{cfg['dataset']['name']}-main"


def cmd_prepare(cfg: dict, args) -> int:
    store = _store(cfg)
    ds = cfg["dataset"]
    df, target = load_dataset(ds)
    md = infer_metadata(df, target)
    unknown = sorted(set(ds.get("column_types") or {}) - set(df.columns))
    if unknown:
        print(f"error: dataset.column_types names columns not in dataset {ds['name']!r}: "
              f"{', '.join(repr(c) for c in unknown)}", file=sys.stderr)
        return 2
    if ds.get("column_types"):
        md = md.model_copy(update={"columns": {
            **md.columns, **{c: ColumnType(t) for c, t in ds["column_types"].items()}}})
    public_info = {"public_bounds": dict(ds.get("public_bounds") or {}),
                   "public_log_scale": list(ds.get("public_log_scale") or [])}
    numerical = {c for c, t in md.columns.items() if t == ColumnType.numerical}
    not_numerical = sorted((set(public_info["public_bounds"])
                            | set(public_info["public_log_scale"])) - numerical)
    if not_numerical:
        print(f"error: dataset.public_bounds/public_log_scale name columns that are not "
              f"numerical in dataset {ds['name']!r}: "
              f"{', '.join(repr(c) for c in not_numerical)}", file=sys.stderr)
        return 2
    unbounded = sorted(set(public_info["public_log_scale"]) - set(public_info["public_bounds"]))
    if unbounded:
        print(f"error: dataset.public_log_scale names columns without public_bounds: "
              f"{', '.join(repr(c) for c in unbounded)}", file=sys.stderr)
        return 2
    bad_values = bound_value_problems(public_info["public_bounds"],
                                      public_info["public_log_scale"])
    if bad_values:
        print(f"error: dataset.public_bounds has invalid values: {'; '.join(bad_values)}",
              file=sys.stderr)
        return 2
    prepared = register_dataset(store, ds["name"], df, md, max_rows=cfg["max_rows"],
                                public_info=public_info)
    for seed in cfg["seeds"]:
        ensure_splits(store, ds["name"], seed)
    jobs = plan_run(store, _run_id(cfg, args), ds["name"], cfg["generators_to_run"],
                    cfg["seeds"], params=cfg["generator_params"],
                    time_cap_seconds=cfg["budget"]["time_cap_seconds"],
                    budget_profile=cfg["budget"].get("profile"))
    print(f"registered {ds['name']} ({len(prepared)} rows); planned {len(jobs)} jobs "
          f"in run {_run_id(cfg, args)}")
    return 0


def cmd_generate(cfg: dict, args) -> int:
    """Exit 1 if any job run here failed (a job skipped as unavailable is not a failure)."""
    store = _store(cfg)
    retry = {"queued", "failed"} if args.retry_failed else {"queued"}
    allow_gpu_only = getattr(args, "allow_gpu_only", False)
    any_failed = False
    for job in load_jobs(store, _run_id(cfg, args)):
        if job_state(store, job)["state"] not in retry:
            continue
        if gpu_only_blocked(job.generator, allow_gpu_only):
            print(f"skipped {job.job_id}: GPU-only generator (use --allow-gpu-only to force)")
            continue
        if not is_available(job.generator):
            print(f"skipped {job.job_id}: not installed in this environment")
            continue
        state = run_generate_with_cap(store, job, allow_gpu_only=allow_gpu_only)
        any_failed |= state == "failed"
        print(f"{job.job_id}: {state}", flush=True)
    return 1 if any_failed else 0


def _metrics_stage():
    """The synthcity metrics module (compute_stored_job, METRICS_VERSION)."""
    from stl_synthcity import metrics

    return metrics


def _needs_recompute(result: dict, version: int) -> bool:
    return ("error" in result or bool(result.get("errors"))
            or result.get("metrics_version") != version)


def cmd_metrics(cfg: dict, args) -> int:
    """synthcity metrics (detection AUC, alpha-precision/beta-recall/authenticity, DOMIAS MIA)
    for every generated job without a synthcity_metrics.json. With --recompute, also for
    generated or evaluated ("done") jobs whose file has an error or an older metrics_version;
    an evaluated job's report is then deleted and its state reset to "generated" so that
    evaluate re-runs it. Needs the synthcity env (exit 2 elsewhere). Exit 1 if any job's
    metrics failed, wholly or in part."""
    try:
        stage = _metrics_stage()
    except ImportError as exc:
        if exc.name in ("stl_synthcity", "stl_synthcity.metrics"):
            print("error: the metrics stage runs in the synthcity env "
                  f"(bash scripts/envrun.sh synthcity ...): {exc}", file=sys.stderr)
        else:
            print("error: importing the synthcity metrics failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    recompute = getattr(args, "recompute", False)
    store = _store(cfg)
    any_failed = False
    for job in load_jobs(store, _run_id(cfg, args)):
        state = job_state(store, job)["state"]
        key = keys.synthcity_metrics(job.run_id, job.job_id)
        if state == "generated" and not store.exists(key):
            pass
        elif (recompute and state in ("generated", "done") and store.exists(key)
              and _needs_recompute(store.read_json(key), stage.METRICS_VERSION)):
            if state == "done":
                store.delete(keys.report(job.run_id, job.job_id))
                set_state(store, job, "generated")
                print(f"{job.job_id}: recomputing synthcity metrics; report reset for "
                      "re-evaluation", flush=True)
        else:
            continue
        result = stage.compute_stored_job(store, job)
        if "error" in result:
            any_failed = True
            print(f"{job.job_id}: synthcity metrics failed: {result['error']}", flush=True)
        elif result.get("errors"):
            any_failed = True
            errors = "; ".join(f"{k}: {v}" for k, v in result["errors"].items())
            print(f"{job.job_id}: synthcity metrics done with errors: {errors}", flush=True)
        else:
            print(f"{job.job_id}: synthcity metrics done", flush=True)
    return 1 if any_failed else 0


def cmd_evaluate(cfg: dict, args) -> int:
    from stl_eval.evaluate import EvalConfig, evaluate_stored_job
    from stl_runner.tracking import log_report

    if cfg.get("tracking_uri"):
        os.environ.setdefault("MLFLOW_TRACKING_URI", cfg["tracking_uri"])
    store = _store(cfg)
    jobs = load_jobs(store, _run_id(cfg, args))
    try:  # a config error must stop the run, not be recorded per job as a metric warning
        config = EvalConfig(**cfg["eval"])
        for dataset_id in sorted({j.dataset_id for j in jobs}):
            config.check(load_metadata(store, dataset_id))
    except (TypeError, ValueError) as exc:
        print(f"error: invalid eval config: {exc}", file=sys.stderr)
        return 2
    any_failed = False
    for job in jobs:
        state = job_state(store, job)["state"]
        has_report = store.exists(keys.report(job.run_id, job.job_id))
        if state in ("queued", "fitting", "sampling"):
            print(f"{job.job_id}: skipped (state {state})", flush=True)
            continue
        if state == "evaluating":  # a previous evaluation was killed mid-way
            print(f"{job.job_id}: was interrupted; re-evaluating", flush=True)
        elif not (state == "generated" or (state == "failed" and not has_report)):
            continue
        report = evaluate_stored_job(store, job, config)
        log_report(report, job)
        any_failed |= report.status == "failed"
        print(f"{job.job_id}: {report.status}", flush=True)
    return 1 if any_failed else 0


def _reports(store: ArtifactStore, run_id: str) -> list[MetricReport]:
    return [
        MetricReport.model_validate(store.read_json(keys.report(j.run_id, j.job_id)))
        for j in load_jobs(store, run_id)
        if store.exists(keys.report(j.run_id, j.job_id))
    ]


def cmd_report(cfg: dict, args) -> int:
    run_id = _run_id(cfg, args)
    store = _store(cfg)
    reports = _reports(store, run_id)
    n_total = len(load_jobs(store, run_id))
    if not reports:
        print(f"no reports yet for run {run_id}; run evaluate first")
        return 1
    versions = sorted({r.metrics_version for r in reports if r.status == "done"},
                      key=lambda v: -1 if v is None else v)
    if len(versions) > 1:
        print(f"warning: run {run_id} mixes metric versions "
              f"({', '.join(str(v) for v in versions)}): re-run metrics --recompute and "
              "evaluate", file=sys.stderr)
    aggs = aggregate(reports)
    try:
        weights = tuple(float(w) for w in args.weights.split(",")) if args.weights else None
        rec = recommend(aggs, preset=args.preset, weights=weights)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"recommendation": rec.model_dump(mode="json"),
                          "aggregated": [a.model_dump(mode="json") for a in aggs],
                          "jobs_reported": len(reports), "jobs_total": n_total}, indent=2))
        return 0
    print(f"{len(reports)} of {n_total} jobs reported")
    print(f"{'generator':<24}{'fidelity':>10}{'utility':>10}{'privacy':>10}{'score':>8}  gate")
    for c in rec.ranking:
        s = c.scores
        cells = [f"{v:>10.3f}" if v is not None else f"{'—':>10}"
                 for v in (s.fidelity, s.utility, s.privacy)]
        score = f"{c.score:>8.3f}" if c.score is not None else f"{'—':>8}"
        gate = "pass" if c.gate_passed else "FAIL: " + "; ".join(c.gate_failures)
        print(f"{c.generator:<24}{''.join(cells)}{score}  {gate}")
    print()
    print(rec.reason)
    return 0


def cmd_export_results(cfg: dict, args) -> int:
    from stl_runner.bundle import export_bundle

    run_id = _run_id(cfg, args)
    store = _store(cfg)
    if not store.exists(keys.run_manifest(run_id)):
        print(f"error: no run {run_id!r} in {store.root}", file=sys.stderr)
        return 2
    try:
        manifest = export_bundle(store, run_id, args.out,
                                 include_synthetic=args.include_synthetic)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"exported {len(manifest.jobs)} jobs from run {run_id} to {args.out}")
    return 0


def cmd_import_results(cfg: dict, args) -> int:
    import zipfile

    from stl_runner.bundle import import_bundle, read_manifest
    from stl_runner.tracking import log_report

    if cfg.get("tracking_uri"):
        os.environ.setdefault("MLFLOW_TRACKING_URI", cfg["tracking_uri"])
    store = _store(cfg)
    notes: list[str] = []
    try:
        run_id = read_manifest(args.path).run_id
        reports = import_bundle(store, args.path, force=args.force, warnings=notes)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for note in notes:
        print(f"warning: {note}", file=sys.stderr)
    jobs = {j.job_id: j for j in load_jobs(store, run_id)}
    for report in reports:
        log_report(report, jobs[report.job_id])
    print(f"imported {len(reports)} reports into run {run_id}")
    return 0


def cmd_metafeatures(cfg: dict, args) -> int:
    """pymfe meta-features of the dataset's seed-0 TRAIN split (or of every registered
    dataset with --all-registered) -> datasets/<id>/metafeatures.json. Needs the meta env
    (exit 2 elsewhere, or if the dataset is not registered). Exit 1 if any dataset failed."""
    try:
        from stl_meta import features
    except ImportError as exc:
        print("error: the metafeatures stage runs in the meta env "
              f"(bash scripts/envrun.sh meta ...): {exc}", file=sys.stderr)
        return 2
    store = _store(cfg)
    if args.all_registered:
        ids = sorted({k.split("/")[1] for k in store.list("datasets")
                      if k.endswith("/metadata.json") and k.count("/") == 2})
    else:
        ids = [cfg["dataset"]["name"]]
    missing = [d for d in ids if not store.exists(keys.dataset_metadata(d))]
    if missing or not ids:
        print(f"error: dataset(s) not registered in {store.root}: {', '.join(missing) or '-'}; "
              "run prepare first", file=sys.stderr)
        return 2
    any_failed = False
    for dataset_id in ids:
        try:
            rec = features.compute_for_dataset(store, dataset_id, seed=0,
                                               max_rows=args.max_rows)
        except features.MetaFeatureError as exc:
            any_failed = True
            print(f"{dataset_id}: meta-features failed: {exc}", flush=True)
            continue
        n_nan = sum(v is None for v in rec["features"].values())
        print(f"{dataset_id}: {len(rec['features'])} meta-features ({n_nan} NaN) from "
              f"{rec['n_rows_used']} rows", flush=True)
    return 1 if any_failed else 0


def cmd_meta_eval(cfg: dict, args) -> int:
    """RQ3: build the meta-dataset from the frozen datasets YAML (runs <run-prefix>cc18_<id>),
    run leave-one-dataset-out evaluation -> meta/lodo.json; with --train also fit the final
    predictor -> meta/predictor.joblib + meta/predictor.json. Needs the meta env (exit 2
    elsewhere). Exit 1 if --train was asked but there is too little data to train on."""
    try:
        from stl_meta import meta
    except ImportError as exc:
        print("error: the meta-eval stage runs in the meta env "
              f"(bash scripts/envrun.sh meta ...): {exc}", file=sys.stderr)
        return 2
    store = _store(cfg)
    meta_df = meta.build_meta_dataset(store, args.datasets_yaml, run_prefix=args.run_prefix)
    summary = meta.lodo_evaluate(meta_df)
    meta.write_json(store, keys.meta_lodo(), summary)

    def num(v) -> str:
        return "—" if v is None else f"{v:.3f}"

    for e in summary["excluded"]:
        print(f"excluded {e['dataset_id']}: {e['reason']}")
    if summary["none_datasets"]:
        print(f"no gate-passing generator (label none): {', '.join(summary['none_datasets'])}")
    if summary["predictions"]:
        print(f"{'dataset':<14}{'true':<24}{'predicted':<24}{'baseline':<24}"
              f"{'regret':>8}{'base_regret':>12}")
        for p in summary["predictions"]:
            print(f"{p['dataset_id']:<14}{p['true']:<24}{p['predicted']:<24}"
                  f"{p['baseline']:<24}{num(p['regret']):>8}{num(p['baseline_regret']):>12}")
    print(f"datasets={summary['n_datasets']} labelled={summary['n_labelled']} "
          f"none={summary['n_none']} excluded={len(summary['excluded'])}")
    if summary["insufficient_data"]:
        print(summary["reason"])
    else:
        print(f"{'':<10}{'accuracy':>10}{'bal_acc':>10}{'regret':>10}{'gate_fail':>10}")
        for who, pre in (("model", ""), ("baseline", "baseline_")):
            print(f"{who:<10}{num(summary[pre + 'accuracy']):>10}"
                  f"{num(summary[pre + 'balanced_accuracy']):>10}"
                  f"{num(summary[pre + 'mean_regret']):>10}"
                  f"{summary[pre + 'n_gate_failing']:>10}")
    print(f"wrote {keys.meta_lodo()}")
    if args.train:
        if summary["insufficient_data"]:
            print("error: not training the predictor: " + summary["reason"], file=sys.stderr)
            return 1
        meta.save_artifact(store, meta.train_final(meta_df), summary)
        print(f"wrote {keys.meta_predictor()} and {keys.meta_predictor_summary()}")
    return 0


def cmd_all(cfg: dict, args) -> int:
    for fn in (cmd_prepare, cmd_generate, cmd_metrics, cmd_evaluate, cmd_report):
        rc = fn(cfg, args)
        if fn is cmd_metrics and rc == 2:
            continue  # not the synthcity env: evaluate warns and the gate fails closed
        if rc != 0:
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="Hydra override, e.g. -o seeds=[0]")
    parser.add_argument("--run", help="run id (default: <dataset>-main)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    gen = sub.add_parser("generate")
    gen.add_argument("--retry-failed", action="store_true")
    gen.add_argument("--allow-gpu-only", action="store_true",
                     help="run GPU-only generators (tabddpm, great) even without CUDA")
    met = sub.add_parser("metrics", help="synthcity metrics for generated jobs (synthcity env)")
    met.add_argument("--recompute", action="store_true",
                     help="also re-run jobs whose metrics file has an error or is stale")
    sub.add_parser("evaluate")
    mf = sub.add_parser("metafeatures", help="pymfe meta-features of the seed-0 TRAIN split "
                        "(meta env)")
    mf.add_argument("--all-registered", action="store_true",
                    help="every dataset registered in the store, not just dataset.name")
    mf.add_argument("--max-rows", type=int, default=5000,
                    help="stratified sample cap for pymfe (default 5000)")
    me = sub.add_parser("meta-eval", help="RQ3 LODO evaluation and predictor training "
                        "(meta env)")
    me.add_argument("--datasets-yaml", default=str(CONF_DIR / "meta" / "datasets.yaml"))
    me.add_argument("--run-prefix", default="meta-",
                    help="run id of dataset cc18_<id> is <prefix>cc18_<id> (default meta-)")
    me.add_argument("--train", action="store_true",
                    help="also train and save the final predictor artifact")
    exp = sub.add_parser("export-results", help="bundle a run's results into a zip")
    exp.add_argument("--run", default=argparse.SUPPRESS, help="run id (also accepted globally)")
    exp.add_argument("--out", required=True)
    exp.add_argument("--include-synthetic", action="store_true")
    imp = sub.add_parser("import-results", help="import a results bundle zip")
    imp.add_argument("path")
    imp.add_argument("--force", action="store_true",
                     help="import even if the dataset fingerprint differs")
    for name in ("report", "all"):
        p = sub.add_parser(name)
        p.add_argument("--preset", default="balanced")
        p.add_argument("--weights", help="custom fidelity,utility,privacy weights")
        p.add_argument("--json", action="store_true")
    sub.choices["all"].add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.override)
    commands = {"prepare": cmd_prepare, "generate": cmd_generate, "metrics": cmd_metrics,
                "evaluate": cmd_evaluate, "metafeatures": cmd_metafeatures,
                "meta-eval": cmd_meta_eval, "report": cmd_report, "all": cmd_all,
                "export-results": cmd_export_results, "import-results": cmd_import_results}
    return commands[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
