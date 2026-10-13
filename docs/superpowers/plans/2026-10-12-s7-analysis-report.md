# S7 — Analysis & report artefacts (RQ1, RQ2, cost) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn stored benchmark results into the thesis evidence: a tidy results table, RQ1 rankings with autorank critical-difference (CD) diagrams, RQ2 ε curves and Pareto-front shifts, a cost table, and a reproducible `make report` that regenerates every figure/table from the store (local runs, imported owner-compute bundles, and the RQ3 meta-benchmark runs).

**Spec:** `docs/superpowers/specs/2026-10-05-synth-trust-lab-design.md` §1 (RQ1/RQ2), §6 (metrics/decision), §3 (`notebooks/` = analysis, autorank CD diagrams, ε curves).

## Global Constraints
- **No git commits/pushes/add/stash/checkout.** Python only via `bash scripts/envrun.sh <env> …`.
- New uv project `packages/analysis` (Python 3.11, env `analysis` at `$HOME/.stl/analysis`): deps `stl-core`, `stl-runner[s3]`, pandas, numpy, scipy, matplotlib, `autorank` **editable from `third_party/autorank`** (owner rule: use the pulled repos' libraries). Add `analysis` to `scripts/envrun.sh` guard list.
- **Never recompute metrics** in analysis — read `report.json`/`aggregated` outputs only; the decision engine (`stl_core.aggregate`, `stl_core.decision.recommend`) is the single source of scores/gates.
- Honesty: every figure/table records its provenance (run ids, n datasets, n seeds, budget profile, generator_params/library versions if present, metrics_version). Results from smoke/local runs are labelled **"smoke — not for the thesis"** automatically (budget profile `smoke`, or epochs < profile default, or < 3 seeds).
- Gate-failing candidates are shown but never ranked as winners; train_copy appears only as a sanity row; PrivBayes (`nominal_dp_audit`) is plotted separately from ε-DP MST.
- Outputs land in `reports/` (gitignored except `reports/README.md`): `reports/figures/*.{png,pdf}`, `reports/tables/*.{csv,tex,md}`, `reports/results.md` (auto-generated summary with every number cited from tables).

## Task 1: `packages/analysis` env + `collect`
`stl_analysis.collect.collect(stores: list[str], run_ids: list[str] | None, include_meta: bool) -> pd.DataFrame` — one row per (store, run, dataset, generator, family, seed) with flattened `report.json` metrics (fidelity.*, utility.*, privacy.*, cost.*), gate result + reasons, balanced score (via core), budget profile, generator_params summary, metrics_version, job state (failed/cancelled rows kept with state + error last line). CLI `experiments/run.py collect` is NOT used (separate env) — add `python -m stl_analysis collect --store outputs/store --store outputs/store_integrated -o reports/tables/results.parquet` (+ CSV). Tests: fake store with done/failed/cancelled jobs, missing optional metrics, old reports without generator_params, smoke labelling rule.

## Task 2: RQ1 — rankings + CD diagrams + dataset-characteristic analysis
`stl_analysis.rq1`: per-dataset aggregated balanced score (mean over seeds via core aggregate, gate per core), rank table (datasets × generators, gate-failing marked), **autorank** (Friedman + Nemenyi or Bayesian per autorank's own decision, alpha 0.05) over datasets → `plot_stats` CD diagram (figure) + autorank `latex_table`/report text saved; requires ≥ 3 datasets with ≥ 2 common generators — otherwise writes an "insufficient datasets" note (no crash). Datasets = Adult/credit/diabetes benchmark runs **plus** the CC18 meta-benchmark runs (`meta-cc18_*`, equal50) as a second, larger RQ1 panel (separate CD diagram; budget profiles never mixed in one diagram). Dataset characteristics (n rows, n features, max cardinality, class imbalance, mean |corr|) from stored metadata/metafeatures.json → table + "winner vs characteristic" plots (strip/box by winning generator). Tests: synthetic results where one generator dominates → autorank finds it; insufficient-data path; profiles not mixed.

## Task 3: RQ2 — ε curves + Pareto-front shift
`stl_analysis.rq2`: for each dataset, MST ε ∈ {0.1, 1, 10} curves of fidelity (quality_score), utility (TSTR mean), privacy risks (singling-out/linkability/inference lower bounds, mia_advantage, dcr_share) with seed ranges (min–max or 95% t-CI when ≥ 3 seeds); horizontal reference lines for the best non-DP gate-passing generator; Pareto front (utility-fidelity score vs privacy score) with and without DP candidates, highlighting how the front moves; privacy.json ε-split/path annotated (public vs private bounds — the MST fallback). PrivBayes audit panel separate, titled "nominal DP (not ε-DP)". Tests on synthetic results (monotone curve plotted, missing ε handled, front computed via core).

## Task 4: Cost + report assembly
`stl_analysis.cost`: fit/sample seconds, peak RSS per generator (median, IQR) per dataset + time-cap/failed counts → table + bar chart (log scale). `stl_analysis.report`: `python -m stl_analysis report --store … -o reports/` runs collect → rq1 → rq2 → cost, writes all artefacts + `reports/results.md` (sections RQ1/RQ2/cost/RQ3-pointer, provenance header, smoke warnings, every number from the tables). `scripts/make_report.sh` wrapper. Notebooks `notebooks/rq1_benchmark.ipynb`, `notebooks/rq2_dp.ipynb` (thin: call the module, display figures; committed without outputs; executed headless in verification). Tests: end-to-end on a fake store produces every expected file.

## Task 5: Run on real local data + docs
Run `make_report.sh` on `outputs/store` + `outputs/store_integrated` (current runs: adult-integrated 20k, smoke runs, meta-cc18_3/11) and paste `reports/results.md`; verify smoke labelling. README "Analysis & report" section (how to import owner-compute bundles then regenerate), spec §3/§13 updates, `reports/README.md`.

## Phase gate
All suites green (incl. `packages/analysis`), ruff clean; `make_report.sh` regenerates every artefact from the store; CD diagram + ε curves produced on the CC18 smoke panel / adult runs (or explicit insufficient-data notes); results.md numbers trace to tables.
