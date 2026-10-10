# S2b — Library-default hyperparameters + pretrained-LLM generators — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Owner direction (2026-10-10): "use the weights/parameters from these 3rd-party repos, through their libraries" — clarified as BOTH (a) every generator trains with its repo's own default hyperparameters (no blanket overrides), recorded per job; (b) wherever a library publishes pretrained model weights, load them through that library's own loader.

**Evidence:** `.superpowers/sdd/weights-research.md` (file:line citations). Key facts: no repo ships pretrained *synthesizers*; pretrained weights = base LLM checkpoints fine-tuned on TRAIN (be-great `distilgpt2` already used; `tabularisai/Qwen3-0.3B-distil` documented by be-great; NVIDIA Safe-Synthesizer `HuggingFaceTB/SmolLM3-3B` QLoRA, GPU-only Linux). Only unjustified override today: shared `budget.epochs=300` (synthcity defaults: ctgan 2000, tvae 1000, tabddpm 1000 — ctgan/tvae early-stop by default; be-great 100; our great wrapper hard-codes 50).

## Global Constraints
- **No git commits/pushes/add/stash/checkout.** Python only via `bash scripts/envrun.sh <env> …`.
- **Methodology change (owner-approved):** the benchmark default becomes `budget.profile=library` — no epoch override; each library's defaults apply. The old equal-budget rule survives as an explicit profile `equal50` (CPU benchmark, 50 epochs) and `smoke` (5 epochs, small rows). Justified overrides stay and are documented as such: PrivBayes `K=2,n_bins=20` (OOM), GReaT `guided_sampling=True`, MST public bounds / preprocessor share, `save_strategy="no"`.
- Every job records `generator_params` = resolved kwargs actually passed to the library **merged with the library signature defaults** (`inspect.signature`), plus `{library, library_version, overrides: [...], profile}` — written to the job's manifest/cost area and copied into `report.json` (schema field, optional for old reports).
- Time cap: `library` profile default `time_cap_seconds=86400` (owner compute); `equal50` keeps 7200; platform: API setting `STL_BUDGET_PROFILE` (default `library`) + per-run override field `budget_profile` in POST /runs (validated against known profiles); the run manifest records the profile.
- Pretrained checkpoints are loaded only through the library loaders (be-great `GReaT(llm=…)`, Safe-Synthesizer config `training.pretrained_model`). Model card licence + size must be verified (network) and recorded in the catalog entry docs before a checkpoint is enabled; GPU-only entries `cpu_ok=False`.
- Families: be-great entries `llm` (not DP); Safe-Synthesizer `llm` and its DP-SGD variant `llm_dp_sgd` (NOT `differential_privacy`, which stays reserved for the vetted MST sweep).

## Task 1: `library` / `equal50` / `smoke` budget profiles + generator_params
Files: `experiments/conf/config.yaml` (+ `experiments/conf/budget/{library,equal50,smoke}.yaml` Hydra group), `experiments/conf/generators/default.yaml`, `workers/synthcity/src/stl_synthcity/{generators,great}.py` (drop None kwargs; great default epochs → be-great default 100 i.e. don't pass when None), `packages/runner/src/stl_runner/pipeline.py` (record generator_params; profile in manifest), `packages/core/src/stl_core/schemas.py` (optional `generator_params` on report), `workers/eval` report writer if that is where report.json is assembled, `scripts/smoke_adult.sh` (+ run_benchmark.sh / run_meta_benchmark.sh: meta-benchmark keeps `equal50`), each generator wrapper exposes `resolved_params()` (library signature defaults ⊕ overrides; library name/version). Tests: None-drop (synthcity receives no n_iter when profile=library), resolved params contain library defaults (e.g. ctgan n_iter=2000 under library, 50 under equal50), report carries generator_params, old reports without it still parse, profile validation.

## Task 2: API + web show the profile and parameters
API: `STL_BUDGET_PROFILE`, `budget_profile` in POST /runs (422 on unknown), echoed in GET /runs/{id}; job report passthrough of `generator_params`. Web: new-run select "Training budget: Library defaults (slow, faithful) / Equal 50 epochs (CPU) / Smoke"; run page per-job "Parameters" disclosure showing generator_params (library + version + overrides highlighted). MSW + tests.

## Task 3: SDV CTGAN/TVAE with SDV's shipped defaults
`sdv_ctgan`, `sdv_tvae` (family `deep`, queue `q.sdv`, cpu_ok True) via `CTGANSynthesizer` / `TVAESynthesizer` defaults (300 epochs, batch 500); seeds; generator_params; tests (tiny data, few epochs override only in tests).

## Task 4: `great_qwen` — tabular-pretrained checkpoint via be-great
Verify `tabularisai/Qwen3-0.3B-distil` on the HF Hub (exists, size, licence) — STOP and report if missing/gated/non-permissive. Catalog `great_qwen` (family `llm`, cpu_ok False) = GReaT wrapper with `llm` set; params per be-great README for that checkpoint (`batch_size=32, epochs=5`; `fp16` only when CUDA). CPU smoke (tiny rows, 1 epoch, few samples) proves loading + sampling; GPU path via notebook.

## Task 5: NVIDIA Safe-Synthesizer (owner GPU compute → import-results)
`workers/safesynth` (Linux/GPU env, editable `third_party/Safe-Synthesizer`): generator wrapper building the library config with its defaults (`HuggingFaceTB/SmolLM3-3B`, QLoRA defaults) for `nss_smollm3` and DP variant `nss_smollm3_dp_eps8` (library DP-SGD defaults, ε=8); writes the standard job artefacts so `metrics`/`evaluate` run unchanged; `cpu_ok=False`, Windows: config-validation tests only (CPU install). Notebook cell in `notebooks/gpu_offline_run.ipynb` to run it on owner GPU and export a bundle for `import-results`. Verify checkpoint licence/size (network).

## Task 6: Spec + docs
Spec §2 (new deviation: library-default profile replaces equal-budget rule; equal50 retained for CPU comparability), §5 step 3, §7 generator table (+ sdv_ctgan, sdv_tvae, great_qwen, nss_*), README "Budget profiles" + runtime table (estimates from research §3), CHANGELOG-style note in the S2b ledger.

## Phase gate
All suites green, ruff clean, web checks green; smoke run (profile smoke) on Adult for the new CPU entries; `library` profile verified to pass no epoch override (unit + one real tiny run inspecting generator_params).
