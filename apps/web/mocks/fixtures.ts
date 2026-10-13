import type {
  AggregatedReport,
  Dataset,
  SuggestionAvailable,
  SuggestionUnavailable,
  GeneratorInfo,
  GeneratorParams,
  JobState,
  RankedCandidate,
  Recommendation,
  TableMetadata,
} from "@/lib/types";

/** Mirrors packages/core/src/stl_core/catalog.py (plus the API's `audit` flag). */
export const GENERATORS: GeneratorInfo[] = [
  { name: "train_copy", family: "sanity_ceiling", queue: "q.sdv", cpu_ok: true, params: {}, audit: false },
  { name: "independent_marginals", family: "sanity_floor", queue: "q.sdv", cpu_ok: true, params: {}, audit: false },
  { name: "marginals", family: "sanity_floor", queue: "q.synthcity", cpu_ok: true, params: {}, audit: false },
  { name: "gaussian_copula", family: "statistical", queue: "q.sdv", cpu_ok: true, params: {}, audit: false },
  { name: "bayesian_network", family: "graphical", queue: "q.synthcity", cpu_ok: true, params: {}, audit: false },
  { name: "ctgan", family: "deep", queue: "q.synthcity", cpu_ok: true, params: {}, audit: false },
  { name: "tvae", family: "deep", queue: "q.synthcity", cpu_ok: true, params: {}, audit: false },
  { name: "tabddpm", family: "diffusion", queue: "q.synthcity", cpu_ok: false, params: {}, audit: false },
  { name: "great", family: "llm", queue: "q.synthcity", cpu_ok: false, params: {}, audit: false },
  { name: "mst_eps0.1", family: "differential_privacy", queue: "q.smartnoise", cpu_ok: true, params: { epsilon: 0.1 }, audit: false },
  { name: "mst_eps1", family: "differential_privacy", queue: "q.smartnoise", cpu_ok: true, params: { epsilon: 1.0 }, audit: false },
  { name: "mst_eps10", family: "differential_privacy", queue: "q.smartnoise", cpu_ok: true, params: { epsilon: 10.0 }, audit: false },
  { name: "privbayes_eps0.1", family: "nominal_dp_audit", queue: "q.synthcity", cpu_ok: true, params: { epsilon: 0.1 }, audit: true },
  { name: "privbayes_eps1", family: "nominal_dp_audit", queue: "q.synthcity", cpu_ok: true, params: { epsilon: 1.0 }, audit: true },
  { name: "privbayes_eps10", family: "nominal_dp_audit", queue: "q.synthcity", cpu_ok: true, params: { epsilon: 10.0 }, audit: true },
];

/** GET /datasets/{id}/suggestion bodies (mirror services/api/tests/fixtures/suggestion.json). */
export const SUGGESTION_AVAILABLE: SuggestionAvailable = {
  available: true,
  generator: "tvae",
  probabilities: { tvae: 0.62, gaussian_copula: 0.24, mst_eps1: 0.14 },
  model: { n_datasets: 25, n_none: 1, trained_at: "2026-10-10T00:00:00+00:00" },
};

export const SUGGESTION_NO_PREDICTOR: SuggestionUnavailable = {
  available: false,
  reason: "no meta-predictor has been trained yet (RQ3 meta-benchmark)",
};

export const SUGGESTION_NOT_REGISTERED: SuggestionUnavailable = {
  available: false,
  reason: "dataset is not registered",
};

export const BUILTINS = ["adult", "credit", "diabetes"] as const;

const ADULT_META: TableMetadata = {
  columns: {
    age: "numerical",
    workclass: "categorical",
    education: "categorical",
    "hours-per-week": "numerical",
    sex: "categorical",
    income: "categorical",
  },
  target: "income",
  task: "classification",
};

export const ADULT_PREVIEW = [
  { age: 39, workclass: "State-gov", education: "Bachelors", "hours-per-week": 40, sex: "Male", income: "<=50K" },
  { age: 50, workclass: "Self-emp", education: "Bachelors", "hours-per-week": 13, sex: "Male", income: "<=50K" },
  { age: 38, workclass: "Private", education: "HS-grad", "hours-per-week": 40, sex: "Male", income: "<=50K" },
  { age: 53, workclass: "Private", education: "11th", "hours-per-week": 40, sex: "Male", income: "<=50K" },
  { age: 28, workclass: "Private", education: "Bachelors", "hours-per-week": 40, sex: "Female", income: ">50K" },
];

/** A mock dataset row: the API shape plus whether an upload is stored (import placeholders have none). */
export type MockDataset = Dataset & { hasUpload: boolean };

export function initialDatasets(): MockDataset[] {
  return [
    {
      id: "adult",
      name: "Adult",
      source: "builtin",
      target: "income",
      n_rows: 20000,
      metadata: structuredClone(ADULT_META),
      registered: true,
      fingerprint: "sha256:3f9a1c0de2",
      created_at: "2026-10-09T08:00:00",
      preview: ADULT_PREVIEW,
      hasUpload: true,
    },
    {
      id: "uploaded-1",
      name: "Uploaded sample",
      source: "upload",
      target: "label",
      n_rows: 1200,
      metadata: {
        columns: { id: "id", x1: "numerical", x2: "numerical", colour: "categorical", label: "categorical" },
        target: "label",
        task: "classification",
      },
      registered: false,
      fingerprint: null,
      created_at: "2026-10-09T09:00:00",
      preview: [
        { id: 1, x1: 0.3, x2: 1.2, colour: "red", label: "a" },
        { id: 2, x1: 0.9, x2: -0.4, colour: "blue", label: "b" },
      ],
      hasUpload: true,
    },
  ];
}

export function builtinDataset(name: string): MockDataset {
  return {
    ...initialDatasets()[0],
    id: name,
    name,
    fingerprint: `sha256:builtin${name}0001`,
  };
}

export const JOB_STATES: JobState[] = ["queued", "fitting", "sampling", "generated", "evaluating", "done"];

type Axes = [number, number, number];

export interface FixtureCandidate {
  generator: string;
  /** Axis scores [fidelity, utility, privacy]. */
  axes: Axes;
  /** check_gate() output; empty means the gate passed. */
  gate_failures: string[];
  /** Singling-out was not attackable in every seed (and the safeguards passed). */
  notAttackable?: boolean;
}

/** decision.NOT_ATTACKABLE_NOTE */
export const NOT_ATTACKABLE_NOTE = "singling-out not attackable (safeguards passed)";

/** Finished Adult run: train_copy fails the privacy gate, tvae wins (balanced), MST "not attackable". */
export const DONE_CANDIDATES: FixtureCandidate[] = [
  {
    generator: "train_copy",
    axes: [0.99, 0.98, 0.05],
    gate_failures: [
      "failed the singling-out check (lower bound 0.98 > 0.10)",
      "failed the DCR check (share 1.00 > 0.60)",
    ],
  },
  { generator: "independent_marginals", axes: [0.55, 0.6, 0.9], gate_failures: [] },
  { generator: "gaussian_copula", axes: [0.7, 0.72, 0.85], gate_failures: [] },
  { generator: "tvae", axes: [0.82, 0.84, 0.78], gate_failures: [] },
  { generator: "mst_eps1", axes: [0.66, 0.62, 0.95], gate_failures: [], notAttackable: true },
];

/** Candidates for a run's own generators: known fixtures reused, others synthesised deterministically. */
export function candidatesFor(generators: string[]): FixtureCandidate[] {
  return generators.map((g) => {
    const known = DONE_CANDIDATES.find((c) => c.generator === g);
    if (known) return known;
    const h = [...g].reduce((n, ch) => (n * 31 + ch.charCodeAt(0)) % 997, 7) / 997;
    const axes: Axes = [0.5 + 0.35 * h, 0.5 + 0.3 * (1 - h), 0.6 + 0.3 * ((h * 7) % 1)];
    return { generator: g, axes, gate_failures: [] };
  });
}

const AXES = ["fidelity", "utility", "privacy"] as const;

/** decision.pareto_front over gate-passing candidates with a score. */
function paretoFront(points: Map<string, Axes>): Set<string> {
  const front = new Set<string>();
  for (const [a, pa] of points) {
    const dominated = [...points].some(
      ([b, pb]) => b !== a && pb.every((v, i) => v >= pa[i]) && pb.some((v, i) => v > pa[i]),
    );
    if (!dominated) front.add(a);
  }
  return front;
}

/** decision.recommend's ranking: every candidate scored (gate or not), sorted gate-passing first, then by score. */
export function rank(weights: Axes, candidates: FixtureCandidate[] = DONE_CANDIDATES): RankedCandidate[] {
  const total = weights.reduce((a, b) => a + b, 0);
  const passed = (c: FixtureCandidate) => c.gate_failures.length === 0;
  const front = paretoFront(new Map(candidates.filter(passed).map((c) => [c.generator, c.axes])));
  const ranked = candidates.map<RankedCandidate>((c) => ({
    generator: c.generator,
    scores: { fidelity: c.axes[0], utility: c.axes[1], privacy: c.axes[2] },
    score: (c.axes[0] * weights[0] + c.axes[1] * weights[1] + c.axes[2] * weights[2]) / total,
    gate_passed: passed(c),
    gate_failures: c.gate_failures,
    on_pareto_front: front.has(c.generator),
    notes: c.notAttackable && passed(c) ? [NOT_ATTACKABLE_NOTE] : [],
  }));
  return ranked.sort((a, b) => {
    if (a.gate_passed !== b.gate_passed) return a.gate_passed ? -1 : 1;
    if ((a.score === null) !== (b.score === null)) return a.score === null ? 1 : -1;
    if (a.score !== b.score) return (b.score ?? 0) - (a.score ?? 0);
    return a.generator < b.generator ? -1 : 1;
  });
}

/** decision._reason */
function reason(ranking: RankedCandidate[], label: string): string {
  const passing = ranking.filter((c) => c.gate_passed && c.score !== null);
  const winner = passing[0];
  if (!winner) {
    const parts = ranking.map((c) => `${c.generator} ${c.gate_failures[0] ?? "has incomplete scores"}`);
    const prefix = ranking.some((c) => c.gate_passed)
      ? "No generator passed the privacy gate with complete scores. "
      : "No generator passed the privacy gate. ";
    return `${prefix}${parts.join("; ")}.`;
  }
  const s = winner.scores as Record<(typeof AXES)[number], number>;
  const best = AXES.reduce((a, b) => (s[b] > s[a] ? b : a));
  let text =
    `${winner.generator}: highest ${label} score (${winner.score!.toFixed(2)}) among models passing ` +
    `the privacy gate; strongest on ${best} (${s[best].toFixed(2)}).`;
  for (const note of winner.notes) text += ` Note: ${note}.`;
  if (passing.length > 1) text += ` Runner-up: ${passing[1].generator} (${passing[1].score!.toFixed(2)}).`;
  for (const c of ranking) {
    if (c.gate_passed) continue;
    const better = (["fidelity", "utility"] as const).find((a) => (c.scores[a] ?? -1) > s[a]);
    if (better) text += ` ${c.generator} had higher ${better} but ${c.gate_failures[0]}.`;
  }
  return text;
}

export function recommendation(label: string, weights: Axes, candidates: FixtureCandidate[]): Recommendation {
  const ranking = rank(weights, candidates);
  const winner = ranking.find((c) => c.gate_passed && c.score !== null);
  return { preset: label, weights, winner: winner?.generator ?? null, reason: reason(ranking, label), ranking };
}

/** Flat per-generator mean metrics with the real MetricReport.flat() keys. */
function meanMetrics(c: FixtureCandidate): Record<string, number | null> {
  const [f, u, p] = c.axes;
  const copy = c.generator === "train_copy";
  const r = (x: number) => Math.round(x * 1000) / 1000;
  return {
    "fidelity.quality_score": f,
    "fidelity.column_shapes": r(Math.min(1, f + 0.01)),
    "fidelity.column_pair_trends": r(f - 0.02),
    "fidelity.detection_auc": copy ? 0.312 : r(0.5 + (1 - f) / 2),
    "fidelity.alpha_precision": r(f - 0.03),
    "fidelity.beta_recall": r(f - 0.05),
    "utility.tstr.logreg": r(u),
    "utility.tstr.random_forest": r(u - 0.02),
    "utility.tstr.xgboost": r(u - 0.01),
    "utility.importance_spearman": r(u - 0.04),
    "privacy.dcr_share": copy ? 1 : r(0.5 + (1 - p) * 0.1),
    "privacy.singling_out_risk": c.notAttackable ? null : copy ? 0.99 : r((1 - p) * 0.1),
    "privacy.singling_out_risk_lower": c.notAttackable ? null : copy ? 0.98 : r((1 - p) * 0.05),
    "privacy.linkability_risk": copy ? 0.62 : r((1 - p) * 0.08),
    "privacy.linkability_risk_lower": copy ? 0.55 : r((1 - p) * 0.04),
    "privacy.inference_risk": copy ? 0.71 : r((1 - p) * 0.12),
    "privacy.inference_risk_lower": copy ? 0.66 : r((1 - p) * 0.06),
    "privacy.mia_advantage": copy ? 0.64 : r((1 - p) * 0.2),
    "privacy.exact_match_excess_scaled": copy ? 1 : 0,
    "privacy.qi_unique_share": copy ? 0.31 : r((1 - p) * 0.04),
    "cost.fit_seconds": r(5 + 40 * f),
    "cost.sample_seconds": r(0.5 + 2 * f),
    "cost.peak_memory_mb": r(300 + 500 * f),
  };
}

/** stl_core.schemas.WORST_KEYS present in the fixture metrics. */
const WORST_KEYS = [
  "privacy.singling_out_risk_lower",
  "privacy.linkability_risk_lower",
  "privacy.inference_risk_lower",
  "privacy.singling_out_risk",
  "privacy.linkability_risk",
  "privacy.inference_risk",
  "privacy.dcr_share",
  "privacy.exact_match_excess_scaled",
  "privacy.qi_unique_share",
];

/** aggregate(): one entry per generator with at least one report, sorted by generator. */
export function aggregated(candidates: FixtureCandidate[], seedsDone: Map<string, number>): AggregatedReport[] {
  return [...candidates]
    .sort((a, b) => (a.generator < b.generator ? -1 : 1))
    .map((c) => {
      const n = seedsDone.get(c.generator) ?? 1;
      const mean = meanMetrics(c);
      const std = Object.fromEntries(
        Object.entries(mean).map(([k, v]) => [k, v === null ? null : n > 1 ? (k.startsWith("cost.") ? 1.5 : 0.01) : 0]),
      );
      const worst = Object.fromEntries(
        WORST_KEYS.map((k) => [k, mean[k] === null ? null : Math.round(Math.min(1, (mean[k] as number) + 0.005) * 1000) / 1000]),
      );
      return {
        generator: c.generator,
        n_seeds: n,
        n_failed: 0,
        mean,
        std,
        missing: Object.keys(mean).filter((k) => mean[k] === null),
        worst,
        singling_out_status: Array.from({ length: n }, () => (c.notAttackable ? "not_attackable" : "measured")),
      };
    });
}

/** A realistic worker traceback (multi-line), as stored in the job status by the runner. */
export const OOM_TRACEBACK = [
  "Traceback (most recent call last):",
  '  File "/app/packages/runner/src/stl_runner/pipeline.py", line 155, in generate_job',
  "    generator.fit(splits.train, metadata)",
  '  File "/app/workers/synthcity/src/stl_synthcity/tvae.py", line 48, in fit',
  "    self._model.fit(loader)",
  '  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _call_impl',
  "    return forward_call(*args, **kwargs)",
  "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 7.79 GiB total capacity)",
  "",
].join("\n");

/** Library defaults per generator, as each wrapper's resolved_params() records them. */
const LIBRARY_PARAMS: Record<string, { library: string; library_version: string; params: Record<string, unknown>; epochKey?: string }> = {
  gaussian_copula: {
    library: "sdv",
    library_version: "1.38.5",
    params: { enforce_min_max_values: true, enforce_rounding: true, locales: ["en_US"], numerical_distributions: null, default_distribution: null },
  },
  ctgan: {
    library: "synthcity",
    library_version: "0.2.12",
    params: { n_iter: 2000, batch_size: 200, lr: 0.001, patience: 5, n_iter_min: 100, device: "cpu", random_state: 0 },
    epochKey: "n_iter",
  },
  tvae: {
    library: "synthcity",
    library_version: "0.2.12",
    params: { n_iter: 1000, batch_size: 200, lr: 0.001, patience: 5, n_iter_min: 100, device: "cpu", random_state: 0 },
    epochKey: "n_iter",
  },
  great: {
    library: "be-great",
    library_version: "0.0.14",
    params: { llm: "distilgpt2", epochs: 100, batch_size: 8, save_strategy: "no", sample: { guided_sampling: true, temperature: 0.7 } },
    epochKey: "epochs",
  },
};

const PROFILE_EPOCHS: Record<string, number | null> = { library: null, equal50: 50, smoke: 5 };

/** GET /runs/{id}/report per_job generator_params for a job that has started training. */
export function generatorParamsFor(generator: string, seed: number, profile: string): GeneratorParams {
  const lib = LIBRARY_PARAMS[generator];
  if (!lib) return { library: "stl_core", library_version: null, params: {}, overrides: [], profile };
  const params: Record<string, unknown> = { ...lib.params };
  const overrides: string[] = [];
  if ("random_state" in params) {
    params.random_state = seed;
    overrides.push("random_state");
  }
  const epochs = PROFILE_EPOCHS[profile] ?? null;
  if (lib.epochKey && epochs !== null) {
    params[lib.epochKey] = epochs;
    overrides.push(lib.epochKey);
  }
  if (generator === "great") overrides.push("llm", "sample.guided_sampling", "save_strategy");
  return { library: lib.library, library_version: lib.library_version, params, overrides: overrides.sort(), profile };
}

export const BUDGET_PROFILE_IDS = ["equal50", "library", "smoke"] as const;
