/**
 * MSW handlers that mirror the real FastAPI service (services/api/src/stl_api/routes/*):
 * same shapes, status codes and detail strings, so code that works here works against the API.
 */
import { http, HttpResponse } from "msw";
import type { ColumnType, CreateRunBody, Dataset, DatasetSuggestion, PerJobReport, Run, RunJob, RunReport, RunSummary } from "@/lib/types";
import { PRESETS } from "@/lib/presets";
import {
  BUILTINS,
  DONE_CANDIDATES,
  GENERATORS,
  JOB_STATES,
  OOM_TRACEBACK,
  aggregated,
  builtinDataset,
  candidatesFor,
  initialDatasets,
  recommendation,
  type MockDataset,
} from "./fixtures";

type Axes = [number, number, number];

interface MockRun {
  run_id: string;
  dataset_id: string;
  generators: string[];
  seeds: number[];
  preset: string;
  /** Normalised weights stored with the run; null for imported runs. */
  weights: Axes | null;
  /** Number of polls so far; planned job i is at state index clamp(polls - i). Ignored when `finished`. */
  polls: number;
  finished: boolean;
  /** Job ids that have failed (cleared by POST /runs/{id}/retry). */
  failing?: string[];
  /** Job ids re-queued by a retry: they progress again from `queued` while the rest stay finished. */
  retried?: string[];
  /** The artifact store was unreachable on the last refresh. */
  stale?: boolean;
  /** Polls no longer advance the run (tests pin a live run at one point of its timeline). */
  frozen?: boolean;
}

interface MockState {
  datasets: MockDataset[];
  runs: MockRun[];
  counter: number;
}

const presetWeights = (p: keyof typeof PRESETS): Axes => [...PRESETS[p].weights] as Axes;

function initialState(): MockState {
  return {
    datasets: initialDatasets(),
    runs: [
      {
        run_id: "adult-done",
        dataset_id: "adult",
        generators: DONE_CANDIDATES.map((c) => c.generator),
        seeds: [0, 1],
        preset: "balanced",
        weights: presetWeights("balanced"),
        polls: 0,
        finished: true,
      },
      {
        run_id: "adult-live",
        dataset_id: "adult",
        generators: ["gaussian_copula", "ctgan", "tvae"],
        seeds: [0],
        preset: "balanced",
        weights: presetWeights("balanced"),
        polls: 0,
        finished: false,
      },
    ],
    counter: 0,
  };
}

let state = initialState();

export function resetMockState(): void {
  state = initialState();
}

/** Adds a finished run whose `tvae__s0` job failed (for failure/retry flows). Reset by `resetMockState`. */
export function seedFailedRun(run_id = "adult-partial"): void {
  state.runs.push({
    run_id,
    dataset_id: "adult",
    generators: ["gaussian_copula", "tvae"],
    seeds: [0],
    preset: "balanced",
    weights: presetWeights("balanced"),
    polls: 0,
    finished: true,
    failing: ["tvae__s0"],
  });
}

/** Pins a live run at `polls` polls into its timeline; later polls do not advance it. */
export function freezeRun(run_id: string, polls: number): void {
  const run = state.runs.find((r) => r.run_id === run_id);
  if (!run) throw new Error(`no mock run ${run_id}`);
  run.polls = polls;
  run.frozen = true;
}

export const IMPORTED_RUN_ID = "diabetes-20261009101500-b7c1";

/** What POST /import-results does to the store: the run, plus a placeholder dataset if it is unknown. */
export function seedImport(): string {
  const dataset_id = "diabetes";
  if (!findDataset(dataset_id)) {
    state.datasets.push({
      id: dataset_id,
      name: dataset_id,
      source: "import",
      target: "",
      n_rows: null,
      metadata: {},
      registered: false,
      fingerprint: "sha256:bundle5e0d2f",
      hasUpload: false,
    });
  }
  if (!state.runs.some((r) => r.run_id === IMPORTED_RUN_ID)) {
    state.runs.unshift({
      run_id: IMPORTED_RUN_ID,
      dataset_id,
      generators: ["gaussian_copula", "ctgan"],
      seeds: [0, 1],
      preset: "balanced",
      weights: null,
      polls: 0,
      finished: true,
    });
  }
  return IMPORTED_RUN_ID;
}

const jobId = (generator: string, seed: number) => `${generator}__s${seed}`;
const clampState = (i: number) => JOB_STATES[Math.min(Math.max(i, 0), JOB_STATES.length - 1)];

function toRun(r: MockRun): Run {
  const planned = r.generators.flatMap((g) => r.seeds.map((s) => [g, s] as const));
  const jobs: RunJob[] = planned.map(([generator, seed], i) => {
    const job_id = jobId(generator, seed);
    if (r.failing?.includes(job_id)) return { job_id, generator, seed, state: "failed", error: OOM_TRACEBACK };
    const retriedAt = r.retried?.indexOf(job_id) ?? -1;
    const st = retriedAt >= 0 ? clampState(r.polls - retriedAt) : r.finished ? "done" : clampState(r.polls - i);
    return { job_id, generator, seed, state: st, error: null };
  });
  jobs.sort((a, b) => (a.job_id < b.job_id ? -1 : 1));
  const counts: Run["counts"] = {};
  for (const j of jobs) counts[j.state] = (counts[j.state] ?? 0) + 1;
  return {
    run_id: r.run_id,
    dataset_id: r.dataset_id,
    generators: r.generators,
    seeds: r.seeds,
    preset: r.preset,
    counts,
    done: jobs.length > 0 && jobs.every((j) => j.state === "done" || j.state === "failed"),
    stale: r.stale ?? false,
    jobs,
    weights: r.weights,
  };
}

/** GET /runs list item: the detail without per-job rows or weights. */
function toSummary(r: MockRun): RunSummary {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { jobs, weights, ...summary } = toRun(r);
  return summary;
}

const err = (status: number, detail: string) => HttpResponse.json({ detail }, { status });
/** A pydantic validation failure as FastAPI renders it (422 with a list of errors). */
const validation = (field: string, msg: string) =>
  HttpResponse.json(
    { detail: [{ type: "value_error", loc: ["body", field], msg: `Value error, ${msg}`, input: null }] },
    { status: 422 },
  );
const pyList = (xs: (string | number)[]) => `[${xs.map((x) => (typeof x === "string" ? `'${x}'` : x)).join(", ")}]`;
const pyTuple = (xs: number[]) => `(${xs.map((x) => (Number.isInteger(x) ? x.toFixed(1) : String(x))).join(", ")})`;

/** decision.resolve_weights: custom weights are normalised; returns [label, weights] or an error detail. */
function resolveWeights(preset: string, custom: number[] | null): [string, Axes] | string {
  if (custom) {
    if (custom.length !== 3 || !custom.every((w) => Number.isFinite(w) && w >= 0) || !(custom.reduce((a, b) => a + b, 0) > 0)) {
      return (
        "custom weights must be exactly 3 finite non-negative numbers " +
        `(fidelity, utility, privacy) with a positive sum; got ${pyTuple(custom)}`
      );
    }
    const total = custom.reduce((a, b) => a + b, 0);
    return ["custom", custom.map((w) => w / total) as Axes];
  }
  if (!(preset in PRESETS)) return `unknown preset '${preset}'; choose from ${pyList(Object.keys(PRESETS).sort())}`;
  return [preset, presetWeights(preset as keyof typeof PRESETS)];
}

/** DatasetOut: every dataset endpoint returns this shape. */
function out(d: MockDataset): Dataset {
  return {
    id: d.id,
    name: d.name,
    source: d.source,
    target: d.target,
    n_rows: d.n_rows,
    metadata: d.metadata,
    registered: d.registered,
    fingerprint: d.fingerprint ?? null,
  };
}

const findDataset = (id: unknown) => state.datasets.find((x) => x.id === String(id));
const runsUsing = (id: string) => state.runs.filter((r) => r.dataset_id === id).length;

/** datasets._guard_runs: an import placeholder may always be (re)registered, which promotes it. */
function guard(id: string) {
  const d = findDataset(id);
  if (d?.source === "import") return null;
  const n = runsUsing(id);
  return n > 0 ? err(409, `dataset ${id} is used by ${n} run(s); create a new dataset instead`) : null;
}

/** datasets._promote: warn when the registered data differs from what the imported runs used. */
function promote(d: MockDataset, importedFp: string | null | undefined, source: string): string[] {
  d.source = source;
  return importedFp && importedFp !== d.fingerprint
    ? [`fingerprint differs from imported run(s) of dataset ${d.id}; results not comparable`]
    : [];
}

const slug = (name: string) =>
  (name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40).replace(/-+$/, "") || "dataset") +
  "-a1b2c3";

export const handlers = [
  http.get("*/health", () => HttpResponse.json({ status: "ok", db: true, store: true })),

  http.get("*/datasets", () => HttpResponse.json(state.datasets.map(out))),

  http.get("*/datasets/:id/suggestion", ({ params }) => {
    const d = findDataset(params.id);
    if (!d) return err(404, `dataset '${String(params.id)}' not found`);
    const body: DatasetSuggestion =
      d.id === "adult"
        ? {
            available: true,
            generator: "tvae",
            probabilities: { tvae: 0.62, gaussian_copula: 0.24, mst_eps1: 0.14 },
            model: { n_datasets: 25, trained_at: "2026-10-10T00:00:00+00:00" },
          }
        : { available: false, reason: "suggestion has not been computed yet" };
    return HttpResponse.json(body);
  }),

  http.post("*/datasets/builtin/:name", ({ params }) => {
    const name = String(params.name);
    if (!(BUILTINS as readonly string[]).includes(name)) {
      return err(404, `unknown built-in dataset '${name}'; known: ${pyList([...BUILTINS])}`);
    }
    const blocked = guard(name);
    if (blocked) return blocked;
    let d = findDataset(name);
    if (!d) {
      d = builtinDataset(name);
      state.datasets.push(d);
    }
    const placeholder = d.source === "import";
    const importedFp = placeholder ? d.fingerprint : null;
    Object.assign(d, {
      ...builtinDataset(name),
      name: d.name,
      source: d.source,
    });
    const body: Dataset = out(d);
    if (placeholder) {
      const warnings = promote(d, importedFp, "builtin");
      Object.assign(body, out(d));
      if (warnings.length) body.warnings = warnings;
    }
    return HttpResponse.json(body, { status: 201 });
  }),

  http.get("*/datasets/:id", ({ params }) => {
    const d = findDataset(params.id);
    if (!d) return err(404, `dataset '${String(params.id)}' not found`);
    return HttpResponse.json({ ...out(d), created_at: d.created_at ?? "2026-10-10T00:00:00", preview: d.hasUpload ? (d.preview ?? []) : [] });
  }),

  http.post("*/datasets", async ({ request }) => {
    const form = await request.formData();
    const file = form.get("file");
    const name = String(form.get("name") ?? "");
    const target = String(form.get("target") ?? "");
    if (!(file instanceof Blob)) return validation("file", "Field required");
    if (!name.trim()) return err(422, "name must not be empty");
    const fileName = (file as File).name ?? "";
    const suffix = /\.[^.]*$/.exec(fileName)?.[0].toLowerCase() ?? "";
    if (suffix !== ".csv" && suffix !== ".parquet") {
      return err(422, `unsupported file type '${suffix}'; upload CSV or Parquet`);
    }
    let header: string[] = [target];
    let nRows = 2;
    if (suffix === ".csv") {
      const lines = (await file.text()).split(/\r?\n/).filter((l) => l.trim() !== "");
      header = (lines[0] ?? "").split(",");
      nRows = lines.length - 1;
      if (nRows <= 0) return err(422, "the file has no rows");
    }
    if (!header.includes(target)) {
      return err(422, `target column '${target}' not found; columns are ${pyList(header)}`);
    }
    const d: MockDataset = {
      id: slug(name),
      name: name.trim(),
      source: "upload",
      target,
      n_rows: nRows,
      metadata: {
        columns: Object.fromEntries(header.map((c) => [c, c === target ? ("categorical" as const) : ("numerical" as const)])),
        target,
        task: "classification",
      },
      registered: false,
      fingerprint: null,
      hasUpload: true,
    };
    state.datasets.push(d);
    return HttpResponse.json(out(d), { status: 201 });
  }),

  http.put("*/datasets/:id/metadata", async ({ params, request }) => {
    const d = findDataset(params.id);
    if (!d) return err(404, `dataset '${String(params.id)}' not found`);
    const blocked = guard(d.id);
    if (blocked) return blocked;
    if (!d.hasUpload) return err(409, "dataset has no stored upload");
    const body = (await request.json()) as { columns: Record<string, ColumnType>; target: string };
    if (!(body.target in body.columns)) return err(422, `target column '${body.target}' is not in columns`);
    d.metadata = {
      columns: body.columns,
      target: body.target,
      task: body.columns[body.target] === "categorical" ? "classification" : "regression",
    };
    d.target = body.target;
    d.registered = false;
    d.fingerprint = null;
    return HttpResponse.json(out(d));
  }),

  http.post("*/datasets/:id/register", async ({ params, request }) => {
    const d = findDataset(params.id);
    if (!d) return err(404, `dataset '${String(params.id)}' not found`);
    const blocked = guard(d.id);
    if (blocked) return blocked;
    if (!d.hasUpload) return err(409, "dataset has no stored upload");
    const body = ((await request.json().catch(() => ({}))) ?? {}) as { max_rows?: number };
    if (body.max_rows !== undefined && !(body.max_rows >= 1)) {
      return HttpResponse.json(
        { detail: [{ type: "greater_than_equal", loc: ["body", "max_rows"], msg: "Input should be greater than or equal to 1" }] },
        { status: 422 },
      );
    }
    const placeholder = d.source === "import";
    const importedFp = placeholder ? d.fingerprint : null;
    d.registered = true;
    d.fingerprint = `sha256:mock${d.id}`;
    const res: Dataset = out(d);
    if (placeholder) {
      const warnings = promote(d, importedFp, "upload");
      Object.assign(res, out(d));
      if (warnings.length) res.warnings = warnings;
    }
    return HttpResponse.json(res);
  }),

  http.get("*/generators", () => HttpResponse.json(GENERATORS)),

  http.post("*/runs", async ({ request }) => {
    const body = (await request.json()) as CreateRunBody;
    const known = new Set(GENERATORS.map((g) => g.name));
    if (body.generators.length === 0) return validation("generators", "generators must not be empty");
    if (new Set(body.generators).size !== body.generators.length) return validation("generators", "generators must be unique");
    const unknown = body.generators.filter((g) => !known.has(g));
    if (unknown.length) {
      return validation("generators", `unknown generators ${pyList(unknown)}; known: ${pyList([...known].sort())}`);
    }
    const seeds = body.seeds ?? [0, 1, 2];
    if (seeds.some((x) => !Number.isInteger(x) || x < 0)) {
      return HttpResponse.json(
        { detail: [{ type: "greater_than_equal", loc: ["body", "seeds", 0], msg: "Input should be greater than or equal to 0" }] },
        { status: 422 },
      );
    }
    if (seeds.length === 0) return validation("seeds", "seeds must not be empty");
    if (new Set(seeds).size !== seeds.length) return validation("seeds", "seeds must be unique");
    const d = findDataset(body.dataset_id);
    if (!d || !d.registered) return err(409, `dataset '${body.dataset_id}' is not registered`);
    const resolved = resolveWeights(body.preset ?? "balanced", body.weights ?? null);
    if (typeof resolved === "string") return err(422, resolved);
    state.counter += 1;
    const run: MockRun = {
      run_id: `${d.id}-20261010120000-${(0xa000 + state.counter).toString(16)}`,
      dataset_id: d.id,
      generators: body.generators,
      seeds,
      preset: resolved[0],
      weights: resolved[1],
      polls: 0,
      finished: false,
    };
    state.runs.unshift(run);
    const jobs = body.generators.flatMap((g) =>
      seeds.map((s) => ({ job_id: jobId(g, s), generator: g, seed: s, state: "queued" as const, error: null })),
    );
    return HttpResponse.json({ run_id: run.run_id, jobs }, { status: 201 });
  }),

  http.get("*/runs", () => HttpResponse.json(state.runs.map(toSummary))),

  http.get("*/runs/:id/report", ({ params, request }) => {
    const run = state.runs.find((r) => r.run_id === params.id);
    if (!run) return err(404, `run '${String(params.id)}' not found`);
    const url = new URL(request.url);
    const rawWeights = url.searchParams.get("weights");
    let custom: number[] | null = null;
    if (rawWeights !== null) {
      custom = rawWeights.split(",").map(Number);
      if (custom.some((x) => Number.isNaN(x))) {
        return err(422, `weights must be 'fidelity,utility,privacy' numbers: '${rawWeights}'`);
      }
    }
    const resolved = resolveWeights(url.searchParams.get("preset") ?? "balanced", custom);
    if (typeof resolved === "string") return err(422, resolved);
    const [label, weights] = resolved;

    const view = toRun(run);
    const reported = view.jobs.filter((j) => j.state === "done");
    const per_job: PerJobReport[] = view.jobs.map((j) => ({
      job_id: j.job_id,
      generator: j.generator,
      seed: j.seed,
      status: j.state,
      warnings:
        j.state === "done" && j.generator === "train_copy"
          ? ["detection_auc 0.312 < 0.4: synthetic rows likely duplicate real rows (memorisation signal)"]
          : [],
    }));
    const report: RunReport = {
      run_id: run.run_id,
      jobs_reported: reported.length,
      jobs_total: view.jobs.length,
      recommendation: null,
      aggregated: [],
      pareto: [],
      per_job,
    };
    if (reported.length > 0) {
      const seedsDone = new Map<string, number>();
      for (const j of reported) seedsDone.set(j.generator, (seedsDone.get(j.generator) ?? 0) + 1);
      const cands = candidatesFor(run.generators.filter((g) => seedsDone.has(g)));
      const rec = recommendation(label, weights, cands);
      report.recommendation = rec;
      report.aggregated = aggregated(cands, seedsDone);
      report.pareto = rec.ranking.map((c) => ({
        generator: c.generator,
        fidelity: c.scores.fidelity,
        utility: c.scores.utility,
        privacy: c.scores.privacy,
        on_front: c.on_pareto_front,
        gate_passed: c.gate_passed,
      }));
    }
    return HttpResponse.json(report);
  }),

  http.get("*/runs/:id", ({ params }) => {
    const run = state.runs.find((r) => r.run_id === params.id);
    if (!run) return err(404, `run '${String(params.id)}' not found`);
    if (!run.frozen && (!run.finished || run.retried?.length)) run.polls += 1;
    return HttpResponse.json(toRun(run));
  }),

  http.post("*/runs/:id/retry", ({ params }) => {
    const run = state.runs.find((r) => r.run_id === params.id);
    if (!run) return err(404, `run '${String(params.id)}' not found`);
    const retried = run.failing ?? [];
    if (retried.length) {
      run.failing = [];
      run.retried = retried;
      run.polls = 0;
    }
    return HttpResponse.json({ run_id: run.run_id, retried });
  }),

  /** POST /import-results: the bundle's run lands in the run list; an unknown dataset gets a placeholder row. */
  http.post("*/import-results", async ({ request }) => {
    const form = await request.formData();
    if (!(form.get("file") instanceof Blob)) return validation("file", "Field required");
    const run_id = seedImport();
    return HttpResponse.json({ run_id, imported: 4, warnings: [] });
  }),
];
