/** API types mirroring the Python schemas (stl_core.schemas / stl_api routes). snake_case kept. */

export type ColumnType = "numerical" | "categorical" | "datetime" | "id";
export type Task = "classification" | "regression" | string;

export interface TableMetadata {
  columns: Record<string, ColumnType>;
  target: string;
  task: Task;
}

export interface Dataset {
  id: string;
  name: string;
  /** "import" marks a placeholder created by a results-bundle import (no stored rows). */
  source: "upload" | "builtin" | "import" | string;
  /** Empty string for an import placeholder. */
  target: string;
  n_rows: number | null;
  /** `{}` for an import placeholder: always check with `hasMetadata` before reading columns. */
  metadata: Partial<TableMetadata>;
  registered: boolean;
  fingerprint?: string | null;
  /** Only on builtin/register responses that promoted an import placeholder. */
  warnings?: string[];
  /** Only on GET /datasets/{id}. */
  created_at?: string;
  /** First 5 rows; only present on GET /datasets/{id}. */
  preview?: Record<string, unknown>[];
}

/** GET /datasets/{id}/suggestion: the exploratory RQ3 meta-model's best-generator guess. */
export interface SuggestionAvailable {
  available: true;
  generator: string;
  probabilities: Record<string, number>;
  model?: {
    n_datasets?: number;
    trained_at?: string;
    [key: string]: unknown;
  };
}

/** No suggestion: not registered, no predictor trained, not computed yet, extraction failed. */
export interface SuggestionUnavailable {
  available: false;
  reason: string;
}

export type DatasetSuggestion = SuggestionAvailable | SuggestionUnavailable;

/** A dataset whose metadata is complete (every dataset except an import placeholder). */
export type DatasetWithMetadata = Dataset & { metadata: TableMetadata };

export function hasMetadata(d: Dataset): d is DatasetWithMetadata {
  const m = d.metadata as Partial<TableMetadata> | null | undefined;
  return !!m && typeof m.columns === "object" && m.columns !== null && typeof m.target === "string";
}

export type GeneratorFamily =
  | "sanity_ceiling"
  | "sanity_floor"
  | "statistical"
  | "graphical"
  | "deep"
  | "diffusion"
  | "llm"
  | "differential_privacy"
  | "nominal_dp_audit";

export interface GeneratorInfo {
  name: string;
  family: GeneratorFamily | string;
  queue: string;
  cpu_ok: boolean;
  params: Record<string, unknown>;
  audit?: boolean;
}

export type JobState =
  | "queued"
  | "fitting"
  | "sampling"
  | "generated"
  | "evaluating"
  | "done"
  | "failed";

export interface RunJob {
  job_id: string;
  generator: string;
  seed: number;
  state: JobState;
  error: string | null;
}

/** GET /runs list item: no per-job rows. */
export interface RunSummary {
  run_id: string;
  dataset_id: string;
  generators: string[];
  seeds: number[];
  preset: string;
  counts: Partial<Record<JobState, number>>;
  done: boolean;
  /** True when the artifact store was unreachable, so job states may be out of date. */
  stale?: boolean;
}

/** GET /runs/{id}. */
export interface Run extends RunSummary {
  jobs: RunJob[];
  /** Normalised [fidelity, utility, privacy] the run was created with; null when unknown (e.g. imported runs). */
  weights?: [number, number, number] | null;
}

/** POST /runs/{id}/retry response (200; the 503 body has the same shape plus `detail`). */
export interface RetryResult {
  run_id: string;
  retried: string[];
}

/** POST /runs response (201): only the id and the planned jobs. */
export interface CreatedRun {
  run_id: string;
  jobs: RunJob[];
}

export interface AxisScores {
  fidelity: number | null;
  utility: number | null;
  privacy: number | null;
}

export interface RankedCandidate {
  generator: string;
  scores: AxisScores;
  score: number | null;
  gate_passed: boolean;
  gate_failures: string[];
  on_pareto_front: boolean;
  notes: string[];
}

export interface Recommendation {
  preset: string;
  weights: [number, number, number];
  winner: string | null;
  reason: string;
  ranking: RankedCandidate[];
}

export interface AggregatedReport {
  generator: string;
  n_seeds: number;
  n_failed: number;
  mean: Record<string, number | null>;
  std: Record<string, number | null>;
  missing: string[];
  worst: Record<string, number | null>;
  singling_out_status?: (string | null)[];
}

export interface ParetoPoint {
  generator: string;
  fidelity: number | null;
  utility: number | null;
  privacy: number | null;
  on_front: boolean;
  gate_passed: boolean;
}

export interface PerJobReport {
  job_id: string;
  generator: string;
  seed: number;
  /** A job state while the run is live; "unknown" when the store could not be read. */
  status: JobState | "unknown";
  warnings: string[];
}

export interface RunReport {
  run_id: string;
  jobs_reported: number;
  jobs_total: number;
  recommendation: Recommendation | null;
  aggregated: AggregatedReport[];
  pareto: ParetoPoint[];
  per_job: PerJobReport[];
}

export interface Health {
  status: string;
  db?: boolean;
  store?: boolean;
}

export interface CreateRunBody {
  dataset_id: string;
  generators: string[];
  seeds: number[];
  preset: string;
  weights?: [number, number, number];
  allow_gpu_only?: boolean;
  params?: Record<string, Record<string, unknown>>;
}

export interface ImportResult {
  run_id: string;
  imported: number;
  warnings: string[];
}

export interface RegisterBody {
  max_rows?: number;
  public_bounds?: Record<string, unknown>;
  public_log_scale?: string[];
}
