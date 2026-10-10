import type {
  CreateRunBody,
  CreatedRun,
  Dataset,
  DatasetSuggestion,
  GeneratorInfo,
  Health,
  ImportResult,
  Run,
  RegisterBody,
  RetryResult,
  RunReport,
  RunSummary,
  TableMetadata,
} from "@/lib/types";

export class ApiError extends Error {
  status: number;
  detail: string;
  /** Parsed JSON response body (null when absent or not JSON). */
  body: unknown;

  constructor(status: number, detail: string, body: unknown = null) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}

export function apiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8010").replace(/\/+$/, "");
}

/** FastAPI `detail` is a string, or a list of `{msg}` validation errors. */
function parseDetail(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((d) => {
        if (typeof d === "string") return d;
        const item = d as { msg?: unknown; loc?: unknown };
        if (typeof item?.msg !== "string") return undefined;
        let msg = item.msg.replace(/^Value error, /, "");
        const field = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : undefined;
        if (typeof field === "string" && field !== "body") {
          if (msg.startsWith(`${field} `)) msg = msg.slice(field.length + 1);
          return `${field}: ${msg}`;
        }
        return msg;
      })
      .filter((m): m is string => typeof m === "string" && m.length > 0);
    if (msgs.length) return msgs.join("; ");
  }
  return fallback;
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${apiBase()}${path}`, init);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new ApiError(0, "Request aborted");
    }
    // CORS-less error responses (e.g. a proxy rejecting a large upload) look the same as an unreachable host.
    throw new ApiError(0, `Cannot reach the API, or it rejected the request (${apiBase()})`);
  }
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, parseDetail(body, res.statusText || `HTTP ${res.status}`), body);
  }
  if (res.status === 204) return undefined as T;
  try {
    return (await res.json()) as T;
  } catch {
    throw new ApiError(res.status, "The API returned a response that is not valid JSON");
  }
}

function jsonInit(method: string, body: unknown): RequestInit {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

export const health = () => apiFetch<Health>("/health");

export const listDatasets = () => apiFetch<Dataset[]>("/datasets");
export const getDataset = (id: string) => apiFetch<Dataset>(`/datasets/${encodeURIComponent(id)}`);
export const getSuggestion = (id: string) =>
  apiFetch<DatasetSuggestion>(`/datasets/${encodeURIComponent(id)}/suggestion`);

export function uploadDataset(file: File, name: string, target: string) {
  const form = new FormData();
  form.append("file", file);
  form.append("name", name);
  form.append("target", target);
  return apiFetch<Dataset>("/datasets", { method: "POST", body: form });
}

export const updateMetadata = (
  id: string,
  metadata: { columns: TableMetadata["columns"]; target: string },
) => apiFetch<Dataset>(`/datasets/${encodeURIComponent(id)}/metadata`, jsonInit("PUT", metadata));

export const registerDataset = (id: string, body: RegisterBody = {}) =>
  apiFetch<Dataset>(`/datasets/${encodeURIComponent(id)}/register`, jsonInit("POST", body));

export const loadBuiltin = (name: string) =>
  apiFetch<Dataset>(`/datasets/builtin/${encodeURIComponent(name)}`, { method: "POST" });

export const listGenerators = () => apiFetch<GeneratorInfo[]>("/generators");

export const createRun = (body: CreateRunBody) => apiFetch<CreatedRun>("/runs", jsonInit("POST", body));
export const listRuns = () => apiFetch<RunSummary[]>("/runs");
export const getRun = (id: string) => apiFetch<Run>(`/runs/${encodeURIComponent(id)}`);

export const retryRun = (id: string) =>
  apiFetch<RetryResult>(`/runs/${encodeURIComponent(id)}/retry`, { method: "POST" });

export function getReport(id: string, opts: { preset?: string; weights?: number[] } = {}) {
  const q = new URLSearchParams();
  if (opts.preset) q.set("preset", opts.preset);
  if (opts.weights) q.set("weights", opts.weights.join(","));
  const qs = q.toString();
  return apiFetch<RunReport>(`/runs/${encodeURIComponent(id)}/report${qs ? `?${qs}` : ""}`);
}

export function importBundle(file: File, force: boolean) {
  const form = new FormData();
  form.append("file", file);
  form.append("force", String(force));
  return apiFetch<ImportResult>("/import-results", { method: "POST", body: form });
}
