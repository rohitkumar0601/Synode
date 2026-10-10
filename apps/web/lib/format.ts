import type { JobState } from "@/lib/types";

const DASH = "—";

export function fmtScore(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return x.toFixed(digits);
}

export function fmtMeanStd(
  mean: number | null | undefined,
  std: number | null | undefined,
  digits = 3,
): string {
  if (mean === null || mean === undefined || !Number.isFinite(mean)) return DASH;
  if (std === null || std === undefined || !Number.isFinite(std)) return fmtScore(mean, digits);
  return `${fmtScore(mean, digits)} ± ${fmtScore(std, digits)}`;
}

/** Dataset `source` as shown to people. */
export function sourceLabel(source: string): string {
  if (source === "builtin") return "Built-in";
  if (source === "upload") return "Upload";
  if (source === "import") return "Imported";
  return source;
}

export function fmtState(state: JobState | string): string {
  return state.charAt(0).toUpperCase() + state.slice(1);
}

/** "sha256:3f9a1c0de2…" -> first 12 hex chars. */
export function shortFingerprint(fp: string | null | undefined): string {
  if (!fp) return "—";
  return fp.replace(/^[a-z0-9]+:/i, "").slice(0, 12);
}
