import type { StatusState } from "@/components/status-badge";
import type { JobState, RunSummary } from "@/lib/types";

type Counts = RunSummary["counts"];

/** Job states that never change again without a retry. */
export const TERMINAL_STATES: readonly JobState[] = ["done", "failed", "cancelled"];

const count = (counts: Counts, s: JobState) => counts[s] ?? 0;

export function jobTotals(counts: Counts): { total: number; finished: number; failed: number; cancelled: number } {
  const total = Object.values(counts).reduce<number>((a, b) => a + (b ?? 0), 0);
  const finished = TERMINAL_STATES.reduce((a, s) => a + count(counts, s), 0);
  return { total, finished, failed: count(counts, "failed"), cancelled: count(counts, "cancelled") };
}

/**
 * Overall state of a run. While any job is unfinished the run is "running" (failures so far are a
 * note, see `failedNote`). Once every job is terminal: cancelled if any job was cancelled, failed only
 * if every job failed, done with failures if some did, else done.
 */
export function runState(run: Pick<RunSummary, "counts" | "done">): StatusState {
  const { total, finished, failed, cancelled } = jobTotals(run.counts);
  if (!run.done || finished < total) return "running";
  if (cancelled > 0) return "cancelled";
  if (total > 0 && failed === total) return "failed";
  if (failed > 0) return "done_with_failures";
  return "done";
}

/** "· 2 failed" next to a running run's badge; null when nothing failed or the run is finished. */
export function failedNote(run: Pick<RunSummary, "counts" | "done">): string | null {
  const failed = count(run.counts, "failed");
  return runState(run) === "running" && failed > 0 ? `· ${failed} failed` : null;
}

/** How far through the pipeline each state is, for the weighted progress percentage. */
export const STAGE_WEIGHTS: Record<JobState, number> = {
  queued: 0,
  fitting: 0.25,
  sampling: 0.4,
  generated: 0.5,
  evaluating: 0.75,
  done: 1,
  failed: 1,
  cancelled: 1,
};

/** Stage-aware progress in percent (0-100, rounded): moves as soon as jobs fit, sample or evaluate. */
export function weightedProgress(counts: Counts): number {
  let total = 0;
  let sum = 0;
  for (const [state, n] of Object.entries(counts)) {
    if (!n) continue;
    total += n;
    sum += n * (STAGE_WEIGHTS[state as JobState] ?? 0);
  }
  return total === 0 ? 0 : Math.round((sum / total) * 100);
}

/** Bar/legend order: finished work first, then the pipeline from its far end back to the queue. */
export const SEGMENT_ORDER: readonly JobState[] = [
  "done",
  "failed",
  "cancelled",
  "evaluating",
  "generated",
  "sampling",
  "fitting",
  "queued",
];

export function stageSegments(counts: Counts): { state: JobState; count: number }[] {
  const known = new Set<string>(SEGMENT_ORDER);
  const segs = SEGMENT_ORDER.map((state) => ({ state, count: count(counts, state) }));
  // States this app does not know count as queued, so the bar always adds up.
  const other = Object.entries(counts).reduce((a, [s, n]) => a + (known.has(s) ? 0 : (n ?? 0)), 0);
  if (other) segs[segs.length - 1].count += other;
  return segs.filter((s) => s.count > 0);
}

/** "9 generated · 2 fitting · 19 queued": the unfinished states, pipeline order. */
export function legendText(counts: Counts): string {
  return stageSegments(counts)
    .map((s) => `${s.count} ${s.state}`)
    .join(" · ");
}

/** "0 of 30 finished (2 failed) · 9 generated, awaiting metrics". */
export function progressSummary(counts: Counts): string {
  const { total, finished, failed, cancelled } = jobTotals(counts);
  const notes = [failed ? `${failed} failed` : null, cancelled ? `${cancelled} cancelled` : null].filter(Boolean);
  let text = `${finished} of ${total} finished${notes.length ? ` (${notes.join(", ")})` : ""}`;
  const generated = count(counts, "generated");
  if (generated) text += ` · ${generated} generated, awaiting metrics`;
  return text;
}

/** Unfinished jobs left behind with a synthetic table but no evaluation (or a stale view): offer resume. */
export function canResume(run: Pick<RunSummary, "counts" | "done" | "stale" | "cancelled">): boolean {
  if (run.done || run.cancelled) return false;
  return Boolean(run.stale) || count(run.counts, "generated") > 0;
}
