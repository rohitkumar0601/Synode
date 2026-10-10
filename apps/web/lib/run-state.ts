import type { StatusState } from "@/components/status-badge";
import type { RunSummary } from "@/lib/types";

/** Overall state of a run: any failed job wins, then done, else still running. */
export function runState(run: Pick<RunSummary, "counts" | "done">): StatusState {
  if ((run.counts.failed ?? 0) > 0) return "failed";
  return run.done ? "done" : "running";
}

export function jobTotals(counts: RunSummary["counts"]): { total: number; finished: number } {
  const total = Object.values(counts).reduce<number>((a, b) => a + (b ?? 0), 0);
  return { total, finished: (counts.done ?? 0) + (counts.failed ?? 0) };
}
