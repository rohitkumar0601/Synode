import { stageSegments, weightedProgress } from "@/lib/run-state";
import type { JobState, RunSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

/** Segment fill per state; the legend (or the text next to a mini bar) carries the meaning, not colour. */
export const STAGE_FILL: Record<JobState, string> = {
  done: "bg-primary",
  failed: "bg-destructive",
  cancelled: "bg-muted-foreground/40",
  evaluating: "bg-chart-5",
  generated: "bg-chart-4",
  sampling: "bg-chart-3",
  fitting: "bg-chart-2",
  queued: "bg-transparent",
};

/**
 * Stacked bar of job states (done, failed, cancelled, evaluating, generated, sampling, fitting,
 * queued) with a weighted percentage as its progressbar value, so it moves before any job is done.
 */
export function StageBar({
  counts,
  label,
  size = "md",
  legend = false,
}: {
  counts: RunSummary["counts"];
  label: string;
  size?: "sm" | "md";
  legend?: boolean;
}) {
  const segs = stageSegments(counts);
  const total = segs.reduce((a, s) => a + s.count, 0);
  const pct = weightedProgress(counts);
  const text = segs.map((s) => `${s.count} ${s.state}`).join(" · ");
  return (
    <div className={cn(size === "md" ? "space-y-2" : "min-w-24")}>
      <div
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        aria-valuetext={`${pct}% (${text || "no jobs"})`}
        className={cn("flex w-full overflow-hidden rounded-full bg-muted", size === "md" ? "h-2.5" : "h-1.5")}
      >
        {segs.map((s) => (
          <div
            key={s.state}
            data-state={s.state}
            className={cn("h-full transition-all", STAGE_FILL[s.state])}
            style={{ width: `${total ? (s.count / total) * 100 : 0}%` }}
          />
        ))}
      </div>
      {legend && (
        <ul aria-label="Job states" className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground tabular-nums">
          {segs.map((s) => (
            <li key={s.state} className="inline-flex items-center gap-1.5">
              <span aria-hidden="true" className={cn("size-2.5 rounded-sm border border-border", STAGE_FILL[s.state])} />
              {s.count} {s.state}
            </li>
          ))}
          <li className="ml-auto">{pct}% through the pipeline</li>
        </ul>
      )}
    </div>
  );
}
