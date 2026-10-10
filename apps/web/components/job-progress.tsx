"use client";

import { toast } from "sonner";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, RotateCw } from "lucide-react";
import { ErrorState } from "@/components/error-state";
import { StatusBadge } from "@/components/status-badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ApiError, retryRun } from "@/lib/api";
import type { Run } from "@/lib/types";

/** The last non-empty line of a traceback is the exception itself, e.g. "torch.OutOfMemoryError: ...". */
export function lastErrorLine(error: string): string {
  const lines = error
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  return lines.at(-1) ?? error;
}

const plural = (n: number) => `${n} job${n === 1 ? "" : "s"}`;

/** A 503 from the retry endpoint still re-queued jobs: its body lists them. */
function retriedFrom503(e: unknown): number | null {
  if (!(e instanceof ApiError) || e.status !== 503) return null;
  const retried = (e.body as { retried?: unknown } | null)?.retried;
  return Array.isArray(retried) ? retried.length : null;
}

export function JobProgress({ run }: { run: Run }) {
  const qc = useQueryClient();
  const total = run.jobs.length;
  const finished = run.jobs.filter((j) => j.state === "done" || j.state === "failed").length;
  const failed = run.jobs.filter((j) => j.state === "failed").length;
  const pct = total === 0 ? 0 : Math.round((finished / total) * 100);

  const retry = useMutation({
    mutationFn: () => retryRun(run.run_id),
    onSuccess: (r) => {
      toast.success(`Retried ${r.retried.length} failed job${r.retried.length === 1 ? "" : "s"}`);
    },
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ["run", run.run_id] });
      void qc.invalidateQueries({ queryKey: ["report", run.run_id] });
      void qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
  const requeued = retry.isError ? retriedFrom503(retry.error) : null;

  return (
    <section aria-labelledby="progress-heading" className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="progress-heading" className="text-lg font-semibold">
          Progress
        </h2>
        <p className="text-sm text-muted-foreground tabular-nums">
          {finished} of {total} jobs finished{failed > 0 ? ` (${failed} failed)` : ""}
        </p>
      </div>

      <Progress value={pct} aria-label="Run progress" />

      {run.stale && (
        <Alert>
          <AlertTriangle aria-hidden="true" />
          <AlertDescription>
            Job status may be out of date: the artifact store could not be reached, so some states are the last known
            ones.
          </AlertDescription>
        </Alert>
      )}

      {failed > 0 && (
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="outline" size="sm" onClick={() => retry.mutate()} disabled={retry.isPending}>
            <RotateCw aria-hidden="true" />
            {retry.isPending ? "Retrying…" : "Retry failed jobs"}
          </Button>
          <div role="status" className="text-sm text-muted-foreground">
            {retry.isSuccess ? `Retried ${plural(retry.data.retried.length)}.` : null}
          </div>
        </div>
      )}
      {retry.isError &&
        (requeued !== null ? (
          <Alert variant="destructive">
            <AlertTriangle aria-hidden="true" />
            <AlertTitle>Retry could not be dispatched</AlertTitle>
            <AlertDescription>
              <p>{retry.error instanceof ApiError ? retry.error.detail : "Retry failed."}</p>
              <p>
                {requeued === 1 ? "1 job was re-queued" : `${requeued} jobs were re-queued`}; they stay queued until a
                worker is reachable.
              </p>
            </AlertDescription>
          </Alert>
        ) : (
          <ErrorState title="Could not retry" error={retry.error} />
        ))}

      <ul className="divide-y rounded-lg border">
        {run.jobs.map((j) => (
          <li key={j.job_id} aria-label={`${j.generator} seed ${j.seed}`} className="space-y-1 px-4 py-2 text-sm">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-medium">
                {j.generator} <span className="font-normal text-muted-foreground">seed {j.seed}</span>
              </span>
              <StatusBadge state={j.state} />
            </div>
            {j.state === "failed" && j.error && (
              <div className="space-y-1">
                <p className="break-words font-mono text-xs text-destructive">{lastErrorLine(j.error)}</p>
                <details className="text-xs">
                  <summary className="cursor-pointer py-1 text-muted-foreground underline-offset-4 hover:underline">
                    Full error
                  </summary>
                  <pre
                    tabIndex={0}
                    aria-label={`Full error for ${j.generator} seed ${j.seed}`}
                    className="mt-1 max-h-72 overflow-auto rounded-md bg-muted p-2 font-mono text-xs whitespace-pre"
                  >
                    {j.error}
                  </pre>
                </details>
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
