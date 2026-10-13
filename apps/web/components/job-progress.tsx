"use client";

import { toast } from "sonner";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, RotateCw } from "lucide-react";
import { ErrorState } from "@/components/error-state";
import { GeneratorParamsDisclosure } from "@/components/generator-params";
import { StatusBadge } from "@/components/status-badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { ResumeRunButton } from "@/components/run-controls";
import { StageBar } from "@/components/stage-bar";
import { ApiError, retryRun } from "@/lib/api";
import { canResume, progressSummary } from "@/lib/run-state";
import type { GeneratorParams, Run } from "@/lib/types";

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

/**
 * `params` maps job_id to what the job trained with (from the run report); undefined while the
 * report has not loaded, so no "Parameters" disclosure is shown yet.
 */
export function JobProgress({ run, params }: { run: Run; params?: Record<string, GeneratorParams | null> }) {
  const qc = useQueryClient();
  const counts: Run["counts"] = {};
  for (const j of run.jobs) counts[j.state] = (counts[j.state] ?? 0) + 1;
  const failed = counts.failed ?? 0;
  const cancelled = counts.cancelled ?? 0;

  const retry = useMutation({
    mutationFn: () => retryRun(run.run_id),
    onSuccess: (r) => {
      toast.success(`Retried ${plural(r.retried.length)}`);
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
        <p className="text-sm text-muted-foreground tabular-nums">{progressSummary(counts)}</p>
      </div>

      <StageBar counts={counts} label="Run progress" legend />

      {run.stale && (
        <Alert>
          <AlertTriangle aria-hidden="true" />
          <AlertDescription>
            Job status may be out of date: the artifact store could not be reached, so some states are the last known
            ones.
          </AlertDescription>
        </Alert>
      )}

      {(failed > 0 || cancelled > 0 || canResume(run)) && (
        <div className="flex flex-wrap items-center gap-3">
          {canResume(run) && <ResumeRunButton runId={run.run_id} />}
          {(failed > 0 || cancelled > 0) && (
            <Button variant="outline" size="sm" onClick={() => retry.mutate()} disabled={retry.isPending}>
              <RotateCw aria-hidden="true" />
              {retry.isPending ? "Retrying…" : cancelled > 0 ? "Retry cancelled and failed jobs" : "Retry failed jobs"}
            </Button>
          )}
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
            <GeneratorParamsDisclosure
              params={params === undefined ? undefined : (params[j.job_id] ?? null)}
              state={j.state}
              label={`${j.generator} seed ${j.seed}`}
            />
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
