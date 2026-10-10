"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";
import Link from "next/link";
import { ErrorState } from "@/components/error-state";
import { StatusBadge } from "@/components/status-badge";
import { buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { listRuns } from "@/lib/api";
import { pollMs } from "@/lib/poll";
import { jobTotals, runState } from "@/lib/run-state";
import { cn } from "@/lib/utils";

export default function RunsPage() {
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: listRuns,
    refetchInterval: (q) => (q.state.data?.some((r) => !r.done) ? pollMs() : false),
    refetchIntervalInBackground: false,
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">Runs</h1>
        <Link href="/runs/new" className={cn(buttonVariants())}>
          New run
        </Link>
      </div>
      {runs.isError ? (
        <ErrorState title="Could not load runs" error={runs.error} />
      ) : runs.isPending ? (
        <Skeleton className="h-32 w-full" />
      ) : runs.data.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No runs yet.{" "}
          <Link href="/runs/new" className="underline">
            Start a new run
          </Link>
          .
        </p>
      ) : (
        <Table aria-label="Runs">
          <TableHeader>
            <TableRow>
              <TableHead>Run</TableHead>
              <TableHead>Dataset</TableHead>
              <TableHead>Generators</TableHead>
              <TableHead>Seeds</TableHead>
              <TableHead>Jobs finished</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {runs.data.map((r) => {
              const { total, finished } = jobTotals(r.counts);
              return (
                <TableRow key={r.run_id}>
                  <TableCell>
                    <Link href={`/runs/${encodeURIComponent(r.run_id)}`} className="font-medium underline-offset-4 hover:underline">
                      {r.run_id}
                    </Link>
                  </TableCell>
                  <TableCell>{r.dataset_id}</TableCell>
                  <TableCell>{r.generators.length}</TableCell>
                  <TableCell>{r.seeds.length}</TableCell>
                  <TableCell className="tabular-nums">
                    {finished} / {total}
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-wrap items-center gap-2">
                      <StatusBadge state={runState(r)} />
                      {r.stale && (
                        <span
                          className="inline-flex items-center gap-1 text-xs text-muted-foreground"
                          title="The artifact store could not be reached; job states are the last known ones."
                        >
                          <AlertTriangle aria-hidden="true" className="size-3.5" />
                          May be out of date
                        </span>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
