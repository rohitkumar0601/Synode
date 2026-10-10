"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useId } from "react";
import { ErrorState } from "@/components/error-state";
import { StatusBadge } from "@/components/status-badge";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { health, listDatasets, listRuns } from "@/lib/api";
import { runState } from "@/lib/run-state";

function Count({ label, value, loading }: { label: string; value: number | undefined; loading: boolean }) {
  const id = useId();
  return (
    <Card>
      <CardHeader>
        <CardTitle id={id} className="text-sm font-medium text-muted-foreground">
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-8 w-12" />
        ) : (
          <p role="status" aria-labelledby={id} className="text-3xl font-semibold tabular-nums">
            {value ?? "—"}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const healthQ = useQuery({ queryKey: ["health"], queryFn: health, refetchInterval: 15000, retry: false });
  const datasetsQ = useQuery({ queryKey: ["datasets"], queryFn: listDatasets });
  const runsQ = useQuery({
    queryKey: ["runs"],
    queryFn: listRuns,
    refetchInterval: (q) => (q.state.data?.some((r) => !r.done) ? 5000 : false),
  });

  const runs = runsQ.data;
  // /health answers "ok" whenever the process is up; db/store say whether it can do any work.
  const degraded = [
    healthQ.data?.db === false ? "database" : null,
    healthQ.data?.store === false ? "artifact store" : null,
  ].filter((x): x is string => x !== null);
  const active = runs?.filter((r) => !r.done).length;

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
        <div role="status" aria-live="polite">
          {healthQ.isPending ? (
            <Badge variant="secondary">Checking API…</Badge>
          ) : healthQ.isError || healthQ.data?.status !== "ok" ? (
            <Badge variant="destructive">API unreachable</Badge>
          ) : degraded.length > 0 ? (
            <span className="flex flex-wrap items-center gap-2">
              <Badge variant="destructive">API degraded</Badge>
              <span className="text-sm text-muted-foreground">{degraded.join(" and ")} unavailable</span>
            </span>
          ) : (
            <Badge variant="default">API online</Badge>
          )}
        </div>
      </div>

      <section aria-label="Counts" className="grid gap-4 sm:grid-cols-3">
        <Count label="Datasets" value={datasetsQ.data?.length} loading={datasetsQ.isPending} />
        <Count label="Runs" value={runs?.length} loading={runsQ.isPending} />
        <Count label="Active runs" value={active} loading={runsQ.isPending} />
      </section>

      <section aria-labelledby="recent-runs" className="space-y-3">
        <h2 id="recent-runs" className="text-lg font-semibold">
          Recent runs
        </h2>
        {runsQ.isError ? (
          <ErrorState title="Could not load runs" error={runsQ.error} />
        ) : runsQ.isPending ? (
          <Skeleton className="h-24 w-full" />
        ) : runs && runs.length > 0 ? (
          <ul className="divide-y rounded-lg border">
            {runs.slice(0, 8).map((r) => (
              <li key={r.run_id} className="flex flex-wrap items-center justify-between gap-2 px-4 py-3">
                <div className="min-w-0">
                  <Link href={`/runs/${encodeURIComponent(r.run_id)}`} className="font-medium underline-offset-4 hover:underline">
                    {r.run_id}
                  </Link>
                  <p className="text-sm text-muted-foreground">
                    {r.dataset_id} · {r.generators.length} generators · {r.seeds.length} seeds
                  </p>
                </div>
                <StatusBadge state={runState(r)} />
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted-foreground">
            No runs yet.{" "}
            <Link href="/runs/new" className="underline">
              Start a new run
            </Link>
            .
          </p>
        )}
      </section>
    </div>
  );
}
