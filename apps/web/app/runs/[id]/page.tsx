"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { useState } from "react";
import { ErrorState, errorDetail } from "@/components/error-state";
import { JobProgress } from "@/components/job-progress";
import { initialRanking, PresetSwitcher, type Ranking } from "@/components/preset-switcher";
import { RecommendationCard } from "@/components/recommendation-card";
import { ScoreChart } from "@/components/score-chart";
import { JobWarnings } from "@/components/job-warnings";
import { ParetoPlot } from "@/components/pareto-plot";
import { Scorecard } from "@/components/scorecard";
import { StatusBadge } from "@/components/status-badge";
import { Skeleton } from "@/components/ui/skeleton";
import { getReport, getRun, listGenerators } from "@/lib/api";
import { pollMs } from "@/lib/poll";
import { runState } from "@/lib/run-state";

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  // null until the person picks a ranking; until then the run's own preset/weights apply.
  const [picked, setPicked] = useState<Ranking | null>(null);

  const run = useQuery({
    queryKey: ["run", id],
    queryFn: () => getRun(id),
    refetchInterval: (q) => (q.state.data && !q.state.data.done ? pollMs() : false),
    refetchIntervalInBackground: false,
  });
  const runDone = run.data?.done ?? false;
  const initial = run.data ? initialRanking(run.data) : null;
  const ranking: Ranking = picked ?? initial?.ranking ?? { kind: "preset", preset: "balanced" };

  const rankKey = ranking.kind === "preset" ? ranking.preset : ranking.weights.join(",");
  const report = useQuery({
    // runDone is part of the key so the report is re-fetched once the run finishes.
    queryKey: ["report", id, ranking.kind, rankKey, runDone],
    queryFn: () =>
      getReport(
        id,
        ranking.kind === "preset" ? { preset: ranking.preset } : { weights: ranking.weights },
      ),
    enabled: run.data !== undefined,
    placeholderData: keepPreviousData,
    refetchInterval: runDone ? false : pollMs(),
    refetchIntervalInBackground: false,
  });
  const generators = useQuery({ queryKey: ["generators"], queryFn: listGenerators });

  if (run.isError && run.data === undefined) return <ErrorState title="Could not load this run" error={run.error} />;
  if (!run.data) return <Skeleton className="h-48 w-full" aria-label="Loading run" />;

  const data = run.data;
  const rep = report.data;
  const rec = rep?.recommendation ?? null;
  const refreshError = run.isError ? run.error : report.isError && rep ? report.error : null;

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Run {data.run_id}</h1>
          <p className="text-sm text-muted-foreground">
            Dataset {data.dataset_id} · {data.generators.length} generators · seeds {data.seeds.join(", ")}
          </p>
        </div>
        <StatusBadge state={runState(data)} />
      </header>

      {refreshError !== null && (
        <div role="status" className="rounded-md border px-3 py-2 text-sm text-muted-foreground">
          Couldn&apos;t refresh — retrying… ({errorDetail(refreshError)})
        </div>
      )}

      <JobProgress run={data} />

      <section aria-labelledby="results-heading" className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 id="results-heading" className="text-lg font-semibold">
            Results
          </h2>
          {rep && (
            <p className="text-sm text-muted-foreground tabular-nums">{`${rep.jobs_reported} of ${rep.jobs_total} jobs reported`}</p>
          )}
        </div>

        <PresetSwitcher value={ranking} onChange={setPicked} />
        {initial?.missing && picked === null && (
          <p role="note" className="text-sm text-muted-foreground">
            This run used custom weights that aren&apos;t available. The report is ranked with the Balanced preset instead.
          </p>
        )}

        {report.isError && !rep ? (
          <ErrorState title="Could not load the report" error={report.error} />
        ) : !rep ? (
          <Skeleton className="h-40 w-full" />
        ) : rec ? (
          <div className={report.isPlaceholderData ? "space-y-6 opacity-60" : "space-y-6"} aria-busy={report.isPlaceholderData}>
            <RecommendationCard rec={rec} />
            <Scorecard ranking={rec.ranking} aggregated={rep.aggregated} generators={generators.data} />
            <ScoreChart ranking={rec.ranking} />
            <ParetoPlot report={rep} />
            <JobWarnings jobs={rep.per_job} />
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            No results yet. The recommendation appears as soon as the first job has been evaluated.
          </p>
        )}
      </section>
    </div>
  );
}
