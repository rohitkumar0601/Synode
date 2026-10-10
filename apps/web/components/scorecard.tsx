"use client";

import { ArrowDown, ArrowUp, ChevronDown, ChevronRight, ChevronsUpDown, Star } from "lucide-react";
import { Fragment, useMemo, useState } from "react";
import { FamilyBadge } from "@/components/family-badge";
import { GateCell, GateFailureList } from "@/components/gate-failures";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { fmtMeanStd, fmtScore } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { AggregatedReport, GeneratorInfo, RankedCandidate } from "@/lib/types";

type SortDir = "asc" | "desc" | null;

/**
 * Sorts by weighted score within gate verdicts: gate-passing candidates always come first (a
 * gate-failing candidate is scored too, but can never be recommended), and candidates without a
 * score go last within their group. Null direction keeps API order.
 */
export function sortRanking(ranking: RankedCandidate[], dir: SortDir): RankedCandidate[] {
  if (dir === null) return ranking;
  const sign = dir === "asc" ? 1 : -1;
  return [...ranking].sort((a, b) => {
    if (a.gate_passed !== b.gate_passed) return a.gate_passed ? -1 : 1;
    if (a.score === null && b.score === null) return 0;
    if (a.score === null) return 1;
    if (b.score === null) return -1;
    return sign * (a.score - b.score);
  });
}

function DetailRow({ cand, agg }: { cand: RankedCandidate; agg: AggregatedReport | undefined }) {
  const metrics = agg ? Object.keys(agg.mean).sort() : [];
  return (
    <TableRow id={`detail-${cand.generator}`} className="bg-muted/30 hover:bg-muted/30">
      <TableCell colSpan={8} className="space-y-3 whitespace-normal py-3">
        <GateFailureList failures={cand.gate_failures} />
        {(cand.notes ?? []).length > 0 && (
          <ul className="list-disc pl-5 text-xs text-muted-foreground">
            {cand.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        )}
        {agg ? (
          <div className="space-y-1">
            <p className="text-xs font-medium">
              Per-metric mean ± std over {agg.n_seeds} seed{agg.n_seeds === 1 ? "" : "s"}
              {agg.n_failed > 0 ? ` (${agg.n_failed} failed)` : ""}
            </p>
            <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-6 gap-y-0.5 text-xs sm:max-w-md">
              {metrics.map((k) => (
                <Fragment key={k}>
                  <dt className="font-mono text-muted-foreground">{k}</dt>
                  <dd className="text-right tabular-nums">{fmtMeanStd(agg.mean[k], agg.std[k])}</dd>
                </Fragment>
              ))}
            </dl>
            {agg.missing.length > 0 && (
              <p className="text-xs text-muted-foreground">Missing metrics: {agg.missing.join(", ")}</p>
            )}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">No per-metric data for this generator.</p>
        )}
      </TableCell>
    </TableRow>
  );
}

export function Scorecard({
  ranking,
  aggregated,
  generators,
}: {
  ranking: RankedCandidate[];
  aggregated: AggregatedReport[];
  generators: GeneratorInfo[] | undefined;
}) {
  const [dir, setDir] = useState<SortDir>(null);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const rows = useMemo(() => sortRanking(ranking, dir), [ranking, dir]);
  const info = useMemo(() => new Map((generators ?? []).map((g) => [g.name, g])), [generators]);
  const aggBy = useMemo(() => new Map(aggregated.map((a) => [a.generator, a])), [aggregated]);

  const toggle = (g: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (!next.delete(g)) next.add(g);
      return next;
    });
  const cycle = () => setDir((d) => (d === null ? "asc" : d === "asc" ? "desc" : null));
  const SortIcon = dir === "asc" ? ArrowUp : dir === "desc" ? ArrowDown : ChevronsUpDown;

  return (
    <div className="space-y-2">
      <Table aria-label="Scorecard">
        <TableHeader>
          <TableRow>
            <TableHead>Generator</TableHead>
            <TableHead>Family</TableHead>
            <TableHead className="text-right">Fidelity</TableHead>
            <TableHead className="text-right">Utility</TableHead>
            <TableHead className="text-right">Privacy</TableHead>
            <TableHead
              className="text-right"
              aria-sort={dir === "asc" ? "ascending" : dir === "desc" ? "descending" : "none"}
            >
              <Button variant="ghost" size="sm" onClick={cycle} aria-label={`Weighted score, sort ${dir === "asc" ? "descending next" : dir === "desc" ? "by default order next" : "ascending next"}`}>
                Weighted score
                <SortIcon aria-hidden="true" />
              </Button>
            </TableHead>
            <TableHead>Privacy gate</TableHead>
            <TableHead>
              <span className="sr-only">Pareto front and details</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((c) => {
            const g = info.get(c.generator);
            const audit = g?.family === "nominal_dp_audit";
            const expanded = open.has(c.generator);
            return (
              <Fragment key={c.generator}>
                <TableRow
                  data-generator={c.generator}
                  data-gate={c.gate_passed ? "pass" : "fail"}
                  data-audit={audit ? "true" : "false"}
                  className={cn(!c.gate_passed && "bg-muted/40 text-muted-foreground", audit && "italic")}
                >
                  <TableCell className="font-medium">{c.generator}</TableCell>
                  <TableCell>{g ? <FamilyBadge family={g.family} cpuOk={g.cpu_ok} /> : null}</TableCell>
                  <TableCell className="text-right tabular-nums">{fmtScore(c.scores.fidelity)}</TableCell>
                  <TableCell className="text-right tabular-nums">{fmtScore(c.scores.utility)}</TableCell>
                  <TableCell className="text-right tabular-nums">{fmtScore(c.scores.privacy)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {c.gate_passed ? (
                      <span className="font-medium">{fmtScore(c.score)}</span>
                    ) : (
                      <span className="inline-flex flex-col items-end">
                        <span className="text-muted-foreground line-through decoration-1">{fmtScore(c.score)}</span>
                        <span className="text-xs text-muted-foreground">gate failed</span>
                      </span>
                    )}
                  </TableCell>
                  <TableCell>
                    <GateCell passed={c.gate_passed} failures={c.gate_failures} />
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center justify-end gap-1">
                      {c.on_pareto_front && (
                        <span role="img" aria-label="On Pareto front" title="On the Pareto front">
                          <Star aria-hidden="true" className="size-4 fill-current" />
                        </span>
                      )}
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-expanded={expanded}
                        aria-controls={expanded ? `detail-${c.generator}` : undefined}
                        aria-label={`Details for ${c.generator}`}
                        onClick={() => toggle(c.generator)}
                      >
                        {expanded ? <ChevronDown aria-hidden="true" /> : <ChevronRight aria-hidden="true" />}
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
                {expanded && <DetailRow cand={c} agg={aggBy.get(c.generator)} />}
              </Fragment>
            );
          })}
        </TableBody>
      </Table>
      <p className="text-xs text-muted-foreground">
        Axis scores run 0 to 1 (higher is better). Expand a row for per-metric mean ± std across seeds. The star marks
        generators on the Pareto front. Generators that fail the privacy gate are scored but struck through and
        always listed after the ones that pass: they cannot be recommended.
      </p>
    </div>
  );
}
