"use client";

import { CheckCircle2, XCircle } from "lucide-react";
import { useTheme } from "next-themes";
import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { fmtScore } from "@/lib/format";
import { buildParetoTraces, paletteFor, type ParetoMode } from "@/lib/pareto";
import type { RunReport } from "@/lib/types";

// Plotly touches `window` and is several MB, so it loads client-side only, and only when this component mounts.
const PlotlyChart = dynamic(() => import("@/components/plotly-chart"), {
  ssr: false,
  loading: () => <Skeleton className="h-[480px] w-full" aria-label="Loading plot" />,
});

const RANGE = [0, 1.05];

export function ParetoPlot({ report }: { report: Pick<RunReport, "pareto"> }) {
  const [mode, setMode] = useState<ParetoMode>("2d");
  const { resolvedTheme } = useTheme();
  const palette = paletteFor(resolvedTheme);
  const { traces, plotted, note } = useMemo(() => buildParetoTraces(report, mode, palette), [report, mode, palette]);
  const layout = useMemo(() => {
    const axis = { range: RANGE, gridcolor: palette.grid, zerolinecolor: palette.grid, color: palette.fg };
    return {
      autosize: true,
      height: 480,
      margin: { l: 56, r: 16, t: 16, b: 56 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: palette.fg },
      legend: { orientation: "h" as const, y: -0.2 },
      xaxis: { title: { text: "Fidelity" }, ...axis },
      yaxis: { title: { text: "Utility" }, ...axis },
      scene: {
        xaxis: { title: { text: "Fidelity" }, ...axis },
        yaxis: { title: { text: "Utility" }, ...axis },
        zaxis: { title: { text: "Privacy" }, ...axis },
      },
    };
  }, [palette]);
  if (report.pareto.length === 0) return null;

  const frontCount = plotted.filter((p) => p.on_front).length;
  const failCount = plotted.filter((p) => !p.gate_passed).length;
  const summary = `${plotted.length} generators plotted by fidelity and utility with privacy as colour; ${frontCount} on the Pareto front; ${failCount} fail the gate.`;

  return (
    <figure className="space-y-3" aria-labelledby="pareto-heading">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 id="pareto-heading" className="text-base font-semibold">
          Pareto front
        </h3>
        <div role="group" aria-label="Plot dimensions" className="flex gap-1">
          {(["2d", "3d"] as const).map((m) => (
            <Button
              key={m}
              type="button"
              size="sm"
              variant={mode === m ? "default" : "outline"}
              aria-pressed={mode === m}
              onClick={() => setMode(m)}
            >
              {m === "2d" ? "2D" : "3D"}
            </Button>
          ))}
        </div>
      </div>
      {note && (
        <p role="status" className="text-sm text-muted-foreground">
          {note}
        </p>
      )}
      <div role="img" aria-label={summary}>
        <PlotlyChart
          key={mode}
          data={traces}
          layout={layout}
          config={{ displaylogo: false, responsive: true }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      </div>
      <figcaption className="text-xs text-muted-foreground">
        Circle = gate passed, diamond = gate failed. Ringed points joined by a dotted line are on the Pareto front (joined in order of increasing fidelity, also in 3D).
        Colour = privacy score.
      </figcaption>
      <div className="overflow-x-auto">
        <Table aria-label="Pareto plot data">
          <TableHeader>
            <TableRow>
              <TableHead>Generator</TableHead>
              <TableHead className="text-right">Fidelity</TableHead>
              <TableHead className="text-right">Utility</TableHead>
              <TableHead className="text-right">Privacy</TableHead>
              <TableHead>Pareto front</TableHead>
              <TableHead>Gate</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {report.pareto.map((p) => (
              <TableRow key={p.generator}>
                <TableCell className="font-mono text-xs">{p.generator}</TableCell>
                <TableCell className="text-right tabular-nums">{fmtScore(p.fidelity)}</TableCell>
                <TableCell className="text-right tabular-nums">{fmtScore(p.utility)}</TableCell>
                <TableCell className="text-right tabular-nums">{fmtScore(p.privacy)}</TableCell>
                <TableCell>{p.on_front ? "On front" : p.gate_passed ? "Dominated" : "Not eligible (gate failed)"}</TableCell>
                <TableCell>
                  {p.gate_passed ? (
                    <span className="inline-flex items-center gap-1">
                      <CheckCircle2 aria-hidden="true" className="size-4 text-primary" />
                      Pass
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 font-medium text-destructive">
                      <XCircle aria-hidden="true" className="size-4" />
                      Fail
                    </span>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </figure>
  );
}
