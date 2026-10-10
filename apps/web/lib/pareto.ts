import type { PlotData } from "plotly.js/index";
import type { ParetoPoint, RunReport } from "@/lib/types";

export type ParetoMode = "2d" | "3d";

/** Gate pass is a filled circle, gate fail a diamond; the legend names the shape too. */
export const PASS_SYMBOL = "circle";
export const FAIL_SYMBOL = "diamond";

export type PlotTrace = Partial<PlotData>;

/** Theme-dependent colours; keeps buildParetoTraces pure. */
export interface Palette {
  /** Text and axis labels. */
  fg: string;
  grid: string;
  /** Front ring and dotted line. */
  front: string;
  /** Outline of ordinary markers. */
  outline: string;
}

export const LIGHT_PALETTE: Palette = { fg: "#374151", grid: "rgba(0,0,0,0.12)", front: "#111827", outline: "#4b5563" };
export const DARK_PALETTE: Palette = { fg: "#d1d5db", grid: "rgba(255,255,255,0.18)", front: "#f9fafb", outline: "#9ca3af" };
export const paletteFor = (theme: string | undefined): Palette => (theme === "dark" ? DARK_PALETTE : LIGHT_PALETTE);

export interface Plotted {
  generator: string;
  fidelity: number;
  utility: number;
  privacy: number;
  on_front: boolean;
  gate_passed: boolean;
}

export interface ParetoTraces {
  traces: PlotTrace[];
  /** Points actually drawn. */
  plotted: Plotted[];
  /** Generators dropped because a score is null. */
  missing: string[];
  /** Visible note, e.g. "2 generators missing scores", or null. */
  note: string | null;
}

const HOVER =
  "<b>%{customdata[0]}</b><br>Fidelity %{customdata[1]:.3f}<br>Utility %{customdata[2]:.3f}<br>Privacy %{customdata[3]:.3f}<extra></extra>";

const isNum = (v: number | null): v is number => v !== null && Number.isFinite(v);

function complete(p: ParetoPoint): p is ParetoPoint & { fidelity: number; utility: number; privacy: number } {
  return isNum(p.fidelity) && isNum(p.utility) && isNum(p.privacy);
}

/** Builds the Plotly traces for the Pareto plot. Pure: no Plotly import, safe to unit-test and to SSR. */
export function buildParetoTraces(report: Pick<RunReport, "pareto">, mode: ParetoMode,
  palette: Palette = LIGHT_PALETTE,
): ParetoTraces {
  const plotted: Plotted[] = [];
  const missing: string[] = [];
  for (const p of report.pareto) {
    if (complete(p)) plotted.push(p);
    else missing.push(p.generator);
  }
  const is3d = mode === "3d";
  const coords = (pts: Plotted[]) => ({
    x: pts.map((p) => p.fidelity),
    y: pts.map((p) => p.utility),
    ...(is3d ? { z: pts.map((p) => p.privacy) } : {}),
  });
  const type = is3d ? "scatter3d" : "scatter";

  const traces: PlotTrace[] = [];
  let colourbarUsed = false;
  const group = (pts: Plotted[], name: string, symbol: string) => {
    if (pts.length === 0) return;
    const marker: Record<string, unknown> = {
      symbol,
      size: is3d ? 5 : 11,
      color: pts.map((p) => p.privacy),
      colorscale: "Viridis",
      cmin: 0,
      cmax: 1,
      line: { width: 1, color: palette.outline },
    };
    if (!colourbarUsed) {
      marker.showscale = true;
      marker.colorbar = { title: { text: "Privacy" }, thickness: 14 };
      colourbarUsed = true;
    }
    traces.push({
      type,
      mode: "markers",
      name,
      ...coords(pts),
      text: pts.map((p) => p.generator),
      customdata: pts.map((p) => [p.generator, p.fidelity, p.utility, p.privacy]),
      hovertemplate: HOVER,
      marker,
    });
  };
  group(
    plotted.filter((p) => p.gate_passed),
    `Gate passed (${PASS_SYMBOL})`,
    PASS_SYMBOL,
  );
  group(
    plotted.filter((p) => !p.gate_passed),
    `Gate failed (${FAIL_SYMBOL})`,
    FAIL_SYMBOL,
  );

  const front = plotted.filter((p) => p.on_front).sort((a, b) => a.fidelity - b.fidelity);
  if (front.length > 0) {
    traces.push({
      type,
      mode: "lines+markers",
      name: "Pareto front (outlined, connected)",
      ...coords(front),
      hoverinfo: "skip",
      marker: { symbol: "circle-open", size: is3d ? 9 : 19, color: palette.front, line: { width: 2, color: palette.front } },
      line: { color: palette.front, width: 1.5, dash: "dot" },
    });
  }

  const k = missing.length;
  return { traces, plotted, missing, note: k > 0 ? `${k} generator${k === 1 ? "" : "s"} missing scores` : null };
}
