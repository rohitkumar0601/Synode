import { describe, expect, it } from "vitest";
import { buildParetoTraces, DARK_PALETTE, LIGHT_PALETTE, PASS_SYMBOL, FAIL_SYMBOL } from "@/lib/pareto";
import type { ParetoPoint, RunReport } from "@/lib/types";

const pt = (
  generator: string,
  f: number | null,
  u: number | null,
  p: number | null,
  on_front: boolean,
  gate_passed = true,
): ParetoPoint => ({ generator, fidelity: f, utility: u, privacy: p, on_front, gate_passed });

function report(pareto: ParetoPoint[]): RunReport {
  return { run_id: "r", jobs_reported: 1, jobs_total: 1, recommendation: null, aggregated: [], pareto, per_job: [] };
}

const POINTS = [
  pt("a", 0.9, 0.5, 0.2, true),
  pt("b", 0.3, 0.9, 0.5, true),
  pt("c", 0.6, 0.7, 0.9, true),
  pt("d", 0.5, 0.4, 0.4, false),
  pt("bad", 0.99, 0.98, 0.05, false, false),
];

describe("buildParetoTraces", () => {
  it("2D: splits gate pass/fail into distinct symbols, with legend text naming the symbol", () => {
    const { traces } = buildParetoTraces(report(POINTS), "2d");
    const pass = traces.find((t) => t.name?.startsWith("Gate passed"));
    const fail = traces.find((t) => t.name?.startsWith("Gate failed"));
    expect(pass?.marker?.symbol).toBe(PASS_SYMBOL);
    expect(fail?.marker?.symbol).toBe(FAIL_SYMBOL);
    expect(PASS_SYMBOL).not.toBe(FAIL_SYMBOL);
    expect(pass?.name).toMatch(/circle/);
    expect(fail?.name).toMatch(/diamond/);
    expect(pass?.x).toEqual([0.9, 0.3, 0.6, 0.5]);
    expect(pass?.y).toEqual([0.5, 0.9, 0.7, 0.4]);
    expect(fail?.x).toEqual([0.99]);
  });

  it("2D: colours markers by privacy with one shared colourbar", () => {
    const { traces } = buildParetoTraces(report(POINTS), "2d");
    const pass = traces.find((t) => t.name?.startsWith("Gate passed"));
    const fail = traces.find((t) => t.name?.startsWith("Gate failed"));
    expect(pass?.marker?.color).toEqual([0.2, 0.5, 0.9, 0.4]);
    expect(fail?.marker?.color).toEqual([0.05]);
    const bars = traces.filter((t) => (t.marker as { showscale?: boolean } | undefined)?.showscale);
    expect(bars).toHaveLength(1);
    expect(JSON.stringify(bars[0].marker)).toContain("Privacy");
  });

  it("outlines front points with open rings and connects them in fidelity order", () => {
    const { traces } = buildParetoTraces(report(POINTS), "2d");
    const front = traces.find((t) => t.name?.startsWith("Pareto front"));
    expect(front?.mode).toBe("lines+markers");
    expect(front?.marker?.symbol).toBe("circle-open");
    expect(front?.name).toMatch(/outlined/);
    expect(front?.x).toEqual([0.3, 0.6, 0.9]);
    expect(front?.y).toEqual([0.9, 0.7, 0.5]);
    expect(front?.hoverinfo).toBe("skip");
  });

  it("hover shows all three scores", () => {
    const { traces } = buildParetoTraces(report(POINTS), "2d");
    const pass = traces.find((t) => t.name?.startsWith("Gate passed"));
    expect(pass?.hovertemplate).toMatch(/Fidelity/);
    expect(pass?.hovertemplate).toMatch(/Utility/);
    expect(pass?.hovertemplate).toMatch(/Privacy/);
    expect((pass?.customdata as unknown[])[0]).toEqual(["a", 0.9, 0.5, 0.2]);
  });

  it("3D: scatter3d with privacy on z", () => {
    const { traces } = buildParetoTraces(report(POINTS), "3d");
    const pass = traces.find((t) => t.name?.startsWith("Gate passed"));
    expect(pass?.type).toBe("scatter3d");
    expect(pass?.z).toEqual([0.2, 0.5, 0.9, 0.4]);
    const front = traces.find((t) => t.name?.startsWith("Pareto front"));
    expect(front?.type).toBe("scatter3d");
    expect(front?.z).toEqual([0.5, 0.9, 0.2]);
  });

  it("excludes points with null scores and reports a visible note", () => {
    const r = buildParetoTraces(
      report([...POINTS, pt("n1", null, 0.5, 0.5, false), pt("n2", 0.5, null, null, true)]),
      "2d",
    );
    expect(r.missing).toEqual(["n1", "n2"]);
    expect(r.note).toBe("2 generators missing scores");
    expect(r.plotted.map((p) => p.generator)).not.toContain("n1");
    const pass = r.traces.find((t) => t.name?.startsWith("Gate passed"));
    expect(pass?.x).toHaveLength(4);
    const front = r.traces.find((t) => t.name?.startsWith("Pareto front"));
    expect(front?.x).toHaveLength(3);
  });

  it("singular note, no note when nothing is missing, omits empty traces", () => {
    expect(buildParetoTraces(report([...POINTS, pt("n1", null, 0.5, 0.5, false)]), "2d").note).toBe(
      "1 generator missing scores",
    );
    const ok = buildParetoTraces(report([pt("a", 0.5, 0.5, 0.5, false)]), "2d");
    expect(ok.note).toBeNull();
    expect(ok.traces).toHaveLength(1);
    expect(buildParetoTraces(report([]), "2d").traces).toEqual([]);
  });

  it.each([
    ["light", LIGHT_PALETTE],
    ["dark", DARK_PALETTE],
  ])("applies the %s palette to the front trace and marker outlines (2D and 3D)", (_n, palette) => {
    for (const mode of ["2d", "3d"] as const) {
      const { traces } = buildParetoTraces(report(POINTS), mode, palette);
      const front = traces.find((t) => t.name?.startsWith("Pareto front"));
      expect((front?.marker?.line as { color: string }).color).toBe(palette.front);
      expect(front?.marker?.color).toBe(palette.front);
      expect(front?.line?.color).toBe(palette.front);
      const pass = traces.find((t) => t.name?.startsWith("Gate passed"));
      expect((pass?.marker?.line as { color: string }).color).toBe(palette.outline);
    }
  });

  it("light and dark palettes differ and meet 4.5:1 text contrast on their backgrounds", () => {
    expect(LIGHT_PALETTE.front).not.toBe(DARK_PALETTE.front);
    const lum = (hex: string) => {
      const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
      return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    };
    const ratio = (a: string, b: string) => (Math.max(lum(a), lum(b)) + 0.05) / (Math.min(lum(a), lum(b)) + 0.05);
    expect(ratio(LIGHT_PALETTE.fg, "#ffffff")).toBeGreaterThan(4.5);
    expect(ratio(DARK_PALETTE.fg, "#0a0a0a")).toBeGreaterThan(4.5);
  });
});
