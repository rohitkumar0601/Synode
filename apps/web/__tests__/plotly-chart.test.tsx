import { render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const plotly = vi.hoisted(() => ({
  react: vi.fn(() => Promise.resolve()),
  newPlot: vi.fn(() => Promise.resolve()),
  purge: vi.fn(),
  Plots: { resize: vi.fn() },
}));
vi.mock("plotly.js-dist-min", () => ({ default: plotly }));

import PlotlyChart from "@/components/plotly-chart";

describe("PlotlyChart", () => {
  it("passes data, layout and config through to Plotly", async () => {
    const data = [{ type: "scatter" as const, x: [1], y: [2], name: "t" }];
    const layout = { height: 480, font: { color: "#fff" } };
    const config = { displaylogo: false };
    render(<PlotlyChart data={data} layout={layout} config={config} />);
    await waitFor(() => expect(plotly.react.mock.calls.length + plotly.newPlot.mock.calls.length).toBeGreaterThan(0));
    const call = (plotly.newPlot.mock.calls[0] ?? plotly.react.mock.calls[0]) as unknown as [unknown, Record<string, unknown>];
    const args = call.slice(1);
    const flat = JSON.stringify(args);
    expect(flat).toContain('"name":"t"');
    expect(flat).toContain('"font":{"color":"#fff"}');
    expect(flat).toContain('"displaylogo":false');
  });
});
