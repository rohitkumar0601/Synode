import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "adult-done" }),
  usePathname: () => "/runs/adult-done",
}));
// Plotly needs a real canvas/WebGL; the stub exposes the traces it was given.
vi.mock("@/components/plotly-chart", () => ({
  default: ({ data }: { data: { type: string; name: string }[] }) => (
    <div data-testid="plotly" data-types={data.map((d) => d.type).join(",")} data-names={data.map((d) => d.name).join("|")} />
  ),
}));

import RunPage from "@/app/runs/[id]/page";
import { ParetoPlot } from "@/components/pareto-plot";

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("Run detail: Pareto plot and per-job details", () => {
  it("renders the Pareto plot with a text alternative table", async () => {
    renderWithClient(<RunPage />);
    const plot = await screen.findByTestId("plotly");
    expect(plot.dataset.types).toBe("scatter,scatter,scatter");
    expect(plot.dataset.names).toBe("Gate passed (circle)|Gate failed (diamond)|Pareto front (outlined, connected)");
    const table = screen.getByRole("table", { name: "Pareto plot data" });
    const row = within(table).getByText("train_copy").closest("tr") as HTMLElement;
    expect(within(row).getByText("Fail")).toBeInTheDocument();
    expect(within(row).getByText("Not eligible (gate failed)")).toBeInTheDocument();
    const tvae = within(table).getByText("tvae").closest("tr") as HTMLElement;
    expect(within(tvae).getByText("On front")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /generators plotted by fidelity and utility/ })).toBeInTheDocument();
  });

  it("toggles to the 3D scatter", async () => {
    const user = userEvent.setup();
    renderWithClient(<RunPage />);
    await screen.findByTestId("plotly");
    expect(screen.getByRole("button", { name: "2D" })).toHaveAttribute("aria-pressed", "true");
    await user.click(screen.getByRole("button", { name: "3D" }));
    expect((await screen.findByTestId("plotly")).dataset.types).toBe("scatter3d,scatter3d,scatter3d");
    expect(screen.getByRole("button", { name: "3D" })).toHaveAttribute("aria-pressed", "true");
  });

  it("lists per-job warnings in a dialog", async () => {
    const user = userEvent.setup();
    renderWithClient(<RunPage />);
    const table = await screen.findByRole("table", { name: "Per-job details" });
    expect(screen.getByText(/jobs have warnings/)).toBeInTheDocument();
    const buttons = within(table).getAllByRole("button", { name: /1 warning/ });
    await user.click(buttons[0]);
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/^detection_auc 0.312 < 0.4: synthetic rows likely duplicate real rows/)).toBeInTheDocument();
  });

  it("shows the missing-scores note", async () => {
    renderWithClient(
      <ParetoPlot
        report={{
          pareto: [
            { generator: "a", fidelity: 0.5, utility: 0.5, privacy: 0.5, on_front: true, gate_passed: true },
            { generator: "b", fidelity: null, utility: null, privacy: null, on_front: false, gate_passed: true },
          ],
        }}
      />,
    );
    expect(await screen.findByText("1 generator missing scores")).toBeInTheDocument();
    expect((await screen.findByTestId("plotly")).dataset.names).toBe("Gate passed (circle)|Pareto front (outlined, connected)");
  });
});
