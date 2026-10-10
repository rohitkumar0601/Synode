import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import Dashboard from "@/app/page";
import { server } from "@/mocks/server";

function renderDashboard() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Dashboard />
    </QueryClientProvider>,
  );
}

describe("Dashboard", () => {
  it("shows health, counts and recent runs from the mock API", async () => {
    renderDashboard();
    expect(await screen.findByText(/API online/i)).toBeInTheDocument();
    expect(await screen.findByText("adult-done")).toBeInTheDocument();
    expect(screen.getByText("adult-live")).toBeInTheDocument();
    expect(screen.getByRole("status", { name: "Datasets" })).toHaveTextContent("2");
    expect(screen.getByRole("status", { name: "Runs" })).toHaveTextContent("2");
  });

  it("shows degraded when the database or store is unavailable", async () => {
    server.use(http.get("http://api.test/health", () => HttpResponse.json({ status: "ok", db: true, store: false })));
    renderDashboard();
    expect(await screen.findByText(/API degraded/i)).toBeInTheDocument();
    expect(screen.getByText(/artifact store unavailable/i)).toBeInTheDocument();
    expect(screen.queryByText(/API online/i)).not.toBeInTheDocument();
  });

  it("encodes run ids in links", async () => {
    server.use(
      http.get("http://api.test/runs", () =>
        HttpResponse.json([
          { run_id: "odd id#1", dataset_id: "adult", generators: ["tvae"], seeds: [0], preset: "balanced", counts: { done: 1 }, done: true, stale: false },
        ]),
      ),
    );
    renderDashboard();
    expect(await screen.findByRole("link", { name: "odd id#1" })).toHaveAttribute("href", "/runs/odd%20id%231");
  });

  it("shows an error state when the API is down", async () => {
    server.use(http.get("http://api.test/health", () => HttpResponse.error()));
    server.use(
      http.get("http://api.test/runs", () => HttpResponse.json({ detail: "db down" }, { status: 500 })),
    );
    renderDashboard();
    expect(await screen.findByText(/API unreachable/i)).toBeInTheDocument();
    expect(await screen.findByText("db down")).toBeInTheDocument();
  });
});
