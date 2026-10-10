import { QueryClient, QueryClientProvider, timeoutManager } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { freezeRun, seedFailedRun } from "@/mocks/handlers";
import { server } from "@/mocks/server";
import { DONE_CANDIDATES, rank, recommendation } from "@/mocks/fixtures";
import type { Run, RunReport } from "@/lib/types";

const nav = vi.hoisted(() => ({ id: "adult-done" }));
vi.mock("next/navigation", () => ({
  useParams: () => ({ id: nav.id }),
  usePathname: () => `/runs/${nav.id}`,
}));

// Plotly (several MB, needs canvas) is stubbed; the Pareto plot is covered in run-pareto.test.tsx.
vi.mock("@/components/plotly-chart", () => ({ default: () => null }));

import RunPage from "@/app/runs/[id]/page";
import RunsPage from "@/app/runs/page";

/**
 * react-query's refetchInterval timers are routed to this registry instead of real intervals, so
 * polling advances only when a test calls `poll()` and "no longer polling" is simply an empty
 * registry: no wall-clock windows, no sleeps. Installed before any QueryClient exists.
 */
const intervals = new Map<number, () => void>();
let nextInterval = 0;
timeoutManager.setTimeoutProvider({
  setTimeout: (cb, ms) => setTimeout(cb, ms) as unknown as number,
  clearTimeout: (id) => clearTimeout(id as unknown as ReturnType<typeof setTimeout>),
  setInterval: (cb) => {
    nextInterval += 1;
    intervals.set(nextInterval, cb);
    return nextInterval;
  },
  clearInterval: (id) => {
    intervals.delete(Number(id));
  },
});
/** Fires every active poll once (one refetchInterval tick). */
const poll = () => {
  for (const cb of [...intervals.values()]) cb();
};

const API = "http://api.test";

function renderUi(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}
const renderPage = () => renderUi(<RunPage />);

beforeEach(() => {
  nav.id = "adult-done";
});
afterEach(() => {
  intervals.clear();
});

function emptyReport(over: Partial<RunReport> = {}): RunReport {
  return {
    run_id: "adult-done",
    jobs_reported: 0,
    jobs_total: 6,
    recommendation: null,
    aggregated: [],
    pareto: [],
    per_job: [],
    ...over,
  };
}

function runBody(over: Partial<Run> = {}): Run {
  return {
    run_id: "adult-done",
    dataset_id: "adult",
    generators: ["tvae"],
    seeds: [0],
    preset: "balanced",
    weights: [1 / 3, 1 / 3, 1 / 3],
    jobs: [{ job_id: "tvae__s0", generator: "tvae", seed: 0, state: "done", error: null }],
    counts: { done: 1 },
    done: true,
    stale: false,
    ...over,
  };
}

const rowOrder = (table: HTMLElement) =>
  within(table)
    .getAllByRole("row")
    .slice(1)
    .map((r) => r.getAttribute("data-generator"))
    .filter(Boolean);

describe("Run detail: finished run", () => {
  it("shows the winner, reason, notes and the header count", async () => {
    renderPage();
    expect(await screen.findByText("10 of 10 jobs reported")).toBeInTheDocument();
    const card = await screen.findByRole("region", { name: "Recommendation" });
    expect(within(card).getByText("tvae")).toBeInTheDocument();
    expect(within(card).getByText(/highest balanced score \(0\.81\)/)).toBeInTheDocument();
    expect(within(card).getByText(/singling-out not attackable \(safeguards passed\)/)).toBeInTheDocument();
  });

  it("renders the scorecard with gate icon+text, failure detail, a score for gate-failing rows and the pareto marker", async () => {
    renderPage();
    const table = await screen.findByRole("table", { name: "Scorecard" });
    const train = within(table).getByRole("row", { name: /^train_copy/ });
    expect(train).toHaveAttribute("data-gate", "fail");
    expect(within(train).getByText(/Fail/)).toBeInTheDocument();
    expect(within(train).getByText("failed the singling-out check (lower bound 0.98 > 0.10)")).toBeInTheDocument();
    expect(within(train).getByText("0.673")).toBeInTheDocument(); // decision.py scores gate-failing candidates too
    expect(within(train).getByText("gate failed")).toBeInTheDocument();
    expect(within(train).queryByLabelText("On Pareto front")).not.toBeInTheDocument();
    const tvae = within(table).getByRole("row", { name: /^tvae/ });
    expect(tvae).toHaveAttribute("data-gate", "pass");
    expect(within(tvae).getByText(/Pass/)).toBeInTheDocument();
    expect(within(tvae).getByLabelText("On Pareto front")).toBeInTheDocument();
    expect(within(table).getByRole("row", { name: /^independent_marginals/ })).not.toHaveTextContent(
      "On Pareto front",
    );
  });

  it("expands a row to the full failure list and per-metric mean ± std", async () => {
    const user = userEvent.setup();
    renderPage();
    const table = await screen.findByRole("table", { name: "Scorecard" });
    const train = within(table).getByRole("row", { name: /^train_copy/ });
    expect(screen.queryByText("failed the DCR check (share 1.00 > 0.60)")).not.toBeInTheDocument();
    await user.click(within(train).getByRole("button", { name: /Details for train_copy/ }));
    expect(screen.getByText("failed the DCR check (share 1.00 > 0.60)")).toBeInTheDocument();
    expect(screen.getByText("fidelity.quality_score")).toBeInTheDocument();
    expect(screen.getByText("utility.tstr.logreg")).toBeInTheDocument();
    expect(screen.getAllByText("0.990 ± 0.010").length).toBeGreaterThan(0);
  });

  it("sorts by weighted score, keeping gate-passing candidates ahead of gate-failing ones", async () => {
    const user = userEvent.setup();
    renderPage();
    const table = await screen.findByRole("table", { name: "Scorecard" });
    expect(rowOrder(table)).toEqual(["tvae", "gaussian_copula", "mst_eps1", "independent_marginals", "train_copy"]);
    await user.click(within(table).getByRole("button", { name: /Weighted score, sort/ }));
    expect(rowOrder(table)).toEqual(["independent_marginals", "mst_eps1", "gaussian_copula", "tvae", "train_copy"]);
    await user.click(within(table).getByRole("button", { name: /Weighted score, sort/ }));
    expect(rowOrder(table)).toEqual(["tvae", "gaussian_copula", "mst_eps1", "independent_marginals", "train_copy"]);
  });

  it("renders the weighted-score chart with an accessible summary", async () => {
    renderPage();
    expect(await screen.findByRole("img", { name: /Weighted scores/ })).toBeInTheDocument();
  });
});

describe("Run detail: report states", () => {
  it("labels a partial report", async () => {
    server.use(
      http.get(`${API}/runs/adult-done/report`, () =>
        HttpResponse.json(emptyReport({ jobs_reported: 2, jobs_total: 6 })),
      ),
    );
    renderPage();
    expect(await screen.findByText("2 of 6 jobs reported")).toBeInTheDocument();
  });

  it("shows a partial recommendation while the run is live, with every job's current state", async () => {
    nav.id = "adult-live";
    freezeRun("adult-live", 6); // gaussian_copula and ctgan evaluated, tvae still evaluating
    renderPage();
    expect(await screen.findByText("2 of 3 jobs reported")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Recommendation" })).toBeInTheDocument();
    const perJob = screen.getByRole("table", { name: "Per-job details" });
    expect(within(perJob).getAllByRole("row").slice(1)).toHaveLength(3);
    expect(within(perJob).getAllByText("Done")).toHaveLength(2);
    expect(within(perJob).getByText("Evaluating")).toBeInTheDocument();
    expect(within(perJob).queryByText("Failed")).not.toBeInTheDocument();
    expect(intervals.size).toBeGreaterThan(0); // still live
  });

  it("renders unknown and in-progress per-job states with status badges, not as failures", async () => {
    server.use(
      http.get(`${API}/runs/adult-done/report`, () =>
        HttpResponse.json(
          emptyReport({
            jobs_reported: 1,
            jobs_total: 3,
            recommendation: recommendation("balanced", [1 / 3, 1 / 3, 1 / 3], [DONE_CANDIDATES[3]]),
            per_job: [
              { job_id: "ctgan__s0", generator: "ctgan", seed: 0, status: "fitting", warnings: [] },
              { job_id: "tvae__s0", generator: "tvae", seed: 0, status: "done", warnings: [] },
              { job_id: "tvae__s1", generator: "tvae", seed: 1, status: "unknown", warnings: [] },
            ],
          }),
        ),
      ),
    );
    renderPage();
    const perJob = await screen.findByRole("table", { name: "Per-job details" });
    expect(within(perJob).getByText("Fitting")).toBeInTheDocument();
    expect(within(perJob).getByText("Unknown")).toBeInTheDocument();
    expect(within(perJob).getByText("Done")).toBeInTheDocument();
    expect(within(perJob).queryByText("Failed")).not.toBeInTheDocument();
  });

  it("shows the no-winner state with the reason", async () => {
    const reason = "No generator passed the privacy gate. train_copy failed the singling-out check (lower bound 0.98 > 0.10).";
    server.use(
      http.get(`${API}/runs/adult-done/report`, () =>
        HttpResponse.json(
          emptyReport({
            jobs_reported: 2,
            jobs_total: 2,
            recommendation: {
              preset: "balanced",
              weights: [1 / 3, 1 / 3, 1 / 3],
              winner: null,
              reason,
              ranking: rank([1 / 3, 1 / 3, 1 / 3], [DONE_CANDIDATES[0]]),
            },
          }),
        ),
      ),
    );
    renderPage();
    expect(await screen.findByText("No generator passed the privacy gate")).toBeInTheDocument();
    const card = screen.getByRole("region", { name: "Recommendation" });
    expect(within(card).getByText(reason)).toBeInTheDocument();
  });

  it("renders null axis scores as an em dash", async () => {
    server.use(
      http.get(`${API}/runs/adult-done/report`, () =>
        HttpResponse.json(
          emptyReport({
            jobs_reported: 1,
            jobs_total: 1,
            recommendation: {
              preset: "balanced",
              weights: [1 / 3, 1 / 3, 1 / 3],
              winner: "tvae",
              reason: "ok",
              ranking: [
                {
                  generator: "tvae",
                  scores: { fidelity: null, utility: 0.5, privacy: null },
                  score: null,
                  gate_passed: true,
                  gate_failures: [],
                  on_pareto_front: false,
                  notes: [],
                },
              ],
            },
          }),
        ),
      ),
    );
    renderPage();
    const row = await screen.findByRole("row", { name: /^tvae/ });
    expect(within(row).getAllByText("—")).toHaveLength(3);
    expect(within(row).getByText("0.500")).toBeInTheDocument();
  });

  it("flags audit generators and GPU-only ones", async () => {
    server.use(
      http.get(`${API}/runs/adult-done/report`, () =>
        HttpResponse.json(
          emptyReport({
            jobs_reported: 1,
            jobs_total: 1,
            recommendation: {
              preset: "balanced",
              weights: [1 / 3, 1 / 3, 1 / 3],
              winner: "privbayes_eps1",
              reason: "ok",
              ranking: rank([1, 1, 1], [
                { generator: "privbayes_eps1", axes: [0.5, 0.5, 0.5], gate_failures: [] },
                { generator: "tabddpm", axes: [0.6, 0.6, 0.6], gate_failures: [] },
              ]),
            },
          }),
        ),
      ),
    );
    renderPage();
    const audit = await screen.findByRole("row", { name: /^privbayes_eps1/ });
    expect(audit).toHaveAttribute("data-audit", "true");
    expect(await within(audit).findByText("audit — not DP")).toBeInTheDocument();
    expect(await within(screen.getByRole("row", { name: /^tabddpm/ })).findByText("GPU")).toBeInTheDocument();
  });

  it("shows an error state when the report store is unavailable (503)", async () => {
    server.use(
      http.get(`${API}/runs/adult-done/report`, () =>
        HttpResponse.json({ detail: "artifact store unavailable: boom" }, { status: 503 }),
      ),
    );
    renderPage();
    expect(await screen.findByText(/artifact store unavailable: boom/)).toBeInTheDocument();
    // the run itself still renders
    expect(screen.getAllByText("gaussian_copula").length).toBeGreaterThan(0);
  });
});

describe("Run detail: presets", () => {
  function captureReports(run: Partial<Run>) {
    const urls: string[] = [];
    server.use(
      http.get(`${API}/runs/adult-done/report`, ({ request }) => {
        urls.push(new URL(request.url).search);
        return HttpResponse.json(emptyReport());
      }),
      http.get(`${API}/runs/adult-done`, () => HttpResponse.json(runBody(run))),
    );
    return urls;
  }

  it("refetches only the report with the chosen preset", async () => {
    const user = userEvent.setup();
    const reportUrls = captureReports({});
    let runCalls = 0;
    server.events.on("request:start", ({ request }) => {
      if (new URL(request.url).pathname === "/runs/adult-done") runCalls += 1;
    });
    try {
      renderPage();
      await screen.findByRole("tab", { name: "Privacy first" });
      await waitFor(() => expect(reportUrls.length).toBe(1));
      expect(reportUrls[0]).toContain("preset=balanced");
      const before = runCalls;
      await user.click(screen.getByRole("tab", { name: "Privacy first" }));
      await waitFor(() => expect(reportUrls.some((u) => u.includes("preset=privacy_first"))).toBe(true));
      expect(runCalls).toBe(before);
    } finally {
      server.events.removeAllListeners("request:start");
    }
  });

  it("starts from the run's own preset", async () => {
    const urls = captureReports({ preset: "privacy_first", weights: [0.2, 0.2, 0.6] });
    renderPage();
    await waitFor(() => expect(urls.length).toBeGreaterThan(0));
    expect(urls[0]).toContain("preset=privacy_first");
    expect(screen.getByRole("tab", { name: "Privacy first" })).toHaveAttribute("aria-selected", "true");
  });

  it("starts from the run's custom weights on the Custom tab", async () => {
    const urls = captureReports({ preset: "custom", weights: [0.5, 0.25, 0.25] });
    renderPage();
    await waitFor(() => expect(urls.length).toBeGreaterThan(0));
    expect(new URLSearchParams(urls[0]).get("weights")).toBe("0.5,0.25,0.25");
    expect(screen.getByRole("tab", { name: "Custom" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText("Fidelity weight")).toHaveValue("0.5");
    expect(screen.queryByText(/custom weights that aren.t available/)).not.toBeInTheDocument();
  });

  it("explains when the run's custom weights are not available", async () => {
    const urls = captureReports({ preset: "custom", weights: null });
    renderPage();
    expect(await screen.findByText("This run used custom weights that aren't available.", { exact: false })).toBeInTheDocument();
    await waitFor(() => expect(urls.length).toBeGreaterThan(0));
    expect(urls[0]).toContain("preset=balanced");
  });

  it("applies custom weights via the weights query param", async () => {
    const user = userEvent.setup();
    const urls: string[] = [];
    server.use(
      http.get(`${API}/runs/adult-done/report`, ({ request }) => {
        urls.push(new URL(request.url).search);
        return HttpResponse.json(emptyReport());
      }),
    );
    renderPage();
    await user.click(await screen.findByRole("tab", { name: "Custom" }));
    fireEvent.change(screen.getByLabelText("Fidelity weight"), { target: { value: "0.8" } });
    await user.click(screen.getByRole("button", { name: "Apply weights" }));
    await waitFor(() => expect(urls.some((u) => u.includes("weights="))).toBe(true));
    const q = new URLSearchParams(urls.find((u) => u.includes("weights="))!);
    expect(q.get("weights")).toBe("0.8,0.3,0.3");
    expect(q.get("preset")).toBeNull();
  });

  it("blocks invalid custom weights", async () => {
    const user = userEvent.setup();
    renderPage();
    await user.click(await screen.findByRole("tab", { name: "Custom" }));
    for (const label of ["Fidelity weight", "Utility weight", "Privacy weight"]) {
      fireEvent.change(screen.getByLabelText(label), { target: { value: "0" } });
    }
    expect(screen.getByRole("alert")).toHaveTextContent(/positive sum/);
    expect(screen.getByRole("button", { name: "Apply weights" })).toBeDisabled();
  });
});

describe("Run detail: live progress", () => {
  it("polls while not done, then stops polling and refetches the report once on completion", async () => {
    nav.id = "adult-live";
    const log: string[] = [];
    const onStart = ({ request }: { request: Request }) => {
      const p = new URL(request.url).pathname;
      if (p === "/runs/adult-live") log.push("run");
      else if (p === "/runs/adult-live/report") log.push("report");
    };
    server.events.on("request:start", onStart);
    try {
      renderPage();
      expect(await screen.findByRole("progressbar", { name: /Run progress/ })).toBeInTheDocument();
      expect(intervals.size).toBeGreaterThan(0); // live: polling is scheduled
      await waitFor(() => {
        poll();
        expect(screen.getByText("3 of 3 jobs reported")).toBeInTheDocument();
      });
      expect(screen.getByRole("progressbar", { name: /Run progress/ })).toHaveAttribute("aria-valuenow", "100");
      // both the run and the report query cancelled their intervals: nothing can poll any more
      await waitFor(() => expect(intervals.size).toBe(0));
      expect(log.filter((x) => x === "run").length).toBeGreaterThan(3);
      const lastRun = log.lastIndexOf("run");
      expect(log.slice(lastRun + 1).filter((x) => x === "report").length).toBeGreaterThanOrEqual(1);
    } finally {
      server.events.removeListener("request:start", onStart);
    }
  });

  it("keeps content and shows a banner when a poll fails, then recovers", async () => {
    let calls = 0;
    const body = (state: "fitting" | "done") =>
      runBody({
        jobs: [{ job_id: "tvae__s0", generator: "tvae", seed: 0, state, error: null }],
        counts: { [state]: 1 },
        done: state === "done",
      });
    server.use(
      http.get(`${API}/runs/adult-done`, () => {
        calls += 1;
        if (calls === 2) return HttpResponse.json({ detail: "store hiccup" }, { status: 503 });
        return HttpResponse.json(body(calls < 4 ? "fitting" : "done"));
      }),
    );
    renderPage();
    await screen.findByRole("progressbar", { name: /Run progress/ });
    poll(); // second call fails
    expect(await screen.findByText(/Couldn.t refresh/)).toBeInTheDocument();
    expect(screen.getByText(/store hiccup/)).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: /Run progress/ })).toBeInTheDocument(); // content stays
    expect(screen.queryByText("Could not load this run")).not.toBeInTheDocument();
    expect(intervals.size).toBeGreaterThan(0); // still polling after the error
    await waitFor(() => {
      poll();
      expect(screen.getByRole("progressbar", { name: /Run progress/ })).toHaveAttribute("aria-valuenow", "100");
    });
    expect(screen.queryByText(/Couldn.t refresh/)).not.toBeInTheDocument();
  });

  it("does not poll a finished run", async () => {
    server.use(http.get(`${API}/runs/adult-done`, () => HttpResponse.json(runBody())));
    renderPage();
    await screen.findByText(/jobs reported/);
    expect(intervals.size).toBe(0);
  });

  it("keeps polling an unfinished run", async () => {
    let runCalls = 0;
    server.use(
      http.get(`${API}/runs/adult-done`, () => {
        runCalls += 1;
        return HttpResponse.json(
          runBody({
            jobs: [{ job_id: "tvae__s0", generator: "tvae", seed: 0, state: "fitting", error: null }],
            counts: { fitting: 1 },
            done: false,
          }),
        );
      }),
    );
    renderPage();
    await waitFor(() => {
      poll();
      expect(runCalls).toBeGreaterThanOrEqual(3);
    });
    expect(intervals.size).toBeGreaterThan(0);
  });

  it("shows the stale notice", async () => {
    server.use(
      http.get(`${API}/runs/adult-done`, () =>
        HttpResponse.json(
          runBody({
            jobs: [{ job_id: "tvae__s0", generator: "tvae", seed: 0, state: "fitting", error: null }],
            counts: { fitting: 1 },
            done: false,
            stale: true,
          }),
        ),
      ),
    );
    renderPage();
    expect(await screen.findByText(/status may be out of date/i)).toBeInTheDocument();
  });

  it("shows an error for an unknown run", async () => {
    nav.id = "nope";
    renderPage();
    expect(await screen.findByText(/run 'nope' not found/)).toBeInTheDocument();
  });
});

describe("Run detail: failed jobs and retry", () => {
  beforeEach(() => {
    seedFailedRun();
    nav.id = "adult-partial";
  });

  it("shows failed jobs with the last traceback line, the full traceback on demand, and progress counting failures", async () => {
    const user = userEvent.setup();
    renderPage();
    const row = await screen.findByRole("listitem", { name: /tvae seed 0/ });
    expect(within(row).getByText("Failed")).toBeInTheDocument();
    expect(within(row).getByText(/^torch\.OutOfMemoryError: CUDA out of memory/)).toBeInTheDocument();
    const details = row.querySelector("details")!;
    expect(details).not.toHaveAttribute("open");
    await user.click(within(row).getByText("Full error"));
    expect(details).toHaveAttribute("open");
    expect(within(row).getByText(/Traceback \(most recent call last\)/)).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: /Run progress/ })).toHaveAttribute("aria-valuenow", "100");
    expect(await screen.findByText("1 of 2 jobs reported")).toBeInTheDocument();
  });

  it("retries failed jobs", async () => {
    const user = userEvent.setup();
    let called = 0;
    server.use(
      http.post(`${API}/runs/adult-partial/retry`, () => {
        called += 1;
        return HttpResponse.json({ run_id: "adult-partial", retried: ["tvae__s0"] });
      }),
    );
    renderPage();
    await user.click(await screen.findByRole("button", { name: /Retry failed jobs/ }));
    await waitFor(() => expect(called).toBe(1));
    expect(await screen.findByText(/Retried 1 job/)).toBeInTheDocument();
  });

  it("surfaces the 503 retry body", async () => {
    const user = userEvent.setup();
    server.use(
      http.post(`${API}/runs/adult-partial/retry`, () =>
        HttpResponse.json(
          { detail: "run adult-partial: no retried job could be dispatched: broker down", run_id: "adult-partial", retried: ["tvae__s0"] },
          { status: 503 },
        ),
      ),
    );
    renderPage();
    await user.click(await screen.findByRole("button", { name: /Retry failed jobs/ }));
    expect(await screen.findByText(/broker down/)).toBeInTheDocument();
    expect(screen.getByText(/1 job was re-queued/)).toBeInTheDocument();
  });

  it("hides retry when nothing failed", async () => {
    nav.id = "adult-done";
    renderPage();
    await screen.findByText("10 of 10 jobs reported");
    expect(screen.queryByRole("button", { name: /Retry failed jobs/ })).not.toBeInTheDocument();
  });
});

describe("Runs list page", () => {
  it("lists runs with links, counts and status", async () => {
    seedFailedRun();
    renderUi(<RunsPage />);
    const link = await screen.findByRole("link", { name: "adult-done" });
    expect(link).toHaveAttribute("href", "/runs/adult-done");
    expect(screen.getByRole("link", { name: "adult-live" })).toBeInTheDocument();
    const partial = screen.getByRole("link", { name: "adult-partial" }).closest("tr")!;
    expect(within(partial).getByText("Failed")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /New run/ })).toHaveAttribute("href", "/runs/new");
  });

  it("encodes run ids in links and flags stale rows", async () => {
    const { jobs, weights, ...summary } = runBody({ run_id: "odd id#1", stale: true, done: false, counts: { fitting: 1 } });
    void jobs;
    void weights;
    server.use(http.get(`${API}/runs`, () => HttpResponse.json([summary])));
    renderUi(<RunsPage />);
    const link = await screen.findByRole("link", { name: "odd id#1" });
    expect(link).toHaveAttribute("href", "/runs/odd%20id%231");
    expect(within(link.closest("tr")!).getByText(/may be out of date/i)).toBeInTheDocument();
  });

  it("shows an empty state", async () => {
    server.use(http.get(`${API}/runs`, () => HttpResponse.json([])));
    renderUi(<RunsPage />);
    expect(await screen.findByText(/No runs yet/)).toBeInTheDocument();
  });

  it("shows an error state", async () => {
    server.use(http.get(`${API}/runs`, () => HttpResponse.json({ detail: "db down" }, { status: 500 })));
    renderUi(<RunsPage />);
    expect(await screen.findByText(/db down/)).toBeInTheDocument();
  });
});
