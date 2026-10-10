import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/server";
import type { CreateRunBody } from "@/lib/types";

const nav = vi.hoisted(() => ({ push: vi.fn(), search: "" }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: nav.push }),
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => "/runs/new",
}));

import NewRunPage from "@/app/runs/new/page";
import { generatorDescription, GENERATOR_INFO } from "@/lib/generator-info";
import { GENERATORS } from "@/mocks/fixtures";

const API = "http://api.test";

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <NewRunPage />
    </QueryClientProvider>,
  );
}

function captureRun() {
  const seen: { body: CreateRunBody | null } = { body: null };
  server.use(
    http.post(`${API}/runs`, async ({ request }) => {
      seen.body = (await request.json()) as CreateRunBody;
      return HttpResponse.json({ run_id: "adult-20261010-ab12", jobs: [] }, { status: 201 });
    }),
  );
  return seen;
}

beforeEach(() => {
  nav.push.mockReset();
  nav.search = "";
});

describe("generator-info", () => {
  it("describes every catalog generator", () => {
    for (const g of GENERATORS) expect(GENERATOR_INFO[g.name], g.name).toBeTruthy();
    expect(generatorDescription("train_copy")).toMatch(/must fail the gate/);
    expect(generatorDescription("privbayes_eps1")).toMatch(/not actually differentially private/);
  });
});

describe("New run page", () => {
  it("lists only registered datasets and preselects from ?dataset=", async () => {
    nav.search = "dataset=adult";
    renderPage();
    const select = await screen.findByLabelText("Dataset");
    expect(select).toHaveValue("adult");
    expect(screen.getByRole("option", { name: "Adult (adult)" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /Uploaded sample/ })).not.toBeInTheDocument();
  });

  it("ignores a ?dataset= that is not registered", async () => {
    nav.search = "dataset=uploaded-1";
    renderPage();
    expect(await screen.findByLabelText("Dataset")).toHaveValue("");
  });

  it("checks CPU non-audit generators by default; GPU and audit start unchecked", async () => {
    renderPage();
    await screen.findByLabelText("Dataset");
    for (const g of GENERATORS) {
      const box = screen.getByRole("checkbox", { name: new RegExp(`^${g.name.replace(".", "\\.")}\\b`) });
      const expected = g.cpu_ok && !g.audit;
      expect(box, g.name).toHaveProperty("checked", expected);
    }
    expect(screen.getAllByText("GPU").length).toBe(GENERATORS.filter((g) => !g.cpu_ok).length);
    for (const g of GENERATORS.filter((x) => x.audit)) {
      expect(screen.getByRole("checkbox", { name: new RegExp(`^${g.name.replace(".", "\\.")} audit — not DP`) })).toBeInTheDocument();
    }
    // badges are part of the checkbox's accessible name
    expect(screen.getByRole("checkbox", { name: /^tabddpm diffusion\s*GPU$/ })).toBeInTheDocument();
  });

  it("Clear and Select all CPU change the selection", async () => {
    const user = userEvent.setup();
    renderPage();
    await screen.findByLabelText("Dataset");
    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByRole("checkbox", { name: /^ctgan\b/ })).not.toBeChecked();
    await user.click(screen.getByRole("button", { name: "Select all CPU" }));
    expect(screen.getByRole("checkbox", { name: /^ctgan\b/ })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: /^tabddpm\b/ })).not.toBeChecked();
  });

  it("disables submit with reasons until a dataset is chosen", async () => {
    renderPage();
    await screen.findByLabelText("Dataset");
    expect(screen.getByRole("button", { name: "Start benchmark" })).toBeDisabled();
    expect(screen.getByText("Choose a registered dataset.")).toBeInTheDocument();
  });

  it("validates seeds (non-negative, unique, non-empty)", async () => {
    const user = userEvent.setup();
    nav.search = "dataset=adult";
    renderPage();
    const seeds = await screen.findByLabelText("Seeds");
    const submit = screen.getByRole("button", { name: "Start benchmark" });
    expect(submit).toBeEnabled();
    for (const [text, msg] of [
      ["0,0", "Seeds must be unique."],
      ["-1", /non-negative whole numbers/],
      ["1.5", /non-negative whole numbers/],
      ["", "Enter at least one seed."],
    ] as const) {
      await user.clear(seeds);
      if (text) await user.type(seeds, text);
      expect(submit).toBeDisabled();
      expect(screen.getByText(msg)).toBeInTheDocument();
    }
  });

  it("blocks submit on invalid custom weights", async () => {
    const user = userEvent.setup();
    nav.search = "dataset=adult";
    renderPage();
    await screen.findByLabelText("Dataset");
    await user.click(screen.getByRole("radio", { name: /Custom/ }));
    for (const axis of ["Fidelity", "Utility", "Privacy"]) {
      fireEvent.change(screen.getByLabelText(`${axis} weight`), { target: { value: "0" } });
    }
    expect(screen.getByText("Weights must have a positive sum.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start benchmark" })).toBeDisabled();
  });

  it("requires Allow GPU-only when a GPU generator is selected", async () => {
    const user = userEvent.setup();
    nav.search = "dataset=adult";
    renderPage();
    await screen.findByLabelText("Dataset");
    await user.click(screen.getByRole("checkbox", { name: /^tabddpm\b/ }));
    const submit = screen.getByRole("button", { name: "Start benchmark" });
    expect(submit).toBeDisabled();
    await user.click(screen.getByRole("checkbox", { name: "Allow GPU-only generators" }));
    expect(submit).toBeEnabled();
  });

  it("posts a body matching the contract and navigates to the run", async () => {
    const user = userEvent.setup();
    const seen = captureRun();
    nav.search = "dataset=adult";
    renderPage();
    await screen.findByLabelText("Dataset");
    await user.click(screen.getByRole("radio", { name: /Privacy first/ }));
    await user.click(screen.getByRole("button", { name: "Start benchmark" }));
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/adult-20261010-ab12"));
    expect(seen.body).toEqual({
      dataset_id: "adult",
      generators: GENERATORS.filter((g) => g.cpu_ok && !g.audit).map((g) => g.name),
      seeds: [0, 1, 2],
      preset: "privacy_first",
      allow_gpu_only: false,
    });
  });

  it("shows the exploratory generator suggestion and can select it", async () => {
    const user = userEvent.setup();
    const seen = captureRun();
    nav.search = "dataset=adult";
    renderPage();
    await screen.findByLabelText("Dataset");
    expect(await screen.findByText(/Suggested:/)).toHaveTextContent("tvae");
    expect(screen.getByText("25 datasets")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Use suggestion" }));
    await user.click(screen.getByRole("button", { name: "Start benchmark" }));
    await waitFor(() => expect(nav.push).toHaveBeenCalled());
    expect(seen.body?.generators).toEqual(["tvae"]);
  });

  it("sends custom weights as a 3-array", async () => {
    const user = userEvent.setup();
    const seen = captureRun();
    nav.search = "dataset=adult";
    renderPage();
    await screen.findByLabelText("Dataset");
    await user.click(screen.getByRole("radio", { name: /Custom/ }));
    fireEvent.change(screen.getByLabelText("Privacy weight"), { target: { value: "0.5" } });
    await user.click(screen.getByRole("button", { name: "Start benchmark" }));
    await waitFor(() => expect(nav.push).toHaveBeenCalled());
    expect(seen.body?.weights).toEqual([0.4, 0.3, 0.5]);
    expect(seen.body?.preset).toBe("balanced");
  });

  it("on 503 shows the message and a link to the created run", async () => {
    const user = userEvent.setup();
    server.use(
      http.post(`${API}/runs`, () =>
        HttpResponse.json(
          {
            detail: "run adult-20261010-ab12 created but no job could be dispatched: dispatch failed: down",
            run_id: "adult-20261010-ab12",
            jobs: [],
          },
          { status: 503 },
        ),
      ),
    );
    nav.search = "dataset=adult";
    renderPage();
    await screen.findByLabelText("Dataset");
    await user.click(screen.getByRole("button", { name: "Start benchmark" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/no job could be dispatched/);
    expect(screen.getByRole("link", { name: /Open run adult-20261010-ab12/ })).toHaveAttribute(
      "href",
      "/runs/adult-20261010-ab12",
    );
    expect(nav.push).not.toHaveBeenCalled();
  });

  it("shows API validation errors in an alert", async () => {
    const user = userEvent.setup();
    server.use(
      http.post(`${API}/runs`, () =>
        HttpResponse.json(
          { detail: [{ type: "value_error", loc: ["body", "seeds"], msg: "Value error, seeds must be unique" }] },
          { status: 422 },
        ),
      ),
    );
    nav.search = "dataset=adult";
    renderPage();
    await screen.findByLabelText("Dataset");
    await user.click(screen.getByRole("button", { name: "Start benchmark" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("seeds: must be unique");
  });

  it("503 without a run_id in the body falls back to the generic error", async () => {
    const user = userEvent.setup();
    server.use(http.post(`${API}/runs`, () => HttpResponse.json({ detail: "queue down" }, { status: 503 })));
    nav.search = "dataset=adult";
    renderPage();
    await screen.findByLabelText("Dataset");
    await user.click(screen.getByRole("button", { name: "Start benchmark" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("queue down");
    expect(screen.queryByRole("link", { name: /Open run/ })).not.toBeInTheDocument();
  });

  it("409 unregistered dataset shows the message with a link to /datasets", async () => {
    const user = userEvent.setup();
    server.use(
      http.post(`${API}/runs`, () => HttpResponse.json({ detail: "dataset 'adult' is not registered" }, { status: 409 })),
    );
    nav.search = "dataset=adult";
    renderPage();
    await screen.findByLabelText("Dataset");
    await user.click(screen.getByRole("button", { name: "Start benchmark" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("is not registered");
    expect(screen.getByRole("link", { name: /Datasets page/ })).toHaveAttribute("href", "/datasets");
  });

  it("sends allow_gpu_only true with a GPU generator", async () => {
    const user = userEvent.setup();
    const seen = captureRun();
    nav.search = "dataset=adult";
    renderPage();
    await screen.findByLabelText("Dataset");
    await user.click(screen.getByRole("checkbox", { name: /^tabddpm\b/ }));
    await user.click(screen.getByRole("checkbox", { name: "Allow GPU-only generators" }));
    await user.click(screen.getByRole("button", { name: "Start benchmark" }));
    await waitFor(() => expect(nav.push).toHaveBeenCalled());
    expect(seen.body?.allow_gpu_only).toBe(true);
    expect(seen.body?.generators).toContain("tabddpm");
  });
});
