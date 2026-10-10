import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { seedImport } from "@/mocks/handlers";
import { server } from "@/mocks/server";

const toasts = vi.hoisted(() => ({ success: vi.fn(), warning: vi.fn(), error: vi.fn() }));
vi.mock("sonner", () => ({ toast: Object.assign(vi.fn(), toasts) }));

const nav = vi.hoisted(() => ({ push: vi.fn(), params: { id: "uploaded-1" } }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: nav.push }),
  useParams: () => nav.params,
  usePathname: () => "/datasets",
}));

import DatasetDetail from "@/app/datasets/[id]/page";
import Datasets from "@/app/datasets/page";
import { parseCsvHeader } from "@/lib/csv";

const API = "http://api.test";
const registeredFlag = { value: false };

function renderWith(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

afterEach(() => {
  vi.restoreAllMocks();
});

beforeEach(() => {
  nav.push.mockReset();
  for (const f of Object.values(toasts)) f.mockReset();
  nav.params = { id: "uploaded-1" };
  registeredFlag.value = false;
});

describe("parseCsvHeader", () => {
  it("handles BOM, quotes, CRLF and escaped quotes", () => {
    expect(parseCsvHeader('﻿a,"b,c","d ""q"""\r\n1,2,3\r\n')).toEqual(["a", "b,c", 'd "q"']);
    expect(parseCsvHeader("x,y")).toEqual(["x", "y"]);
    expect(parseCsvHeader(" a , b ,")).toEqual([" a ", " b ", ""]); // exactly what pandas sees
    expect(parseCsvHeader("")).toEqual([]);
  });
});

describe("Datasets list page", () => {
  it("lists datasets with source, rows, target and registered badge", async () => {
    renderWith(<Datasets />);
    const adult = (await screen.findByRole("link", { name: "Adult" })).closest("tr")!;
    expect(adult).toHaveAttribute("data-dataset", "adult");
    expect(within(adult).getByText("Built-in")).toBeInTheDocument();
    expect(within(adult).getByText("20,000")).toBeInTheDocument();
    expect(within(adult).getByText("income")).toBeInTheDocument();
    expect(within(adult).getByText("Registered")).toBeInTheDocument();
    const up = screen.getByRole("link", { name: "Uploaded sample" }).closest("tr")!;
    expect(within(up).getByText("Upload")).toBeInTheDocument();
    expect(within(up).getByText("Not registered")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Adult" })).toHaveAttribute("href", "/datasets/adult");
  });

  it("shows an error when listing fails", async () => {
    server.use(http.get(`${API}/datasets`, () => HttpResponse.json({ detail: "db down" }, { status: 500 })));
    renderWith(<Datasets />);
    expect(await screen.findByRole("alert")).toHaveTextContent("db down");
  });

  it("uploads a CSV: header-derived target, multipart fields, navigates to the detail page", async () => {
    // jsdom Files cannot be streamed by Node fetch, so capture the multipart FormData at the fetch boundary
    // (real multipart parsing is exercised by api.test.ts under the node environment).
    let seen: { url: string; method?: string; form: FormData } | null = null;
    const realFetch = globalThis.fetch;
    const spy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (init?.method === "POST" && url === `${API}/datasets`) {
        seen = { url, method: init.method, form: init.body as FormData };
        return Response.json(
          { id: "tiny-abc123", name: "Tiny", target: "b", n_rows: 2, metadata: { columns: {}, target: "b", task: "x" }, registered: false },
          { status: 201 },
        );
      }
      return realFetch(input, init);
    });
    const user = userEvent.setup();
    renderWith(<Datasets />);
    await user.click(await screen.findByRole("button", { name: /upload csv\/parquet/i }));
    const dialog = await screen.findByRole("dialog");
    const file = new File(["a,b,c\n1,2,3\n"], "tiny.csv", { type: "text/csv" });
    await user.upload(within(dialog).getByLabelText("File"), file);
    const target = await within(dialog).findByLabelText("Target column");
    await waitFor(() =>
      expect(within(target).queryAllByRole("option").map((o) => o.textContent)).toEqual(["a", "b", "c"]),
    );
    expect(within(dialog).getByLabelText("Name")).toHaveValue("tiny"); // defaulted from file name
    await user.clear(within(dialog).getByLabelText("Name"));
    await user.type(within(dialog).getByLabelText("Name"), "Tiny");
    await user.selectOptions(target, "b");
    await user.click(within(dialog).getByRole("button", { name: "Upload" }));
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/datasets/tiny-abc123"));
    expect(spy).toHaveBeenCalled();
    expect(seen).not.toBeNull();
    expect(seen!.form).toBeInstanceOf(FormData);
    expect(seen!.form.get("name")).toBe("Tiny");
    expect(seen!.form.get("target")).toBe("b");
    expect((seen!.form.get("file") as File).name).toBe("tiny.csv");
  });

  it("uses a free-text target for Parquet files", async () => {
    const user = userEvent.setup();
    renderWith(<Datasets />);
    await user.click(await screen.findByRole("button", { name: /upload csv\/parquet/i }));
    const dialog = await screen.findByRole("dialog");
    await user.upload(
      within(dialog).getByLabelText("File"),
      new File([new Uint8Array([80, 65, 82, 49])], "x.parquet", { type: "application/octet-stream" }),
    );
    const target = await within(dialog).findByLabelText("Target column");
    expect(target.tagName).toBe("INPUT");
    await user.type(target, "label");
    expect(target).toHaveValue("label");
  });

  it("shows a 422 parse/target error verbatim and keeps the dialog open", async () => {
    server.use(
      http.post(`${API}/datasets`, () =>
        HttpResponse.json({ detail: "target 'b' has only one class" }, { status: 422 }),
      ),
    );
    const user = userEvent.setup();
    renderWith(<Datasets />);
    await user.click(await screen.findByRole("button", { name: /upload csv\/parquet/i }));
    const dialog = await screen.findByRole("dialog");
    await user.upload(within(dialog).getByLabelText("File"), new File(["a,b\n1,2\n"], "t.csv", { type: "text/csv" }));
    await within(dialog).findByLabelText("Target column");
    await user.click(within(dialog).getByRole("button", { name: "Upload" }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("target 'b' has only one class");
    expect(nav.push).not.toHaveBeenCalled();
  });

  it("shows the 413 message verbatim", async () => {
    server.use(
      http.post(`${API}/datasets`, () =>
        HttpResponse.json({ detail: "upload exceeds STL_MAX_UPLOAD_MB=50" }, { status: 413 }),
      ),
    );
    const user = userEvent.setup();
    renderWith(<Datasets />);
    await user.click(await screen.findByRole("button", { name: /upload csv\/parquet/i }));
    const dialog = await screen.findByRole("dialog");
    await user.upload(within(dialog).getByLabelText("File"), new File(["a,b\n1,2\n"], "t.csv", { type: "text/csv" }));
    await within(dialog).findByLabelText("Target column");
    await user.click(within(dialog).getByRole("button", { name: "Upload" }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("upload exceeds STL_MAX_UPLOAD_MB=50");
  });

  it("offers the missing built-ins; loading one posts to /datasets/builtin/{name}", async () => {
    const user = userEvent.setup();
    renderWith(<Datasets />);
    await screen.findByRole("link", { name: "Adult" });
    for (const label of ["Credit default", "Diabetes 130-US"]) {
      expect(screen.getByRole("button", { name: `Load built-in: ${label}` })).toBeInTheDocument();
    }
    await user.click(screen.getByRole("button", { name: "Load built-in: Credit default" }));
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/datasets/credit"));
  });

  it("links to a built-in that is already loaded instead of re-loading it (which would 409 once used)", async () => {
    renderWith(<Datasets />);
    await screen.findByRole("link", { name: "Adult" });
    expect(screen.queryByRole("button", { name: "Load built-in: Adult" })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open built-in: Adult" })).toHaveAttribute("href", "/datasets/adult");
  });

  it("shows a built-in 409 verbatim and disables that built-in", async () => {
    const msg = "dataset credit is used by 1 run(s); create a new dataset instead";
    server.use(http.post(`${API}/datasets/builtin/credit`, () => HttpResponse.json({ detail: msg }, { status: 409 })));
    const user = userEvent.setup();
    renderWith(<Datasets />);
    await user.click(await screen.findByRole("button", { name: "Load built-in: Credit default" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(msg);
    expect(screen.getByRole("button", { name: "Load built-in: Credit default" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Load built-in: Diabetes 130-US" })).toBeEnabled();
  });

  it("labels an import placeholder as Imported and still offers to load the matching built-in", async () => {
    seedImport();
    renderWith(<Datasets />);
    const row = (await screen.findByRole("link", { name: "diabetes" })).closest("tr")!;
    expect(within(row).getByText("Imported")).toBeInTheDocument();
    expect(within(row).getByText("—")).toBeInTheDocument(); // n_rows null
    expect(screen.getByRole("button", { name: "Load built-in: Diabetes 130-US" })).toBeEnabled();
  });

  it("shows promotion warnings after loading a built-in over an import placeholder", async () => {
    seedImport();
    const user = userEvent.setup();
    renderWith(<Datasets />);
    await user.click(await screen.findByRole("button", { name: "Load built-in: Diabetes 130-US" }));
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/datasets/diabetes"));
    expect(toasts.warning).toHaveBeenCalledWith(
      "fingerprint differs from imported run(s) of dataset diabetes; results not comparable",
    );
  });

  it("blocks uploads over the size limit before sending", async () => {
    const user = userEvent.setup();
    const spy = vi.spyOn(globalThis, "fetch");
    renderWith(<Datasets />);
    await user.click(await screen.findByRole("button", { name: /upload csv\/parquet/i }));
    const dialog = await screen.findByRole("dialog");
    const big = new File(["a,b\n1,2\n"], "big.csv", { type: "text/csv" });
    Object.defineProperty(big, "size", { value: 201 * 1024 * 1024 });
    await user.upload(within(dialog).getByLabelText("File"), big);
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("larger than the server's upload limit (200 MB)");
    expect(within(dialog).getByRole("button", { name: "Upload" })).toBeDisabled();
    expect(spy.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(0);
  });

  it("hints at the size limit when a large upload fails at the network level", async () => {
    server.use(http.post(`${API}/datasets`, () => HttpResponse.error()));
    const user = userEvent.setup();
    renderWith(<Datasets />);
    await user.click(await screen.findByRole("button", { name: /upload csv\/parquet/i }));
    const dialog = await screen.findByRole("dialog");
    const large = new File(["a,b\n1,2\n"], "large.csv", { type: "text/csv" });
    Object.defineProperty(large, "size", { value: 60 * 1024 * 1024 });
    await user.upload(within(dialog).getByLabelText("File"), large);
    await within(dialog).findByLabelText("Target column");
    await user.click(within(dialog).getByRole("button", { name: "Upload" }));
    const alert = await within(dialog).findByRole("alert");
    expect(alert).toHaveTextContent("Cannot reach the API, or it rejected the request");
    expect(alert).toHaveTextContent("A large upload may have exceeded the server's size limit");
  });
});

describe("Dataset detail page: imported datasets", () => {
  beforeEach(() => {
    seedImport();
    nav.params = { id: "diabetes" };
  });

  it("shows a read-only view without metadata editor or register panel", async () => {
    renderWith(<DatasetDetail />);
    expect(await screen.findByRole("heading", { name: "diabetes" })).toBeInTheDocument();
    expect(
      screen.getByText("This dataset came from an imported results bundle; its rows are not stored here."),
    ).toBeInTheDocument();
    expect(screen.getByText(/Imported/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save metadata" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /register/i })).not.toBeInTheDocument();
  });

  it("promotes a built-in id with Load built-in and keeps the warnings on the page", async () => {
    const user = userEvent.setup();
    renderWith(<DatasetDetail />);
    await user.click(await screen.findByRole("button", { name: "Load built-in" }));
    const msg = "fingerprint differs from imported run(s) of dataset diabetes; results not comparable";
    expect(await screen.findByText(msg)).toBeInTheDocument();
    expect(toasts.warning).toHaveBeenCalledWith(msg);
    // the promoted dataset is a normal built-in now
    expect(await screen.findByRole("button", { name: "Save metadata" })).toBeInTheDocument();
    expect(screen.getByText(msg)).toBeInTheDocument();
  });

  it("does not offer Load built-in for an id that is not a built-in", async () => {
    server.use(
      http.get(`${API}/datasets/my-data`, () =>
        HttpResponse.json({
          id: "my-data", name: "my-data", source: "import", target: "", n_rows: null, metadata: {},
          registered: false, fingerprint: "sha256:x", preview: [],
        }),
      ),
    );
    nav.params = { id: "my-data" };
    renderWith(<DatasetDetail />);
    await screen.findByRole("heading", { name: "my-data" });
    expect(screen.getByText(/imported results bundle/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Load built-in" })).not.toBeInTheDocument();
  });

  it("renders a dataset with missing column metadata read-only instead of crashing", async () => {
    nav.params = { id: "weird" };
    server.use(
      http.get(`${API}/datasets/weird`, () =>
        HttpResponse.json({
          id: "weird", name: "Weird", source: "upload", target: "y", n_rows: 3, metadata: {},
          registered: false, fingerprint: null, preview: [],
        }),
      ),
    );
    renderWith(<DatasetDetail />);
    expect(await screen.findByRole("heading", { name: "Weird" })).toBeInTheDocument();
    expect(screen.getByText(/no column metadata/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save metadata" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Load built-in" })).not.toBeInTheDocument();
  });

  it("shows register promotion warnings persistently", async () => {
    nav.params = { id: "uploaded-1" };
    const msg = "fingerprint differs from imported run(s) of dataset uploaded-1; results not comparable";
    server.use(
      http.post(`${API}/datasets/uploaded-1/register`, () =>
        HttpResponse.json({
          id: "uploaded-1", name: "Uploaded sample", source: "upload", target: "label", n_rows: 1200,
          metadata: { columns: { label: "categorical" }, target: "label", task: "classification" },
          registered: true, fingerprint: "sha256:abc", warnings: [msg],
        }),
      ),
    );
    const user = userEvent.setup();
    renderWith(<DatasetDetail />);
    await user.click(await screen.findByRole("button", { name: "Register" }));
    expect(await screen.findByText(msg)).toBeInTheDocument();
    expect(toasts.warning).toHaveBeenCalledWith(msg);
  });
});

describe("Dataset detail page", () => {
  it("shows the 5-row preview, metadata editor and task", async () => {
    renderWith(<DatasetDetail />);
    expect(await screen.findByRole("heading", { name: "Uploaded sample" })).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "Data preview" });
    expect(within(table).getAllByRole("row")).toHaveLength(1 + 2); // header + fixture's 2 rows
    expect(within(table).getByText("red")).toBeInTheDocument();
    expect(screen.getByLabelText("Type of column x1")).toHaveValue("numerical");
    expect(screen.getByLabelText("Target column")).toHaveValue("label");
    expect(screen.getByText("classification")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /start a run/i })).not.toBeInTheDocument();
  });

  it("sends the edited metadata in the PUT body", async () => {
    let body: unknown = null;
    server.use(
      http.put(`${API}/datasets/uploaded-1/metadata`, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json({
          id: "uploaded-1", name: "Uploaded sample", target: "label", n_rows: 1200, registered: false,
          metadata: { columns: {}, target: "label", task: "classification" },
        });
      }),
    );
    const user = userEvent.setup();
    renderWith(<DatasetDetail />);
    await screen.findByRole("heading", { name: "Uploaded sample" });
    await user.selectOptions(screen.getByLabelText("Type of column x1"), "categorical");
    await user.click(screen.getByRole("button", { name: "Save metadata" }));
    await waitFor(() => expect(body).not.toBeNull());
    expect(body).toEqual({
      columns: { id: "id", x1: "categorical", x2: "numerical", colour: "categorical", label: "categorical" },
      target: "label",
    });
  });

  it("updates the task live when the target type changes", async () => {
    const user = userEvent.setup();
    renderWith(<DatasetDetail />);
    await screen.findByRole("heading", { name: "Uploaded sample" });
    await user.selectOptions(screen.getByLabelText("Target column"), "x1");
    expect(screen.getByText("regression")).toBeInTheDocument();
  });

  it("shows a 409 on metadata save verbatim", async () => {
    server.use(
      http.put(`${API}/datasets/uploaded-1/metadata`, () =>
        HttpResponse.json(
          { detail: "dataset uploaded-1 is used by 3 run(s); create a new dataset instead" },
          { status: 409 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderWith(<DatasetDetail />);
    await screen.findByRole("heading", { name: "Uploaded sample" });
    await user.selectOptions(screen.getByLabelText("Type of column x1"), "categorical");
    await user.click(screen.getByRole("button", { name: "Save metadata" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "dataset uploaded-1 is used by 3 run(s); create a new dataset instead",
    );
    expect(screen.getByRole("button", { name: "Save metadata" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Register" })).toBeDisabled();
  });

  it("shows a register 409 verbatim and disables Register and Save metadata", async () => {
    const msg = "dataset uploaded-1 is used by 2 run(s); create a new dataset instead";
    server.use(http.post(`${API}/datasets/uploaded-1/register`, () => HttpResponse.json({ detail: msg }, { status: 409 })));
    const user = userEvent.setup();
    renderWith(<DatasetDetail />);
    await screen.findByRole("heading", { name: "Uploaded sample" });
    await user.click(screen.getByRole("button", { name: "Register" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(msg);
    expect(screen.getByRole("button", { name: "Register" })).toBeDisabled();
    await user.selectOptions(screen.getByLabelText("Type of column x1"), "categorical");
    expect(screen.getByRole("button", { name: "Save metadata" })).toBeDisabled();
  });

  it("keeps id columns out of the target select and blocks saving an id target", async () => {
    const user = userEvent.setup();
    renderWith(<DatasetDetail />);
    await screen.findByRole("heading", { name: "Uploaded sample" });
    const target = screen.getByLabelText("Target column");
    expect(within(target).queryAllByRole("option").map((o) => o.textContent)).toEqual(["x1", "x2", "colour", "label"]);
    await user.selectOptions(screen.getByLabelText("Type of column label"), "id");
    expect(await screen.findByRole("alert")).toHaveTextContent(/target column cannot be an id column/i);
    expect(screen.getByRole("button", { name: "Save metadata" })).toBeDisabled();
  });

  it("re-enables Register after reverting unsaved edits", async () => {
    const user = userEvent.setup();
    renderWith(<DatasetDetail />);
    await screen.findByRole("heading", { name: "Uploaded sample" });
    const sel = screen.getByLabelText("Type of column x1");
    await user.selectOptions(sel, "categorical");
    expect(screen.getByRole("button", { name: "Register" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Save metadata" })).toBeEnabled();
    await user.selectOptions(sel, "numerical");
    expect(screen.getByRole("button", { name: "Register" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Save metadata" })).toBeDisabled();
  });

  it("drops the stale Registered state after a failed re-register", async () => {
    let registered = true;
    server.use(
      http.get(`${API}/datasets/uploaded-1`, () =>
        HttpResponse.json({
          id: "uploaded-1", name: "Uploaded sample", source: "upload", target: "label", n_rows: 1200,
          metadata: { columns: { x1: "numerical", label: "categorical" }, target: "label", task: "classification" },
          registered, fingerprint: registered ? "sha256:abcdef0123456789" : null, preview: [],
        }),
      ),
      http.post(`${API}/datasets/uploaded-1/register`, () => {
        registered = false; // the API clears the registration when register fails
        return HttpResponse.json({ detail: "target class 'b' has only 3 rows" }, { status: 422 });
      }),
    );
    const user = userEvent.setup();
    renderWith(<DatasetDetail />);
    expect(await screen.findByText("abcdef012345")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Re-register" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("target class 'b' has only 3 rows");
    await waitFor(() => expect(screen.queryByText("abcdef012345")).not.toBeInTheDocument());
    expect(screen.queryByRole("link", { name: /start a run/i })).not.toBeInTheDocument();
    expect(screen.getAllByText("Not registered").length).toBeGreaterThan(0);
  });

  it("registers with max_rows default 20000, disabled while pending, then shows fingerprint and Start a run", async () => {
    let body: unknown = null;
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    server.use(
      http.post(`${API}/datasets/uploaded-1/register`, async ({ request }) => {
        body = await request.json();
        await gate;
        return HttpResponse.json({
          id: "uploaded-1", name: "Uploaded sample", target: "label", n_rows: 1200, registered: true,
          metadata: { columns: {}, target: "label", task: "classification" },
        });
      }),
      // after registering, the refetched detail carries the fingerprint
      http.get(`${API}/datasets/uploaded-1`, () =>
        HttpResponse.json({
          id: "uploaded-1", name: "Uploaded sample", source: "upload", target: "label", n_rows: 1200,
          metadata: { columns: { x1: "numerical", label: "categorical" }, target: "label", task: "classification" },
          registered: registeredFlag.value, fingerprint: registeredFlag.value ? "sha256:abcdef0123456789ffff" : null,
          preview: [{ x1: 1, label: "a" }],
        }),
      ),
    );
    const user = userEvent.setup();
    renderWith(<DatasetDetail />);
    await screen.findByRole("heading", { name: "Uploaded sample" });
    expect(screen.getByLabelText("Max rows")).toHaveValue(20000);
    await user.click(screen.getByRole("button", { name: "Register" }));
    const busy = await screen.findByRole("button", { name: /registering/i });
    expect(busy).toBeDisabled();
    expect(screen.getByLabelText("Max rows")).toBeDisabled();
    registeredFlag.value = true;
    release();
    expect(await screen.findByText("abcdef012345")).toBeInTheDocument();
    expect(body).toEqual({ max_rows: 20000 });
    expect(screen.getByRole("link", { name: /start a run/i })).toHaveAttribute("href", "/runs/new?dataset=uploaded-1");
  });

  it("shows a registration 422 (rare class) verbatim", async () => {
    const msg = "target class 'b' has only 3 rows; need at least 10";
    server.use(
      http.post(`${API}/datasets/uploaded-1/register`, () => HttpResponse.json({ detail: msg }, { status: 422 })),
    );
    const user = userEvent.setup();
    renderWith(<DatasetDetail />);
    await screen.findByRole("heading", { name: "Uploaded sample" });
    await user.click(screen.getByRole("button", { name: "Register" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(msg);
    expect(screen.getByRole("button", { name: "Register" })).toBeEnabled();
  });

  it("shows fingerprint and Start a run for a registered dataset", async () => {
    nav.params = { id: "adult" };
    renderWith(<DatasetDetail />);
    expect(await screen.findByRole("heading", { name: "Adult" })).toBeInTheDocument();
    expect(screen.getByText("3f9a1c0de2")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /start a run/i })).toHaveAttribute("href", "/runs/new?dataset=adult");
  });

  it("shows a not-found error", async () => {
    nav.params = { id: "nope" };
    renderWith(<DatasetDetail />);
    expect(await screen.findByRole("alert")).toHaveTextContent("dataset 'nope' not found");
  });
});
