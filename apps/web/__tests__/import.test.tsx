import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/server";
import ImportPage from "@/app/import/page";

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ImportPage />
    </QueryClientProvider>,
  );
}
/** jsdom Files cannot be streamed by Node fetch: capture the FormData at the fetch boundary (real multipart is covered in api.test.ts). */
function captureImport(body: unknown) {
  const seen: FormData[] = [];
  const realFetch = globalThis.fetch;
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    if (init?.method === "POST" && String(input).endsWith("/import-results")) {
      seen.push(init.body as FormData);
      return Response.json(body);
    }
    return realFetch(input, init);
  });
  return seen;
}
const zip = () => new File(["PK"], "bundle.zip", { type: "application/zip" });

describe("Import page", () => {
  afterEach(() => vi.restoreAllMocks());
  it("posts multipart with file and force=false and shows the summary", async () => {
    const seen = captureImport({ run_id: "run-9", imported: 4, warnings: ["manifest older than 1.2"] });
    const user = userEvent.setup();
    renderPage();
    const submit = screen.getByRole("button", { name: "Import" });
    expect(submit).toBeDisabled();
    await user.upload(screen.getByLabelText("Results bundle (.zip)"), zip());
    await user.click(submit);
    expect(await screen.findByText(/Imported 4 job reports/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "run-9" })).toHaveAttribute("href", "/runs/run-9");
    expect(screen.getByText("manifest older than 1.2")).toBeInTheDocument();
    expect(seen).toHaveLength(1);
    expect(seen[0].get("file")).toBeInstanceOf(Blob);
    expect((seen[0].get("file") as File).name).toBe("bundle.zip");
    expect(seen[0].get("force")).toBe("false");
  });

  it("force checkbox shows a warning and sends force=true", async () => {
    const seen = captureImport({ run_id: "r", imported: 1, warnings: [] });
    const user = userEvent.setup();
    renderPage();
    expect(screen.queryByRole("note")).toBeNull();
    await user.click(screen.getByRole("checkbox", { name: /Force/ }));
    expect(screen.getByRole("note")).toHaveTextContent(/overwrite/i);
    await user.upload(screen.getByLabelText("Results bundle (.zip)"), zip());
    await user.click(screen.getByRole("button", { name: "Import" }));
    expect(await screen.findByText(/Imported 1 job report\b/)).toBeInTheDocument();
    expect(seen[0].get("force")).toBe("true");
  });

  it("blocks files over the size limit before sending", async () => {
    const seen = captureImport({ run_id: "r", imported: 1, warnings: [] });
    const user = userEvent.setup();
    renderPage();
    const big = new File(["x"], "big.zip");
    Object.defineProperty(big, "size", { value: 201 * 1024 * 1024 });
    await user.upload(screen.getByLabelText("Results bundle (.zip)"), big);
    expect(screen.getByRole("alert")).toHaveTextContent("larger than the server's upload limit (200 MB)");
    expect(screen.getByRole("button", { name: "Import" })).toBeDisabled();
    expect(seen).toHaveLength(0);
  });

  it.each([
    [422, "manifest.json missing from bundle", /manifest\.json missing/, /Nothing was imported/],
    [413, "upload exceeds STL_MAX_UPLOAD_MB=200", /exceeds STL_MAX_UPLOAD_MB=200/, null],
    [503, "artifact store unavailable: disk", /artifact store unavailable/, null],
  ])("shows %i errors", async (status, detail, re, extra) => {
    server.use(http.post("*/import-results", () => HttpResponse.json({ detail }, { status })));
    const user = userEvent.setup();
    renderPage();
    await user.upload(screen.getByLabelText("Results bundle (.zip)"), zip());
    await user.click(screen.getByRole("button", { name: "Import" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(re);
    if (extra) expect(alert).toHaveTextContent(extra);
    expect(screen.queryByText(/^Imported/)).toBeNull();
  });
});
