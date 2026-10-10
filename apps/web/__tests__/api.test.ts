// @vitest-environment node
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import {
  ApiError,
  apiFetch,
  createRun,
  getDataset,
  getReport,
  getRun,
  health,
  importBundle,
  listDatasets,
  listGenerators,
  listRuns,
  loadBuiltin,
  registerDataset,
  updateMetadata,
  uploadDataset,
} from "@/lib/api";
import { seedFailedRun } from "@/mocks/handlers";
import { server } from "@/mocks/server";

const BASE = "http://api.test";
const ORDER = ["queued", "fitting", "sampling", "generated", "evaluating", "done"];

async function fail(p: Promise<unknown>): Promise<ApiError> {
  try {
    await p;
  } catch (e) {
    return e as ApiError;
  }
  throw new Error("expected the call to reject");
}

describe("apiFetch error parsing", () => {
  it("uses a string detail", async () => {
    server.use(http.get(`${BASE}/x`, () => HttpResponse.json({ detail: "nope" }, { status: 409 })));
    const err = await fail(apiFetch("/x"));
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.detail).toBe("nope");
    expect(err.message).toBe("nope");
  });

  it("joins a FastAPI validation list of {msg}", async () => {
    server.use(
      http.get(`${BASE}/x`, () =>
        HttpResponse.json({ detail: [{ msg: "field required" }, { msg: "bad value" }] }, { status: 422 }),
      ),
    );
    const err = await fail(apiFetch("/x"));
    expect(err.status).toBe(422);
    expect(err.detail).toBe("field required; bad value");
  });

  it("keeps the parsed body and formats FastAPI loc/Value error items", async () => {
    server.use(
      http.get(`${BASE}/x`, () =>
        HttpResponse.json(
          { detail: [{ type: "value_error", loc: ["body", "seeds"], msg: "Value error, seeds must be unique" }] },
          { status: 422 },
        ),
      ),
      http.get(`${BASE}/y`, () =>
        HttpResponse.json({ detail: "created but failed", run_id: "r1", jobs: [] }, { status: 503 }),
      ),
    );
    expect((await fail(apiFetch("/x"))).detail).toBe("seeds: must be unique");
    const err = await fail(apiFetch("/y"));
    expect(err.detail).toBe("created but failed");
    expect(err.body).toMatchObject({ run_id: "r1" });
  });

  it("falls back to status text when the body is not JSON", async () => {
    server.use(http.get(`${BASE}/x`, () => new HttpResponse("boom", { status: 500, statusText: "Server Error" })));
    const err = await fail(apiFetch("/x"));
    expect(err.status).toBe(500);
    expect(err.detail).toBe("Server Error");
  });

  it("reports an unreachable API as status 0", async () => {
    server.use(http.get(`${BASE}/x`, () => HttpResponse.error()));
    const err = await fail(apiFetch("/x"));
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
    expect(err.detail).toMatch(/^Cannot reach the API, or it rejected the request/);
    expect(err.detail).toContain(BASE);
  });

  it("treats a non-JSON 2xx body as an ApiError", async () => {
    server.use(http.get(`${BASE}/x`, () => new HttpResponse("<html>", { status: 200 })));
    const err = await fail(apiFetch("/x"));
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(200);
    expect(err.detail).toMatch(/not valid JSON/i);
  });

  it("reports an aborted request as aborted, not unreachable", async () => {
    server.use(http.get(`${BASE}/x`, () => HttpResponse.json({})));
    const ctl = new AbortController();
    ctl.abort();
    const err = await fail(apiFetch("/x", { signal: ctl.signal }));
    expect(err.status).toBe(0);
    expect(err.detail).toMatch(/aborted/i);
    expect(err.detail).not.toMatch(/reach/i);
  });

  it("returns parsed JSON on success", async () => {
    server.use(http.get(`${BASE}/x`, () => HttpResponse.json({ a: 1 })));
    expect(await apiFetch<{ a: number }>("/x")).toEqual({ a: 1 });
  });
});

describe("typed client against the mock API", () => {
  it("health", async () => {
    expect((await health()).status).toBe("ok");
  });

  it("lists and gets datasets", async () => {
    const all = await listDatasets();
    expect(all).toHaveLength(2);
    expect(all.filter((d) => d.registered)).toHaveLength(1);
    const d = await getDataset("adult");
    expect(d.target).toBe("income");
    expect(d.preview?.length).toBeGreaterThan(0);
  });

  it("uploads a dataset and surfaces a 422 on a bad target", async () => {
    const file = new File(["a,b\n1,2\n"], "t.csv", { type: "text/csv" });
    const ok = await uploadDataset(file, "tiny", "b");
    expect(ok.registered).toBe(false);
    await expect(uploadDataset(file, "tiny", "zzz")).rejects.toMatchObject({
      status: 422,
      detail: expect.stringContaining("zzz"),
    });
  });

  it("updates metadata, registers and loads a builtin", async () => {
    const d = await updateMetadata("uploaded-1", { columns: { a: "numerical" }, target: "a" });
    expect(d.registered).toBe(false);
    const r = await registerDataset("uploaded-1", { max_rows: 500 });
    expect(r.registered).toBe(true);
    const b = await loadBuiltin("credit");
    expect(b.registered).toBe(true);
    expect(b.source).toBe("builtin");
    await expect(loadBuiltin("adult")).rejects.toMatchObject({
      status: 409,
      detail: "dataset adult is used by 2 run(s); create a new dataset instead",
    });
  });

  it("lists the full generator catalog", async () => {
    const g = await listGenerators();
    expect(g).toHaveLength(15);
    expect(g.find((x) => x.name === "tabddpm")?.cpu_ok).toBe(false);
    expect(g.filter((x) => x.audit)).toHaveLength(3);
  });

  it("creates runs (409 for unregistered, 422 for unknown generator)", async () => {
    const run = await createRun({ dataset_id: "adult", generators: ["tvae"], seeds: [0], preset: "balanced" });
    expect(run.jobs).toHaveLength(1);
    await expect(
      createRun({ dataset_id: "uploaded-1", generators: ["tvae"], seeds: [0], preset: "balanced" }),
    ).rejects.toMatchObject({ status: 409 });
    await expect(
      createRun({ dataset_id: "adult", generators: ["nope"], seeds: [0], preset: "balanced" }),
    ).rejects.toMatchObject({ status: 422 });
  });

  it("an in-progress run advances on each poll", async () => {
    const first = await getRun("adult-live");
    const second = await getRun("adult-live");
    const third = await getRun("adult-live");
    expect(first.done).toBe(false);
    const rank = (r: typeof first) => r.jobs.reduce((n, j) => n + ORDER.indexOf(j.state), 0);
    expect(rank(second)).toBeGreaterThan(rank(first));
    expect(rank(third)).toBeGreaterThan(rank(second));
    expect((await listRuns()).length).toBeGreaterThanOrEqual(2);
  });

  it("reports: tvae wins on Adult, train_copy fails the gate, weights are honoured", async () => {
    const rep = await getReport("adult-done");
    expect(rep.recommendation?.winner).toBe("tvae");
    const tc = rep.recommendation?.ranking.find((r) => r.generator === "train_copy");
    expect(tc?.gate_passed).toBe(false);
    const mst = rep.recommendation?.ranking.find((r) => r.generator === "mst_eps1");
    expect(mst?.notes?.join(" ")).toMatch(/not attackable/i);
    const priv = await getReport("adult-done", { preset: "privacy_first" });
    expect(priv.recommendation?.preset).toBe("privacy_first");
    const custom = await getReport("adult-done", { weights: [0.5, 0.25, 0.25] });
    expect(custom.recommendation?.weights).toEqual([0.5, 0.25, 0.25]);
    await expect(getReport("adult-done", { weights: [0, 0, 0] })).rejects.toMatchObject({ status: 422 });
  });

  it("a live run yields a full report once all jobs are done", async () => {
    let run = await getRun("adult-live");
    expect((await getReport("adult-live")).recommendation).toBeNull();
    for (let i = 0; i < 10 && !run.done; i++) run = await getRun("adult-live");
    expect(run.done).toBe(true);
    const rep = await getReport("adult-live");
    expect(rep.recommendation).not.toBeNull();
    expect(rep.jobs_reported).toBe(rep.jobs_total);
    expect(rep.jobs_total).toBe(run.jobs.length);
    expect(rep.aggregated.map((a) => a.generator).sort()).toEqual([...run.generators].sort());
  });

  it("imports a bundle", async () => {
    const res = await importBundle(new File(["zip"], "b.zip"), false);
    expect(res.imported).toBeGreaterThan(0);
  });
});

/** The MSW mock must behave like the real FastAPI service (services/api), or it hides real bugs. */
describe("mock fidelity to the real API", () => {
  it("returns a partial report during a live run, listing every job with its current state", async () => {
    let run = await getRun("adult-live");
    while (!run.jobs.some((j) => j.state === "done")) run = await getRun("adult-live");
    expect(run.done).toBe(false);
    const rep = await getReport("adult-live");
    expect(rep.recommendation).not.toBeNull();
    expect(rep.jobs_reported).toBeGreaterThan(0);
    expect(rep.jobs_reported).toBeLessThan(rep.jobs_total);
    expect(rep.per_job.map((j) => j.job_id)).toEqual(run.jobs.map((j) => j.job_id).sort());
    expect(rep.per_job.some((j) => j.status !== "done")).toBe(true);
    for (const j of rep.per_job) expect(ORDER).toContain(j.status);
    expect(rep.aggregated.map((a) => a.generator)).toEqual(
      [...new Set(run.jobs.filter((j) => j.state === "done").map((j) => j.generator))].sort(),
    );
  });

  it("uses the real job id format and a multi-line traceback for failed jobs", async () => {
    seedFailedRun();
    const run = await getRun("adult-partial");
    expect(run.jobs.map((j) => j.job_id)).toEqual(["gaussian_copula__s0", "tvae__s0"]);
    const failed = run.jobs.find((j) => j.state === "failed");
    expect(failed?.error).toMatch(/^Traceback \(most recent call last\):\n/);
    expect(failed?.error?.trim().split("\n").at(-1)).toMatch(/OutOfMemoryError/);
    const rep = await getReport("adult-partial");
    expect(rep.per_job.map((j) => [j.job_id, j.status])).toEqual([
      ["gaussian_copula__s0", "done"],
      ["tvae__s0", "failed"],
    ]);
  });

  it("scores gate-failing candidates, keeps them off the Pareto front and uses the real gate strings", async () => {
    const rep = await getReport("adult-done");
    const ranking = rep.recommendation!.ranking;
    const tc = ranking.find((r) => r.generator === "train_copy")!;
    expect(typeof tc.score).toBe("number");
    expect(tc.on_pareto_front).toBe(false);
    expect(tc.gate_failures).toContain("failed the DCR check (share 1.00 > 0.60)");
    expect(tc.gate_failures).toContain("failed the singling-out check (lower bound 0.98 > 0.10)");
    // the real sort: gate-passing first, then by score
    expect(ranking.at(-1)!.generator).toBe("train_copy");
    for (const p of rep.pareto) if (p.on_front) expect(p.gate_passed).toBe(true);
    expect(rep.pareto.find((p) => p.generator === "train_copy")?.on_front).toBe(false);
    const mst = ranking.find((r) => r.generator === "mst_eps1")!;
    expect(mst.notes).toEqual(["singling-out not attackable (safeguards passed)"]);
    expect(rep.recommendation!.reason).toMatch(
      /^tvae: highest balanced score \(0\.81\) among models passing the privacy gate/,
    );
  });

  it("aggregates the real metric keys", async () => {
    const rep = await getReport("adult-done");
    const tvae = rep.aggregated.find((a) => a.generator === "tvae")!;
    for (const k of [
      "fidelity.quality_score",
      "fidelity.detection_auc",
      "utility.tstr.logreg",
      "utility.tstr.random_forest",
      "utility.importance_spearman",
      "privacy.dcr_share",
      "privacy.singling_out_risk_lower",
      "privacy.mia_advantage",
      "cost.fit_seconds",
    ]) {
      expect(tvae.mean).toHaveProperty([k]);
      expect(tvae.std).toHaveProperty([k]);
    }
    expect(tvae.worst).toHaveProperty(["privacy.dcr_share"]);
    expect(tvae.n_seeds).toBe(2);
  });

  it("normalises custom weights and stores a custom-weighted run as preset 'custom'", async () => {
    const rep = await getReport("adult-done", { weights: [2, 1, 1] });
    expect(rep.recommendation?.preset).toBe("custom");
    expect(rep.recommendation?.weights).toEqual([0.5, 0.25, 0.25]);
    const created = await createRun({
      dataset_id: "adult",
      generators: ["tvae"],
      seeds: [0],
      preset: "balanced",
      weights: [2, 1, 1],
    });
    const run = await getRun(created.run_id);
    expect(run.preset).toBe("custom");
    expect(run.weights).toEqual([0.5, 0.25, 0.25]);
    const plain = await createRun({ dataset_id: "adult", generators: ["tvae"], seeds: [0], preset: "privacy_first" });
    expect((await getRun(plain.run_id)).weights).toEqual([0.2, 0.2, 0.6]);
    expect((await getRun("adult-done")).weights).toEqual([1 / 3, 1 / 3, 1 / 3]);
  });

  it("an import creates a placeholder dataset and a run without weights", async () => {
    const res = await importBundle(new File(["zip"], "b.zip"), false);
    const ds = (await listDatasets()).find((d) => d.source === "import")!;
    expect(ds).toMatchObject({ target: "", metadata: {}, n_rows: null, registered: false });
    expect(typeof ds.fingerprint).toBe("string");
    const run = await getRun(res.run_id);
    expect(run.dataset_id).toBe(ds.id);
    expect(run.weights).toBeNull();
    expect((await listRuns()).map((r) => r.run_id)).toContain(res.run_id);
  });

  it("promotes an import placeholder through a built-in load and returns warnings", async () => {
    await importBundle(new File(["zip"], "b.zip"), false);
    const ds = (await listDatasets()).find((d) => d.source === "import")!;
    expect(ds.id).toBe("diabetes");
    await expect(registerDataset(ds.id)).rejects.toMatchObject({ status: 409, detail: "dataset has no stored upload" });
    const promoted = await loadBuiltin("diabetes");
    expect(promoted.source).toBe("builtin");
    expect(promoted.registered).toBe(true);
    expect(promoted.warnings).toEqual([
      "fingerprint differs from imported run(s) of dataset diabetes; results not comparable",
    ]);
  });

  it("accepts only CSV or Parquet uploads with rows", async () => {
    await expect(uploadDataset(new File(["a,b\n1,2\n"], "t.txt"), "t", "b")).rejects.toMatchObject({
      status: 422,
      detail: "unsupported file type '.txt'; upload CSV or Parquet",
    });
    await expect(uploadDataset(new File(["a,b\n"], "t.csv"), "t", "b")).rejects.toMatchObject({
      status: 422,
      detail: "the file has no rows",
    });
  });
});
