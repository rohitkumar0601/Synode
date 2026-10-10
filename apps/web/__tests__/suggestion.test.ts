// @vitest-environment node
import { describe, expect, it } from "vitest";
import { getSuggestion } from "@/lib/api";
import { suggestionView } from "@/lib/suggestion";
import { SUGGESTION_AVAILABLE, SUGGESTION_NO_PREDICTOR, SUGGESTION_NOT_REGISTERED } from "@/mocks/fixtures";

describe("suggestionView", () => {
  it("summarises an available suggestion", () => {
    expect(suggestionView(SUGGESTION_AVAILABLE)).toEqual({ generator: "tvae", probability: 0.62, nDatasets: 25 });
  });

  it("is null when nothing is available", () => {
    expect(suggestionView(undefined)).toBeNull();
    expect(suggestionView(SUGGESTION_NO_PREDICTOR)).toBeNull();
  });

  it("tolerates missing probability and model", () => {
    expect(suggestionView({ available: true, generator: "ctgan", probabilities: {} })).toEqual({
      generator: "ctgan",
      probability: null,
      nDatasets: null,
    });
    expect(
      suggestionView({ available: true, generator: "ctgan", probabilities: { ctgan: Number.NaN }, model: { n_datasets: 0 } }),
    ).toEqual({ generator: "ctgan", probability: null, nDatasets: null });
  });
});

describe("getSuggestion (MSW)", () => {
  it("returns the available and the unavailable shapes", async () => {
    expect(await getSuggestion("adult")).toEqual(SUGGESTION_AVAILABLE);
    expect(await getSuggestion("uploaded-1")).toEqual(SUGGESTION_NOT_REGISTERED);
  });

  it("404s for an unknown dataset", async () => {
    await expect(getSuggestion("nope")).rejects.toMatchObject({ status: 404 });
  });
});
