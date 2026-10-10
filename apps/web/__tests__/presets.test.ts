import { describe, expect, it } from "vitest";
import { PRESETS, validateWeights } from "@/lib/presets";

describe("PRESETS", () => {
  it("defines the three named presets", () => {
    expect(Object.keys(PRESETS)).toEqual(["balanced", "privacy_first", "utility_first"]);
    expect(PRESETS.privacy_first.weights).toEqual([0.2, 0.2, 0.6]);
    expect(PRESETS.utility_first.weights).toEqual([0.2, 0.6, 0.2]);
    const sum = PRESETS.balanced.weights.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
  });
});

describe("validateWeights", () => {
  it("accepts non-negative weights with a positive sum", () => {
    expect(validateWeights([1, 2, 3])).toBeNull();
    expect(validateWeights([0, 0, 1])).toBeNull();
  });
  it("rejects negatives", () => {
    expect(validateWeights([-0.1, 0.5, 0.6])).toMatch(/non-negative/i);
  });
  it("rejects a zero sum", () => {
    expect(validateWeights([0, 0, 0])).toMatch(/positive/i);
  });
  it("rejects non-finite values", () => {
    expect(validateWeights([Number.NaN, 1, 1])).toMatch(/number/i);
  });
});
