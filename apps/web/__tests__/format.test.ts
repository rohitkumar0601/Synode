import { describe, expect, it } from "vitest";
import { fmtMeanStd, fmtScore, fmtState } from "@/lib/format";

describe("fmtScore", () => {
  it("formats to three digits by default", () => {
    expect(fmtScore(0.7421)).toBe("0.742");
  });
  it("honours digits", () => {
    expect(fmtScore(0.7421, 1)).toBe("0.7");
  });
  it("renders an em dash for missing or non-finite values", () => {
    expect(fmtScore(null)).toBe("—");
    expect(fmtScore(undefined)).toBe("—");
    expect(fmtScore(Number.NaN)).toBe("—");
  });
  it("keeps zero", () => {
    expect(fmtScore(0)).toBe("0.000");
  });
});

describe("fmtMeanStd", () => {
  it("formats mean ± std", () => {
    expect(fmtMeanStd(0.5, 0.0123)).toBe("0.500 ± 0.012");
  });
  it("omits std when missing", () => {
    expect(fmtMeanStd(0.5, null)).toBe("0.500");
  });
  it("is a dash when the mean is missing", () => {
    expect(fmtMeanStd(null, 0.1)).toBe("—");
  });
});

describe("fmtState", () => {
  it("capitalises job states", () => {
    expect(fmtState("fitting")).toBe("Fitting");
    expect(fmtState("done")).toBe("Done");
  });
});
