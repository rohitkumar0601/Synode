export type Weights = [number, number, number];

export interface Preset {
  label: string;
  weights: Weights;
}

/** Axis weights are ordered [fidelity, utility, privacy]. */
export const PRESETS: Record<"balanced" | "privacy_first" | "utility_first", Preset> = {
  balanced: { label: "Balanced", weights: [1 / 3, 1 / 3, 1 / 3] },
  privacy_first: { label: "Privacy first", weights: [0.2, 0.2, 0.6] },
  utility_first: { label: "Utility first", weights: [0.2, 0.6, 0.2] },
};

/** Returns an error message, or null when the weights are valid. */
export function validateWeights(w: readonly number[]): string | null {
  if (w.length !== 3 || w.some((x) => typeof x !== "number" || !Number.isFinite(x))) {
    return "Weights must be three finite numbers.";
  }
  if (w.some((x) => x < 0)) return "Weights must be non-negative.";
  if (w.reduce((a, b) => a + b, 0) <= 0) return "Weights must have a positive sum.";
  return null;
}
