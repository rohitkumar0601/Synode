/** Training-budget profiles (mirror packages/core/src/stl_core/budget.py BUDGET_PROFILES). */

export type BudgetProfile = "library" | "equal50" | "smoke";

export const DEFAULT_BUDGET_PROFILE: BudgetProfile = "library";

export interface BudgetProfileInfo {
  id: BudgetProfile;
  /** Short name, e.g. for the run header chip. */
  label: string;
  /** Option text in the new-run select. */
  option: string;
  /** Expected cost / what it means. */
  hint: string;
}

export const BUDGET_PROFILES: BudgetProfileInfo[] = [
  {
    id: "library",
    label: "Library defaults",
    option: "Library defaults (faithful, slow — use on GPU/owner compute)",
    hint:
      "Every generator trains with its library's own defaults (e.g. synthcity CTGAN up to 2000 epochs with early " +
      "stopping, TVAE 1000, GReaT 100). Faithful to each library, but CTGAN/TVAE can take many hours on CPU. " +
      "Time cap 24 h per job.",
  },
  {
    id: "equal50",
    label: "Equal 50 epochs",
    option: "Equal 50 epochs (CPU comparison)",
    hint:
      "50 epochs for every deep generator: comparable across generators on a CPU, but not each library's own " +
      "setting. Time cap 2 h per job.",
  },
  {
    id: "smoke",
    label: "Smoke (5 epochs)",
    option: "Smoke (5 epochs, quick check)",
    hint: "5 epochs, only to check the pipeline end to end; the scores are not meaningful. Time cap 30 min per job.",
  },
];

/** Display name of a profile; unknown ids are shown as-is. */
export function budgetLabel(profile: string): string {
  return BUDGET_PROFILES.find((p) => p.id === profile)?.label ?? profile;
}
