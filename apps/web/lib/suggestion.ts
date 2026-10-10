import { useQuery } from "@tanstack/react-query";
import { getSuggestion } from "@/lib/api";
import type { DatasetSuggestion } from "@/lib/types";

export const suggestionKey = (datasetId: string) => ["dataset-suggestion", datasetId] as const;

/** The RQ3 suggestion for a dataset; disabled until a dataset is chosen. */
export function useDatasetSuggestion(datasetId: string) {
  return useQuery({
    queryKey: suggestionKey(datasetId),
    queryFn: () => getSuggestion(datasetId),
    enabled: !!datasetId,
    retry: false,
    staleTime: 30_000,
  });
}

export interface SuggestionView {
  generator: string;
  /** Probability of the suggested generator, or null if the response lacks it. */
  probability: number | null;
  /** Number of benchmark datasets the meta-model was trained on, or null if unknown. */
  nDatasets: number | null;
}

/** What to show for a suggestion response; null when there is nothing to suggest. */
export function suggestionView(s: DatasetSuggestion | undefined): SuggestionView | null {
  if (!s || s.available !== true || typeof s.generator !== "string" || !s.generator) return null;
  const p = s.probabilities?.[s.generator];
  const n = s.model?.n_datasets;
  return {
    generator: s.generator,
    probability: typeof p === "number" && Number.isFinite(p) ? p : null,
    nDatasets: typeof n === "number" && Number.isInteger(n) && n > 0 ? n : null,
  };
}
