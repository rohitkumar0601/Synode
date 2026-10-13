import { useQuery } from "@tanstack/react-query";
import { apiFetch, getSuggestion } from "@/lib/api";
import type { DatasetSuggestion } from "@/lib/types";

export const suggestionKey = (datasetId: string) => ["dataset-suggestion", datasetId] as const;

/** Poll every 10 s while the worker computes the suggestion, for at most ~5 minutes. */
export const SUGGESTION_POLL_MS = 10_000;
export const SUGGESTION_POLL_MAX_MS = 5 * 60_000;

/** API reasons (services/api routes/datasets.py get_suggestion). */
const NOT_COMPUTED = "suggestion has not been computed yet";
const NO_PREDICTOR = "no meta-predictor has been trained yet";

const reasonOf = (s: DatasetSuggestion | undefined) => (s && s.available === false ? s.reason : "");

/** True while the meta-feature task has been queued but has not written a result yet. */
export function suggestionPending(s: DatasetSuggestion | undefined): boolean {
  return reasonOf(s).startsWith(NOT_COMPUTED);
}

/** A refresh can help when nothing was computed yet or no predictor was trained then. */
export function canRefreshSuggestion(s: DatasetSuggestion | undefined): boolean {
  const reason = reasonOf(s);
  return reason.startsWith(NOT_COMPUTED) || reason.startsWith(NO_PREDICTOR);
}

/** react-query refetchInterval: poll while pending and within the time budget. */
export function suggestionPollInterval(
  s: DatasetSuggestion | undefined,
  pollStartedAt: number,
  now: number,
): number | false {
  return suggestionPending(s) && now - pollStartedAt < SUGGESTION_POLL_MAX_MS ? SUGGESTION_POLL_MS : false;
}

/** POST /datasets/{id}/suggestion/refresh: re-queue the meta-feature task (202). */
export const refreshSuggestion = (id: string) =>
  apiFetch<{ dataset_id: string; queued: boolean }>(`/datasets/${encodeURIComponent(id)}/suggestion/refresh`, {
    method: "POST",
  });

/**
 * The RQ3 suggestion for a dataset; disabled until a dataset is chosen. While the suggestion
 * is being computed it is polled (see suggestionPollInterval), counting from `pollStartedAt`
 * (a stable timestamp: when the dataset was chosen or a refresh was requested).
 */
export function useDatasetSuggestion(datasetId: string, pollStartedAt: number) {
  return useQuery({
    queryKey: suggestionKey(datasetId),
    queryFn: () => getSuggestion(datasetId),
    enabled: !!datasetId,
    retry: false,
    staleTime: 30_000,
    refetchInterval: (query) => suggestionPollInterval(query.state.data, pollStartedAt, Date.now()),
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
