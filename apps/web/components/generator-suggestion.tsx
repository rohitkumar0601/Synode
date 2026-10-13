"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Info, Sparkles } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  canRefreshSuggestion,
  refreshSuggestion,
  suggestionKey,
  suggestionView,
  useDatasetSuggestion,
} from "@/lib/suggestion";
import type { GeneratorInfo } from "@/lib/types";

export const EXPLORATORY_HELP =
  "RQ3 is exploratory: a small meta-model trained on the profiles of only about 25 benchmark " +
  "datasets guesses which generator would win the balanced preset. With so few datasets it may " +
  "well be wrong. Treat it as a hint and let the benchmark decide.";

/**
 * "Suggested: <generator> (p=…)" for the chosen dataset, with a one-click "Use suggestion"
 * that adds the generator to the selection (others stay selected). Shows a subtle
 * "No generator suggestion yet" when the API has none (with a "Refresh suggestion" link when
 * nothing was computed yet or no predictor existed), and nothing while loading or on error.
 * While the suggestion is being computed it is polled (lib/suggestion).
 */
export function GeneratorSuggestion(props: SuggestionProps) {
  // Keyed by dataset so the polling window restarts when another dataset is chosen.
  return <SuggestionBox key={props.datasetId} {...props} />;
}

interface SuggestionProps {
  datasetId: string;
  generators: GeneratorInfo[];
  selected: string[];
  onChange: (next: string[]) => void;
}

function SuggestionBox({ datasetId, generators, selected, onChange }: SuggestionProps) {
  const [pollStartedAt, setPollStartedAt] = useState(() => Date.now());
  const queryClient = useQueryClient();
  const query = useDatasetSuggestion(datasetId, pollStartedAt);
  const refresh = useMutation({
    mutationFn: () => refreshSuggestion(datasetId),
    onSuccess: () => {
      setPollStartedAt(Date.now());
      return queryClient.invalidateQueries({ queryKey: suggestionKey(datasetId) });
    },
  });
  if (!datasetId || !query.data) return null;
  const view = suggestionView(query.data);
  if (!view) {
    const reason = query.data.available ? undefined : query.data.reason;
    return (
      <p className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span title={reason}>No generator suggestion yet.</span>
        {canRefreshSuggestion(query.data) ? (
          <button
            type="button"
            className="text-primary underline-offset-4 hover:underline disabled:opacity-50"
            disabled={refresh.isPending}
            onClick={() => refresh.mutate()}
          >
            Refresh suggestion
          </button>
        ) : null}
        {refresh.isError ? <span className="text-destructive">{refresh.error.message}</span> : null}
      </p>
    );
  }
  const known = generators.some((g) => g.name === view.generator);
  const inSelection = selected.includes(view.generator);
  return (
    <div
      role="status"
      aria-label="Generator suggestion"
      className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-muted/30 px-3 py-2 text-sm"
    >
      <Sparkles className="size-4 text-primary" aria-hidden="true" />
      <span>
        Suggested: <span className="font-mono">{view.generator}</span>
        {view.probability !== null ? ` (p=${view.probability.toFixed(2)})` : ""}
        {" — exploratory"}
        {view.nDatasets !== null ? `, trained on ${view.nDatasets} datasets` : ""}
      </span>
      <Tooltip>
        <TooltipTrigger
          render={<button type="button" aria-label="Why the suggestion is exploratory" className="text-muted-foreground" />}
        >
          <Info className="size-4" aria-hidden="true" />
        </TooltipTrigger>
        <TooltipContent>{EXPLORATORY_HELP}</TooltipContent>
      </Tooltip>
      {!known ? (
        <span className="text-xs text-muted-foreground">(not offered by this API)</span>
      ) : inSelection ? (
        <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
          <Check className="size-3.5" aria-hidden="true" />
          Selected
        </span>
      ) : (
        <Button type="button" variant="outline" size="sm" onClick={() => onChange(generators.map((g) => g.name).filter((n) => n === view.generator || selected.includes(n)))}
        >
          Use suggestion
        </Button>
      )}
    </div>
  );
}
