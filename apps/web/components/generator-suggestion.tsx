"use client";

import { Check, Info, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { suggestionView, useDatasetSuggestion } from "@/lib/suggestion";
import type { GeneratorInfo } from "@/lib/types";

export const EXPLORATORY_HELP =
  "RQ3 is exploratory: a small meta-model trained on the profiles of only about 25 benchmark " +
  "datasets guesses which generator would win the balanced preset. With so few datasets it may " +
  "well be wrong. Treat it as a hint and let the benchmark decide.";

/**
 * "Suggested: <generator> (p=…)" for the chosen dataset, with a one-click "Use suggestion"
 * that adds the generator to the selection (others stay selected). Shows a subtle
 * "No generator suggestion yet" when the API has none, and nothing while loading or on error.
 */
export function GeneratorSuggestion({
  datasetId,
  generators,
  selected,
  onChange,
}: {
  datasetId: string;
  generators: GeneratorInfo[];
  selected: string[];
  onChange: (next: string[]) => void;
}) {
  const query = useDatasetSuggestion(datasetId);
  if (!datasetId || !query.data) return null;
  const view = suggestionView(query.data);
  if (!view) {
    const reason = query.data.available ? undefined : query.data.reason;
    return (
      <p className="text-xs text-muted-foreground" title={reason}>
        No generator suggestion yet.
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
