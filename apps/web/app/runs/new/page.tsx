"use client";

import { toast } from "sonner";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, Info, Sparkles } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { ErrorState } from "@/components/error-state";
import { defaultSelection, GeneratorPicker } from "@/components/generator-picker";
import { presetError, WeightsEditor, type PresetChoice } from "@/components/weights-editor";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { NativeSelect } from "@/components/ui/native-select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ApiError, createRun, getSuggestion, listDatasets, listGenerators } from "@/lib/api";
import type { Weights } from "@/lib/presets";
import type { CreateRunBody } from "@/lib/types";

/** Parses "0, 1 2" into unique non-negative integers; returns an error message otherwise. */
function parseSeeds(raw: string): { seeds: number[]; error: string | null } {
  const parts = raw.split(/[\s,]+/).filter(Boolean);
  if (parts.length === 0) return { seeds: [], error: "Enter at least one seed." };
  const seeds: number[] = [];
  for (const p of parts) {
    if (!/^\d+$/.test(p) || !Number.isSafeInteger(Number(p))) {
      return { seeds: [], error: `Seeds must be non-negative whole numbers; "${p}" is not.` };
    }
    seeds.push(Number(p));
  }
  if (new Set(seeds).size !== seeds.length) return { seeds: [], error: "Seeds must be unique." };
  return { seeds, error: null };
}

/** A 503 from POST /runs still created the run: its body carries `run_id`. */
function createdRunId(e: unknown): string | null {
  if (!(e instanceof ApiError) || e.status !== 503) return null;
  const id = (e.body as { run_id?: unknown } | null)?.run_id;
  return typeof id === "string" && id ? id : null;
}

function NewRunForm() {
  const router = useRouter();
  const qc = useQueryClient();
  const preselect = useSearchParams().get("dataset");
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: listDatasets });
  const generators = useQuery({ queryKey: ["generators"], queryFn: listGenerators });

  const [datasetChoice, setDatasetChoice] = useState<string | null>(null);
  const [selection, setSelection] = useState<string[] | null>(null);
  const [seedsText, setSeedsText] = useState("0,1,2");
  const [preset, setPreset] = useState<PresetChoice>("balanced");
  const [custom, setCustom] = useState<Weights>([0.4, 0.3, 0.3]);
  const [allowGpu, setAllowGpu] = useState(false);

  const registered = (datasets.data ?? []).filter((d) => d.registered);
  const datasetId = datasetChoice ?? (preselect && registered.some((d) => d.id === preselect) ? preselect : "");
  const selected = selection ?? (generators.data ? defaultSelection(generators.data) : []);
  const suggestion = useQuery({
    queryKey: ["dataset-suggestion", datasetId],
    queryFn: () => getSuggestion(datasetId),
    enabled: !!datasetId,
  });

  const run = useMutation({
    mutationFn: (body: CreateRunBody) => createRun(body),
    onSuccess: async (r) => {
      await qc.invalidateQueries({ queryKey: ["runs"] });
      toast.success(`Run ${r.run_id} created`);
      router.push(`/runs/${encodeURIComponent(r.run_id)}`);
    },
  });

  const seedCheck = parseSeeds(seedsText);
  const gpuPicked = (generators.data ?? []).filter((g) => !g.cpu_ok && selected.includes(g.name)).map((g) => g.name);
  const reasons: string[] = [];
  if (!datasetId) reasons.push("Choose a registered dataset.");
  if (selected.length === 0) reasons.push("Select at least one generator.");
  if (seedCheck.error) reasons.push("Fix the seeds (see the error under the field).");
  const wErr = presetError(preset, custom);
  if (wErr) reasons.push("Fix the custom weights (see the error above).");
  if (gpuPicked.length > 0 && !allowGpu) {
    reasons.push(`GPU-only generators selected (${gpuPicked.join(", ")}); enable "Allow GPU-only generators" or deselect them.`);
  }

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (reasons.length > 0) return;
    const body: CreateRunBody = {
      dataset_id: datasetId,
      generators: selected,
      seeds: seedCheck.seeds,
      preset: preset === "custom" ? "balanced" : preset,
      allow_gpu_only: allowGpu,
    };
    if (preset === "custom") body.weights = custom;
    run.mutate(body);
  };

  const suggested = suggestion.data?.available && suggestion.data.generator ? suggestion.data : null;
  const suggestedProbability =
    suggested?.generator && suggested.probabilities?.[suggested.generator] !== undefined
      ? suggested.probabilities[suggested.generator]
      : null;

  if (datasets.isError) return <ErrorState title="Could not load datasets" error={datasets.error} />;
  if (generators.isError) return <ErrorState title="Could not load generators" error={generators.error} />;
  if (datasets.isPending || generators.isPending) return <Skeleton className="h-64 w-full" />;

  const createdId = createdRunId(run.error);

  return (
    <form onSubmit={submit} className="space-y-8" aria-label="New benchmark run">
      <section className="space-y-2">
        <Label htmlFor="dataset">Dataset</Label>
        <NativeSelect id="dataset" value={datasetId} onChange={(e) => setDatasetChoice(e.target.value)}>
          <option value="">Select a registered dataset…</option>
          {registered.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name === d.id ? d.id : `${d.name} (${d.id})`}
            </option>
          ))}
        </NativeSelect>
        {registered.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No registered datasets yet.{" "}
            <Link href="/datasets" className="underline">
              Register one on the Datasets page
            </Link>
            .
          </p>
        )}
        {suggested && (
          <div className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-muted/30 px-3 py-2 text-sm">
            <Sparkles className="size-4 text-primary" aria-hidden="true" />
            <span>
              Suggested: <span className="font-mono">{suggested.generator}</span>
              {suggestedProbability !== null ? ` (p=${suggestedProbability.toFixed(2)})` : ""}
            </span>
            {suggested.model?.n_datasets ? (
              <Badge variant="outline">{suggested.model.n_datasets} datasets</Badge>
            ) : null}
            <Tooltip>
              <TooltipTrigger
                render={<button type="button" aria-label="Why the suggestion is exploratory" className="text-muted-foreground" />}
              >
                <Info className="size-4" aria-hidden="true" />
              </TooltipTrigger>
              <TooltipContent>
                Exploratory RQ3 model; use it as a starting point, not a substitute for the benchmark.
              </TooltipContent>
            </Tooltip>
            <Button type="button" variant="outline" size="sm" onClick={() => setSelection([suggested.generator!])}>
              Use suggestion
            </Button>
          </div>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-sm font-medium">Generators</h2>
        {generators.data.length === 0 && (
          <p className="text-sm text-muted-foreground">No generators are available from the API.</p>
        )}
        <GeneratorPicker generators={generators.data} selected={selected} onChange={setSelection} />
      </section>

      <section className="space-y-2">
        <Label htmlFor="seeds">Seeds</Label>
        <input
          id="seeds"
          value={seedsText}
          onChange={(e) => setSeedsText(e.target.value)}
          aria-invalid={seedCheck.error ? true : undefined}
          aria-describedby={seedCheck.error ? "seeds-help seeds-error" : "seeds-help"}
          className="h-8 w-full max-w-xs rounded-lg border border-input bg-transparent px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        />
        <p id="seeds-help" className="text-xs text-muted-foreground">
          Comma-separated non-negative integers, e.g. 0,1,2. Each generator runs once per seed.
        </p>
        {seedCheck.error && (
          <p id="seeds-error" className="text-sm text-destructive">
            {seedCheck.error}
          </p>
        )}
      </section>

      <WeightsEditor preset={preset} custom={custom} onPresetChange={setPreset} onCustomChange={setCustom} />

      <section className="flex items-center gap-2">
        <input
          id="allow-gpu"
          type="checkbox"
          className="size-4 accent-primary"
          checked={allowGpu}
          onChange={(e) => setAllowGpu(e.target.checked)}
        />
        <Label htmlFor="allow-gpu">Allow GPU-only generators</Label>
        <Tooltip>
          <TooltipTrigger
            render={<button type="button" aria-label="About GPU-only generators" className="text-muted-foreground" />}
          >
            <Info className="size-4" aria-hidden="true" />
          </TooltipTrigger>
          <TooltipContent>GPU-only generators need a GPU worker. Without one, their jobs stay queued or fail.</TooltipContent>
        </Tooltip>
      </section>

      {run.isError &&
        (createdId ? (
          <Alert variant="destructive">
            <AlertCircle aria-hidden="true" />
            <AlertTitle>Run created, but no job could be dispatched</AlertTitle>
            <AlertDescription>
              <span>{run.error.message} </span>
              <Link href={`/runs/${encodeURIComponent(createdId)}`} className="underline">
                Open run {createdId}
              </Link>{" "}
              to retry the failed jobs.
            </AlertDescription>
          </Alert>
        ) : run.error instanceof ApiError && run.error.status === 409 ? (
          <Alert variant="destructive">
            <AlertCircle aria-hidden="true" />
            <AlertTitle>Could not start the run</AlertTitle>
            <AlertDescription>
              {run.error.detail}{" "}
              <Link href="/datasets" className="underline">
                Go to the Datasets page
              </Link>{" "}
              to register it.
            </AlertDescription>
          </Alert>
        ) : (
          <ErrorState title="Could not start the run" error={run.error} />
        ))}

      <div className="space-y-2">
        <Button type="submit" disabled={reasons.length > 0 || run.isPending} aria-describedby="submit-reasons">
          {run.isPending ? "Starting…" : "Start benchmark"}
        </Button>
        <ul id="submit-reasons" className="list-disc pl-5 text-sm text-muted-foreground">
          {reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      </div>
    </form>
  );
}

export default function NewRunPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">New run</h1>
      <Suspense fallback={<Skeleton className="h-64 w-full" />}>
        <NewRunForm />
      </Suspense>
    </div>
  );
}
