"use client";

import { toast } from "sonner";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { AlertTriangle } from "lucide-react";
import { useId, useState } from "react";
import { DataPreview } from "@/components/data-preview";
import { ErrorState } from "@/components/error-state";
import { MetadataEditor } from "@/components/metadata-editor";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, getDataset, loadBuiltin, registerDataset } from "@/lib/api";
import { isBuiltin } from "@/lib/builtins";
import { rememberDatasetWarnings, useDatasetWarnings } from "@/lib/dataset-warnings";
import { shortFingerprint, sourceLabel } from "@/lib/format";
import { hasMetadata, type Dataset } from "@/lib/types";

function RegisterPanel({
  dataset,
  dirty,
  locked,
  onLocked,
}: {
  dataset: Dataset;
  dirty: boolean;
  locked: boolean;
  onLocked: () => void;
}) {
  const qc = useQueryClient();
  const maxId = useId();
  const [maxRows, setMaxRows] = useState("20000");
  const n = Number(maxRows);
  const valid = Number.isInteger(n) && n >= 1;

  const register = useMutation({
    mutationFn: () => registerDataset(dataset.id, { max_rows: n }),
    onError: (e) => {
      if (e instanceof ApiError && e.status === 409) onLocked();
    },
    onSuccess: (d) => {
      toast.success(`Registered dataset "${dataset.name}"`);
      rememberDatasetWarnings(qc, d);
    },
    // A failed register resets the stored registration server-side, so refetch either way.
    onSettled: async () => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["dataset", dataset.id] }),
        qc.invalidateQueries({ queryKey: ["datasets"] }),
      ]);
    },
  });

  return (
    <section aria-labelledby="register-heading" className="space-y-3">
      <h2 id="register-heading" className="text-lg font-semibold">
        Registration
      </h2>
      {dataset.registered && (
        <p className="text-sm">
          Registered · fingerprint{" "}
          <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs" title={dataset.fingerprint ?? undefined}>
            {shortFingerprint(dataset.fingerprint)}
          </code>
        </p>
      )}
      <div className="flex flex-wrap items-end gap-3">
        <div className="grid w-40 gap-1.5">
          <Label htmlFor={maxId}>Max rows</Label>
          <Input
            id={maxId}
            type="number"
            min={1}
            step={1}
            value={maxRows}
            disabled={register.isPending}
            aria-invalid={!valid}
            onChange={(e) => setMaxRows(e.target.value)}
          />
        </div>
        <Button
          onClick={() => register.mutate()}
          disabled={register.isPending || !valid || dirty || locked}
          aria-busy={register.isPending}
        >
          {register.isPending ? "Registering…" : dataset.registered ? "Re-register" : "Register"}
        </Button>
        {dataset.registered && (
          <Link
            href={`/runs/new?dataset=${encodeURIComponent(dataset.id)}`}
            className={buttonVariants({ variant: "default" })}
          >
            Start a run
          </Link>
        )}
      </div>
      <p role="status" className="text-xs text-muted-foreground">
        {register.isPending
          ? "Registering: sampling at most max rows and creating splits. This can take a while."
          : dirty
            ? "Save your metadata changes before registering."
            : !valid
              ? "Max rows must be a whole number of at least 1."
              : ""}
      </p>
      {register.isError && <ErrorState title="Registration failed" error={register.error} />}
    </section>
  );
}

/** Import placeholders (and any dataset without column metadata) have no stored rows to edit or register. */
function ReadOnlyDataset({ dataset }: { dataset: Dataset }) {
  const qc = useQueryClient();
  const imported = dataset.source === "import";
  const promote = useMutation({
    mutationFn: () => loadBuiltin(dataset.id),
    onSuccess: async (d) => {
      toast.success(`Loaded built-in dataset "${d.name}"`);
      rememberDatasetWarnings(qc, d);
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["dataset", dataset.id] }),
        qc.invalidateQueries({ queryKey: ["datasets"] }),
      ]);
    },
  });
  return (
    <section aria-labelledby="readonly-heading" className="space-y-3">
      <h2 id="readonly-heading" className="text-lg font-semibold">
        {imported ? "From a results bundle" : "Read-only dataset"}
      </h2>
      <p className="text-sm">
        {imported
          ? "This dataset came from an imported results bundle; its rows are not stored here."
          : "This dataset has no column metadata, so it cannot be edited or registered here."}
      </p>
      {dataset.fingerprint && (
        <p className="text-sm text-muted-foreground">
          Fingerprint of the data the imported runs used{" "}
          <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs" title={dataset.fingerprint}>
            {shortFingerprint(dataset.fingerprint)}
          </code>
        </p>
      )}
      {isBuiltin(dataset.id) && (
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">
            {`"${dataset.id}" is a built-in dataset: load it to store its rows and register it, so you can start new runs on it.`}
          </p>
          <Button onClick={() => promote.mutate()} disabled={promote.isPending} aria-busy={promote.isPending}>
            {promote.isPending ? "Loading…" : "Load built-in"}
          </Button>
        </div>
      )}
      {promote.isError && <ErrorState title="Could not load the built-in dataset" error={promote.error} />}
    </section>
  );
}

function DatasetWarnings({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <Alert role="status">
      <AlertTriangle aria-hidden="true" />
      <AlertTitle>Registration warnings</AlertTitle>
      <AlertDescription>
        <ul className="list-disc pl-5">
          {warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      </AlertDescription>
    </Alert>
  );
}

export default function DatasetDetailPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const q = useQuery({ queryKey: ["dataset", id], queryFn: () => getDataset(id) });
  const [dirty, setDirty] = useState(false);
  const [locked, setLocked] = useState(false);
  const warnings = useDatasetWarnings(id);

  if (q.isError) return <ErrorState title="Could not load the dataset" error={q.error} />;
  if (q.isPending) return <Skeleton className="h-48 w-full" />;
  const d = q.data;

  return (
    <div className="space-y-8">
      <div className="space-y-1">
        <Link href="/datasets" className="text-sm text-muted-foreground underline-offset-4 hover:underline">
          All datasets
        </Link>
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold tracking-tight">{d.name}</h1>
          {d.registered ? <Badge>Registered</Badge> : <Badge variant="secondary">Not registered</Badge>}
        </div>
        <p className="text-sm text-muted-foreground">
          {sourceLabel(d.source)} · {d.n_rows === null ? "—" : d.n_rows.toLocaleString("en-US")} rows · target{" "}
          {d.target || "—"}
        </p>
      </div>

      <DatasetWarnings warnings={warnings} />

      {hasMetadata(d) && d.source !== "import" ? (
        <>
          <section aria-labelledby="preview-heading" className="space-y-3">
            <h2 id="preview-heading" className="text-lg font-semibold">
              Preview
            </h2>
            <DataPreview rows={d.preview ?? []} columns={Object.keys(d.metadata.columns)} />
          </section>

          {/* key: remount from fresh server metadata after a save/register refetch */}
          <MetadataEditor
            key={`${d.target}:${JSON.stringify(d.metadata.columns)}`}
            dataset={d}
            locked={locked}
            onLocked={() => setLocked(true)}
            onDirtyChange={setDirty}
          />
          <RegisterPanel dataset={d} dirty={dirty} locked={locked} onLocked={() => setLocked(true)} />
        </>
      ) : (
        <ReadOnlyDataset dataset={d} />
      )}
    </div>
  );
}
