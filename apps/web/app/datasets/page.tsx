"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ErrorState } from "@/components/error-state";
import { UploadDataset } from "@/components/upload-dataset";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ApiError, listDatasets, loadBuiltin } from "@/lib/api";
import { BUILTINS } from "@/lib/builtins";
import { rememberDatasetWarnings } from "@/lib/dataset-warnings";
import { sourceLabel } from "@/lib/format";

export default function DatasetsPage() {
  const router = useRouter();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["datasets"], queryFn: listDatasets });
  // Built-ins the API refused with 409 (used by runs); retrying cannot succeed.
  const [locked, setLocked] = useState<string[]>([]);
  const builtin = useMutation({
    mutationFn: (name: string) => loadBuiltin(name),
    onError: (e, name) => {
      if (e instanceof ApiError && e.status === 409) setLocked((l) => [...l, name]);
    },
    onSuccess: async (d) => {
      rememberDatasetWarnings(qc, d);
      await qc.invalidateQueries({ queryKey: ["datasets"] });
      await qc.invalidateQueries({ queryKey: ["dataset", d.id] });
      router.push(`/datasets/${encodeURIComponent(d.id)}`);
    },
  });

  const loaded = new Set((q.data ?? []).filter((d) => d.source !== "import").map((d) => d.id));

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">Datasets</h1>
        <UploadDataset />
      </div>

      <section aria-labelledby="builtin-heading" className="space-y-2">
        <h2 id="builtin-heading" className="text-sm font-medium text-muted-foreground">
          Built-in datasets
        </h2>
        <div className="flex flex-wrap gap-2">
          {BUILTINS.map((b) =>
            // Already loaded (an import placeholder still needs loading): link to it; re-loading would
            // re-register it, which the API refuses with 409 once a run uses it.
            loaded.has(b.name) ? (
              <Link
                key={b.name}
                href={`/datasets/${encodeURIComponent(b.name)}`}
                aria-label={`Open built-in: ${b.label}`}
                className={buttonVariants({ variant: "outline" })}
              >
                {b.label}
              </Link>
            ) : (
              <Button
                key={b.name}
                variant="outline"
                aria-label={`Load built-in: ${b.label}`}
                disabled={builtin.isPending || locked.includes(b.name)}
                onClick={() => builtin.mutate(b.name)}
              >
                {builtin.isPending && builtin.variables === b.name ? "Loading…" : b.label}
              </Button>
            ),
          )}
        </div>
        <p role="status" className="text-xs text-muted-foreground">
          {builtin.isPending ? "Loading and registering the dataset; this can take a minute." : ""}
        </p>
        {builtin.isError && <ErrorState title="Could not load the built-in dataset" error={builtin.error} />}
      </section>

      {q.isError ? (
        <ErrorState title="Could not load datasets" error={q.error} />
      ) : q.isPending ? (
        <Skeleton className="h-32 w-full" />
      ) : q.data.length === 0 ? (
        <p className="text-sm text-muted-foreground">No datasets yet. Upload a file or load a built-in dataset.</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Source</TableHead>
                <TableHead className="text-right">Rows</TableHead>
                <TableHead>Target</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {q.data.map((d) => (
                <TableRow key={d.id} data-dataset={d.id}>
                  <TableCell className="font-medium">
                    <Link href={`/datasets/${encodeURIComponent(d.id)}`} className="underline-offset-4 hover:underline">
                      {d.name}
                    </Link>
                  </TableCell>
                  <TableCell>{sourceLabel(d.source)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {d.n_rows === null ? "—" : d.n_rows.toLocaleString("en-US")}
                  </TableCell>
                  <TableCell>{d.target}</TableCell>
                  <TableCell>
                    {d.registered ? <Badge>Registered</Badge> : <Badge variant="secondary">Not registered</Badge>}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
