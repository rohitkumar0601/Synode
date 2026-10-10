"use client";

import { toast } from "sonner";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { TriangleAlert } from "lucide-react";
import Link from "next/link";
import { useId, useState } from "react";
import { ErrorState, errorDetail } from "@/components/error-state";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiError, importBundle } from "@/lib/api";
import { tooBigForUpload, tooBigMessage, uploadError } from "@/lib/upload";

export default function ImportPage() {
  const qc = useQueryClient();
  const ids = { file: useId(), force: useId() };
  const [file, setFile] = useState<File | null>(null);
  const [force, setForce] = useState(false);

  const imp = useMutation({
    mutationFn: () => importBundle(file as File, force),
    onSuccess: (r) => {
      toast.success(`Imported ${r.imported} job report${r.imported === 1 ? "" : "s"} into run ${r.run_id}`);
      void qc.invalidateQueries({ queryKey: ["runs"] });
      void qc.invalidateQueries({ queryKey: ["datasets"] });
    },
  });

  const tooBig = tooBigForUpload(file);
  const result = imp.data;
  return (
    <div className="max-w-2xl space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Import results</h1>
        <p className="text-sm text-muted-foreground">
          Upload a results bundle (.zip) produced by a remote or GPU worker. The run then appears in the run list.
        </p>
      </header>
      <form
        className="grid gap-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (file && !tooBig && !imp.isPending) imp.mutate();
        }}
      >
        <div className="grid gap-1.5">
          <Label htmlFor={ids.file}>Results bundle (.zip)</Label>
          <Input
            id={ids.file}
            type="file"
            accept=".zip,application/zip"
            disabled={imp.isPending}
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              imp.reset();
            }}
          />
        </div>
        <div className="flex items-center gap-2">
          <Checkbox id={ids.force} checked={force} onCheckedChange={(c) => setForce(c === true)} disabled={imp.isPending} />
          <Label htmlFor={ids.force}>Force import</Label>
        </div>
        {force && (
          <p role="note" className="flex items-start gap-2 rounded-md border border-destructive/50 px-3 py-2 text-sm">
            <TriangleAlert aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-destructive" />
            <span>
              Force will overwrite existing job results that have the same job id, even when their contents differ. Use
              it only when you are sure the bundle is the one you want.
            </span>
          </p>
        )}
        {tooBig && (
          <p role="alert" className="text-sm text-destructive">
            {tooBigMessage()}
          </p>
        )}
        <div>
          <Button type="submit" disabled={!file || tooBig || imp.isPending} aria-busy={imp.isPending}>
            {imp.isPending ? "Importing…" : "Import"}
          </Button>
        </div>
      </form>

      {imp.isError && (
        <ErrorState
          title="Import failed"
          error={
            imp.error instanceof ApiError && imp.error.status === 422
              ? new Error(`${errorDetail(imp.error)}. Nothing was imported.`)
              : uploadError(imp.error, file)
          }
        />
      )}

      {result && (
        <section aria-labelledby="import-result" className="space-y-2 rounded-md border p-4">
          <h2 id="import-result" className="font-semibold">
            Import finished
          </h2>
          <p className="text-sm">
            {`Imported ${result.imported} job report${result.imported === 1 ? "" : "s"} into run `}
            <Link href={`/runs/${encodeURIComponent(result.run_id)}`} className="font-mono underline underline-offset-2">
              {result.run_id}
            </Link>
            .
          </p>
          {result.warnings.length > 0 && (
            <div>
              <p className="text-sm font-medium">Warnings</p>
              <ul className="list-disc pl-5 text-sm text-muted-foreground">
                {result.warnings.map((w, i) => (
                  <li key={`${i}:${w}`}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
