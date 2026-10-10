"use client";

import { toast } from "sonner";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useState } from "react";
import { ErrorState } from "@/components/error-state";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { NativeSelect } from "@/components/ui/native-select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ApiError, updateMetadata } from "@/lib/api";
import type { ColumnType, DatasetWithMetadata } from "@/lib/types";

const TYPES: ColumnType[] = ["numerical", "categorical", "datetime", "id"];

/** Mirrors the API: a categorical target is classification, anything else regression. */
export function taskFor(columns: Record<string, ColumnType>, target: string): string {
  return columns[target] === "categorical" ? "classification" : "regression";
}

export function MetadataEditor({
  dataset,
  locked = false,
  onLocked,
  onDirtyChange,
}: {
  dataset: DatasetWithMetadata;
  /** The API refused edits with 409 (dataset used by runs). */
  locked?: boolean;
  onLocked?: () => void;
  onDirtyChange?: (dirty: boolean) => void;
}) {
  const qc = useQueryClient();
  const targetId = useId();
  const [columns, setColumns] = useState<Record<string, ColumnType>>(dataset.metadata.columns);
  const [target, setTarget] = useState(dataset.metadata.target);
  const [saved, setSaved] = useState(false);

  const dirty =
    target !== dataset.metadata.target ||
    Object.keys(columns).some((c) => columns[c] !== dataset.metadata.columns[c]);

  useEffect(() => {
    onDirtyChange?.(dirty);
  }, [dirty, onDirtyChange]);

  const idTarget = columns[target] === "id";

  const save = useMutation({
    mutationFn: () => updateMetadata(dataset.id, { columns, target }),
    onError: (e) => {
      if (e instanceof ApiError && e.status === 409) onLocked?.();
    },
    onSuccess: async () => {
      setSaved(true);
      toast.success("Metadata saved");
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["dataset", dataset.id] }),
        qc.invalidateQueries({ queryKey: ["datasets"] }),
      ]);
    },
  });

  function touch() {
    setSaved(false);
    save.reset();
  }

  const names = Object.keys(columns);
  return (
    <section aria-labelledby="metadata-heading" className="space-y-3">
      <h2 id="metadata-heading" className="text-lg font-semibold">
        Column metadata
      </h2>
      <div className="flex flex-wrap items-end gap-4">
        <div className="grid w-56 gap-1.5">
          <Label htmlFor={targetId}>Target column</Label>
          <NativeSelect
            id={targetId}
            value={target}
            onChange={(e) => {
              setTarget(e.target.value);
              touch();
            }}
          >
            {names
              .filter((c) => columns[c] !== "id" || c === target)
              .map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
          </NativeSelect>
        </div>
        <p className="pb-1.5 text-sm">
          <span className="text-muted-foreground">Task:</span>{" "}
          <span className="font-medium">{taskFor(columns, target)}</span>
        </p>
      </div>
      <div className="overflow-x-auto rounded-lg border">
        <Table aria-label="Column types">
          <TableHeader>
            <TableRow>
              <TableHead>Column</TableHead>
              <TableHead>Type</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {names.map((c) => (
              <TableRow key={c}>
                <TableCell className="font-medium">
                  {c}
                  {c === target && <span className="ml-2 text-xs text-muted-foreground">(target)</span>}
                </TableCell>
                <TableCell>
                  <NativeSelect
                    aria-label={`Type of column ${c}`}
                    className="w-40"
                    value={columns[c]}
                    onChange={(e) => {
                      setColumns({ ...columns, [c]: e.target.value as ColumnType });
                      touch();
                    }}
                  >
                    {TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </NativeSelect>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      {idTarget && (
        <p role="alert" className="text-sm text-destructive">
          The target column cannot be an id column. Pick another target or change this column&apos;s type.
        </p>
      )}
      {save.isError && <ErrorState title="Could not save metadata" error={save.error} />}
      <div className="flex items-center gap-3">
        <Button
          onClick={() => save.mutate()}
          disabled={save.isPending || locked || !dirty || idTarget}
          aria-busy={save.isPending}>
          {save.isPending ? "Saving…" : "Save metadata"}
        </Button>
        <span role="status" className="text-sm text-muted-foreground">
          {saved ? "Metadata saved. Register the dataset again to use it." : dirty ? "Unsaved changes" : ""}
        </span>
      </div>
    </section>
  );
}
