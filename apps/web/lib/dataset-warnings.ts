"use client";

import { useQuery, type QueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import type { Dataset } from "@/lib/types";

/**
 * Warnings the API attaches to a builtin/register response when it promotes an import placeholder
 * (e.g. "fingerprint differs from imported run(s) ..."). They exist only in that one response, so
 * they are kept in the query cache for the dataset page to show after navigation.
 */
const key = (id: string) => ["dataset-warnings", id] as const;

export function rememberDatasetWarnings(qc: QueryClient, d: Dataset): void {
  if (!d.warnings?.length) return;
  for (const w of d.warnings) toast.warning(w);
  qc.setQueryData<string[]>(key(d.id), d.warnings);
}

export function useDatasetWarnings(id: string): string[] {
  const q = useQuery({ queryKey: key(id), queryFn: () => [] as string[], staleTime: Infinity, gcTime: Infinity });
  return q.data ?? [];
}
