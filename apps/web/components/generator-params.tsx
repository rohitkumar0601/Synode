import { budgetLabel } from "@/lib/budget";
import type { GeneratorParams, JobState } from "@/lib/types";
import { cn } from "@/lib/utils";

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** Nested objects become dotted keys (matching the `overrides` names), sorted by key. */
export function flattenParams(params: Record<string, unknown>, prefix = ""): [string, unknown][] {
  const out: [string, unknown][] = [];
  for (const [k, v] of Object.entries(params)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (isPlainObject(v) && Object.keys(v).length > 0) out.push(...flattenParams(v, key));
    else out.push([key, v]);
  }
  return prefix ? out : out.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
}

const show = (v: unknown) => (typeof v === "string" ? v : JSON.stringify(v) ?? String(v));

/**
 * A job's "Parameters" disclosure: library + version, budget profile, the parameters we set
 * instead of the library default, and every resolved parameter. Renders nothing while
 * `params` is undefined (the report has not loaded).
 */
export function GeneratorParamsDisclosure({
  params,
  state,
  label,
}: {
  params: GeneratorParams | null | undefined;
  state: JobState | "unknown";
  label: string;
}) {
  if (params === undefined) return null;
  const overrides = new Set(params?.overrides ?? []);
  const rows = params ? flattenParams(params.params ?? {}) : [];
  return (
    <details className="text-xs">
      <summary className="cursor-pointer py-1 text-muted-foreground underline-offset-4 hover:underline">
        Parameters
      </summary>
      <div role="group" aria-label={`Parameters for ${label}`} className="mt-1 space-y-2">
        {params === null ? (
          <p className="text-muted-foreground">
            {state === "queued"
              ? "Not recorded yet: the parameters are written when training starts."
              : "Not recorded (older run)."}
          </p>
        ) : (
          <>
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
              <dt className="text-muted-foreground">Library</dt>
              <dd>{params.library ? `${params.library}${params.library_version ? ` ${params.library_version}` : ""}` : "—"}</dd>
              <dt className="text-muted-foreground">Budget profile</dt>
              <dd>{params.profile ? budgetLabel(params.profile) : "not recorded"}</dd>
              <dt className="text-muted-foreground">Overrides</dt>
              <dd>
                {params.overrides.length === 0 ? (
                  "none (library defaults)"
                ) : (
                  <ul className="flex flex-wrap gap-1" aria-label="Overridden parameters">
                    {params.overrides.map((o) => (
                      <li key={o} className="rounded bg-amber-100 px-1.5 font-mono text-amber-900 dark:bg-amber-900/40 dark:text-amber-100">
                        {o}
                      </li>
                    ))}
                  </ul>
                )}
              </dd>
            </dl>
            {rows.length === 0 ? (
              <p className="text-muted-foreground">No library parameters (built-in generator).</p>
            ) : (
              <div tabIndex={0} aria-label={`Resolved parameters for ${label}`} className="max-h-72 overflow-auto rounded-md border">
                <table className="w-full font-mono text-xs">
                  <thead className="sticky top-0 bg-muted text-left">
                    <tr>
                      <th scope="col" className="px-2 py-1 font-medium">Parameter</th>
                      <th scope="col" className="px-2 py-1 font-medium">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(([k, v]) => {
                      const over = overrides.has(k);
                      return (
                        <tr
                          key={k}
                          data-override={over ? "true" : undefined}
                          className={cn("border-t", over && "bg-amber-50 dark:bg-amber-900/20")}
                        >
                          <td className="px-2 py-0.5 align-top whitespace-nowrap">{k}</td>
                          <td className="px-2 py-0.5 align-top break-all">
                            {show(v)}
                            {over && (
                              <span className="ml-2 rounded bg-amber-100 px-1 font-sans text-amber-900 dark:bg-amber-900/40 dark:text-amber-100">
                                override
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </details>
  );
}
