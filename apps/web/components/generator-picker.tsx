"use client";

import { useId } from "react";
import { FamilyBadge, familyLabel } from "@/components/family-badge";
import { Button } from "@/components/ui/button";
import { generatorDescription } from "@/lib/generator-info";
import type { GeneratorInfo } from "@/lib/types";

const FAMILY_ORDER = [
  "sanity_ceiling",
  "sanity_floor",
  "statistical",
  "graphical",
  "deep",
  "diffusion",
  "llm",
  "differential_privacy",
  "nominal_dp_audit",
];

const isAudit = (g: GeneratorInfo) => g.audit === true || g.family === "nominal_dp_audit";

/** Defaults: CPU-capable, non-audit generators. */
export function defaultSelection(generators: GeneratorInfo[]): string[] {
  return generators.filter((g) => g.cpu_ok && !isAudit(g)).map((g) => g.name);
}

export function GeneratorPicker({
  generators,
  selected,
  onChange,
}: {
  generators: GeneratorInfo[];
  selected: string[];
  onChange: (next: string[]) => void;
}) {
  const hintId = useId();
  const families = [...new Set(generators.map((g) => g.family))].sort((a, b) => {
    const ia = FAMILY_ORDER.indexOf(a);
    const ib = FAMILY_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  const chosen = new Set(selected);
  const toggle = (name: string, on: boolean) => {
    const next = new Set(chosen);
    if (on) next.add(name);
    else next.delete(name);
    onChange(generators.map((g) => g.name).filter((n) => next.has(n)));
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Button type="button" variant="outline" size="sm" aria-describedby={hintId} onClick={() => onChange(defaultSelection(generators))}>
          Select all CPU
        </Button>
        <Button type="button" variant="outline" size="sm" onClick={() => onChange([])}>
          Clear
        </Button>
        <span id={hintId} className="text-xs text-muted-foreground">
          &ldquo;Select all CPU&rdquo; picks every CPU-capable generator except the audit-only ones.
        </span>
      </div>
      {families.map((family) => (
        <fieldset key={family} className="space-y-2">
          <legend className="mb-1 text-sm font-medium">{familyLabel(family)}</legend>
          <ul className="space-y-2">
            {generators
              .filter((g) => g.family === family)
              .map((g) => (
                <li key={g.name} className={isAudit(g) ? "opacity-80" : undefined}>
                  <label className="flex cursor-pointer items-start gap-3 rounded-md border p-2 has-[:focus-visible]:ring-3 has-[:focus-visible]:ring-ring/50">
                    <input
                      type="checkbox"
                      className="mt-1 size-4 accent-primary"
                      aria-labelledby={`${hintId}-${g.name}-name ${hintId}-${g.name}-badge`}
                      aria-describedby={`${hintId}-${g.name}`}
                      checked={chosen.has(g.name)}
                      onChange={(e) => toggle(g.name, e.target.checked)}
                    />
                    <span className="space-y-0.5">
                      <span className="flex flex-wrap items-center gap-2">
                        <span id={`${hintId}-${g.name}-name`} className="font-mono text-sm">
                          {g.name}
                        </span>
                        <span id={`${hintId}-${g.name}-badge`}>
                          <FamilyBadge family={g.family} cpuOk={g.cpu_ok} />
                        </span>
                      </span>
                      <span id={`${hintId}-${g.name}`} className="block text-xs text-muted-foreground">{generatorDescription(g.name)}</span>
                    </span>
                  </label>
                </li>
              ))}
          </ul>
        </fieldset>
      ))}
    </div>
  );
}
