"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PRESETS, validateWeights, type Weights } from "@/lib/presets";
import type { Run } from "@/lib/types";

export type PresetKey = keyof typeof PRESETS;
/** What the report is currently ranked with. */
export type Ranking = { kind: "preset"; preset: PresetKey } | { kind: "custom"; weights: Weights };

/**
 * The ranking a run was created with: its preset, or its stored custom weights. `missing` is true
 * when the run used custom weights the API did not keep (e.g. an imported run); the caller then
 * falls back to Balanced and says so.
 */
export function initialRanking(run: Pick<Run, "preset" | "weights">): { ranking: Ranking; missing: boolean } {
  if (run.preset in PRESETS) return { ranking: { kind: "preset", preset: run.preset as PresetKey }, missing: false };
  const w = run.weights;
  if (run.preset === "custom" && w && validateWeights(w) === null) {
    return { ranking: { kind: "custom", weights: [w[0], w[1], w[2]] }, missing: false };
  }
  return { ranking: { kind: "preset", preset: "balanced" }, missing: run.preset === "custom" };
}

const AXES = ["Fidelity", "Utility", "Privacy"] as const;
const fmt = (n: number) => (Number.isFinite(n) ? n.toFixed(2) : "?");

export function PresetSwitcher({ value, onChange }: { value: Ranking; onChange: (r: Ranking) => void }) {
  const [tab, setTab] = useState<PresetKey | "custom">(value.kind === "preset" ? value.preset : "custom");
  const [draft, setDraft] = useState<Weights>(value.kind === "custom" ? value.weights : [0.4, 0.3, 0.3]);
  const error = validateWeights(draft);

  const pick = (next: PresetKey | "custom") => {
    setTab(next);
    if (next !== "custom") onChange({ kind: "preset", preset: next });
  };

  return (
    <div className="space-y-3">
      <Tabs value={tab} onValueChange={(v) => pick(v as PresetKey | "custom")}>
        <TabsList aria-label="Ranking preset">
          {(Object.keys(PRESETS) as PresetKey[]).map((k) => (
            <TabsTrigger key={k} value={k}>
              {PRESETS[k].label}
            </TabsTrigger>
          ))}
          <TabsTrigger value="custom">Custom</TabsTrigger>
        </TabsList>
      </Tabs>
      {tab === "custom" ? (
        <div className="space-y-2">
          {AXES.map((axis, i) => (
            <div key={axis} className="grid grid-cols-[7rem_1fr_3rem] items-center gap-3">
              <Label htmlFor={`rw-${axis}`}>{axis} weight</Label>
              <input
                id={`rw-${axis}`}
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={draft[i]}
                className="w-full accent-primary"
                onChange={(e) => {
                  const next = [...draft] as Weights;
                  next[i] = Number(e.target.value);
                  setDraft(next);
                }}
              />
              <span className="text-right text-sm tabular-nums">{fmt(draft[i])}</span>
            </div>
          ))}
          <p className="text-xs text-muted-foreground">Weights are normalised to sum to 1 (fidelity, utility, privacy).</p>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <Button size="sm" disabled={error !== null} onClick={() => onChange({ kind: "custom", weights: draft })}>
            Apply weights
          </Button>
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">
          Weights (fidelity / utility / privacy): {PRESETS[tab].weights.map(fmt).join(" / ")}
        </p>
      )}
    </div>
  );
}
