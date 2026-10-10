"use client";

import { Label } from "@/components/ui/label";
import { PRESETS, validateWeights, type Weights } from "@/lib/presets";

export type PresetChoice = keyof typeof PRESETS | "custom";

const AXES = ["Fidelity", "Utility", "Privacy"] as const;
const fmt = (n: number) => (Number.isFinite(n) ? n.toFixed(2) : "?");

/** Returns the validation error for the current choice (null when valid). */
export function presetError(preset: PresetChoice, custom: Weights): string | null {
  return preset === "custom" ? validateWeights(custom) : null;
}

export function WeightsEditor({
  preset,
  custom,
  onPresetChange,
  onCustomChange,
}: {
  preset: PresetChoice;
  custom: Weights;
  onPresetChange: (p: PresetChoice) => void;
  onCustomChange: (w: Weights) => void;
}) {
  const error = presetError(preset, custom);
  const set = (i: number, v: number) => {
    const next = [...custom] as Weights;
    next[i] = v;
    onCustomChange(next);
  };
  return (
    <fieldset className="space-y-3">
      <legend className="mb-1 text-sm font-medium">Ranking preset</legend>
      <div className="flex flex-wrap gap-x-6 gap-y-2">
        {(Object.keys(PRESETS) as (keyof typeof PRESETS)[]).map((k) => (
          <label key={k} className="flex items-center gap-2 text-sm">
            <input type="radio" name="preset" className="accent-primary" checked={preset === k} onChange={() => onPresetChange(k)} />
            {PRESETS[k].label}
            <span className="text-xs text-muted-foreground">({PRESETS[k].weights.map(fmt).join(" / ")})</span>
          </label>
        ))}
        <label className="flex items-center gap-2 text-sm">
          <input type="radio" name="preset" className="accent-primary" checked={preset === "custom"} onChange={() => onPresetChange("custom")} />
          Custom
        </label>
      </div>
      {preset === "custom" && (
        <div className="space-y-2">
          {AXES.map((axis, i) => (
            <div key={axis} className="grid grid-cols-[7rem_1fr_3rem] items-center gap-3">
              <Label htmlFor={`weight-${axis}`}>{axis} weight</Label>
              <input
                id={`weight-${axis}`}
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={custom[i]}
                aria-describedby="weights-hint"
                className="w-full accent-primary"
                onChange={(e) => set(i, Number(e.target.value))}
              />
              <span className="text-right text-sm tabular-nums">{fmt(custom[i])}</span>
            </div>
          ))}
          <p id="weights-hint" className="text-xs text-muted-foreground">
            Weights are normalised to sum to 1 (fidelity, utility, privacy).
          </p>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </div>
      )}
    </fieldset>
  );
}
