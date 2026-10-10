"use client";

import { Bar, BarChart, Rectangle, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtScore } from "@/lib/format";
import type { RankedCandidate } from "@/lib/types";

interface Datum {
  generator: string;
  score: number;
  gate_passed: boolean;
}

/** Horizontal bar chart of weighted scores. Gate-failing bars are grey and hatched, so colour is not the only cue. */
export function ScoreChart({ ranking }: { ranking: RankedCandidate[] }) {
  const data: Datum[] = ranking
    .filter((c): c is RankedCandidate & { score: number } => c.score !== null && Number.isFinite(c.score))
    .map((c) => ({ generator: c.generator, score: c.score, gate_passed: c.gate_passed }));
  if (data.length === 0) return null;
  const summary = data.map((d) => `${d.generator} ${fmtScore(d.score)}${d.gate_passed ? "" : " (fails gate)"}`).join(", ");
  const height = 40 + data.length * 32;

  return (
    <figure className="space-y-2">
      <div role="img" aria-label={`Weighted scores by generator: ${summary}`}>
        <ResponsiveContainer width="100%" height={height} initialDimension={{ width: 640, height }}>
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
            <defs>
              <pattern id="gate-fail-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="6" height="6" fill="var(--muted)" />
                <line x1="0" y1="0" x2="0" y2="6" stroke="var(--muted-foreground)" strokeWidth="2" />
              </pattern>
            </defs>
            <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" />
            <YAxis type="category" dataKey="generator" width={150} tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" />
            <Tooltip formatter={(v) => fmtScore(typeof v === "number" ? v : Number(v))} />
            <Bar
              dataKey="score"
              isAnimationActive={false}
              shape={(p: unknown) => {
                const props = p as React.ComponentProps<typeof Rectangle> & { payload: Datum };
                const pass = props.payload.gate_passed;
                return (
                  <Rectangle
                    {...props}
                    fill={pass ? "var(--primary)" : "url(#gate-fail-hatch)"}
                    stroke={pass ? undefined : "var(--muted-foreground)"}
                  />
                );
              }}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="text-xs text-muted-foreground">
        Solid bars pass the privacy gate; grey hatched bars fail it.
      </figcaption>
    </figure>
  );
}
