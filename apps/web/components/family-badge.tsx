import { Badge } from "@/components/ui/badge";

const LABELS: Record<string, string> = {
  sanity_ceiling: "sanity ceiling",
  sanity_floor: "sanity floor",
  statistical: "statistical",
  graphical: "graphical",
  deep: "deep",
  diffusion: "diffusion",
  llm: "LLM",
  differential_privacy: "differential privacy",
  nominal_dp_audit: "audit — not DP",
};

export function familyLabel(family: string): string {
  return LABELS[family] ?? family;
}

export function FamilyBadge({ family, cpuOk = true }: { family: string; cpuOk?: boolean }) {
  const audit = family === "nominal_dp_audit";
  return (
    <span className="inline-flex items-center gap-1">
      <Badge variant={audit ? "ghost" : "outline"} className={audit ? "italic text-muted-foreground" : undefined}>
        {familyLabel(family)}
      </Badge>
      {cpuOk ? null : <Badge variant="secondary">GPU</Badge>}
    </span>
  );
}
