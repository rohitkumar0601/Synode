import { CheckCircle2, CircleDashed, CircleHelp, Loader2, XCircle, type LucideIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { fmtState } from "@/lib/format";
import type { JobState } from "@/lib/types";

export type StatusState = JobState | "running" | "unknown";

const ICONS: Record<StatusState, LucideIcon> = {
  queued: CircleDashed,
  fitting: Loader2,
  sampling: Loader2,
  generated: Loader2,
  evaluating: Loader2,
  running: Loader2,
  done: CheckCircle2,
  failed: XCircle,
  unknown: CircleHelp,
};

const isKnown = (s: string): s is StatusState => Object.hasOwn(ICONS, s);

/**
 * Job/run state: icon + text, so colour is never the only signal. A state this app does not know
 * (the API reports "unknown" when the store cannot be read) renders neutrally as "Unknown".
 */
export function StatusBadge({ state: raw }: { state: StatusState | string }) {
  const state: StatusState = isKnown(raw) ? raw : "unknown";
  const Icon = ICONS[state];
  const variant = state === "failed" ? "destructive" : state === "done" ? "default" : "secondary";
  const spin = state !== "queued" && state !== "done" && state !== "failed" && state !== "unknown";
  return (
    <Badge variant={variant} data-state={state}>
      <Icon aria-hidden="true" className={spin ? "motion-safe:animate-spin" : undefined} />
      {fmtState(state)}
    </Badge>
  );
}
