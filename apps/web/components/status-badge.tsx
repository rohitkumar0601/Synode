import {
  Ban,
  CheckCircle2,
  CircleAlert,
  CircleDashed,
  CircleHelp,
  Loader2,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { fmtState } from "@/lib/format";
import type { JobState } from "@/lib/types";

export type StatusState = JobState | "running" | "done_with_failures" | "unknown";

const ICONS: Record<StatusState, LucideIcon> = {
  queued: CircleDashed,
  fitting: Loader2,
  sampling: Loader2,
  generated: Loader2,
  evaluating: Loader2,
  running: Loader2,
  done: CheckCircle2,
  done_with_failures: CircleAlert,
  failed: XCircle,
  cancelled: Ban,
  unknown: CircleHelp,
};

const LABELS: Partial<Record<StatusState, string>> = { done_with_failures: "Done with failures" };

/** States explained on hover/focus (the label alone is ambiguous). */
export const STATE_HELP: Partial<Record<StatusState, string>> = {
  generated: "Generated = synthetic data ready, waiting for metrics/evaluation",
  cancelled: "Cancelled: the run was terminated before this job finished",
};

const STILL = new Set<StatusState>(["queued", "done", "done_with_failures", "failed", "cancelled", "unknown"]);

const isKnown = (s: string): s is StatusState => Object.hasOwn(ICONS, s);

/**
 * Job/run state: icon + text, so colour is never the only signal. A state this app does not know
 * (the API reports "unknown" when the store cannot be read) renders neutrally as "Unknown".
 */
export function StatusBadge({ state: raw }: { state: StatusState | string }) {
  const state: StatusState = isKnown(raw) ? raw : "unknown";
  const Icon = ICONS[state];
  const variant = state === "failed" ? "destructive" : state === "done" ? "default" : state === "cancelled" ? "outline" : "secondary";
  const badge = (
    <Badge variant={variant} data-state={state}>
      <Icon aria-hidden="true" className={STILL.has(state) ? undefined : "motion-safe:animate-spin"} />
      {LABELS[state] ?? fmtState(state)}
    </Badge>
  );
  const help = STATE_HELP[state];
  if (!help) return badge;
  return (
    <Tooltip>
      <TooltipTrigger render={<span tabIndex={0} title={help} className="inline-flex rounded-4xl outline-none focus-visible:ring-2 focus-visible:ring-ring" />}>
        {badge}
      </TooltipTrigger>
      <TooltipContent>{help}</TooltipContent>
    </Tooltip>
  );
}
