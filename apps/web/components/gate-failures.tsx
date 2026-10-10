import { CheckCircle2, XCircle } from "lucide-react";

/** Gate verdict as icon + text (never colour alone) with the first failure inline. */
export function GateCell({ passed, failures }: { passed: boolean; failures: string[] }) {
  if (passed) {
    return (
      <span className="inline-flex items-center gap-1 text-sm">
        <CheckCircle2 aria-hidden="true" className="size-4 text-primary" />
        Pass
      </span>
    );
  }
  return (
    <div className="space-y-0.5">
      <span className="inline-flex items-center gap-1 text-sm font-medium text-destructive">
        <XCircle aria-hidden="true" className="size-4" />
        Fail
      </span>
      {failures[0] && <p className="max-w-56 text-xs text-muted-foreground">{failures[0]}</p>}
    </div>
  );
}

export function GateFailureList({ failures }: { failures: string[] }) {
  if (failures.length === 0) return null;
  return (
    <div>
      <p className="text-xs font-medium">Gate failures</p>
      <ul className="list-disc pl-5 text-xs text-muted-foreground">
        {failures.map((f) => (
          <li key={f}>{f}</li>
        ))}
      </ul>
    </div>
  );
}
