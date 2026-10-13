"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Octagon, Play } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ApiError, cancelRun, resumeRun } from "@/lib/api";

const jobs = (n: number) => `${n} job${n === 1 ? "" : "s"}`;
const message = (e: unknown) => (e instanceof ApiError ? e.detail : e instanceof Error ? e.message : String(e));

function useInvalidateRun(runId: string) {
  const qc = useQueryClient();
  return () => {
    void qc.invalidateQueries({ queryKey: ["run", runId] });
    void qc.invalidateQueries({ queryKey: ["report", runId] });
    void qc.invalidateQueries({ queryKey: ["runs"] });
  };
}

export const TERMINATE_PROMPT = "Stop all remaining jobs? Running training is killed; finished results are kept.";

/** "Terminate run" with a confirmation dialog. `compact` renders an icon button (runs list rows). */
export function TerminateRunButton({ runId, compact = false }: { runId: string; compact?: boolean }) {
  const [open, setOpen] = useState(false);
  const invalidate = useInvalidateRun(runId);
  const cancel = useMutation({
    mutationFn: () => cancelRun(runId),
    onSuccess: (r) => {
      setOpen(false);
      toast.success(r.cancelled > 0 ? `Run terminated: ${jobs(r.cancelled)} cancelled` : "Nothing left to terminate");
    },
    onError: (e) => toast.error(`Could not terminate the run: ${message(e)}`),
    onSettled: invalidate,
  });
  return (
    <>
      {compact ? (
        <Button variant="ghost" size="icon-sm" aria-label={`Terminate run ${runId}`} title="Terminate run" onClick={() => setOpen(true)}>
          <Octagon aria-hidden="true" className="text-destructive" />
        </Button>
      ) : (
        <Button variant="destructive" size="sm" onClick={() => setOpen(true)}>
          <Octagon aria-hidden="true" />
          Terminate run
        </Button>
      )}
      <Dialog open={open} onOpenChange={(o) => !cancel.isPending && setOpen(o)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Terminate run?</DialogTitle>
            <DialogDescription>{TERMINATE_PROMPT}</DialogDescription>
          </DialogHeader>
          <p className="break-all font-mono text-xs text-muted-foreground">{runId}</p>
          <DialogFooter>
            <DialogClose render={<Button variant="outline" disabled={cancel.isPending} />}>Keep running</DialogClose>
            <Button variant="destructive" onClick={() => cancel.mutate()} disabled={cancel.isPending}>
              {cancel.isPending ? "Terminating…" : "Terminate run"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

/** A 503 from resume still lists what was re-queued (always [] today), so read `detail`. */
export function ResumeRunButton({ runId }: { runId: string }) {
  const invalidate = useInvalidateRun(runId);
  const resume = useMutation({
    mutationFn: () => resumeRun(runId),
    onSuccess: (r) =>
      r.resumed.length > 0
        ? toast.success(`Resumed ${jobs(r.resumed.length)}`)
        : toast.info("Nothing to resume: every unfinished job is still queued or running"),
    onError: (e) => toast.error(`Could not resume: ${message(e)}`),
    onSettled: invalidate,
  });
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={() => resume.mutate()}
      disabled={resume.isPending}
      title="Re-queue jobs whose queue messages were lost (stuck at generated, or stalled past the time cap)"
    >
      <Play aria-hidden="true" />
      {resume.isPending ? "Resuming…" : "Resume stalled jobs"}
    </Button>
  );
}
