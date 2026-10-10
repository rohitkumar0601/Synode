"use client";

import { AlertTriangle } from "lucide-react";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { PerJobReport } from "@/lib/types";

function WarningsDialog({ job }: { job: PerJobReport }) {
  const n = job.warnings.length;
  return (
    <Dialog>
      <DialogTrigger render={<Button variant="outline" size="sm" />}>
        <AlertTriangle aria-hidden="true" />
        {`${n} warning${n === 1 ? "" : "s"}`}
        <span className="sr-only">{` for job ${job.job_id}`}</span>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{`Warnings for ${job.job_id}`}</DialogTitle>
          <DialogDescription>{`${job.generator}, seed ${job.seed}`}</DialogDescription>
        </DialogHeader>
        <ul className="list-disc space-y-1 pl-5 text-sm">
          {job.warnings.map((w, i) => (
            <li key={`${i}:${w}`} className="font-mono text-xs">
              {w}
            </li>
          ))}
        </ul>
      </DialogContent>
    </Dialog>
  );
}

export function JobWarnings({ jobs }: { jobs: PerJobReport[] }) {
  if (jobs.length === 0) return null;
  const withWarnings = jobs.filter((j) => j.warnings.length > 0).length;
  return (
    <section aria-labelledby="per-job-heading" className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 id="per-job-heading" className="text-base font-semibold">
          Per-job details
        </h3>
        <p className="text-sm text-muted-foreground">
          {withWarnings === 0 ? "No job reported warnings." : `${withWarnings} of ${jobs.length} jobs have warnings.`}
        </p>
      </div>
      <div className="overflow-x-auto">
        <Table aria-label="Per-job details">
          <TableHeader>
            <TableRow>
              <TableHead>Job</TableHead>
              <TableHead>Generator</TableHead>
              <TableHead className="text-right">Seed</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Warnings</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {jobs.map((j) => (
              <TableRow key={j.job_id}>
                <TableCell className="font-mono text-xs">{j.job_id}</TableCell>
                <TableCell>{j.generator}</TableCell>
                <TableCell className="text-right tabular-nums">{j.seed}</TableCell>
                <TableCell>
                  <StatusBadge state={j.status} />
                </TableCell>
                <TableCell>
                  {j.warnings.length > 0 ? <WarningsDialog job={j} /> : <span className="text-muted-foreground">None</span>}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </section>
  );
}
