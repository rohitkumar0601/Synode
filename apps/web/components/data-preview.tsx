import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

function cell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export function DataPreview({ rows, columns }: { rows: Record<string, unknown>[]; columns?: string[] }) {
  if (rows.length === 0) return <p className="text-sm text-muted-foreground">No preview available.</p>;
  const cols = columns?.length ? columns : Object.keys(rows[0]);
  return (
    <div className="overflow-x-auto rounded-lg border">
      <Table aria-label="Data preview">
        <TableHeader>
          <TableRow>
            {cols.map((c) => (
              <TableHead key={c}>{c}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r, i) => (
            <TableRow key={i}>
              {cols.map((c) => (
                <TableCell key={c} className="tabular-nums">
                  {cell(r[c])}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
