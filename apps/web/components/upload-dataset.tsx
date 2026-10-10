"use client";

import { toast } from "sonner";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { useId, useRef, useState } from "react";
import { ErrorState } from "@/components/error-state";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NativeSelect } from "@/components/ui/native-select";
import { uploadDataset } from "@/lib/api";
import { readCsvHeader } from "@/lib/csv";
import { tooBigForUpload, tooBigMessage, uploadError } from "@/lib/upload";

const isParquet = (f: File) => /\.parquet$/i.test(f.name);
const stem = (f: File) => f.name.replace(/\.[^.]+$/, "");

export function UploadDataset() {
  const router = useRouter();
  const qc = useQueryClient();
  const ids = { file: useId(), name: useId(), target: useId() };
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [nameTouched, setNameTouched] = useState(false);
  const [target, setTarget] = useState("");
  const [header, setHeader] = useState<string[]>([]);
  const [headerError, setHeaderError] = useState<string | null>(null);
  const fileSeq = useRef(0);

  const upload = useMutation({
    mutationFn: () => uploadDataset(file as File, name.trim(), target),
    onSuccess: async (d) => {
      await qc.invalidateQueries({ queryKey: ["datasets"] });
      toast.success(`Uploaded dataset "${d.name}"`);
      setOpen(false);
      reset();
      router.push(`/datasets/${encodeURIComponent(d.id)}`);
    },
  });

  function reset() {
    fileSeq.current += 1;
    setFile(null);
    setName("");
    setNameTouched(false);
    setTarget("");
    setHeader([]);
    setHeaderError(null);
    upload.reset();
  }

  async function onFile(f: File | null) {
    upload.reset();
    const seq = ++fileSeq.current;
    setFile(f);
    setHeader([]);
    setTarget("");
    setHeaderError(null);
    if (!f) return;
    if (!nameTouched) setName(stem(f));
    if (isParquet(f)) return;
    try {
      const { header: cols, complete } = await readCsvHeader(f);
      if (seq !== fileSeq.current) return; // a newer file superseded this read
      setHeader(cols);
      setTarget(cols.filter((c) => c !== "").at(-1) ?? "");
      if (cols.length === 0) setHeaderError("Could not find a header row in this file.");
      else if (!complete) setHeaderError("No line break in the first 64 KB; the column list may be incomplete.");
    } catch {
      if (seq === fileSeq.current) setHeaderError("Could not read the file's header.");
    }
  }

  const parquet = file ? isParquet(file) : false;
  const tooBig = tooBigForUpload(file);
  const canSubmit = !!file && !tooBig && name.trim() !== "" && target.trim() !== "" && !upload.isPending;

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        if (!o) reset();
      }}
    >
      <DialogTrigger render={<Button />}>
        <Upload aria-hidden="true" />
        Upload CSV/Parquet
      </DialogTrigger>
      <DialogContent>
        <form
          className="grid gap-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (canSubmit) upload.mutate();
          }}
        >
          <DialogHeader>
            <DialogTitle>Upload a dataset</DialogTitle>
            <DialogDescription>
              A single table in CSV or Parquet format. Pick the column the generators should be able to predict.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-1.5">
            <Label htmlFor={ids.file}>File</Label>
            <Input
              id={ids.file}
              type="file"
              accept=".csv,.parquet"
              disabled={upload.isPending}
              onChange={(e) => void onFile(e.target.files?.[0] ?? null)}
            />
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor={ids.name}>Name</Label>
            <Input
              id={ids.name}
              value={name}
              disabled={upload.isPending}
              onChange={(e) => {
                setName(e.target.value);
                setNameTouched(true);
              }}
            />
          </div>
          {file && (
            <div className="grid gap-1.5">
              <Label htmlFor={ids.target}>Target column</Label>
              {parquet ? (
                <Input
                  id={ids.target}
                  value={target}
                  placeholder="column name"
                  disabled={upload.isPending}
                  onChange={(e) => setTarget(e.target.value)}
                />
              ) : (
                <NativeSelect
                  id={ids.target}
                  value={target}
                  disabled={upload.isPending || header.length === 0}
                  onChange={(e) => setTarget(e.target.value)}
                >
                  {header.map((c, i) =>
                    c === "" ? null : (
                      <option key={`${i}:${c}`} value={c}>
                        {c}
                      </option>
                    ),
                  )}
                </NativeSelect>
              )}
              {parquet && (
                <p className="text-xs text-muted-foreground">
                  Parquet headers are not read in the browser; type the target column name exactly.
                </p>
              )}
              {headerError && <p className="text-xs text-destructive">{headerError}</p>}
            </div>
          )}
          {tooBig && (
            <p role="alert" className="text-sm text-destructive">
              {tooBigMessage()}
            </p>
          )}
          {upload.isError && <ErrorState title="Upload failed" error={uploadError(upload.error, file)} />}
          <DialogFooter>
            <Button type="submit" disabled={!canSubmit} aria-busy={upload.isPending}>
              {upload.isPending ? "Uploading…" : "Upload"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
