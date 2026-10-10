/** Parse the first record (header row) of a CSV: BOM, quoted fields, "" escapes, CRLF. */
export function parseCsvHeader(text: string): string[] {
  const src = text.replace(/^﻿/, "");
  const fields: string[] = [];
  let cur = "";
  let quoted = false;
  let any = false;
  for (let i = 0; i < src.length; i++) {
    const ch = src[i];
    if (quoted) {
      if (ch === '"') {
        if (src[i + 1] === '"') {
          cur += '"';
          i++;
        } else quoted = false;
      } else cur += ch;
    } else if (ch === '"') {
      quoted = true;
      any = true;
    } else if (ch === ",") {
      fields.push(cur);
      cur = "";
      any = true;
    } else if (ch === "\n" || ch === "\r") {
      break;
    } else {
      cur += ch;
      any = true;
    }
  }
  if (any || cur) fields.push(cur);
  // Names are returned exactly as pandas will see them (no trimming).
  return fields;
}

/** Read the header of a CSV file client-side (only the first 64 KB is read). */
export function readCsvHeader(file: Blob): Promise<{ header: string[]; complete: boolean }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      // Incomplete when the 64 KB window ended before the first line break.
      resolve({ header: parseCsvHeader(text), complete: file.size <= 65536 || text.includes("\n") || text.includes("\r") });
    };
    reader.onerror = () => reject(reader.error ?? new Error("could not read the file"));
    reader.readAsText(file.slice(0, 65536));
  });
}
