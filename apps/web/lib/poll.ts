/** Poll interval for live runs (ms). Overridable so tests need not wait 3 s per poll. */
export function pollMs(): number {
  const n = Number(process.env.NEXT_PUBLIC_POLL_MS);
  return Number.isFinite(n) && n > 0 ? n : 3000;
}
