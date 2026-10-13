#!/usr/bin/env bash
# Generate thesis report figures, tables, and results.md from the artifact store(s).
# Usage:
#   bash scripts/make_report.sh [--allow-smoke] [--store PATH]... [-o OUT_DIR]
# By default, reads outputs/store and outputs/store_integrated (if present) and outputs to reports/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

declare -a ARGS=()
declare -a STORES=()
OUT_DIR="reports"
ALLOW_SMOKE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --store)
      STORES+=("$2")
      shift 2
      ;;
    -o|--output)
      OUT_DIR="$2"
      shift 2
      ;;
    --allow-smoke)
      ALLOW_SMOKE="--allow-smoke"
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [ ${#STORES[@]} -eq 0 ]; then
  if [ -d "outputs/store" ]; then
    STORES+=("outputs/store")
  fi
  if [ -d "outputs/store_integrated" ]; then
    STORES+=("outputs/store_integrated")
  fi
fi

CMD_ARGS=("report")
for s in "${STORES[@]}"; do
  CMD_ARGS+=(--store "$s")
done
CMD_ARGS+=(-o "$OUT_DIR")
if [ -n "$ALLOW_SMOKE" ]; then
  CMD_ARGS+=("$ALLOW_SMOKE")
fi
if [ ${#ARGS[@]} -gt 0 ]; then
  CMD_ARGS+=("${ARGS[@]}")
fi

echo "Running report generation on stores: ${STORES[*]} -> $OUT_DIR"
bash scripts/envrun.sh analysis python -m stl_analysis "${CMD_ARGS[@]}"
