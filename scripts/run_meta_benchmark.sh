#!/usr/bin/env bash
# RQ3 meta-benchmark: run the benchmark on every OpenML-CC18 dataset frozen in
# experiments/conf/meta/datasets.yaml (dataset cc18_<id>, run meta-cc18_<id>), then its meta-features.
#
#   bash scripts/run_meta_benchmark.sh [--limit N] [--ids 3,11] [--seeds 0,1,2] [--epochs 50]
#                                      [--generators a,b,...] [--max-rows 5000]
#   smoke: bash scripts/run_meta_benchmark.sh --limit 2 --seeds 0 --epochs 2
#
# Budget profile equal50 (50 epochs for every deep generator, 7,200 s cap): the meta-benchmark keeps
# the equal budget for CPU comparability; --epochs N overrides the epoch count within that profile.
#
# Per dataset: prepare (eval) -> generate (sdv, synthcity, smartnoise, eval) -> metrics (synthcity)
# -> evaluate (eval) -> report (eval) -> metafeatures (meta). Each stage logs to
# outputs/meta_logs/<id>_<stage>.log. A failed stage never stops the loop (only a failed prepare
# skips that dataset's later stages); a summary table of exit codes is printed at the end and the
# script exits 1 if any stage failed.
# The datasets use datasets=openml_generic (no public bounds), so MST takes its documented private-bounds
# fallback (0.2*epsilon on bounds of continuous columns); the bounds path of every MST job
# (runs/<run>/<job>/privacy.json) is collected into outputs/meta_logs/mst_bounds.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIMIT=0
IDS=""
SEEDS="0,1,2"
EPOCHS=""  # empty: the equal50 profile's 50
MAX_ROWS=5000
GENERATORS="train_copy,independent_marginals,gaussian_copula,bayesian_network,ctgan,tvae,mst_eps1"
while [ $# -gt 0 ]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --ids) IDS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --generators) GENERATORS="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOG_DIR="outputs/meta_logs"
STORE="${STL_STORAGE_ROOT:-outputs/store}"
mkdir -p "$LOG_DIR"

# "<id>\t<name>\t<target>" per selected dataset, in YAML order
mapfile -t ROWS < <(bash scripts/envrun.sh eval python - "$LIMIT" "$IDS" <<'EOF'
import sys
import yaml
limit, ids = int(sys.argv[1]), [s for s in sys.argv[2].split(",") if s.strip()]
rows = yaml.safe_load(open("experiments/conf/meta/datasets.yaml", encoding="utf-8"))["datasets"]
if ids:
    known = {str(r["id"]) for r in rows}
    unknown = [i for i in ids if i.strip() not in known]
    if unknown:
        sys.exit(f"ids not in experiments/conf/meta/datasets.yaml: {', '.join(unknown)}")
    rows = [r for r in rows if str(r["id"]) in {i.strip() for i in ids}]
if limit > 0:
    rows = rows[:limit]
for r in rows:
    print(f"{r['id']}\t{r['name']}\t{r['target']}")
EOF
)
if [ "${#ROWS[@]}" -eq 0 ]; then echo "no datasets selected" >&2; exit 2; fi

STAGES=(prepare gen_sdv gen_synthcity gen_smartnoise gen_eval metrics evaluate report metafeatures)
declare -A RC=()
ANY_FAILED=0
run_stage() {  # run_stage <id> <stage> <env> <subcommand...>: log, record exit code, never abort
  local id="$1" stage="$2" env="$3"; shift 3
  local rc=0 log="$LOG_DIR/${id}_${stage}.log"
  echo "[$(date +%T)] cc18_$id $stage ($env)"
  bash scripts/envrun.sh "$env" python experiments/run.py "${OPTS[@]}" "$@" >"$log" 2>&1 || rc=$?
  RC["$id:$stage"]=$rc
  if [ "$rc" -ne 0 ]; then ANY_FAILED=1; echo "  -> exit $rc (see $log)"; fi
  return 0
}

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r ID NAME TARGET <<<"${row%$'\r'}"
  RUN_ID="meta-cc18_$ID"
  OPTS=(-o datasets=openml_generic -o "dataset.name=cc18_$ID" -o dataset.source=openml
        -o "dataset.openml_id=$ID" -o "dataset.target=$TARGET" -o "run_id=$RUN_ID"
        -o "seeds=[$SEEDS]" -o "max_rows=$MAX_ROWS" -o budget=equal50
        -o "generators_to_run=[$GENERATORS]"
        -o eval.singling_out_mode=univariate -o eval.n_attacks=200 -o eval.min_attacks=100)
  if [ -n "$EPOCHS" ]; then OPTS+=(-o "budget.epochs=$EPOCHS"); fi
  echo "=== cc18_$ID ($NAME, target $TARGET), run $RUN_ID ==="
  run_stage "$ID" prepare eval prepare
  if [ "${RC[$ID:prepare]}" -ne 0 ]; then  # OpenML downloads time out now and then: retry once
    mv "$LOG_DIR/${ID}_prepare.log" "$LOG_DIR/${ID}_prepare.try1.log"
    run_stage "$ID" prepare eval prepare
  fi
  if [ "${RC[$ID:prepare]}" -ne 0 ]; then
    for s in "${STAGES[@]:1}"; do RC["$ID:$s"]=skip; done
    continue
  fi
  run_stage "$ID" gen_sdv sdv generate
  run_stage "$ID" gen_synthcity synthcity generate
  run_stage "$ID" gen_smartnoise smartnoise generate
  run_stage "$ID" gen_eval eval generate
  run_stage "$ID" metrics synthcity metrics
  run_stage "$ID" evaluate eval evaluate
  run_stage "$ID" report eval report
  run_stage "$ID" metafeatures meta metafeatures
done

# MST bounds path of every meta run in the store (rebuilt each time, so resumed/partial runs stay listed)
{
  echo "# MST bounds path per job, from $STORE/runs/meta-cc18_*/<job>/privacy.json ($(date -u +%FT%TZ))"
  echo "# private: no public bounds, 0.2*eps spent privately on bounds (preprocessor), 0.8*eps on MST;"
  echo "# none: no continuous columns, all eps on MST; public/mixed would mean codebook bounds were used"
  shopt -s nullglob
  for d in "$STORE"/runs/meta-cc18_*/mst_*/; do
    d="${d%/}"; f="$d/privacy.json"
    job="$(basename "$d")"; run="$(basename "$(dirname "$d")")"
    field() { grep -o "\"$1\": *[^,}]*" "$f" | sed 's/^[^:]*: *//; s/"//g' || true; }
    if [ -f "$f" ]; then
      echo "$run $job bounds=$(field bounds) preprocessor_eps=$(field preprocessor) mst_eps=$(field synthesizer)"
    else  # no privacy record: the job has not run, or failed (e.g. private bounds not found)
      state="$(grep -o '"state": *"[a-z]*"' "$d/status.json" 2>/dev/null | sed 's/.*"\([a-z]*\)"$/\1/' || true)"
      reason="$(grep -o 'MST could not privately infer bounds[^:]*' "$d/status.json" 2>/dev/null | head -1 || true)"
      echo "$run $job bounds=? state=${state:-?}${reason:+ ($reason)}"
    fi
  done
  shopt -u nullglob
} > "$LOG_DIR/mst_bounds.txt"

echo
printf '%-6s %-22s' id name
for s in "${STAGES[@]}"; do printf ' %-14s' "$s"; done
echo
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r ID NAME TARGET <<<"${row%$'\r'}"
  printf '%-6s %-22s' "$ID" "${NAME:0:22}"
  for s in "${STAGES[@]}"; do printf ' %-14s' "${RC[$ID:$s]:-?}"; done
  echo
done
echo "logs: $LOG_DIR/<id>_<stage>.log; MST bounds paths: $LOG_DIR/mst_bounds.txt"
exit "$ANY_FAILED"
