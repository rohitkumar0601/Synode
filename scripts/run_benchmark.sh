#!/usr/bin/env bash
# Main benchmark for one dataset: bash scripts/run_benchmark.sh DATASET [extra hydra overrides...]
# Budget: profile `library` by default (each library's own epoch defaults, 86,400 s cap per job);
# pass budget=equal50 (50 epochs, 7,200 s cap) or budget=smoke, e.g.
#   bash scripts/run_benchmark.sh adult budget=equal50 seeds=[0]
# A failed generate step does not stop the others; its exit code is recorded and summarised.
set -euo pipefail
DATASET="${1:?usage: run_benchmark.sh DATASET [-o-style overrides, e.g. seeds=[0]]}"
shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OPTS=(-o "datasets=$DATASET")
for o in "$@"; do OPTS+=(-o "$o"); done
RUN=(python experiments/run.py "${OPTS[@]}")

declare -a SUMMARY=()
gen() {  # gen ENV: run generate, record the exit code, never abort
  local rc=0
  bash scripts/envrun.sh "$1" "${RUN[@]}" generate || rc=$?
  SUMMARY+=("generate[$1]=$rc")
}

bash scripts/envrun.sh eval "${RUN[@]}" prepare
gen sdv
gen synthcity
gen smartnoise
gen eval
# synthcity metrics (detection AUC, alpha-precision/beta-recall, DOMIAS MIA) in the synthcity env;
# a failure is recorded and evaluate still runs (missing metrics fail the gate closed)
rc=0
bash scripts/envrun.sh synthcity "${RUN[@]}" metrics || rc=$?
SUMMARY+=("metrics=$rc")
# evaluate exits 1 when some job's report failed; still report on the rest (config error is 2)
rc=0
bash scripts/envrun.sh eval "${RUN[@]}" evaluate || rc=$?
SUMMARY+=("evaluate=$rc")
[ "$rc" -le 1 ] || { echo "exit codes: ${SUMMARY[*]}"; exit "$rc"; }
bash scripts/envrun.sh eval "${RUN[@]}" report

echo "exit codes: ${SUMMARY[*]}"
