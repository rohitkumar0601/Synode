#!/usr/bin/env bash
# Smoke run: Adult capped at 2,000 rows, one seed, the full CPU lineup across the sdv, synthcity and
# smartnoise envs (MST epsilon sweep, plus the synthcity PrivBayes nominal-DP audit entries), then the
# synthcity metrics stage.
# Deep generators use few epochs. A failed stage does not stop the later ones; every stage's exit code
# is summarised on exit and the script exits with the worst one. SMOKE_RUN_ID picks the run (default adult-smoke).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${SMOKE_RUN_ID:-adult-smoke}"
OPTS=(-o seeds=[0] -o max_rows=2000 -o "run_id=$RUN_ID" -o eval.n_attacks=300
      -o eval.singling_out_mode=univariate -o budget.time_cap_seconds=1800 -o budget.epochs=5
      -o "generators_to_run=[train_copy,independent_marginals,marginals,gaussian_copula,bayesian_network,ctgan,tvae,mst_eps0.1,mst_eps1,mst_eps10,privbayes_eps0.1,privbayes_eps1,privbayes_eps10]")
declare -a SUMMARY=()
WORST=0
trap 'echo "exit codes (run $RUN_ID): ${SUMMARY[*]}; worst=$WORST"' EXIT
stage() {  # stage <label> <env> <subcommand>: run it, record its exit code, keep going
  local rc=0
  bash scripts/envrun.sh "$2" python experiments/run.py "${OPTS[@]}" "$3" || rc=$?
  SUMMARY+=("$1=$rc")
  if [ "$rc" -gt "$WORST" ]; then WORST=$rc; fi
}
stage prepare eval prepare
if [ "$WORST" -ne 0 ]; then exit "$WORST"; fi  # nothing to generate without a prepared run
stage generate[sdv] sdv generate
stage generate[synthcity] synthcity generate
stage generate[smartnoise] smartnoise generate
stage metrics synthcity metrics  # synthcity detection / alpha-precision / DOMIAS MIA
stage evaluate eval evaluate
stage report eval report
exit "$WORST"
