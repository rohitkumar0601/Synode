#!/usr/bin/env bash
# End-to-end platform check over HTTP: compose up (build + wait for health) -> upload a 1,000-row
# toy CSV -> register -> POST /runs (5 generators across q.sdv / q.synthcity / q.smartnoise, then
# metrics on q.synthcity and evaluate on q.eval) -> poll until done -> GET the report -> assert
# train_copy failed the gate and every job reported -> print the ranking -> check the web container.
# Env: E2E_SKIP_UP=1 skips `docker compose up`; E2E_TIMEOUT_S (default 2700) bounds the poll;
# STL_API_PORT / STL_WEB_PORT (default 8010 / 3010) are the compose host ports (also read by
# docker-compose.yml; set STL_PUBLIC_API_URL=http://localhost:$STL_API_PORT to match the web build).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
API="${STL_API:-http://127.0.0.1:${STL_API_PORT:-8010}}"
WEB="${STL_WEB:-http://127.0.0.1:${STL_WEB_PORT:-3010}}"
TIMEOUT_S="${E2E_TIMEOUT_S:-2700}"
WORK="outputs/e2e"
mkdir -p "$WORK"
py() { bash scripts/envrun.sh core python "$@"; }
json() { py -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

if [ "${E2E_SKIP_UP:-0}" != "1" ]; then
  echo "== docker compose up -d --build --wait"
  docker compose up -d --build --wait
fi
curl -fsS "$API/health"; echo

echo "== toy dataset (1,000 rows)"
py - "$WORK/toy.csv" <<'EOF'
import sys
import numpy as np
import pandas as pd
rng = np.random.default_rng(0)
n = 1000
age = rng.integers(18, 90, n)
edu = rng.choice(["hs", "bachelor", "master", "phd"], n, p=[0.4, 0.35, 0.2, 0.05])
sex = rng.choice(["f", "m"], n)
bonus = np.select([edu == "bachelor", edu == "master", edu == "phd"], [15000, 25000, 35000], 0)
income = np.round(20000 + 900 * (age - 18) + bonus + rng.normal(0, 8000, n), 2)
label = np.where(income + rng.normal(0, 10000, n) > 60000, "high", "low")
pd.DataFrame({"age": age, "education": edu, "sex": sex, "income": income,
              "label": label}).to_csv(sys.argv[1], index=False)
print(f"wrote {sys.argv[1]}")
EOF

DS=$(curl -fsS -F "file=@$WORK/toy.csv;type=text/csv" -F name=e2e-toy -F target=label \
       "$API/datasets" | json 'd["id"]')
echo "dataset: $DS"
curl -fsS -X POST -H 'Content-Type: application/json' -d '{}' "$API/datasets/$DS/register" \
  | json '"registered=%s n_rows=%s fingerprint=%s" % (d["registered"], d["n_rows"], d["fingerprint"])'

echo "== POST /runs"
BODY=$(printf '{"dataset_id": "%s", "generators": ["train_copy", "independent_marginals", "gaussian_copula", "ctgan", "mst_eps10"], "seeds": [0], "params": {"ctgan": {"epochs": 2}}}' "$DS")
RUN=$(curl -fsS -X POST -H 'Content-Type: application/json' -d "$BODY" "$API/runs" | json 'd["run_id"]')
echo "run: $RUN"

START=$(date +%s); LAST=""
while :; do
  STATUS=$(curl -fsS "$API/runs/$RUN")
  LINE=$(printf '%s' "$STATUS" | json '"done=%s counts=%s" % (d["done"], d["counts"])')
  if [ "$LINE" != "$LAST" ]; then echo "[$(( $(date +%s) - START ))s] $LINE"; LAST="$LINE"; fi
  if printf '%s' "$STATUS" | json 'd["done"]' | grep -q True; then break; fi
  if [ $(( $(date +%s) - START )) -gt "$TIMEOUT_S" ]; then
    echo "TIMEOUT after ${TIMEOUT_S}s"; printf '%s\n' "$STATUS"; exit 1
  fi
  sleep 10
done
printf '%s' "$STATUS" | json '"\n".join("  %-24s %-8s %s" % (j["generator"], j["state"], (j["error"] or "")[:200].replace(chr(10), " ")) for j in d["jobs"])'

echo "== GET /runs/$RUN/report"
curl -fsS "$API/runs/$RUN/report" > "$WORK/report.json"
py - "$WORK/report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
rec = r["recommendation"]
print(f"jobs_reported={r['jobs_reported']} jobs_total={r['jobs_total']}")
for j in r["per_job"]:
    print(f"  job {j['job_id']:<28} {j['status']:<7} warnings={len(j['warnings'])}")
print(f"winner={rec['winner']!r} preset={rec['preset']} reason={rec['reason']}")
print("ranking:")
for i, c in enumerate(rec["ranking"], 1):
    s = c["scores"]
    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
    print(f"  {i}. {c['generator']:<22} score={fmt(c['score'])} F={fmt(s['fidelity'])} "
          f"U={fmt(s['utility'])} P={fmt(s['privacy'])} gate={'pass' if c['gate_passed'] else 'FAIL'}"
          f" pareto={c['on_pareto_front']} {'; '.join(c['gate_failures'])[:120]}")
assert r["jobs_reported"] == r["jobs_total"], "not every job produced a report"
tc = [c for c in rec["ranking"] if c["generator"] == "train_copy"]
assert tc and not tc[0]["gate_passed"], "train_copy must fail the privacy sanity gate"
assert rec["winner"] != "train_copy"
print("ASSERTIONS OK: train_copy failed the gate; jobs_reported == jobs_total")
EOF

echo "== web container"
for path in / "/runs/$RUN"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$WEB$path")
  echo "GET $WEB$path -> $code"
  [ "$code" = "200" ] || { echo "web check failed"; exit 1; }
done
echo "E2E OK (run $RUN)"
