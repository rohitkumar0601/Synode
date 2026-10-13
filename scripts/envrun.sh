#!/usr/bin/env bash
# Run a command inside one of the project's uv environments.
# Environments live outside the repo (Windows MAX_PATH); see plan Global Constraints.
# The worker envs install synthcity, SDV, SDMetrics, Anonymeter and smartnoise-synth
# editable from the upstream clones in third_party/ (scripts/fetch_third_party.sh).
# Usage: scripts/envrun.sh <core|runner|analysis|eval|sdv|synthcity|smartnoise|datadesigner|meta|safesynth|api> <command...>
set -euo pipefail
ENV_NAME="$1"; shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$ENV_NAME" in
  core|runner|analysis) PROJECT="$ROOT/packages/$ENV_NAME" ;;
  api)         PROJECT="$ROOT/services/api" ;;
  *)           PROJECT="$ROOT/workers/$ENV_NAME" ;;
esac
case "$ENV_NAME" in
  eval|sdv|synthcity|smartnoise|meta|analysis|safesynth)
    if [ ! -d "$ROOT/third_party" ]; then
      echo "third_party/ is missing: run bash scripts/fetch_third_party.sh first" >&2
      exit 2
    fi ;;
esac
export UV_PROJECT_ENVIRONMENT="${STL_VENV_ROOT:-$HOME/.stl}/$ENV_NAME"
exec uv run --project "$PROJECT" "$@"
