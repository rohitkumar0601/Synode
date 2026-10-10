#!/usr/bin/env bash
# Clone the upstream repositories this project is built on into third_party/,
# pinned to the exact versions the worker environments use.
# Usage: bash scripts/fetch_third_party.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/third_party"
cd "$ROOT/third_party"

# name | url | ref (tag or commit)
REPOS=(
  "synthcity|https://github.com/vanderschaarlab/synthcity.git|v0.2.12"
  "SDV|https://github.com/sdv-dev/SDV.git|v1.38.5"
  "SDMetrics|https://github.com/sdv-dev/SDMetrics.git|v0.32.0"
  "anonymeter|https://github.com/statice/anonymeter.git|v1.0.0"
  "pymfe|https://github.com/ealcobaca/pymfe.git|0.4.4"
  "autorank|https://github.com/sherbold/autorank.git|10b65d3"
  "openml-python|https://github.com/openml/openml-python.git|4a25027"
  "smartnoise-sdk|https://github.com/opendp/smartnoise-sdk.git|677d0bb"
  "DataDesigner|https://github.com/NVIDIA-NeMo/DataDesigner.git|88ecb59"
  "Safe-Synthesizer|https://github.com/NVIDIA-NeMo/Safe-Synthesizer.git|fb4429f"
)

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url ref <<<"$entry"
  if [ -d "$name/.git" ]; then
    echo "$name: present ($(git -C "$name" rev-parse --short HEAD))"
    continue
  fi
  git -c core.longpaths=true init -q "$name"
  git -C "$name" remote add origin "$url"
  git -C "$name" -c core.longpaths=true fetch -q --depth 1 origin "$ref"
  git -C "$name" -c core.longpaths=true checkout -q FETCH_HEAD
  echo "$name @ $ref"
done
