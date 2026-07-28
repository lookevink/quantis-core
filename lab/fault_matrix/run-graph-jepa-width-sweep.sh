#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
source="${repository}/artifacts/jepa-world-model-v2/contextual-confirmation-v2"
pilot="${repository}/artifacts/jepa-world-model-v3/graph-observability-pilot-v1"
output="${pilot}/width-sweep"

cd "${repository}"
.venv/bin/python \
  "${lab_directory}/run_graph_jepa_width_sweep.py" \
  --captures-directory "${source}/cases" \
  --manifests-directory "${source}/inputs/manifests" \
  --metric-feature-spec \
    "${lab_directory}/feature-spec.json" \
  --log-feature-spec \
    "${lab_directory}/contextual-v2-log-feature-spec.json" \
  --split-spec "${source}/inputs/split.json" \
  --output "${output}"
