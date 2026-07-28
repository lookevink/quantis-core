#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
source="${repository}/artifacts/jepa-world-model-v3/observability-graph-confirmation-v1"

cd "${repository}"
.venv/bin/python "${lab_directory}/compile_corpus.py" \
  --captures-directory "${source}/cases" \
  --manifests-directory "${source}/inputs/manifests" \
  --metric-feature-spec "${lab_directory}/feature-spec.json" \
  --log-feature-spec \
    "${repository}/lab/fault_matrix/contextual-v2-log-feature-spec.json" \
  --split-spec "${source}/inputs/split.json" \
  --protocol \
    "${lab_directory}/observability-graph-jepa-confirmation-v1.json" \
  --source-git-commit "$(git rev-parse HEAD)" \
  --output "${source}/graph-cache"
