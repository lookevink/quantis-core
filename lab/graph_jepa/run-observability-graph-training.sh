#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
source="${repository}/artifacts/jepa-world-model-v3/observability-graph-confirmation-v1"

cd "${repository}"
.venv/bin/python "${lab_directory}/run_confirmation.py" \
  --cache-index "${source}/graph-cache/cache-index.json" \
  --corpus-protocol \
    "${lab_directory}/observability-graph-jepa-confirmation-v1.json" \
  --training-protocol \
    "${lab_directory}/observability-graph-jepa-training-v1.json" \
  --preregistered-git-commit "$(git rev-parse HEAD)" \
  --output "${source}/training"
