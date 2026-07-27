#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
output="${repository}/artifacts/jepa-world-model-v0/pilot"
captures="${output}/cases"
manifests="${lab_directory}/jepa_pilot_manifests"
export CAPTURE_DIRECTORY="${captures}"
export EXPERIMENT_DIRECTORY="${manifests}"
export EXPERIMENT_PATH="/experiments/schedule-a.json"
export WORKER_REPLICAS=1
compose=(
  docker compose
  --project-name quantis-jepa-pilot
  --file "${lab_directory}/compose.yaml"
)
case_files=(
  "schedule-a.json"
  "schedule-b.json"
  "schedule-c.json"
)

if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing JEPA pilot output: ${output}" >&2
  exit 1
fi
mkdir -p "${captures}"

cleanup() {
  "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${compose[@]}" build api
export APPLICATION_IMAGE_ID="$(
  docker image inspect quantis-fault-matrix-app:local --format '{{.Id}}'
)"
export APPLICATION_BUILD_CONTEXT_SHA256="$(
  "${repository}/.venv/bin/python" "${lab_directory}/hash_build_context.py"
)"

for case_file in "${case_files[@]}"; do
  case_id="$(
    "${repository}/.venv/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["case_id"])' \
      "${manifests}/${case_file}"
  )"
  case_output="${captures}/${case_id}"
  mkdir -p "${case_output}"
  export CAPTURE_DIRECTORY="${case_output}"
  export EXPERIMENT_PATH="/experiments/${case_file}"
  cleanup
  "${compose[@]}" up --detach redis postgres collector api worker
  "${compose[@]}" run --rm runner
  "${compose[@]}" stop collector
  cleanup
done

cd "${repository}"
.venv/bin/python -m quantis_core train-jepa-world-model \
  --captures-directory "${captures}" \
  --manifests-directory "${manifests}" \
  --feature-spec lab/fault_matrix/feature-spec.json \
  --split-spec lab/fault_matrix/jepa-pilot-split.json \
  --latent-dimension 3 \
  --epochs 200 \
  --seed 12 \
  --output "${output}"
