#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
output="${repository}/artifacts/jepa-world-model-v0/multimodal-pilot"
captures="${output}/cases"
inputs="${output}/inputs"
manifests="${lab_directory}/multimodal_pilot_manifests"
split="${lab_directory}/multimodal-pilot-split.json"
compose=(
  docker compose
  --project-name quantis-multimodal-jepa-pilot
  --file "${lab_directory}/compose.yaml"
)

if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite multimodal JEPA pilot: ${output}" >&2
  exit 1
fi
mkdir -p "${captures}" "${inputs}"
git -C "${repository}" rev-parse HEAD >"${inputs}/git-commit.txt"

cleanup() {
  "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

export EXPERIMENT_DIRECTORY="${manifests}"
export CAPTURE_DIRECTORY="${captures}"
export EXPERIMENT_PATH="/experiments/schedule-a.json"
export WORKER_REPLICAS=1
"${compose[@]}" build api
export APPLICATION_IMAGE_ID="$(
  docker image inspect quantis-fault-matrix-app:local --format '{{.Id}}'
)"
export APPLICATION_BUILD_CONTEXT_SHA256="$(
  "${repository}/.venv/bin/python" \
    "${lab_directory}/hash_build_context.py"
)"

for case_path in "${manifests}"/*.json; do
  case_file="$(basename "${case_path}")"
  case_id="$(
    "${repository}/.venv/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["case_id"])' \
      "${case_path}"
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
.venv/bin/python -m quantis_core \
  train-multimodal-jepa-world-model \
  --captures-directory "${captures}" \
  --manifests-directory "${manifests}" \
  --metric-feature-spec lab/fault_matrix/feature-spec.json \
  --log-feature-spec lab/fault_matrix/log-feature-spec.json \
  --split-spec "${split}" \
  --metric-latent-dimension 3 \
  --log-latent-dimension 2 \
  --epochs 200 \
  --seed 44 \
  --output "${output}/training"
