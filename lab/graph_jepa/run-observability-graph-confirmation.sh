#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
protocol="${lab_directory}/observability-graph-jepa-confirmation-v1.json"
output="${repository}/artifacts/jepa-world-model-v3/observability-graph-confirmation-v1"
inputs="${output}/inputs"

cd "${repository}"
if [[ -e "${output}" ]]; then
  echo "refusing to overwrite graph confirmation output: ${output}" >&2
  exit 1
fi

application_build_context_sha256="$(
  .venv/bin/python "${lab_directory}/hash_build_context.py"
)"
preregistered_build_context_sha256="$(
  .venv/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["corpus"]["application_build_context_sha256"])' \
    "${protocol}"
)"
if [[ "${application_build_context_sha256}" != "${preregistered_build_context_sha256}" ]]; then
  echo "application build context differs from protocol" >&2
  exit 1
fi

.venv/bin/python "${lab_directory}/prepare_confirmation.py" \
  --protocol "${protocol}" \
  --output "${inputs}"

CAPTURE_DIRECTORY="${output}/build-capture-placeholder" \
EXPERIMENT_PATH=/experiments/placeholder.json \
docker compose \
  --project-name quantis-observability-graph-build \
  --file "${lab_directory}/compose.yaml" \
  build api
application_image_id="$(
  docker image inspect quantis-graph-jepa-app:local \
    --format '{{.Id}}'
)"

.venv/bin/python "${lab_directory}/collect_confirmation.py" \
  --protocol "${protocol}" \
  --manifests-directory "${inputs}/manifests" \
  --captures-directory "${output}/cases" \
  --compose-file "${lab_directory}/compose.yaml" \
  --project-prefix quantis-observability-graph-v1 \
  --application-image-id "${application_image_id}" \
  --application-build-context-sha256 \
    "${application_build_context_sha256}" \
  --api-request-queue-size 128 \
  --parallel-jobs 3 \
  --attestation "${output}/collection-attestation.json"
