#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
output="${repository}/artifacts/jepa-world-model-v0/normal-corpus-v1"
inputs="${output}/inputs"
captures="${output}/cases"
manifests="${inputs}/manifests"
split="${inputs}/split.json"
compose=(
  docker compose
  --project-name quantis-jepa-normal
  --file "${lab_directory}/compose.yaml"
)

if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing JEPA corpus output: ${output}" >&2
  exit 1
fi
mkdir -p "${output}"
"${repository}/.venv/bin/python" \
  "${lab_directory}/prepare_jepa_normal_corpus.py" \
  --output "${inputs}"
git -C "${repository}" rev-parse HEAD >"${inputs}/git-commit.txt"
mkdir -p "${captures}"

cleanup() {
  "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

case_files=("${manifests}"/*.json)
export EXPERIMENT_DIRECTORY="${manifests}"
export CAPTURE_DIRECTORY="${captures}"
export EXPERIMENT_PATH="/experiments/$(basename "${case_files[0]}")"
export WORKER_REPLICAS=1
"${compose[@]}" build api
export APPLICATION_IMAGE_ID="$(
  docker image inspect quantis-fault-matrix-app:local --format '{{.Id}}'
)"
export APPLICATION_BUILD_CONTEXT_SHA256="$(
  "${repository}/.venv/bin/python" "${lab_directory}/hash_build_context.py"
)"

for case_path in "${case_files[@]}"; do
  case_file="$(basename "${case_path}")"
  read -r case_id worker_replicas < <(
    "${repository}/.venv/bin/python" -c \
      'import json,sys; p=json.load(open(sys.argv[1])); print(p["case_id"], p["worker_replicas"])' \
      "${case_path}"
  )
  case_output="${captures}/${case_id}"
  mkdir -p "${case_output}"
  export CAPTURE_DIRECTORY="${case_output}"
  export EXPERIMENT_PATH="/experiments/${case_file}"
  export WORKER_REPLICAS="${worker_replicas}"
  cleanup
  "${compose[@]}" up --detach --scale worker="${WORKER_REPLICAS}" \
    redis postgres collector api worker
  "${compose[@]}" run --rm runner
  "${compose[@]}" stop collector
  cleanup
done

cd "${repository}"
.venv/bin/python -m quantis_core train-jepa-world-model \
  --captures-directory "${captures}" \
  --manifests-directory "${manifests}" \
  --feature-spec lab/fault_matrix/feature-spec.json \
  --split-spec "${split}" \
  --latent-dimension 4 \
  --epochs 300 \
  --seed 13 \
  --output "${output}/training"
