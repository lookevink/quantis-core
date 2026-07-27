#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
output="${repository}/artifacts/jepa-world-model-v0/multimodal-normal-corpus-v2"
inputs="${output}/inputs"
captures="${output}/cases"
manifests="${inputs}/manifests"
split="${inputs}/split.json"
specification="${repository}/docs/specs/multimodal-jepa-corpus-v2.md"
compose=(
  docker compose
  --project-name quantis-multimodal-jepa-normal
  --file "${lab_directory}/compose.yaml"
)

preregistered_commit="$(git -C "${repository}" rev-parse HEAD)"
if [[ -n "$(git -C "${repository}" status --porcelain)" ]]; then
  echo "Refusing to collect with a dirty worktree" >&2
  exit 1
fi
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite multimodal JEPA corpus: ${output}" >&2
  exit 1
fi
mkdir -p "${output}"
export QUANTIS_API_REQUEST_QUEUE_SIZE=128
"${repository}/.venv/bin/python" \
  "${lab_directory}/prepare_multimodal_normal_corpus.py" \
  --output "${inputs}"
echo "${preregistered_commit}" >"${inputs}/git-commit.txt"
echo "true" >"${inputs}/worktree-clean.txt"
echo "${QUANTIS_API_REQUEST_QUEUE_SIZE}" \
  >"${inputs}/api-request-queue-size.txt"
"${repository}/.venv/bin/python" -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  "${specification}" >"${inputs}/specification-sha256.txt"
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

if [[ -n "$(git -C "${repository}" status --porcelain)" ]]; then
  echo "Refusing to train with a dirty worktree" >&2
  exit 1
fi
if [[ "$(
  git -C "${repository}" rev-parse HEAD
)" != "${preregistered_commit}" ]]; then
  echo "Refusing to train after the preregistered commit changed" >&2
  exit 1
fi
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
  --epochs 300 \
  --learning-rate 0.02 \
  --ema-decay 0.98 \
  --weight-decay 0.0001 \
  --calibration-quantile 0.98 \
  --maximum-validation-alert-rate 0.10 \
  --seed 48 \
  --output "${output}/training"
