#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
output="${repository}/artifacts/jepa-world-model-v2/contextual-development-v2"
inputs="${output}/inputs"
captures="${output}/cases"
manifests="${inputs}/manifests"
split="${inputs}/split.json"
specification="${repository}/docs/specs/contextual-multimodal-jepa-v2-development.md"
compose=(
  docker compose
  --project-name quantis-contextual-jepa-v2-development
  --file "${lab_directory}/compose.yaml"
)

preregistered_commit="$(git -C "${repository}" rev-parse HEAD)"
if [[ -n "$(git -C "${repository}" status --porcelain)" ]]; then
  echo "Refusing to collect with a dirty worktree" >&2
  exit 1
fi
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite contextual JEPA v2 output: ${output}" >&2
  exit 1
fi

mkdir -p "${output}"
export QUANTIS_API_REQUEST_QUEUE_SIZE=128
"${repository}/.venv/bin/python" \
  "${lab_directory}/prepare_contextual_v2_development_corpus.py" \
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

if ! rg -q \
  'dependency\.(redis|postgresql)\.(latency|operation)' \
  "${captures}"/*/collector-logs.jsonl; then
  echo "No bounded dependency pressure events were captured" >&2
  exit 1
fi
if [[ -n "$(git -C "${repository}" status --porcelain)" ]]; then
  echo "Refusing to train with a dirty worktree" >&2
  exit 1
fi
if [[ "$(
  git -C "${repository}" rev-parse HEAD
)" != "${preregistered_commit}" ]]; then
  echo "Refusing to train after the recorded commit changed" >&2
  exit 1
fi

cd "${repository}"
.venv/bin/python -m quantis_core \
  develop-contextual-multimodal-jepa-v2 \
  --captures-directory "${captures}" \
  --manifests-directory "${manifests}" \
  --metric-feature-spec lab/fault_matrix/feature-spec.json \
  --log-feature-spec lab/fault_matrix/contextual-v2-log-feature-spec.json \
  --split-spec "${split}" \
  --horizons 1 3 6 \
  --target-block-size 2 \
  --metric-latent-dimension 3 \
  --pretraining-epochs 200 \
  --predictor-refinement-epochs 100 \
  --cross-validation-epochs 40 \
  --learning-rate 0.02 \
  --ema-decay 0.98 \
  --weight-decay 0.0001 \
  --loss huber \
  --huber-delta 1.0 \
  --auxiliary-loss-weight 0.2 \
  --rollout-loss-weight 0.2 \
  --calibration-quantile 0.98 \
  --seed 89 \
  --output "${output}/training"
