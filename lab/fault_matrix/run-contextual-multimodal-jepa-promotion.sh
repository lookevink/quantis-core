#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
output="${repository}/artifacts/jepa-world-model-v1/contextual-promotion-v1"
inputs="${output}/inputs"
captures="${output}/cases"
manifests="${inputs}/manifests"
split="${inputs}/split.json"
protocol="${lab_directory}/contextual-jepa-promotion-v1.json"
training_a="${output}/training-a"
training_b="${output}/training-b"
compose=(
  docker compose
  --project-name quantis-contextual-jepa-promotion
  --file "${lab_directory}/compose.yaml"
)

preregistered_commit="$(git -C "${repository}" rev-parse HEAD)"
if [[ -n "$(git -C "${repository}" status --porcelain)" ]]; then
  echo "Refusing to collect with a dirty worktree" >&2
  exit 1
fi
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite contextual JEPA promotion: ${output}" >&2
  exit 1
fi
"${repository}/.venv/bin/python" \
  "${lab_directory}/verify_contextual_jepa_promotion.py" \
  --repository "${repository}" \
  --protocol "${protocol}" \
  --commit "${preregistered_commit}"

mkdir -p "${output}"
export QUANTIS_API_REQUEST_QUEUE_SIZE=128
"${repository}/.venv/bin/python" \
  "${lab_directory}/prepare_contextual_promotion_corpus.py" \
  --output "${inputs}"
"${repository}/.venv/bin/python" \
  "${lab_directory}/verify_contextual_jepa_promotion.py" \
  --repository "${repository}" \
  --protocol "${protocol}" \
  --commit "${preregistered_commit}" \
  --inputs "${inputs}"
echo "${preregistered_commit}" >"${inputs}/git-commit.txt"
echo "true" >"${inputs}/worktree-clean.txt"
echo "${QUANTIS_API_REQUEST_QUEUE_SIZE}" \
  >"${inputs}/api-request-queue-size.txt"
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
  "${repository}/.venv/bin/python" \
    "${lab_directory}/hash_build_context.py"
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

train_contextual() {
  local destination="$1"
  cd "${repository}"
  .venv/bin/python -m quantis_core \
    train-contextual-multimodal-jepa-world-model \
    --captures-directory "${captures}" \
    --manifests-directory "${manifests}" \
    --metric-feature-spec lab/fault_matrix/feature-spec.json \
    --log-feature-spec \
      lab/fault_matrix/contextual-promotion-log-feature-spec.json \
    --split-spec "${split}" \
    --horizons 1 3 6 \
    --target-block-size 2 \
    --metric-latent-dimension 3 \
    --log-latent-dimension 1 \
    --pretraining-epochs 200 \
    --predictor-refinement-epochs 100 \
    --cross-validation-epochs 0 \
    --learning-rate 0.02 \
    --ema-decay 0.98 \
    --weight-decay 0.0001 \
    --loss l1 \
    --huber-delta 1.0 \
    --auxiliary-loss-weight 0.2 \
    --rollout-loss-weight 0.2 \
    --calibration-quantile 0.98 \
    --seed 73 \
    --evidence-mode promotion_confirmation \
    --promotion-protocol "${protocol}" \
    --output "${destination}"
}

train_contextual "${training_a}"
train_contextual "${training_b}"
for artifact in \
  corpus.json \
  model.json \
  metrics-only-model.json \
  capacity-matched-metrics-only-model.json \
  shuffled-log-model.json \
  log-only-model.json \
  promotion-training.json \
  report.md
do
  cmp "${training_a}/${artifact}" "${training_b}/${artifact}"
done

if [[ -n "$(git -C "${repository}" status --porcelain)" ]]; then
  echo "Refusing to assess with a dirty worktree" >&2
  exit 1
fi
if [[ "$(
  git -C "${repository}" rev-parse HEAD
)" != "${preregistered_commit}" ]]; then
  echo "Refusing to assess after the preregistered commit changed" >&2
  exit 1
fi
"${repository}/.venv/bin/python" \
  "${lab_directory}/verify_contextual_jepa_promotion.py" \
  --repository "${repository}" \
  --protocol "${protocol}" \
  --commit "${preregistered_commit}" \
  --inputs "${inputs}"

cd "${repository}"
.venv/bin/python -m quantis_core \
  assess-contextual-multimodal-jepa-promotion \
  --training-result "${training_a}/promotion-training.json" \
  --repeat-training-result \
    "${training_b}/promotion-training.json" \
  --training-attestation \
    "${training_a}/execution-attestation.json" \
  --repeat-training-attestation \
    "${training_b}/execution-attestation.json" \
  --promotion-protocol "${protocol}" \
  --repository "${repository}" \
  --preregistered-git-commit "${preregistered_commit}" \
  --output "${output}/promotion"
