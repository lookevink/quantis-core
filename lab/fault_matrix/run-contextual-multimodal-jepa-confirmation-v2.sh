#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
output="${repository}/artifacts/jepa-world-model-v2/contextual-confirmation-v2"
inputs="${output}/inputs"
captures="${output}/cases"
manifests="${inputs}/manifests"
split="${inputs}/split.json"
protocol="${lab_directory}/contextual-jepa-confirmation-v2.json"
collection_attestation="${inputs}/collection-attestation.json"
repeat_training="${output}/training-seed-89-repeat"
assessment="${output}/assessment"
seeds=(89 97 101 103 107)
mode="${1:-collect}"
if [[ "${mode}" != "collect" && "${mode}" != "--resume-training" ]]; then
  echo "Usage: $0 [--resume-training]" >&2
  exit 1
fi

preregistered_commit="$(git -C "${repository}" rev-parse HEAD)"
if [[ -n "$(git -C "${repository}" status --porcelain)" ]]; then
  echo "Refusing to run confirmation with a dirty worktree" >&2
  exit 1
fi
if [[ "${mode}" == "collect" && -e "${output}" ]]; then
  echo "Refusing to overwrite confirmation evidence: ${output}" >&2
  exit 1
fi
if [[ "${mode}" == "--resume-training" ]]; then
  for required in \
    "${inputs}" \
    "${captures}" \
    "${split}" \
    "${collection_attestation}" \
    "${inputs}/git-commit.txt"
  do
    if [[ ! -e "${required}" ]]; then
      echo "Cannot resume without collected evidence: ${required}" >&2
      exit 1
    fi
  done
  for destination in \
    "${output}/training-seed-89" \
    "${output}/training-seed-97" \
    "${output}/training-seed-101" \
    "${output}/training-seed-103" \
    "${output}/training-seed-107" \
    "${repeat_training}" \
    "${assessment}"
  do
    if [[ -e "${destination}" ]]; then
      echo "Refusing to overwrite training evidence: ${destination}" >&2
      exit 1
    fi
  done
fi

"${repository}/.venv/bin/python" \
  "${lab_directory}/verify_contextual_jepa_promotion.py" \
  --repository "${repository}" \
  --protocol "${protocol}" \
  --commit "${preregistered_commit}"

if [[ "${mode}" == "--resume-training" ]]; then
  "${repository}/.venv/bin/python" \
    "${lab_directory}/verify_contextual_jepa_promotion.py" \
    --repository "${repository}" \
    --protocol "${protocol}" \
    --commit "${preregistered_commit}" \
    --inputs "${inputs}" \
    --collection-attestation "${collection_attestation}"
  original_collection_commit="$(
    "${repository}/.venv/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["amendments"][0]["original_preregistered_git_commit"])' \
      "${protocol}"
  )"
  observed_collection_commit="$(
    tr -d '\n' <"${inputs}/git-commit.txt"
  )"
  if [[ "${observed_collection_commit}" != "${original_collection_commit}" ]]; then
    echo "Collected corpus commit differs from the amendment" >&2
    exit 1
  fi
  echo "resume-contextual-confirmation-v2: preserved 72-run corpus verified"
else
  mkdir -p "${output}"
  "${repository}/.venv/bin/python" \
    "${lab_directory}/prepare_contextual_confirmation_corpus.py" \
    --protocol "${protocol}" \
    --output "${inputs}"
  "${repository}/.venv/bin/python" \
    "${lab_directory}/verify_contextual_jepa_promotion.py" \
    --repository "${repository}" \
    --protocol "${protocol}" \
    --commit "${preregistered_commit}" \
    --inputs "${inputs}"
  echo "${preregistered_commit}" >"${inputs}/git-commit.txt"
  echo "true" >"${inputs}/worktree-clean.txt"
  echo "128" >"${inputs}/api-request-queue-size.txt"

  export QUANTIS_API_REQUEST_QUEUE_SIZE=128
  export EXPERIMENT_DIRECTORY="${manifests}"
  export CAPTURE_DIRECTORY="${output}/build-capture-placeholder"
  export EXPERIMENT_PATH="/experiments/placeholder.json"
  docker compose \
    --project-name quantis-contextual-confirmation-build \
    --file "${lab_directory}/compose.yaml" \
    build api
  application_image_id="$(
    docker image inspect \
      quantis-fault-matrix-app:local \
      --format '{{.Id}}'
  )"
  application_build_context_sha256="$(
    "${repository}/.venv/bin/python" \
      "${lab_directory}/hash_build_context.py"
  )"

  echo "collect-contextual-confirmation: 3 isolated lanes"
  "${repository}/.venv/bin/python" \
    "${lab_directory}/collect_contextual_confirmation.py" \
    --protocol "${protocol}" \
    --manifests-directory "${manifests}" \
    --captures-directory "${captures}" \
    --compose-file "${lab_directory}/compose.yaml" \
    --project-prefix quantis-contextual-confirmation-v2 \
    --application-image-id "${application_image_id}" \
    --application-build-context-sha256 \
      "${application_build_context_sha256}" \
    --api-request-queue-size 128 \
    --parallel-jobs 3 \
    --attestation "${collection_attestation}"
fi

assert_preregistered_state() {
  if [[ -n "$(git -C "${repository}" status --porcelain)" ]]; then
    echo "Refusing to continue with a dirty worktree" >&2
    exit 1
  fi
  if [[ "$(
    git -C "${repository}" rev-parse HEAD
  )" != "${preregistered_commit}" ]]; then
    echo "Refusing to continue after the commit changed" >&2
    exit 1
  fi
}

train_contextual_confirmation_v2() {
  local seed="$1"
  local destination="$2"
  echo "train-contextual-confirmation-v2: seed ${seed}"
  cd "${repository}"
  .venv/bin/python -m quantis_core \
    train-contextual-multimodal-jepa-world-model \
    --captures-directory "${captures}" \
    --manifests-directory "${manifests}" \
    --metric-feature-spec lab/fault_matrix/feature-spec.json \
    --log-feature-spec \
      lab/fault_matrix/contextual-v2-log-feature-spec.json \
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
    --loss huber \
    --huber-delta 1.0 \
    --auxiliary-loss-weight 0.2 \
    --rollout-loss-weight 0.2 \
    --calibration-quantile 0.98 \
    --seed "${seed}" \
    --evidence-mode promotion_confirmation \
    --promotion-protocol "${protocol}" \
    --output "${destination}"
}

assert_preregistered_state
training_result_arguments=()
training_attestation_arguments=()
for seed in "${seeds[@]}"; do
  destination="${output}/training-seed-${seed}"
  train_contextual_confirmation_v2 "${seed}" "${destination}"
  training_result_arguments+=(
    --training-result "${destination}/promotion-training.json"
  )
  training_attestation_arguments+=(
    --training-attestation \
      "${destination}/execution-attestation.json"
  )
done

train_contextual_confirmation_v2 89 "${repeat_training}"
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
  cmp \
    "${output}/training-seed-89/${artifact}" \
    "${repeat_training}/${artifact}"
done

assert_preregistered_state
"${repository}/.venv/bin/python" \
  "${lab_directory}/verify_contextual_jepa_promotion.py" \
  --repository "${repository}" \
  --protocol "${protocol}" \
  --commit "${preregistered_commit}" \
  --inputs "${inputs}"

echo "assess-contextual-confirmation-v2"
cd "${repository}"
.venv/bin/python -m quantis_core \
  assess-contextual-confirmation-v2 \
  "${training_result_arguments[@]}" \
  "${training_attestation_arguments[@]}" \
  --collection-attestation "${collection_attestation}" \
  --repeat-training-result \
    "${repeat_training}/promotion-training.json" \
  --repeat-training-attestation \
    "${repeat_training}/execution-attestation.json" \
  --confirmation-protocol "${protocol}" \
  --repository "${repository}" \
  --preregistered-git-commit "${preregistered_commit}" \
  --output "${assessment}"
