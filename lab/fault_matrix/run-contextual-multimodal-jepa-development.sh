#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
source_corpus="${repository}/artifacts/jepa-world-model-v0/multimodal-normal-corpus-v2"
output="${repository}/artifacts/jepa-world-model-v1/contextual-multimodal-development"

if [[ ! -d "${source_corpus}/cases" ]]; then
  echo "Missing preserved multimodal v2 corpus: ${source_corpus}" >&2
  exit 1
fi
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite contextual JEPA output: ${output}" >&2
  exit 1
fi

cd "${repository}"
.venv/bin/python -m quantis_core \
  train-contextual-multimodal-jepa-world-model \
  --captures-directory "${source_corpus}/cases" \
  --manifests-directory "${source_corpus}/inputs/manifests" \
  --metric-feature-spec lab/fault_matrix/feature-spec.json \
  --log-feature-spec lab/fault_matrix/log-feature-spec.json \
  --split-spec "${source_corpus}/inputs/split.json" \
  --horizons 1 3 6 \
  --target-block-size 2 \
  --metric-latent-dimension 3 \
  --log-latent-dimension 1 \
  --pretraining-epochs 200 \
  --predictor-refinement-epochs 100 \
  --cross-validation-epochs 40 \
  --learning-rate 0.02 \
  --ema-decay 0.98 \
  --weight-decay 0.0001 \
  --loss l1 \
  --huber-delta 1.0 \
  --auxiliary-loss-weight 0.2 \
  --rollout-loss-weight 0.2 \
  --calibration-quantile 0.98 \
  --seed 48 \
  --output "${output}"
