#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
output="${repository}/artifacts/demand-conditioned-v2/matched-topology-diagnostic"
captures="${output}/cases"
model="${repository}/artifacts/demand-conditioned-v2/model.json"
experiments="${lab_directory}/experiments_v2_matched"
protocol="${lab_directory}/v2-matched-topology-protocol.json"
protocol_verifier="${lab_directory}/verify_v2_protocol.py"
compose=(
  docker compose
  --project-name quantis-v2-matched
  --file "${lab_directory}/compose.yaml"
)
case_files=(
  "${experiments}/workers-1-cache-outage.json"
  "${experiments}/workers-2-cache-outage.json"
  "${experiments}/workers-3-cache-outage.json"
  "${experiments}/workers-2-database-lock.json"
  "${experiments}/workers-3-database-lock.json"
  "${experiments}/workers-1-database-lock.json"
  "${experiments}/workers-3-worker-crash.json"
  "${experiments}/workers-1-worker-crash.json"
  "${experiments}/workers-2-worker-crash.json"
)

mkdir -p "${captures}"
rm -f "${output}/report.md" "${output}/verification.json"
preregistered_git_commit="$(git -C "${repository}" rev-parse HEAD)"
"${repository}/.venv/bin/python" "${protocol_verifier}" \
  --repository "${repository}" \
  --protocol "${protocol}" \
  --commit "${preregistered_git_commit}"
model_sha256_before="$(
  "${repository}/.venv/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["model_file_sha256"])' \
    "${protocol}"
)"

cleanup() {
  "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

export EXPERIMENT_DIRECTORY="${experiments}"
CAPTURE_DIRECTORY="${captures}" \
EXPERIMENT_PATH="/experiments/$(basename "${case_files[0]}")" \
  "${compose[@]}" build
APPLICATION_IMAGE_ID="$(
  docker image inspect quantis-fault-matrix-app:local --format '{{.Id}}'
)"
APPLICATION_BUILD_CONTEXT_SHA256="$(
  "${repository}/.venv/bin/python" "${lab_directory}/hash_build_context.py"
)"
export APPLICATION_IMAGE_ID APPLICATION_BUILD_CONTEXT_SHA256

for case_path in "${case_files[@]}"; do
  case_file="$(basename "${case_path}")"
  case_id="$(
    "${repository}/.venv/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["case_id"])' \
      "${case_path}"
  )"
  WORKER_REPLICAS="$(
    "${repository}/.venv/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["worker_replicas"])' \
      "${case_path}"
  )"
  case_output="${captures}/${case_id}"
  mkdir -p "${case_output}"
  rm -f "${case_output}/collector-output.jsonl"
  export CAPTURE_DIRECTORY="${case_output}"
  export EXPERIMENT_PATH="/experiments/${case_file}"
  export WORKER_REPLICAS
  cleanup
  "${compose[@]}" up --detach --scale worker="${WORKER_REPLICAS}" \
    redis postgres collector api worker
  "${compose[@]}" run --rm runner
  "${compose[@]}" stop collector
  cleanup
done

model_sha256_after="$(shasum -a 256 "${model}" | awk '{print $1}')"
test "${model_sha256_before}" = "${model_sha256_after}"
"${repository}/.venv/bin/python" "${protocol_verifier}" \
  --repository "${repository}" \
  --protocol "${protocol}" \
  --commit "${preregistered_git_commit}"

cd "${repository}"
.venv/bin/python -m quantis_core evaluate-demand-conditioned-matrix \
  --captures-directory artifacts/demand-conditioned-v2/matched-topology-diagnostic/cases \
  --manifests-directory lab/fault_matrix/experiments_v2_matched \
  --feature-spec lab/fault_matrix/feature-spec.json \
  --model artifacts/demand-conditioned-v2/model.json \
  --model-file-sha256 "${model_sha256_before}" \
  --confirmation-protocol lab/fault_matrix/v2-matched-topology-protocol.json \
  --preregistered-git-commit "${preregistered_git_commit}" \
  --output artifacts/demand-conditioned-v2/matched-topology-diagnostic
