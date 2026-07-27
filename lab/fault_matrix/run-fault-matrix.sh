#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
output="${repository}/artifacts/fault-matrix"
captures="${output}/cases"
compose=(
  docker compose
  --project-name quantis-fault-matrix
  --file "${lab_directory}/compose.yaml"
)
case_files=(
  "cache-outage.json"
  "database-lock.json"
  "worker-crash.json"
)
compiler_artifact="${repository}/artifacts/fault-lab/window-compiler.json"
detector_artifact="${repository}/artifacts/fault-lab/detector.json"

mkdir -p "${captures}"
rm -f "${output}/report.md" "${output}/verification.json"
compiler_sha256_before="$(shasum -a 256 "${compiler_artifact}" | awk '{print $1}')"
detector_sha256_before="$(shasum -a 256 "${detector_artifact}" | awk '{print $1}')"

cleanup() {
  "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

CAPTURE_DIRECTORY="${captures}" \
EXPERIMENT_PATH="/experiments/cache-outage.json" \
  "${compose[@]}" build
APPLICATION_IMAGE_ID="$(
  docker image inspect quantis-fault-matrix-app:local --format '{{.Id}}'
)"
APPLICATION_BUILD_CONTEXT_SHA256="$(
  "${repository}/.venv/bin/python" "${lab_directory}/hash_build_context.py"
)"
export APPLICATION_IMAGE_ID APPLICATION_BUILD_CONTEXT_SHA256

for case_file in "${case_files[@]}"; do
  case_id="$(
    "${repository}/.venv/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["case_id"])' \
      "${lab_directory}/experiments/${case_file}"
  )"
  case_output="${captures}/${case_id}"
  mkdir -p "${case_output}"
  rm -f "${case_output}/collector-output.jsonl"
  export CAPTURE_DIRECTORY="${case_output}"
  export EXPERIMENT_PATH="/experiments/${case_file}"
  cleanup
  "${compose[@]}" up --detach redis postgres collector api worker
  "${compose[@]}" run --rm runner
  "${compose[@]}" stop collector
  cleanup
done

compiler_sha256_after="$(shasum -a 256 "${compiler_artifact}" | awk '{print $1}')"
detector_sha256_after="$(shasum -a 256 "${detector_artifact}" | awk '{print $1}')"
test "${compiler_sha256_before}" = "${compiler_sha256_after}"
test "${detector_sha256_before}" = "${detector_sha256_after}"

cd "${repository}"
.venv/bin/python -m quantis_core evaluate-fault-matrix \
  --captures-directory artifacts/fault-matrix/cases \
  --manifests-directory lab/fault_matrix/experiments \
  --feature-spec lab/fault_matrix/feature-spec.json \
  --window-compiler artifacts/fault-lab/window-compiler.json \
  --detector artifacts/fault-lab/detector.json \
  --window-compiler-file-sha256 "${compiler_sha256_before}" \
  --detector-file-sha256 "${detector_sha256_before}" \
  --output artifacts/fault-matrix
