#!/usr/bin/env bash
set -euo pipefail

lab_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "${lab_directory}/../.." && pwd)"
output="${repository}/artifacts/fault-lab"
compose=(docker compose --project-name quantis-fault-lab --file "${lab_directory}/compose.yaml")

mkdir -p "${output}"
rm -f \
  "${output}/collector-output.jsonl" \
  "${output}/detector.json" \
  "${output}/report.md" \
  "${output}/verification.json" \
  "${output}/window-compiler.json"

cleanup() {
  "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${compose[@]}" build
APPLICATION_IMAGE_ID="$(
  docker image inspect quantis-fault-lab-app:local --format '{{.Id}}'
)"
APPLICATION_BUILD_CONTEXT_SHA256="$(
  .venv/bin/python "${lab_directory}/hash_build_context.py"
)"
export APPLICATION_IMAGE_ID APPLICATION_BUILD_CONTEXT_SHA256

"${compose[@]}" up --detach redis postgres collector api worker
"${compose[@]}" run --rm runner
"${compose[@]}" stop collector

cd "${repository}"
.venv/bin/python -m quantis_core evaluate-fault-lab \
  --capture artifacts/fault-lab/collector-output.jsonl \
  --feature-spec lab/fault/feature-spec.json \
  --manifest lab/fault/experiment.json \
  --output artifacts/fault-lab
