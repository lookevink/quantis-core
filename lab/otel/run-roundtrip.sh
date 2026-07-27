#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
compose_file="$repo_root/lab/otel/compose.yaml"

mkdir -p "$repo_root/artifacts/otlp-replay"

cleanup() {
  docker compose -f "$compose_file" down >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker compose -f "$compose_file" up -d

for _ in $(seq 1 30); do
  if curl --fail --silent http://localhost:13133/ >/dev/null; then
    break
  fi
  sleep 1
done

curl --fail --silent http://localhost:13133/ >/dev/null
"$repo_root/.venv/bin/python" "$repo_root/lab/otel/emit_scenario.py"
sleep 1
docker compose -f "$compose_file" stop

"$repo_root/.venv/bin/python" "$repo_root/lab/otel/verify_roundtrip.py"
