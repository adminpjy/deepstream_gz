#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
ARCHIVE="${ROOT_DIR}/output/deepstream-ai-platform.tar"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: scripts/export-image.sh [--env-file FILE] [--archive FILE]

The exported image is always resolved from the rendered Compose
services.app.image value. There is no independent --image override.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="${2:?missing value for --env-file}"; shift 2 ;;
    --archive) ARCHIVE="${2:?missing value for --archive}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

[[ "${ENV_FILE}" == /* ]] || ENV_FILE="${ROOT_DIR}/${ENV_FILE}"
[[ "${ARCHIVE}" == /* ]] || ARCHIVE="${ROOT_DIR}/${ARCHIVE}"
[[ -f "${ENV_FILE}" ]] || fail "Env file not found: ${ENV_FILE}"
command -v docker >/dev/null 2>&1 || fail "Docker CLI is unavailable."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Host Python is unavailable: ${PYTHON_BIN}"

cd "${ROOT_DIR}"
compose_json="$(docker compose --env-file "${ENV_FILE}" config --format json)" \
  || fail "Compose configuration could not be rendered with ${ENV_FILE}."
image="$(printf '%s' "${compose_json}" | "${PYTHON_BIN}" -c '
import json, sys
try:
    image = json.load(sys.stdin)["services"]["app"]["image"]
except (KeyError, TypeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"services.app.image missing from Compose config: {exc}")
if not isinstance(image, str) or not image.strip():
    raise SystemExit("services.app.image is empty")
print(image)
')" || fail "Unable to resolve services.app.image from Compose."

mkdir -p "$(dirname "${ARCHIVE}")"
docker compose --env-file "${ENV_FILE}" build app
docker image inspect "${image}" >/dev/null \
  || fail "Compose built app image is unavailable: ${image}"
docker save --output "${ARCHIVE}" "${image}"
(
  cd "$(dirname "${ARCHIVE}")"
  sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256"
)
echo "Exported Compose services.app.image ${image} -> ${ARCHIVE}"
echo "Checksum -> ${ARCHIVE}.sha256"

