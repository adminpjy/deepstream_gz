#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD=0
DETACH=0

for arg in "$@"; do
  case "${arg}" in
    --build) BUILD=1 ;;
    --detach|-d) DETACH=1 ;;
    --help|-h)
      echo "Usage: scripts/run.sh [--build] [--detach]"
      exit 0
      ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ -f "${ROOT_DIR}/.env" ]] || {
  echo "Missing .env. Copy .env.example to .env and review it first." >&2
  exit 2
}

args=(up)
[[ "${BUILD}" -eq 1 ]] && args+=(--build)
[[ "${DETACH}" -eq 1 ]] && args+=(-d)

cd "${ROOT_DIR}"
exec docker compose "${args[@]}"

