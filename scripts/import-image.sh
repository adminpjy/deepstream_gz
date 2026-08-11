#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:-${IMAGE_ARCHIVE:-output/deepstream-ai-platform.tar}}"
[[ -f "${ARCHIVE}" ]] || { echo "Image archive not found: ${ARCHIVE}" >&2; exit 2; }

if [[ -f "${ARCHIVE}.sha256" ]]; then
  (cd "$(dirname "${ARCHIVE}")" && sha256sum --check "$(basename "${ARCHIVE}.sha256")")
else
  echo "Warning: checksum file not found: ${ARCHIVE}.sha256" >&2
fi

docker load --input "${ARCHIVE}"

