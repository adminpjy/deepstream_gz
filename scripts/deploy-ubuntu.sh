#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) ARCHIVE="${2:?missing value for --archive}"; shift 2 ;;
    --help|-h)
      echo "Usage: scripts/deploy-ubuntu.sh [--archive /path/image.tar]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || { echo "This deploy helper targets Ubuntu Linux." >&2; exit 2; }
[[ -f "${ROOT_DIR}/.env" ]] || {
  echo "Missing .env. Copy .env.example, set a strong database password and review all values." >&2
  exit 2
}
if grep -q 'change-this-local-password' "${ROOT_DIR}/.env"; then
  echo "Refusing production deployment with the example database password in .env." >&2
  exit 2
fi

if [[ -n "${ARCHIVE}" ]]; then
  "${ROOT_DIR}/scripts/import-image.sh" "${ARCHIVE}"
fi

"${ROOT_DIR}/scripts/preflight.sh"

cd "${ROOT_DIR}"
docker compose up -d --no-build
docker compose ps

cat <<'EOF'
Deployment started. Follow logs with:
  docker compose logs -f app postgres

For production, configure external secret management, backup pgvector data and
site-specific RTSP/model settings before accepting traffic.
EOF
