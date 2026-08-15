#!/usr/bin/env bash
set -euo pipefail

if [[ -f /tmp/deepstream-ai.failed ]]; then
  echo "application reported a fatal state" >&2
  exit 1
fi

if ! pgrep -f 'python(3)? .*[-]m deepstream_ai' >/dev/null 2>&1; then
  echo "deepstream_ai process is not running" >&2
  exit 1
fi

HEALTH_FILE="${APP_HEALTH_FILE:-/tmp/deepstream-ai.ready}"
if [[ ! -s "${HEALTH_FILE}" ]]; then
  echo "application readiness file is missing or empty: ${HEALTH_FILE}" >&2
  exit 1
fi

python3 -c 'import json, urllib.request; response = urllib.request.urlopen("http://127.0.0.1:8080/health/ready", timeout=3); document = json.load(response); assert document.get("status") in {"ready", "degraded"}; assert (document.get("legacy") or {}).get("status") == "ready"'
