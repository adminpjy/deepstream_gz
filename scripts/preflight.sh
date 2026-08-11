#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TEST_IMAGE="${GPU_TEST_IMAGE:-nvidia/cuda:13.0.2-base-ubuntu24.04}"
ENV_FILE="${ROOT_DIR}/.env"
CONFIG_FILE="${ROOT_DIR}/configs/config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_GPU_PULL=0

usage() {
  cat <<'EOF'
Usage: scripts/preflight.sh [--skip-gpu-pull] [--env-file FILE]
                            [--config FILE] [--python EXECUTABLE]
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '[preflight] %s\n' "$*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-gpu-pull) SKIP_GPU_PULL=1; shift ;;
    --env-file) ENV_FILE="${2:?missing value for --env-file}"; shift 2 ;;
    --config) CONFIG_FILE="${2:?missing value for --config}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?missing value for --python}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) fail "未知参数: $1" ;;
  esac
done

[[ "${ENV_FILE}" == /* ]] || ENV_FILE="${ROOT_DIR}/${ENV_FILE}"
[[ "${CONFIG_FILE}" == /* ]] || CONFIG_FILE="${ROOT_DIR}/${CONFIG_FILE}"

load_dotenv() {
  local raw key value first last
  while IFS= read -r raw || [[ -n "${raw}" ]]; do
    raw="${raw%$'\r'}"
    [[ -z "${raw//[[:space:]]/}" || "${raw}" =~ ^[[:space:]]*# ]] && continue
    [[ "${raw}" == export\ * ]] && raw="${raw#export }"
    if [[ ! "${raw}" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      fail "${ENV_FILE} 包含无法解析的行；只允许 KEY=VALUE: ${raw}"
    fi
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    if [[ ${#value} -ge 2 ]]; then
      first="${value:0:1}"
      last="${value: -1}"
      if [[ ( "${first}" == '"' && "${last}" == '"' ) || ( "${first}" == "'" && "${last}" == "'" ) ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi
    # Match Compose precedence: an existing process environment wins over .env.
    if [[ ! -v "${key}" ]]; then
      export "${key}=${value}"
    fi
  done < "${ENV_FILE}"
}

command -v docker >/dev/null 2>&1 || fail "Docker CLI 未安装或不在 PATH。"
docker compose version >/dev/null 2>&1 || fail "需要 Docker Compose v2（docker compose）。"
docker info >/dev/null 2>&1 || fail "Docker daemon 不可用；请启动 Docker Desktop 或 docker.service。"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "找不到 host Python: ${PYTHON_BIN}"
[[ -f "${ENV_FILE}" ]] || fail "缺少 env 文件: ${ENV_FILE}"
[[ -f "${CONFIG_FILE}" ]] || fail "缺少配置文件: ${CONFIG_FILE}"

load_dotenv

(cd "${ROOT_DIR}" && docker compose --env-file "${ENV_FILE}" config --quiet) \
  || fail "docker-compose.yml 或 ${ENV_FILE} 无法解析。"

note "验证 host 配置、默认文件源及所有已启用模型资产..."
if ! PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m deepstream_ai validate \
    --config "${CONFIG_FILE}"; then
  fail "配置或资产验证失败；禁止在缺少文件源、人员 nvinfer/模型资产时启动。"
fi

if ! PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/check-media.py" --config "${CONFIG_FILE}"; then
  fail "文件源编码或 nominal_fps 验证失败。"
fi

if [[ "${SKIP_GPU_PULL}" -eq 0 ]]; then
  note "验证容器 GPU（首次会拉取 ${GPU_TEST_IMAGE}）..."
  docker run --rm --gpus all "${GPU_TEST_IMAGE}" nvidia-smi >/dev/null \
    || fail "容器无法访问 NVIDIA GPU。检查驱动、WSL2/Docker Desktop GPU 集成或 NVIDIA Container Toolkit。"
else
  note "已跳过 GPU 容器拉取；仅检查本机 nvidia-smi。"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null \
    || fail "nvidia-smi 不可用。"
fi

note "通过：Docker、Compose、配置、必需资产、文件媒体与 GPU 运行时可用。"
