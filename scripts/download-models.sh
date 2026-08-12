#!/usr/bin/env bash
set -euo pipefail

PEOPLENET_REF='nvidia/tao/peoplenet:deployable_quantized_onnx_v2.6.3'
REID_REF='nvidia/tao/reidentificationnet:deployable_v1.2'
REID_FILENAME='resnet50_market1501_aicity156.onnx'
REID_SIZE='96398132'
REID_SHA256='0e21d09278508ec835955f422a9fdd3cd59b2a6ecdef98d705f388f33cebac2b'
DESTINATION=''
MODEL=''
ACCEPT_LICENSE=0
STAGING_DIR=''
DESTINATION_ABS=''

usage() {
  cat <<'EOF'
Usage:
  scripts/download-models.sh --peoplenet --accept-license \
    [--destination models/peoplenet]
  scripts/download-models.sh --reid --accept-license \
    [--destination models/tracker]

Requirements:
  - NVIDIA NGC CLI (`ngc`) installed on this host
  - NGC CLI already authenticated by the operator (`ngc config set`)
  - Operator has reviewed and accepted the applicable NGC/model license terms

This command never accepts, prints or stores an NGC API key. The reviewed ReID
artifact is installed only after its exact filename, byte size and SHA256 have
been verified. SCRFD/RetinaFace, AdaFace and behavior checkpoints are not
downloaded because their source and license must be selected by the deployment
organization.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

cleanup() {
  if [[ -n "${STAGING_DIR}" && -d "${STAGING_DIR}" ]]; then
    local staging_name staging_parent
    staging_name="$(basename -- "${STAGING_DIR}")"
    staging_parent="$(dirname -- "${STAGING_DIR}")"
    [[ "${staging_parent}" == "${DESTINATION_ABS}" && \
      "${staging_name}" == .reid-download.* ]] || {
      echo "ERROR: Refusing to remove unexpected staging directory: ${STAGING_DIR}" >&2
      return
    }
    rm -rf -- "${STAGING_DIR}"
  fi
}

trap cleanup EXIT

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

verify_reid_file() {
  local path="$1"
  local actual_size actual_sha256

  [[ -f "${path}" && ! -L "${path}" ]] || return 1
  actual_size="$(stat -c '%s' -- "${path}")"
  [[ "${actual_size}" == "${REID_SIZE}" ]] || return 1
  actual_sha256="$(sha256_file "${path}")"
  [[ "${actual_sha256}" == "${REID_SHA256}" ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --peoplenet)
      [[ -z "${MODEL}" ]] || fail "Only one model may be selected per invocation."
      MODEL='peoplenet'
      shift
      ;;
    --reid)
      [[ -z "${MODEL}" ]] || fail "Only one model may be selected per invocation."
      MODEL='reid'
      shift
      ;;
    --destination)
      DESTINATION="${2:?missing value for --destination}"
      shift 2
      ;;
    --accept-license)
      ACCEPT_LICENSE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1. Use --help for supported models."
      ;;
  esac
done

[[ "${MODEL}" == 'peoplenet' || "${MODEL}" == 'reid' ]] || {
  usage >&2
  fail "Select --peoplenet or --reid. Other model families are intentionally user-supplied."
}
[[ "${ACCEPT_LICENSE}" -eq 1 ]] || fail \
  "Review the NGC/model terms, then rerun with --accept-license to record explicit acknowledgement."

if [[ -z "${DESTINATION}" ]]; then
  if [[ "${MODEL}" == 'peoplenet' ]]; then
    DESTINATION='models/peoplenet'
  else
    DESTINATION='models/tracker'
  fi
fi
[[ -n "${DESTINATION}" && "${DESTINATION}" != '/' ]] || fail "Unsafe destination: ${DESTINATION}"

if [[ "${MODEL}" == 'reid' ]]; then
  command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required to verify the ReID artifact."
  command -v stat >/dev/null 2>&1 || fail "GNU stat is required to verify the ReID artifact size."

  mkdir -p -- "${DESTINATION}"
  DESTINATION_ABS="$(cd -- "${DESTINATION}" && pwd -P)"
  [[ "${DESTINATION_ABS}" != '/' ]] || fail "Unsafe destination: ${DESTINATION}"
  REID_TARGET="${DESTINATION_ABS}/${REID_FILENAME}"
  if [[ -e "${REID_TARGET}" || -L "${REID_TARGET}" ]]; then
    if verify_reid_file "${REID_TARGET}"; then
      echo "Verified ReID model already installed: ${REID_TARGET}"
      exit 0
    fi
    fail "Refusing to overwrite existing unverified ReID artifact: ${REID_TARGET}"
  fi
fi

command -v ngc >/dev/null 2>&1 || fail \
  "NGC CLI is not installed. Install it from NVIDIA's official NGC CLI page."
ngc config current >/dev/null 2>&1 || fail \
  "NGC CLI is not authenticated. Run 'ngc config set' interactively; never put the API key in this repository."

if [[ "${MODEL}" == 'reid' ]]; then
  STAGING_DIR="$(mktemp -d "${DESTINATION_ABS}/.reid-download.XXXXXX")"
  echo "Downloading reviewed ReID version to temporary directory ${STAGING_DIR} ..."
  ngc registry model download-version "${REID_REF}" -d "${STAGING_DIR}"

  mapfile -d '' REID_CANDIDATES < <(
    find "${STAGING_DIR}" -type f -name "${REID_FILENAME}" -print0
  )
  [[ "${#REID_CANDIDATES[@]}" -eq 1 ]] || fail \
    "Expected exactly one ${REID_FILENAME} in ${REID_REF}; found ${#REID_CANDIDATES[@]}."
  verify_reid_file "${REID_CANDIDATES[0]}" || fail \
    "Downloaded ${REID_FILENAME} failed the reviewed size/SHA256 check."

  mv -n -- "${REID_CANDIDATES[0]}" "${REID_TARGET}"
  verify_reid_file "${REID_TARGET}" || fail \
    "Atomic install did not produce the reviewed artifact at ${REID_TARGET}; refusing to continue."

  cat <<EOF

ReID download and verification completed from:
  ${REID_REF}

Installed:
  ${REID_TARGET}
  size=${REID_SIZE}
  sha256=${REID_SHA256}

Build its TensorRT FP16 engine on the target deployment GPU and inside the
target DeepStream/TensorRT container; see models/README.md for the exact
min/opt/max profile command. Do not copy an engine between platforms.
EOF
  exit 0
fi

mkdir -p -- "${DESTINATION}"
echo "Downloading reviewed PeopleNet version to ${DESTINATION} ..."
ngc registry model download-version "${PEOPLENET_REF}" -d "${DESTINATION}"

cat <<EOF

PeopleNet download completed from:
  ${PEOPLENET_REF}

NGC may create a version-named subdirectory. Inspect the downloaded manifest,
license and SHA256 values, then point configs/nvinfer/person.example.txt at the
actual ONNX, labels and calibration files (or copy them to the documented
models/person.* paths) before building a TensorRT engine.

Not downloaded: SCRFD/RetinaFace, AdaFace, smoking, eating, drinking, carrying.
Those remain user-supplied and must have explicit source/license/model metadata.
EOF
