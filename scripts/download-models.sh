#!/usr/bin/env bash
set -euo pipefail

PEOPLENET_REF='nvidia/tao/peoplenet:deployable_quantized_onnx_v2.6.3'
DESTINATION='models/peoplenet'
MODEL=''
ACCEPT_LICENSE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/download-models.sh --peoplenet --accept-license \
    [--destination models/peoplenet]

Requirements:
  - NVIDIA NGC CLI (`ngc`) installed on this host
  - NGC CLI already authenticated by the operator (`ngc config set`)
  - Operator has reviewed and accepted the applicable NGC/model license terms

This command never accepts, prints or stores an NGC API key. SCRFD/RetinaFace,
AdaFace and behavior checkpoints are not downloaded because their source and
license must be selected by the deployment organization.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --peoplenet)
      [[ -z "${MODEL}" ]] || fail "Only one model may be selected per invocation."
      MODEL='peoplenet'
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

[[ "${MODEL}" == 'peoplenet' ]] || {
  usage >&2
  fail "Select --peoplenet. Other model families are intentionally user-supplied."
}
[[ "${ACCEPT_LICENSE}" -eq 1 ]] || fail \
  "Review the NGC/model terms, then rerun with --accept-license to record explicit acknowledgement."
[[ -n "${DESTINATION}" && "${DESTINATION}" != '/' ]] || fail "Unsafe destination: ${DESTINATION}"

command -v ngc >/dev/null 2>&1 || fail \
  "NGC CLI is not installed. Install it from NVIDIA's official NGC CLI page."
ngc config current >/dev/null 2>&1 || fail \
  "NGC CLI is not authenticated. Run 'ngc config set' interactively; never put the API key in this repository."

mkdir -p -- "${DESTINATION}"
echo "Downloading reviewed PeopleNet version to ${DESTINATION} ..."
ngc registry model download-version \
  'nvidia/tao/peoplenet:deployable_quantized_onnx_v2.6.3' \
  -d "${DESTINATION}"

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

