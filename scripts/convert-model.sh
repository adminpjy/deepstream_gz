#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/convert-model.sh \
    --input models/smoking.pt \
    --metadata configs/models/smoking.metadata.json \
    --onnx models/smoking.onnx \
    --engine models/smoking.engine \
    --labels-file models/smoking.labels.txt \
    --nvinfer-config configs/nvinfer/smoking.txt \
    [--precision fp32|fp16|int8] [--calibration-cache FILE] \
    [--device 0] [--workspace-mib 4096] [--force]

  scripts/convert-model.sh ... --validate-only

This command implements one contract only: a trusted Ultralytics YOLOv8,
YOLOv9 or YOLO11 detect checkpoint -> raw-output ONNX -> explicit/dynamic-batch
TensorRT engine -> matching DeepStream SGIE config and labels file.

The metadata is the source of truth. Its min/opt/max TensorRT profile must cover
the configured nvinfer batch and at least batch 16. On successful conversion,
ONNX, engine, labels and nvinfer config are staged and atomically replaced one
file at a time; metadata is atomically updated last as the commit record.

INT8 additionally requires an existing calibration cache generated for the
exact network, profile, preprocessing and representative site data.
EOF
}

INPUT=""
METADATA=""
ONNX=""
ENGINE=""
LABELS_FILE=""
NVINFER_CONFIG=""
PRECISION="fp16"
CALIBRATION_CACHE=""
DEVICE="0"
WORKSPACE_MIB="4096"
FORCE=0
VALIDATE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="${2:?missing value for --input}"; shift 2 ;;
    --metadata) METADATA="${2:?missing value for --metadata}"; shift 2 ;;
    --onnx) ONNX="${2:?missing value for --onnx}"; shift 2 ;;
    --engine) ENGINE="${2:?missing value for --engine}"; shift 2 ;;
    --labels-file) LABELS_FILE="${2:?missing value for --labels-file}"; shift 2 ;;
    --nvinfer-config) NVINFER_CONFIG="${2:?missing value for --nvinfer-config}"; shift 2 ;;
    --precision) PRECISION="${2:?missing value for --precision}"; shift 2 ;;
    --calibration-cache) CALIBRATION_CACHE="${2:?missing value for --calibration-cache}"; shift 2 ;;
    --device) DEVICE="${2:?missing value for --device}"; shift 2 ;;
    --workspace-mib) WORKSPACE_MIB="${2:?missing value for --workspace-mib}"; shift 2 ;;
    --validate-only) VALIDATE_ONLY=1; shift ;;
    --force) FORCE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${INPUT}" && -n "${METADATA}" && -n "${ONNX}" && -n "${ENGINE}" \
  && -n "${LABELS_FILE}" && -n "${NVINFER_CONFIG}" ]] || {
  usage >&2
  exit 2
}
[[ -f "${INPUT}" ]] || { echo "Input not found: ${INPUT}" >&2; exit 2; }
[[ -f "${METADATA}" ]] || { echo "Metadata not found: ${METADATA}" >&2; exit 2; }
[[ "${INPUT,,}" == *.pt ]] || {
  echo "Only an explicit .pt -> ONNX export is supported; got: ${INPUT}" >&2
  exit 2
}
[[ "${ONNX,,}" == *.onnx ]] || { echo "--onnx must end in .onnx" >&2; exit 2; }
[[ "${ENGINE,,}" =~ \.(engine|plan)$ ]] || {
  echo "--engine must end in .engine or .plan" >&2
  exit 2
}
[[ "${PRECISION}" =~ ^(fp32|fp16|int8)$ ]] || {
  echo "Unsupported precision: ${PRECISION}" >&2
  exit 2
}
[[ "${WORKSPACE_MIB}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--workspace-mib must be a positive integer" >&2
  exit 2
}
[[ "${DEVICE}" =~ ^[0-9]+$ ]] || { echo "--device must be a non-negative integer" >&2; exit 2; }

command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 2; }

# Structural validation is deliberately strict. A second validation step below
# inspects the exported ONNX graph so a declaration cannot silently turn a
# classifier, segmenter, end-to-end-NMS model or static-batch model into an
# apparently valid DeepStream config.
mapfile -t META < <(python3 - "${METADATA}" <<'PY'
import json
import math
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

required = {
    "schema_version", "name", "architecture", "task", "exporter", "checkpoint",
    "input", "onnx", "labels", "deepstream", "calibration",
}
missing = sorted(required.difference(data))
if missing:
    raise SystemExit(f"metadata missing keys: {', '.join(missing)}")
if data["schema_version"] != 2:
    raise SystemExit("unsupported metadata schema_version; expected 2")

name = data["name"]
if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
    raise SystemExit("name must be a non-placeholder filesystem-safe identifier")
if name.lower().startswith("replace-me"):
    raise SystemExit("name must be replaced before conversion")

architecture = str(data["architecture"]).lower()
if architecture not in {"yolov8", "yolov9", "yolo11"}:
    raise SystemExit("architecture must be exactly one of: yolov8, yolov9, yolo11")
if data["task"] != "detect":
    raise SystemExit("only task=detect is supported")

checkpoint = data["checkpoint"]
if not isinstance(checkpoint, dict) or not re.fullmatch(
    r"[0-9a-f]{64}", str(checkpoint.get("sha256", ""))
):
    raise SystemExit("checkpoint.sha256 must be the reviewed lowercase 64-character SHA256")

exporter = data["exporter"]
if not isinstance(exporter, dict) or exporter.get("name") != "ultralytics":
    raise SystemExit("exporter.name must be ultralytics")
exporter_version = exporter.get("version")
if not isinstance(exporter_version, str) or not re.fullmatch(
    r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?", exporter_version
):
    raise SystemExit("exporter.version must be an exact reviewed version, for example 8.3.0")

model_input = data["input"]
if not isinstance(model_input, dict):
    raise SystemExit("input must be an object")
shape = model_input.get("shape")
if (
    not isinstance(shape, list)
    or len(shape) != 4
    or shape[0] != -1
    or not all(isinstance(v, int) and v > 0 for v in shape[1:])
):
    raise SystemExit("input.shape must be dynamic-batch [-1,C,H,W] with positive C/H/W")
if shape[1] != 3 or model_input.get("layout") != "NCHW":
    raise SystemExit("this contract requires a 3-channel NCHW input")
if model_input.get("color_format") != "RGB":
    raise SystemExit("the built-in raw YOLO contract requires input.color_format=RGB")
scale = model_input.get("scale_factor")
if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
    raise SystemExit("input.scale_factor must be a positive finite number")
if model_input.get("dynamic_batch") is not True:
    raise SystemExit("input.dynamic_batch must be true; static batch engines are rejected")
if not isinstance(model_input.get("name"), str) or not re.fullmatch(
    r"[A-Za-z_][A-Za-z0-9_.-]*", model_input["name"]
):
    raise SystemExit("input.name must be a valid non-empty tensor name")

profile = model_input.get("profile")
if not isinstance(profile, dict) or set(profile) != {"min", "opt", "max"}:
    raise SystemExit("input.profile must contain exactly min, opt and max shapes")
profile_shapes = []
for key in ("min", "opt", "max"):
    candidate = profile[key]
    if not isinstance(candidate, list) or len(candidate) != 4:
        raise SystemExit(f"input.profile.{key} must be [N,C,H,W]")
    if not all(isinstance(v, int) and v > 0 for v in candidate):
        raise SystemExit(f"input.profile.{key} values must be positive integers")
    if candidate[1:] != shape[1:]:
        raise SystemExit(f"input.profile.{key} C/H/W must equal input.shape")
    profile_shapes.append(candidate)
min_shape, opt_shape, max_shape = profile_shapes
if not min_shape[0] <= opt_shape[0] <= max_shape[0]:
    raise SystemExit("batch profile must satisfy min <= opt <= max")
if max_shape[0] < 16:
    raise SystemExit("input.profile.max batch must be at least 16")

onnx = data["onnx"]
if not isinstance(onnx, dict) or not isinstance(onnx.get("opset"), int):
    raise SystemExit("onnx.opset must be an integer")
if not 13 <= onnx["opset"] <= 20:
    raise SystemExit("onnx.opset must be between 13 and 20")
if onnx.get("raw_output") is not True:
    raise SystemExit("onnx.raw_output must be true; embedded/end-to-end NMS is unsupported")

labels = data["labels"]
if (
    not isinstance(labels, list)
    or not labels
    or not all(
        isinstance(label, str)
        and label == label.strip()
        and "\n" not in label
        and "\r" not in label
        for label in labels
    )
):
    raise SystemExit("labels must be a non-empty ordered list of one-line strings")
if len(set(labels)) != len(labels):
    raise SystemExit("labels must be unique")
if any(label.lower().startswith("replace-me") for label in labels):
    raise SystemExit("every labels entry must be replaced before conversion")

deepstream = data["deepstream"]
if not isinstance(deepstream, dict):
    raise SystemExit("deepstream must be an object")
expected = {
    "network_type": 0,
    "cluster_mode": 2,
    "process_mode": 2,
    "operate_on_gie_id": 1,
    "parser_contract": "ultralytics-yolo8-9-11-detect-raw-v1",
    "custom_lib_path": "/opt/nvidia/deepstream/deepstream/lib/libnvdsinfer_custom_yolo_dynamic.so",
    "parse_bbox_func_name": "NvDsInferParseCustomYoloDynamic",
}
for key, value in expected.items():
    if deepstream.get(key) != value:
        raise SystemExit(f"deepstream.{key} must be {value!r} for the built-in parser contract")
batch_size = deepstream.get("batch_size")
if not isinstance(batch_size, int) or batch_size <= 0:
    raise SystemExit("deepstream.batch_size must be a positive integer")
if batch_size < 16:
    raise SystemExit("behavior SGIE deepstream.batch_size must be at least 16")
if not min_shape[0] <= batch_size <= max_shape[0]:
    raise SystemExit("deepstream.batch_size is outside the TensorRT min/max profile")
if opt_shape[0] > batch_size:
    raise SystemExit("input.profile.opt batch must not exceed deepstream.batch_size")
if deepstream.get("num_detected_classes") != len(labels):
    raise SystemExit("deepstream.num_detected_classes must equal len(labels)")
gie_unique_id = deepstream.get("gie_unique_id")
if not isinstance(gie_unique_id, int) or not 2 <= gie_unique_id <= 255:
    raise SystemExit("deepstream.gie_unique_id must be an integer in [2,255]")
class_ids = deepstream.get("operate_on_class_ids")
if class_ids != [0]:
    raise SystemExit("behavior SGIE deepstream.operate_on_class_ids must be [0]")
for key in ("pre_cluster_threshold", "nms_iou_threshold"):
    value = deepstream.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise SystemExit(f"deepstream.{key} must be a finite number in [0,1]")
topk = deepstream.get("topk")
if not isinstance(topk, int) or topk <= 0:
    raise SystemExit("deepstream.topk must be a positive integer")
for key in ("maintain_aspect_ratio", "symmetric_padding"):
    if deepstream.get(key) is not True:
        raise SystemExit(f"deepstream.{key} must be true")

calibration = data["calibration"]
if not isinstance(calibration, dict):
    raise SystemExit("calibration must be an object")

values = [
    checkpoint["sha256"], model_input["name"], str(shape[2]), str(shape[3]),
    str(onnx["opset"]), architecture, exporter_version, str(len(labels)), deepstream["custom_lib_path"],
    deepstream["parse_bbox_func_name"], str(batch_size), str(min_shape[0]),
    str(opt_shape[0]), str(max_shape[0]), str(gie_unique_id),
    str(deepstream["operate_on_gie_id"]), ";".join(str(v) for v in class_ids),
    str(deepstream["pre_cluster_threshold"]), str(deepstream["nms_iou_threshold"]),
    str(topk), repr(float(scale)),
    "1" if calibration.get("required_for_int8", True) else "0",
    str(calibration.get("cache_file") or "-"),
    str(calibration.get("dataset_id") or "-"),
    str(calibration.get("preprocessing_id") or "-"),
]
if any("\n" in value or "\r" in value for value in values):
    raise SystemExit("metadata string values may not contain newlines")
print("\n".join(values))
PY
)

[[ "${#META[@]}" -eq 25 ]] || {
  echo "Metadata validation failed; no conversion was performed." >&2
  exit 2
}

EXPECTED_CHECKPOINT_SHA256="${META[0]}"
INPUT_NAME="${META[1]}"
HEIGHT="${META[2]}"
WIDTH="${META[3]}"
OPSET="${META[4]}"
ARCHITECTURE="${META[5]}"
EXPECTED_EXPORTER_VERSION="${META[6]}"
CLASS_COUNT="${META[7]}"
PARSER_LIB="${META[8]}"
PARSER_FUNC="${META[9]}"
NVINFER_BATCH="${META[10]}"
MIN_BATCH="${META[11]}"
OPT_BATCH="${META[12]}"
MAX_BATCH="${META[13]}"
GIE_UNIQUE_ID="${META[14]}"
OPERATE_ON_GIE_ID="${META[15]}"
OPERATE_ON_CLASS_IDS="${META[16]}"
PRE_CLUSTER_THRESHOLD="${META[17]}"
NMS_IOU_THRESHOLD="${META[18]}"
TOPK="${META[19]}"
SCALE_FACTOR="${META[20]}"
CALIBRATION_REQUIRED="${META[21]}"
METADATA_CALIBRATION_CACHE="${META[22]}"
CALIBRATION_DATASET_ID="${META[23]}"
CALIBRATION_PREPROCESSING_ID="${META[24]}"

ACTUAL_CHECKPOINT_SHA256="$(python3 - "${INPUT}" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
[[ "${ACTUAL_CHECKPOINT_SHA256}" == "${EXPECTED_CHECKPOINT_SHA256}" ]] || {
  echo "Checkpoint SHA256 mismatch: metadata=${EXPECTED_CHECKPOINT_SHA256}, actual=${ACTUAL_CHECKPOINT_SHA256}" >&2
  exit 2
}

python3 - "${INPUT}" "${METADATA}" "${ONNX}" "${ENGINE}" "${LABELS_FILE}" "${NVINFER_CONFIG}" <<'PY'
import os
import sys

paths = [os.path.realpath(path) for path in sys.argv[1:]]
if len(paths) != len(set(paths)):
    raise SystemExit("input, metadata and all output paths must be distinct")
PY

echo "Contract valid: ${ARCHITECTURE} detect, ${CLASS_COUNT} classes, dynamic batch ${MIN_BATCH}/${OPT_BATCH}/${MAX_BATCH}, nvinfer batch ${NVINFER_BATCH}"
if [[ "${VALIDATE_ONLY}" -eq 1 ]]; then
  exit 0
fi

for output in "${ONNX}" "${ENGINE}" "${LABELS_FILE}" "${NVINFER_CONFIG}"; do
  if [[ "${FORCE}" -eq 0 && -e "${output}" ]]; then
    echo "Output exists. Review it, then use --force to replace atomically: ${output}" >&2
    exit 2
  fi
done

if [[ "${PRECISION}" == "int8" ]]; then
  [[ "${CALIBRATION_REQUIRED}" == "1" ]] || {
    echo "Metadata does not authorize/describe INT8 calibration for this model." >&2
    exit 2
  }
  if [[ -z "${CALIBRATION_CACHE}" && "${METADATA_CALIBRATION_CACHE}" != "-" ]]; then
    CALIBRATION_CACHE="${METADATA_CALIBRATION_CACHE}"
  fi
  [[ -n "${CALIBRATION_CACHE}" ]] || {
    echo "INT8 requires --calibration-cache; this script will not fabricate calibration data." >&2
    exit 2
  }
  [[ -f "${CALIBRATION_CACHE}" ]] || {
    echo "Calibration cache not found: ${CALIBRATION_CACHE}" >&2
    exit 2
  }
  [[ "${CALIBRATION_DATASET_ID}" != "-" && "${CALIBRATION_DATASET_ID}" != replace-me* ]] || {
    echo "INT8 metadata must record a reviewed calibration.dataset_id." >&2
    exit 2
  }
  [[ "${CALIBRATION_PREPROCESSING_ID}" != "-" && "${CALIBRATION_PREPROCESSING_ID}" != replace-me* ]] || {
    echo "INT8 metadata must record calibration.preprocessing_id." >&2
    exit 2
  }
  python3 - \
    "${CALIBRATION_CACHE}" "${INPUT}" "${METADATA}" "${ONNX}" "${ENGINE}" \
    "${LABELS_FILE}" "${NVINFER_CONFIG}" <<'PY'
import os
import sys

calibration, *protected_paths = (os.path.realpath(path) for path in sys.argv[1:])
if calibration in protected_paths:
    raise SystemExit("calibration cache path must be distinct from inputs and generated outputs")
PY
fi

command -v yolo >/dev/null 2>&1 || {
  cat >&2 <<'EOF'
Ultralytics exporter command `yolo` is not installed.
Install the exact exporter.version declared by metadata, plus a pinned ONNX
package, in an isolated conversion image. Runtime containers do not need it.
EOF
  exit 2
}

ACTUAL_EXPORTER_VERSION="$(python3 - <<'PY'
from importlib import metadata

try:
    value = metadata.version("ultralytics")
except Exception as exc:
    raise SystemExit(f"cannot resolve installed ultralytics distribution: {exc}")
print(value)
PY
)"
[[ "${ACTUAL_EXPORTER_VERSION}" == "${EXPECTED_EXPORTER_VERSION}" ]] || {
  echo "Ultralytics version mismatch: metadata=${EXPECTED_EXPORTER_VERSION}, installed=${ACTUAL_EXPORTER_VERSION}" >&2
  exit 2
}

TRTEXEC="$(command -v trtexec || true)"
if [[ -z "${TRTEXEC}" && -x /usr/src/tensorrt/bin/trtexec ]]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
[[ -n "${TRTEXEC}" ]] || {
  echo "trtexec not found; run inside the selected DeepStream/TensorRT image." >&2
  exit 2
}

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/deepstream-model-convert.XXXXXX")"
cleanup() {
  rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT

cp -- "${INPUT}" "${WORK_DIR}/checkpoint.pt"

echo "Exporting ${ARCHITECTURE} (${CLASS_COUNT} classes), dynamic batch, profile ${MIN_BATCH}/${OPT_BATCH}/${MAX_BATCH}, ${HEIGHT}x${WIDTH}, opset ${OPSET}"
yolo export \
  model="${WORK_DIR}/checkpoint.pt" \
  format=onnx \
  imgsz="${HEIGHT},${WIDTH}" \
  batch="${OPT_BATCH}" \
  opset="${OPSET}" \
  dynamic=True \
  simplify=False \
  nms=False \
  device="${DEVICE}"

STAGED_ONNX="${WORK_DIR}/checkpoint.onnx"
[[ -s "${STAGED_ONNX}" ]] || {
  echo "Exporter did not create the expected ONNX file: ${STAGED_ONNX}" >&2
  exit 1
}

python3 - "${STAGED_ONNX}" "${METADATA}" <<'PY'
import ast
import json
import re
import sys

try:
    import onnx
except Exception as exc:
    raise SystemExit(f"the pinned conversion environment must provide onnx: {exc}")

onnx_path, metadata_path = sys.argv[1:]
with open(metadata_path, "r", encoding="utf-8") as handle:
    contract = json.load(handle)
model = onnx.load(onnx_path)
onnx.checker.check_model(model)

if len(model.graph.input) != 1:
    raise SystemExit("raw YOLO contract requires exactly one ONNX input")
model_input = model.graph.input[0]
if model_input.name != contract["input"]["name"]:
    raise SystemExit(
        f"ONNX input name {model_input.name!r} != metadata {contract['input']['name']!r}"
    )
input_dims = model_input.type.tensor_type.shape.dim
if len(input_dims) != 4:
    raise SystemExit("ONNX input must be rank-4 NCHW")
if input_dims[0].dim_value > 0 or not input_dims[0].dim_param:
    raise SystemExit("ONNX batch dimension is static; dynamic batch export is required")
expected_shape = contract["input"]["shape"]
if input_dims[1].dim_value != 3:
    raise SystemExit("ONNX input channel dimension must be statically 3")
for index, label in ((2, "height"), (3, "width")):
    value = input_dims[index].dim_value
    if value > 0 and value != expected_shape[index]:
        raise SystemExit(f"ONNX input {label} does not match metadata")

if len(model.graph.output) != 1:
    raise SystemExit("raw detect parser requires exactly one ONNX output")
output_dims = model.graph.output[0].type.tensor_type.shape.dim
if len(output_dims) != 3:
    raise SystemExit("raw detect output must be rank 3: [N,4+C,rows] or [N,rows,4+C]")
if output_dims[0].dim_value > 0 or not output_dims[0].dim_param:
    raise SystemExit("ONNX output batch dimension is static")
expected_channels = 4 + len(contract["labels"])
static_tail = [dim.dim_value for dim in output_dims[1:] if dim.dim_value > 0]
if expected_channels not in static_tail:
    raise SystemExit(
        f"ONNX output does not expose 4+C={expected_channels}; wrong task/classes/parser contract"
    )

properties = {item.key: item.value for item in model.metadata_props}
if properties.get("task") != "detect":
    raise SystemExit("Ultralytics ONNX metadata task must be detect")
if properties.get("version") != contract["exporter"]["version"]:
    raise SystemExit(
        "Ultralytics ONNX metadata version does not match exporter.version: "
        f"{properties.get('version')!r} != {contract['exporter']['version']!r}"
    )
try:
    exported_names = ast.literal_eval(properties["names"])
except (KeyError, SyntaxError, ValueError) as exc:
    raise SystemExit(f"Ultralytics ONNX metadata must contain parseable names: {exc}")
if isinstance(exported_names, dict):
    exported_labels = [str(exported_names[key]) for key in sorted(exported_names, key=int)]
elif isinstance(exported_names, (list, tuple)):
    exported_labels = [str(value) for value in exported_names]
else:
    raise SystemExit("Ultralytics ONNX names metadata has an unsupported type")
if exported_labels != contract["labels"]:
    raise SystemExit(
        f"exported class order {exported_labels!r} != metadata labels {contract['labels']!r}"
    )

description = " ".join(
    properties.get(key, "") for key in ("description", "model", "name")
)
normalized = re.sub(r"[^a-z0-9]", "", description.lower())
expected_architecture = re.sub(r"[^a-z0-9]", "", contract["architecture"].lower())
if expected_architecture not in normalized:
    raise SystemExit(
        "exported ONNX metadata does not identify the declared architecture; "
        f"expected {contract['architecture']!r} in description/model/name"
    )
PY

STAGED_ENGINE="${WORK_DIR}/model.engine"
min_shape="${INPUT_NAME}:${MIN_BATCH}x3x${HEIGHT}x${WIDTH}"
opt_shape="${INPUT_NAME}:${OPT_BATCH}x3x${HEIGHT}x${WIDTH}"
max_shape="${INPUT_NAME}:${MAX_BATCH}x3x${HEIGHT}x${WIDTH}"
trt_args=(
  "--onnx=${STAGED_ONNX}"
  "--saveEngine=${STAGED_ENGINE}"
  "--minShapes=${min_shape}"
  "--optShapes=${opt_shape}"
  "--maxShapes=${max_shape}"
  "--memPoolSize=workspace:${WORKSPACE_MIB}"
  "--skipInference"
)

case "${PRECISION}" in
  fp16) trt_args+=(--fp16) ;;
  int8) trt_args+=(--int8 "--calib=${CALIBRATION_CACHE}") ;;
esac

"${TRTEXEC}" "${trt_args[@]}"
[[ -s "${STAGED_ENGINE}" ]] || {
  echo "TensorRT engine was not created: ${STAGED_ENGINE}" >&2
  exit 1
}

STAGED_LABELS="${WORK_DIR}/labels.txt"
STAGED_CONFIG="${WORK_DIR}/nvinfer.txt"
STAGED_METADATA="${WORK_DIR}/metadata.json"
python3 - \
  "${METADATA}" "${INPUT}" "${STAGED_ONNX}" "${ONNX}" \
  "${STAGED_ENGINE}" "${ENGINE}" "${STAGED_LABELS}" "${LABELS_FILE}" \
  "${STAGED_CONFIG}" "${NVINFER_CONFIG}" "${STAGED_METADATA}" \
  "${PRECISION}" "${WORKSPACE_MIB}" "${ACTUAL_EXPORTER_VERSION}" \
  "${CALIBRATION_CACHE}" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

(
    metadata_path, checkpoint_path, staged_onnx, onnx_path,
    staged_engine, engine_path, staged_labels, labels_path,
    staged_config, config_path, staged_metadata, precision, workspace_mib,
    exporter_version, calibration_cache,
) = sys.argv[1:]

with open(metadata_path, "r", encoding="utf-8") as handle:
    contract = json.load(handle)

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def portable(path: str) -> str:
    return path.replace(os.sep, "/")

def config_reference(path: str) -> str:
    config_directory = os.path.dirname(os.path.abspath(config_path))
    return portable(os.path.relpath(os.path.abspath(path), config_directory))

labels = contract["labels"]
Path(staged_labels).write_text("\n".join(labels) + "\n", encoding="utf-8", newline="\n")

deepstream = contract["deepstream"]
model_input = contract["input"]
network_mode = {"fp32": 0, "int8": 1, "fp16": 2}[precision]
properties = [
    ("gpu-id", "0"),
    ("net-scale-factor", str(model_input["scale_factor"])),
    ("model-color-format", "0"),
    ("onnx-file", config_reference(onnx_path)),
    ("model-engine-file", config_reference(engine_path)),
    ("labelfile-path", config_reference(labels_path)),
    ("infer-dims", ";".join(str(value) for value in model_input["shape"][1:])),
    ("batch-size", str(deepstream["batch_size"])),
    ("network-mode", str(network_mode)),
    ("num-detected-classes", str(deepstream["num_detected_classes"])),
    ("gie-unique-id", str(deepstream["gie_unique_id"])),
    ("network-type", "0"),
    ("process-mode", "2"),
    ("operate-on-gie-id", str(deepstream["operate_on_gie_id"])),
    ("operate-on-class-ids", ";".join(str(value) for value in deepstream["operate_on_class_ids"])),
    ("cluster-mode", "2"),
    ("maintain-aspect-ratio", "1"),
    ("symmetric-padding", "1"),
    ("scaling-filter", "1"),
    ("parse-bbox-func-name", deepstream["parse_bbox_func_name"]),
    ("disable-output-host-copy", "0"),
    ("custom-lib-path", deepstream["custom_lib_path"]),
]
if precision == "int8":
    properties.insert(9, ("int8-calib-file", config_reference(calibration_cache)))
config_lines = [
    "# Generated by scripts/convert-model.sh from metadata schema v2.",
    "# Do not hand-edit model, class, batch or parser fields; regenerate instead.",
    "[property]",
    *(f"{key}={value}" for key, value in properties),
    "",
    "[class-attrs-all]",
    f"nms-iou-threshold={deepstream['nms_iou_threshold']}",
    f"pre-cluster-threshold={deepstream['pre_cluster_threshold']}",
    f"topk={deepstream['topk']}",
    "",
]
Path(staged_config).write_text("\n".join(config_lines), encoding="utf-8", newline="\n")

profile = model_input["profile"]
artifacts = {
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "checkpoint": {"path": portable(checkpoint_path), "sha256": sha256(checkpoint_path)},
    "onnx": {"path": portable(onnx_path), "sha256": sha256(staged_onnx)},
    "engine": {"path": portable(engine_path), "sha256": sha256(staged_engine)},
    "labels": {"path": portable(labels_path), "sha256": sha256(staged_labels)},
    "nvinfer_config": {"path": portable(config_path), "sha256": sha256(staged_config)},
    "build": {
        "precision": precision,
        "explicit_batch": True,
        "profile": profile,
        "nvinfer_batch_size": deepstream["batch_size"],
        "workspace_mib": int(workspace_mib),
        "ultralytics_version": exporter_version,
        "parser_contract": deepstream["parser_contract"],
    },
}
if precision == "int8":
    artifacts["calibration_cache"] = {
        "path": portable(calibration_cache),
        "sha256": sha256(calibration_cache),
        "dataset_id": contract["calibration"]["dataset_id"],
        "preprocessing_id": contract["calibration"]["preprocessing_id"],
    }
contract["artifacts"] = artifacts
Path(staged_metadata).write_text(
    json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
    newline="\n",
)
PY

# All expensive/semantic work has succeeded. Install every staged artifact via
# a sibling temporary file and os.replace; metadata is last, so it is the commit
# record and never advertises a failed/partial conversion attempt.
python3 - \
  "${FORCE}" \
  "${STAGED_ONNX}" "${ONNX}" \
  "${STAGED_ENGINE}" "${ENGINE}" \
  "${STAGED_LABELS}" "${LABELS_FILE}" \
  "${STAGED_CONFIG}" "${NVINFER_CONFIG}" \
  "${STAGED_METADATA}" "${METADATA}" <<'PY'
import os
import shutil
import sys
import tempfile
from pathlib import Path

force = sys.argv[1] == "1"
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
for _, destination in pairs[:-1]:
    if not force and os.path.exists(destination):
        raise SystemExit(f"output appeared during conversion; refusing overwrite: {destination}")

def atomic_install(source: str, destination: str) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp",
            dir=target.parent, delete=False,
        ) as temporary, open(source, "rb") as stream:
            temporary_name = temporary.name
            shutil.copyfileobj(stream, temporary, length=1024 * 1024)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise

for source, destination in pairs:
    atomic_install(source, destination)
PY

cat <<EOF
Conversion committed:
  ONNX:          ${ONNX}
  TensorRT:      ${ENGINE}
  labels:        ${LABELS_FILE}
  nvinfer SGIE:  ${NVINFER_CONFIG}
  metadata:      ${METADATA}
  batch profile: ${MIN_BATCH}/${OPT_BATCH}/${MAX_BATCH} (nvinfer=${NVINFER_BATCH})
  parser:        ${PARSER_LIB} :: ${PARSER_FUNC}

Conversion success does not prove inference correctness. Before enabling the
model, compare framework/ONNX/TensorRT/DeepStream outputs and validate NMS,
class order, preprocessing, site metrics, throughput and GPU memory.
EOF
