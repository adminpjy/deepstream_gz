#!/usr/bin/env bash
set -euo pipefail

python3 /workspace/scripts/prepare-local-behavior-models-dynamic.py \
  --model-root /workspace/models \
  --config-root /workspace/configs/nvinfer \
  --device "${MODEL_CONVERTER_DEVICE:-0}" \
  --precision "${MODEL_CONVERTER_PRECISION:-fp16}" \
  --imgsz "${MODEL_CONVERTER_IMGSZ:-640}" \
  --force

python3 - /workspace/models/deepstream-local-models.manifest.json <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
models = {item["name"]: item for item in data.get("models", [])}
failures = list(data.get("failures", []))
if failures:
    raise SystemExit("model conversion reported failures: " + "; ".join(failures))
required = {"eat_drink", "smoking", "fire"}
missing = sorted(required.difference(models))
if missing:
    raise SystemExit("model conversion manifest missing: " + ", ".join(missing))

# yolo11n.onnx is the standard COCO detector. The converter keeps its source
# labels in sourceLabels but exposes two business labels through the dedicated
# person-crop proxy parser.
expected = ["eating", "drinking"]
actual = models["eat_drink"].get("labels")
if actual != expected:
    raise SystemExit(
        "eat/drink business labels must be ['eating', 'drinking']; "
        f"actual={actual!r}."
    )
source_labels = models["eat_drink"].get("sourceLabels") or []
for required_source in ("bottle", "cup", "fork", "spoon", "bowl"):
    if required_source not in source_labels:
        raise SystemExit(
            f"yolo11n.onnx source labels missing required COCO class {required_source!r}"
        )

for name, expected_labels in {
    "smoking": ["smoking"],
    "fire": ["fire"],
}.items():
    actual_labels = models[name].get("labels")
    if actual_labels != expected_labels:
        raise SystemExit(
            f"{name} model labels must be {expected_labels!r}; actual={actual_labels!r}. "
            "Review the trained checkpoint classes before production use."
        )

print("Local behavior model contract verified:")
for name in ("eat_drink", "smoking", "fire"):
    item = models[name]
    print(
        f"  {name}: labels={item['labels']} engine={item['engine']} "
        f"config={item['nvinferConfig']}"
    )
PY
