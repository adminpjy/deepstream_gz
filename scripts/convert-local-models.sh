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
required = {"eat_drink", "smoking", "phone", "fire"}
missing = sorted(required.difference(models))
if missing:
    raise SystemExit("model conversion manifest missing: " + ", ".join(missing))

# The current production config deliberately uses one shared SGIE for these
# two independent business switches. Refuse reversed/unknown class order
# rather than silently turning drinking into eating or vice versa.
expected = ["eating", "drinking"]
actual = models["eat_drink"].get("labels")
if actual != expected:
    raise SystemExit(
        "yolo11n.onnx class order must be ['eating', 'drinking'] for the current "
        f"production config; actual canonical labels are {actual!r}. "
        "Do not start the recognition service until the model/config mapping is reviewed."
    )

for name, expected_labels in {
    "smoking": ["smoking"],
    "phone": ["phone"],
    "fire": ["fire"],
}.items():
    actual_labels = models[name].get("labels")
    if actual_labels != expected_labels:
        raise SystemExit(
            f"{name} model labels must be {expected_labels!r}; actual={actual_labels!r}. "
            "Review the trained checkpoint classes before production use."
        )

print("Local behavior model contract verified:")
for name in ("eat_drink", "smoking", "phone", "fire"):
    item = models[name]
    print(
        f"  {name}: labels={item['labels']} engine={item['engine']} "
        f"config={item['nvinferConfig']}"
    )
PY
