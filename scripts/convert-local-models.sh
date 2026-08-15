#!/usr/bin/env bash
set -euo pipefail

# The production behavior path currently integrates only eating/drinking and
# smoking. fire.onnx is a rank-2 whole-frame classifier and must not be forced
# through the raw-YOLO detector converter.
python3 /workspace/scripts/prepare-local-behavior-models-dynamic.py \
  --model-root /workspace/models \
  --config-root /workspace/configs/nvinfer \
  --device "${MODEL_CONVERTER_DEVICE:-0}" \
  --precision "${MODEL_CONVERTER_PRECISION:-fp16}" \
  --imgsz "${MODEL_CONVERTER_IMGSZ:-640}" \
  --only eat_drink \
  --only smoking \
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
required = {"eat_drink", "smoking"}
missing = sorted(required.difference(models))
if missing:
    raise SystemExit("model conversion manifest missing: " + ", ".join(missing))

# yolo11n.onnx is the standard COCO detector. The converter keeps its source
# labels in sourceLabels but exposes two business labels through the dedicated
# person-crop proxy parser. The evidence class set mirrors opsvision exactly.
expected = ["eating", "drinking"]
actual = models["eat_drink"].get("labels")
if actual != expected:
    raise SystemExit(
        "eat/drink business labels must be ['eating', 'drinking']; "
        f"actual={actual!r}."
    )
source_labels = set(models["eat_drink"].get("sourceLabels") or [])
required_source = {
    "bottle", "cup", "wine_glass", "bowl",
    "apple", "banana", "sandwich", "orange",
    "pizza", "donut", "cake", "hot_dog",
}
missing_source = sorted(required_source.difference(source_labels))
if missing_source:
    raise SystemExit(
        "yolo11n.onnx source labels missing opsvision eat/drink classes: "
        + ", ".join(missing_source)
    )

actual_smoking = models["smoking"].get("labels")
if actual_smoking != ["smoking"]:
    raise SystemExit(
        f"smoking model labels must be ['smoking']; actual={actual_smoking!r}. "
        "Review the trained checkpoint classes before production use."
    )

config_path = Path(models["eat_drink"]["nvinferConfig"])
config_text = config_path.read_text(encoding="utf-8")
for required_line in (
    "parse-bbox-func-name=NvDsInferParseCustomYoloEatDrinkCoco",
    "pre-cluster-threshold=0.45",
    "num-detected-classes=2",
):
    if required_line not in config_text:
        raise SystemExit(f"eat/drink nvinfer config missing {required_line!r}")

print("Local behavior model contract verified:")
for name in ("eat_drink", "smoking"):
    item = models[name]
    print(
        f"  {name}: labels={item['labels']} engine={item['engine']} "
        f"config={item['nvinferConfig']}"
    )
PY
