# Dynamic-class YOLO parser

This DeepStream host-output parser supports raw Ultralytics YOLO8/9/11 detect
outputs laid out as either `[rows, 4 + class_count]` or
`[4 + class_count, rows]`. Class count comes from the nvinfer
`num-detected-classes` setting; it is not hard-coded to COCO's 80 classes.

The parser expects center-x, center-y, width, height followed by class
probabilities in network-input pixel coordinates. It does not apply NMS;
configure `cluster-mode=2` and the class thresholds in nvinfer. Use
`disable-output-host-copy=0` because this implementation reads host output.

Do not use this parser for models with objectness, embedded NMS, segmentation,
pose, OBB, or a different tensor contract. Validate the engine output against
the training framework before enabling production events.

Use `scripts/convert-model.sh` with metadata schema v2 to generate a matching
dynamic-batch engine, labels and SGIE config. That flow verifies the exported
ONNX graph exposes exactly this raw `4 + class_count` contract and ensures the
TensorRT max profile covers the generated nvinfer batch (at least 16).
