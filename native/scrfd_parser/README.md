# SCRFD DeepStream parser

`NvDsInferParseCustomSCRFD` implements the same strict deployment contract as
`deepstream_ai.pipeline.scrfd`:

- standard SCRFD detector with five landmarks;
- exactly 9 FP32 output tensors: score, bbox distance and 5 keypoints for each
  of strides 8, 16 and 32;
- two anchors per feature-map location;
- row-major `[N,1|4|10]` or channel-major `[1|4|10,N]` output layout;
- nvinfer performs NMS (`cluster-mode=2`); raw host outputs remain attached with
  `output-tensor-meta=1` for Python landmark matching.

Six-output/no-landmark, ten-output, fifteen-output and anchor-free variants are
rejected. This is deliberate: silently guessing an incompatible SCRFD ABI
would produce believable but wrong face alignment.
