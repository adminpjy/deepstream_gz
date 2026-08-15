#!/usr/bin/env python3
"""Compatibility entry point for local YOLO models with dynamic ONNX inputs.

The primary converter intentionally validates the raw YOLO detector contract.
Older/custom Ultralytics exports can make batch, channel and spatial dimensions
symbolic even though deployment always feeds RGB 640x640 tensors. This entry
point keeps those ONNX graphs intact and supplies an explicit TensorRT profile
instead of rejecting them solely because H/W (or C) is symbolic.

For the shared eat/drink COCO proxy, the generated nvinfer config is normalized
to the previously validated opsvision rule: prop confidence >= 0.45. The custom
DeepStream parser owns the matching top-40%-of-person and COCO class mapping.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BASE_PATH = Path(__file__).with_name("prepare-local-behavior-models.py")
_SPEC = importlib.util.spec_from_file_location("deepstream_local_model_converter_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load base converter: {_BASE_PATH}")
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

_ORIGINAL_WRITE_CONFIG = base._write_nvinfer_config


def _requested_imgsz() -> int:
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--imgsz":
            try:
                return int(sys.argv[index + 1])
            except ValueError:
                break
        if value.startswith("--imgsz="):
            try:
                return int(value.split("=", 1)[1])
            except ValueError:
                break
    return int(os.environ.get("MODEL_CONVERTER_IMGSZ", "640"))


@dataclass(frozen=True, slots=True)
class DynamicOnnxContract:
    input_name: str
    input_shape: tuple[int | None, int | None, int | None, int | None]
    labels: tuple[str, ...]
    dynamic_batch: bool
    output_shape: tuple[int | None, ...]
    profile_shape: tuple[int, int, int, int]

    @property
    def batch_size(self) -> int:
        return 16 if self.dynamic_batch else int(self.profile_shape[0])


def _inspect_onnx(path: Path, spec: Any) -> DynamicOnnxContract:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnx package is required in the model-converter image") from exc

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise ValueError(f"{spec.name}: raw YOLO detect ONNX must have exactly one input")
    if len(model.graph.output) != 1:
        raise ValueError(
            f"{spec.name}: expected one raw YOLO output; embedded NMS/end-to-end models are unsupported"
        )

    model_input = model.graph.input[0]
    dims = model_input.type.tensor_type.shape.dim
    if len(dims) != 4:
        raise ValueError(f"{spec.name}: ONNX input must be NCHW rank 4")
    original = tuple(base._dim_value(dim) for dim in dims)
    batch, channels, height, width = original
    if channels not in (None, 3):
        raise ValueError(
            f"{spec.name}: expected RGB channel dimension 3 or dynamic, got input {original}"
        )

    imgsz = _requested_imgsz()
    resolved_batch = int(batch or 1)
    resolved_height = int(height or imgsz)
    resolved_width = int(width or imgsz)
    if resolved_height <= 0 or resolved_width <= 0:
        raise ValueError(f"{spec.name}: invalid resolved input size {resolved_height}x{resolved_width}")
    profile_shape = (resolved_batch, 3, resolved_height, resolved_width)

    output_dims = model.graph.output[0].type.tensor_type.shape.dim
    if len(output_dims) != 3:
        raise ValueError(
            f"{spec.name}: raw YOLO output must be rank 3 [N,4+C,rows] or [N,rows,4+C]; "
            f"actual rank={len(output_dims)}"
        )
    output_shape = tuple(base._dim_value(dim) for dim in output_dims)
    metadata_labels = base._onnx_metadata_names(model)
    labels = metadata_labels or spec.fallback_labels
    expected_channels = 4 + len(labels)
    static_tail = {value for value in output_shape[1:] if value is not None}
    if static_tail and expected_channels not in static_tail:
        fallback_channels = 4 + len(spec.fallback_labels)
        if metadata_labels or fallback_channels not in static_tail:
            raise ValueError(
                f"{spec.name}: output shape {output_shape} does not expose 4+C={expected_channels}; "
                "the supplied model is not compatible with the repository raw YOLO parser"
            )
        labels = spec.fallback_labels
    elif not static_tail and not metadata_labels:
        # Fully symbolic legacy exports cannot prove C from the graph shape.
        # Keep the reviewed per-model fallback and let TensorRT/parser validation
        # be the next contract gate. The shell verifier still checks exact labels.
        labels = spec.fallback_labels

    return DynamicOnnxContract(
        input_name=model_input.name,
        input_shape=(batch, channels, height, width),
        labels=tuple(labels),
        dynamic_batch=batch is None,
        output_shape=output_shape,
        profile_shape=profile_shape,
    )


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    contract: DynamicOnnxContract,
    *,
    device: int,
    workspace_mib: int,
    precision: str,
) -> None:
    trtexec = base._find_trtexec()
    batch, channels, height, width = contract.profile_shape
    args = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mib}",
        f"--device={device}",
        "--skipInference",
    ]
    if any(value is None for value in contract.input_shape):
        if contract.dynamic_batch:
            min_batch, opt_batch, max_batch = 1, 8, 16
        else:
            min_batch = opt_batch = max_batch = batch
        args.extend(
            [
                f"--minShapes={contract.input_name}:{min_batch}x{channels}x{height}x{width}",
                f"--optShapes={contract.input_name}:{opt_batch}x{channels}x{height}x{width}",
                f"--maxShapes={contract.input_name}:{max_batch}x{channels}x{height}x{width}",
            ]
        )
    if precision == "fp16":
        args.append("--fp16")
    elif precision != "fp32":
        raise ValueError(f"unsupported precision: {precision}")
    print(
        f"TensorRT profile {onnx_path.name}: min/opt/max batch="
        f"{'1/8/16' if contract.dynamic_batch else str(batch)} "
        f"input=3x{height}x{width}",
        flush=True,
    )
    subprocess.run(args, check=True)
    if not engine_path.is_file() or engine_path.stat().st_size <= 0:
        raise RuntimeError(f"TensorRT engine was not created: {engine_path}")


def _normalize_eat_drink_config(config_path: Path) -> None:
    value = config_path.read_text(encoding="utf-8")
    required = "parse-bbox-func-name=NvDsInferParseCustomYoloEatDrinkCoco"
    if required not in value:
        raise RuntimeError(
            "generated eat/drink config is not using NvDsInferParseCustomYoloEatDrinkCoco"
        )
    old = "pre-cluster-threshold=0.35"
    new = "pre-cluster-threshold=0.45"
    if old in value:
        value = value.replace(old, new, 1)
    elif new not in value:
        raise RuntimeError("generated eat/drink config has no recognized confidence threshold")
    config_path.write_text(value, encoding="utf-8", newline="\n")


def _write_nvinfer_config(
    spec: Any,
    contract: DynamicOnnxContract,
    **kwargs: Any,
) -> None:
    # The base writer only needs concrete H/W for infer-dims. Preserve the
    # original dynamic_batch flag so its configured SGIE batch remains 16.
    concrete = base.OnnxContract(
        input_name=contract.input_name,
        input_shape=contract.profile_shape,
        labels=contract.labels,
        dynamic_batch=contract.dynamic_batch,
        output_shape=contract.output_shape,
    )
    _ORIGINAL_WRITE_CONFIG(spec, concrete, **kwargs)
    if spec.name == "eat_drink":
        _normalize_eat_drink_config(Path(kwargs["config_path"]))


base._inspect_onnx = _inspect_onnx
base._build_engine = _build_engine
base._write_nvinfer_config = _write_nvinfer_config


if __name__ == "__main__":
    raise SystemExit(base.main())
