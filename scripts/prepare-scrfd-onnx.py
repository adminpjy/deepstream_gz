#!/usr/bin/env python3
"""Adapt a standard InsightFace SCRFD ONNX export for DeepStream nvinfer.

InsightFace's fixed-640 SCRFD export exposes outputs as ``[rows, width]`` even
though its input has a batch axis.  DeepStream treats the first output axis as
the inference batch and removes it before invoking a custom bbox parser.  This
tool adds a size-one batch axis to all nine terminal outputs while preserving
their binding names and numerical values.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError as exc:  # pragma: no cover - exercised by operators, not CI
    raise SystemExit("onnx is required: python -m pip install 'onnx>=1.16,<2'") from exc


EXPECTED_SHAPES = {
    (12_800, 1),
    (3_200, 1),
    (800, 1),
    (12_800, 4),
    (3_200, 4),
    (800, 4),
    (12_800, 10),
    (3_200, 10),
    (800, 10),
}


def _static_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    tensor = value_info.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT:
        raise ValueError(f"output {value_info.name!r} must be FP32")
    dimensions: list[int] = []
    for dimension in tensor.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError(f"output {value_info.name!r} must have a positive static shape")
        dimensions.append(int(dimension.dim_value))
    return tuple(dimensions)


def _default_opset(model: onnx.ModelProto) -> int:
    versions = [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")]
    if len(versions) != 1:
        raise ValueError("model must declare exactly one default ONNX opset")
    return int(versions[0])


def _all_tensor_names(model: onnx.ModelProto) -> set[str]:
    names = {value.name for value in model.graph.input}
    names.update(value.name for value in model.graph.output)
    names.update(value.name for value in model.graph.value_info)
    names.update(value.name for value in model.graph.initializer)
    for node in model.graph.node:
        names.update(value for value in node.input if value)
        names.update(value for value in node.output if value)
    return names


def add_explicit_output_batch(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return ``model`` with standard SCRFD outputs changed from [N,W] to [1,N,W]."""

    onnx.checker.check_model(model, full_check=True)
    outputs = list(model.graph.output)
    shapes = [_static_shape(output) for output in outputs]
    if len(outputs) != 9 or set(shapes) != EXPECTED_SHAPES:
        raise ValueError(
            "expected exactly the fixed-640, 9-output SCRFD contract "
            f"{sorted(EXPECTED_SHAPES)}, got {sorted(shapes)}"
        )

    opset = _default_opset(model)
    names = _all_tensor_names(model)
    appended_nodes: list[onnx.NodeProto] = []
    for output, shape in zip(outputs, shapes, strict=True):
        binding_name = output.name
        internal_name = f"{binding_name}__deepstream_unbatched"
        if internal_name in names:
            raise ValueError(f"internal tensor name collision: {internal_name!r}")
        producers = [
            (node, index)
            for node in model.graph.node
            for index, name in enumerate(node.output)
            if name == binding_name
        ]
        consumers = [
            node.name or node.op_type for node in model.graph.node if binding_name in node.input
        ]
        if len(producers) != 1 or consumers:
            raise ValueError(
                f"output {binding_name!r} must have one producer and no graph consumers; "
                f"producers={len(producers)}, consumers={consumers}"
            )
        producer, output_index = producers[0]
        producer.output[output_index] = internal_name
        names.add(internal_name)

        node_name = f"DeepStreamOutputBatch_{binding_name}"
        if opset >= 13:
            axes_name = f"{internal_name}__axes"
            if axes_name in names:
                raise ValueError(f"axes tensor name collision: {axes_name!r}")
            model.graph.initializer.append(
                helper.make_tensor(axes_name, TensorProto.INT64, [1], [0])
            )
            names.add(axes_name)
            appended_nodes.append(
                helper.make_node(
                    "Unsqueeze", [internal_name, axes_name], [binding_name], name=node_name
                )
            )
        else:
            appended_nodes.append(
                helper.make_node(
                    "Unsqueeze", [internal_name], [binding_name], name=node_name, axes=[0]
                )
            )

        output.type.tensor_type.shape.ClearField("dim")
        for size in (1, *shape):
            output.type.tensor_type.shape.dim.add().dim_value = size

    model.graph.node.extend(appended_nodes)
    onnx.checker.check_model(model, full_check=True)
    converted_shapes = {_static_shape(output) for output in model.graph.output}
    if converted_shapes != {(1, *shape) for shape in EXPECTED_SHAPES}:
        raise ValueError("converted SCRFD output shapes failed post-conversion validation")
    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, destination: Path, *, force: bool = False) -> None:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination:
        raise ValueError("source and destination must be different files")
    if destination.exists() and not force:
        raise FileExistsError(f"destination exists (use --force): {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    converted = add_explicit_output_batch(onnx.load(source))
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        onnx.save(converted, temporary)
        onnx.checker.check_model(onnx.load(temporary), full_check=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"source_sha256={_sha256(source)}")
    print(f"output_sha256={_sha256(destination)}")
    print(f"output={destination}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original fixed-640 SCRFD ONNX")
    parser.add_argument("destination", type=Path, help="DeepStream-adapted ONNX output")
    parser.add_argument("--force", action="store_true", help="replace an existing destination")
    args = parser.parse_args(argv)
    try:
        convert(args.source, args.destination, force=args.force)
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        ValueError,
        onnx.checker.ValidationError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
