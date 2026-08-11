from __future__ import annotations

import importlib.util
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare-scrfd-onnx.py"
SPEC = importlib.util.spec_from_file_location("prepare_scrfd_onnx", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


STANDARD_SHAPES = (
    (12_800, 1),
    (3_200, 1),
    (800, 1),
    (12_800, 4),
    (3_200, 4),
    (800, 4),
    (12_800, 10),
    (3_200, 10),
    (800, 10),
)


def _model(shapes: tuple[tuple[int, ...], ...] = STANDARD_SHAPES) -> onnx.ModelProto:
    binding_names = ("448", "471", "494", "451", "474", "497", "454", "477", "500")
    inputs = []
    outputs = []
    nodes = []
    for index, (name, shape) in enumerate(zip(binding_names, shapes, strict=True)):
        input_name = f"input_{index}"
        inputs.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        nodes.append(helper.make_node("Identity", [input_name], [name], name=f"Identity_{name}"))
    graph = helper.make_graph(nodes, "scrfd-test", inputs, outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def _shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(int(dimension.dim_value) for dimension in value_info.type.tensor_type.shape.dim)


def test_adds_batch_axis_and_preserves_output_binding_names() -> None:
    original = _model()
    names = [output.name for output in original.graph.output]

    converted = MODULE.add_explicit_output_batch(original)

    onnx.checker.check_model(converted, full_check=True)
    assert [output.name for output in converted.graph.output] == names
    assert [_shape(output) for output in converted.graph.output] == [
        (1, *shape) for shape in STANDARD_SHAPES
    ]
    assert sum(node.op_type == "Unsqueeze" for node in converted.graph.node) == 9
    identity_outputs = {
        output
        for node in converted.graph.node
        if node.op_type == "Identity"
        for output in node.output
    }
    assert identity_outputs == {f"{name}__deepstream_unbatched" for name in names}


def test_rejects_nonstandard_scrfd_contract() -> None:
    invalid_shapes = (*STANDARD_SHAPES[:-1], (801, 10))

    with pytest.raises(ValueError, match="fixed-640, 9-output SCRFD contract"):
        MODULE.add_explicit_output_batch(_model(invalid_shapes))


def test_convert_refuses_to_overwrite_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    destination = tmp_path / "prepared.onnx"
    onnx.save(_model(), source)

    MODULE.convert(source, destination)

    with pytest.raises(FileExistsError, match="destination exists"):
        MODULE.convert(source, destination)
