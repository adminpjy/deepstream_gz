from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from deepstream_ai.domain import BoundingBox
from deepstream_ai.pipeline.scrfd import (
    ScrfdTensorResult,
    _tensor_layers,
    assign_scrfd_landmarks,
    decode_scrfd_outputs,
    map_scrfd_candidate,
    match_scrfd_landmarks,
)


def _outputs() -> dict[str, np.ndarray]:
    layers: dict[str, np.ndarray] = {}
    for stride, rows in ((8, 32), (16, 8), (32, 2)):
        layers[f"score_{stride}"] = np.zeros((rows, 1), dtype=np.float32)
        layers[f"bbox_{stride}"] = np.zeros((rows, 4), dtype=np.float32)
        layers[f"kps_{stride}"] = np.zeros((rows, 10), dtype=np.float32)
    # stride-8 cell (1,1), first anchor: center=(8,8), box=(4,4)-(12,12)
    row = 10
    layers["score_8"][row, 0] = 0.9
    layers["bbox_8"][row] = (0.5, 0.5, 0.5, 0.5)
    layers["kps_8"][row] = (
        -0.25,
        -0.25,
        0.25,
        -0.25,
        0.0,
        0.0,
        -0.2,
        0.25,
        0.2,
        0.25,
    )
    return layers


def test_decodes_strict_nine_output_scrfd_contract() -> None:
    candidates = decode_scrfd_outputs(
        _outputs(), network_width=32, network_height=32, threshold=0.5
    )

    assert len(candidates) == 1
    assert candidates[0].bbox.as_tuple() == pytest.approx((4.0, 4.0, 12.0, 12.0))
    assert candidates[0].landmarks[0] == pytest.approx((6.0, 6.0))
    assert candidates[0].landmarks[4] == pytest.approx((9.6, 10.0))


def test_decodes_numeric_bindings_with_trailing_singleton_dimensions() -> None:
    names = ("448", "451", "454", "471", "474", "477", "494", "497", "500")
    layers = {
        name: values.reshape(*values.shape, 1)
        for name, values in zip(names, _outputs().values(), strict=True)
    }

    candidates = decode_scrfd_outputs(layers, network_width=32, network_height=32, threshold=0.5)

    assert len(candidates) == 1
    assert candidates[0].bbox.as_tuple() == pytest.approx((4.0, 4.0, 12.0, 12.0))
    assert candidates[0].landmarks[0] == pytest.approx((6.0, 6.0))
    assert candidates[0].landmarks[4] == pytest.approx((9.6, 10.0))


def test_maps_parent_roi_and_matches_child_face() -> None:
    candidates = decode_scrfd_outputs(
        _outputs(), network_width=32, network_height=32, threshold=0.5
    )
    tensor = ScrfdTensorResult(candidates, 32, 32, True, True)
    parent = BoundingBox(100, 50, 164, 114)
    mapped = map_scrfd_candidate(candidates[0], parent, tensor)

    assert mapped.bbox.as_tuple() == pytest.approx((108, 58, 124, 74))
    points = match_scrfd_landmarks(mapped.bbox, parent, tensor)
    assert np.asarray(points) == pytest.approx(np.asarray(mapped.landmarks))


def test_assignment_never_reuses_one_proposal_for_two_faces() -> None:
    candidates = decode_scrfd_outputs(
        _outputs(), network_width=32, network_height=32, threshold=0.5
    )
    tensor = ScrfdTensorResult(candidates, 32, 32, False, False)
    parent = BoundingBox(0, 0, 32, 32)
    face = BoundingBox(4, 4, 12, 12)

    assigned = assign_scrfd_landmarks((face, face), parent, tensor)

    assert bool(assigned[0]) is not bool(assigned[1])


def test_incomplete_scrfd_variant_fails_closed() -> None:
    layers = _outputs()
    del layers["kps_32"]

    with pytest.raises(ValueError, match="incomplete"):
        decode_scrfd_outputs(layers, network_width=32, network_height=32)


@pytest.mark.parametrize("dtype", [np.float16, np.int8, np.int32])
def test_non_fp32_scrfd_outputs_fail_closed(dtype: type[np.generic]) -> None:
    layers = {name: values.astype(dtype) for name, values in _outputs().items()}

    with pytest.raises(ValueError, match="must use FP32"):
        decode_scrfd_outputs(layers, network_width=32, network_height=32)


class _FakePyds:
    @staticmethod
    def get_nvds_LayerInfo(tensor_meta: SimpleNamespace, index: int) -> SimpleNamespace:
        return tensor_meta.layers[index]

    @staticmethod
    def get_ptr(buffer: ctypes.Array[ctypes.c_float]) -> int:
        return ctypes.addressof(buffer)


def _tensor_meta_layer(
    *,
    dtype: int = 0,
    num_dims: int = 2,
    dimensions: object = (2, 4),
    count: int = 8,
) -> tuple[SimpleNamespace, ctypes.Array[ctypes.c_float]]:
    storage = (ctypes.c_float * max(count, 1))()
    layer = SimpleNamespace(
        buffer=storage,
        layerName="bbox_32",
        dataType=dtype,
        inferDims=SimpleNamespace(
            numDims=num_dims,
            d=dimensions,
            numElements=count,
        ),
    )
    return layer, storage


def test_tensor_meta_decoder_copies_fp32_tensor() -> None:
    layer, storage = _tensor_meta_layer()
    for index in range(8):
        storage[index] = float(index)
    meta = SimpleNamespace(num_output_layers=1, layers=[layer])

    result = _tensor_layers(_FakePyds, meta)

    assert result["bbox_32"].dtype == np.float32
    assert result["bbox_32"].shape == (2, 4)
    assert result["bbox_32"] == pytest.approx(np.arange(8, dtype=np.float32).reshape(2, 4))


@pytest.mark.parametrize("dtype", [1, 2, 3])
def test_tensor_meta_decoder_rejects_non_fp32(dtype: int) -> None:
    layer, _storage = _tensor_meta_layer(dtype=dtype)
    meta = SimpleNamespace(num_output_layers=1, layers=[layer])

    with pytest.raises(ValueError, match="must use FP32"):
        _tensor_layers(_FakePyds, meta)


def test_tensor_meta_decoder_validates_num_dims_before_indexing() -> None:
    class ExplodingDimensions:
        def __getitem__(self, _index: int) -> int:
            raise AssertionError("dimension array must not be accessed")

    layer, _storage = _tensor_meta_layer(num_dims=0, dimensions=ExplodingDimensions())
    meta = SimpleNamespace(num_output_layers=1, layers=[layer])

    with pytest.raises(ValueError, match="invalid numDims 0"):
        _tensor_layers(_FakePyds, meta)


def test_tensor_meta_decoder_rejects_inconsistent_element_count() -> None:
    layer, _storage = _tensor_meta_layer(count=7)
    meta = SimpleNamespace(num_output_layers=1, layers=[layer])

    with pytest.raises(ValueError, match="contains 8 elements, metadata reports 7"):
        _tensor_layers(_FakePyds, meta)
