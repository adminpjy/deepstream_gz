"""SCRFD tensor decoding and DeepStream secondary-object landmark matching.

DeepStream's detector object contract carries boxes and confidence, but no face
landmarks.  With ``output-tensor-meta=1`` a secondary nvinfer instance attaches
``NvDsInferTensorMeta`` to the parent person object.  This module decodes the
standard 9-output SCRFD contract (score/bbox/kps for strides 8, 16 and 32), maps
the result from network coordinates back to the parent ROI, and matches it to
the face child created by nvinfer.
"""

from __future__ import annotations

import ctypes
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from deepstream_ai.domain import BoundingBox, Point

LOGGER = logging.getLogger(__name__)
_SCRFD_STRIDES = (8, 16, 32)
_SCRFD_ANCHORS = 2
_NVDSINFER_MAX_DIMS = 8


@dataclass(frozen=True, slots=True)
class ScrfdCandidate:
    bbox: BoundingBox
    score: float
    landmarks: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class ScrfdTensorResult:
    candidates: tuple[ScrfdCandidate, ...]
    network_width: int
    network_height: int
    maintain_aspect_ratio: bool
    symmetric_padding: bool


def _role_from_layer(name: str, shape: Sequence[int]) -> str | None:
    lowered = name.lower()
    if any(token in lowered for token in ("kps", "keypoint", "landmark")):
        return "kps"
    if any(token in lowered for token in ("bbox", "box", "loc")):
        return "bbox"
    if any(token in lowered for token in ("score", "cls", "conf")):
        return "score"
    dimensions = tuple(int(value) for value in shape if int(value) > 0)
    if not dimensions:
        return None
    # DeepStream may pad an explicit TensorRT output with trailing singleton
    # dimensions, e.g. [rows,4,1].  Numeric binding names therefore need role
    # inference across every dimension rather than from the final axis alone.
    # Match the native parser's fail-closed priority: keypoints, then boxes;
    # the remaining standard SCRFD output is the scalar score tensor.
    if 10 in dimensions:
        return "kps"
    if 4 in dimensions:
        return "bbox"
    return "score"


def _rows(array: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype != np.dtype(np.float32):
        raise ValueError(f"SCRFD output tensors must use FP32, got {values.dtype}")
    if (
        values.ndim <= 0
        or values.ndim > _NVDSINFER_MAX_DIMS
        or any(dimension <= 0 for dimension in values.shape)
    ):
        raise ValueError(f"invalid SCRFD output tensor dimensions {values.shape}")
    if values.size == 0 or values.size % width:
        raise ValueError(f"SCRFD tensor element count is not divisible by {width}")
    squeezed = np.squeeze(values)
    if width == 1:
        return squeezed.reshape(-1, 1)
    if squeezed.ndim >= 2 and squeezed.shape[-1] == width:
        return squeezed.reshape(-1, width)
    if squeezed.ndim == 2 and squeezed.shape[0] == width:
        return squeezed.T.copy()
    return squeezed.reshape(-1, width)


def _stride_for_rows(rows: int, width: int, height: int) -> int:
    for stride in _SCRFD_STRIDES:
        if (width // stride) * (height // stride) * _SCRFD_ANCHORS == rows:
            return stride
    raise ValueError(
        f"SCRFD rows={rows} does not match 2 anchors at strides 8/16/32 "
        f"for network {width}x{height}"
    )


def decode_scrfd_outputs(
    layers: Mapping[str, np.ndarray] | Iterable[tuple[str, np.ndarray]],
    *,
    network_width: int,
    network_height: int,
    threshold: float = 0.0,
) -> tuple[ScrfdCandidate, ...]:
    """Decode standard SCRFD score/bbox/kps output tensors.

    The supported deployment contract is two anchors per location and strides
    8/16/32.  Both row-major ``[N,W]`` and channel-major ``[W,N]`` layer shapes
    are accepted.  Invalid or incomplete contracts fail closed.
    """

    if network_width <= 0 or network_height <= 0:
        raise ValueError("SCRFD network dimensions must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("SCRFD threshold must be in [0, 1]")
    items = list(layers.items() if isinstance(layers, Mapping) else layers)
    grouped: dict[int, dict[str, np.ndarray]] = {}
    for name, raw in items:
        array = np.asarray(raw)
        if array.dtype != np.dtype(np.float32):
            raise ValueError(f"SCRFD output layer {name!s} must use FP32, got {array.dtype}")
        if (
            array.ndim <= 0
            or array.ndim > _NVDSINFER_MAX_DIMS
            or any(dimension <= 0 for dimension in array.shape)
        ):
            raise ValueError(f"SCRFD output layer {name!s} has invalid dimensions {array.shape}")
        role = _role_from_layer(str(name), array.shape)
        if role is None:
            continue
        width = {"score": 1, "bbox": 4, "kps": 10}[role]
        values = _rows(array, width)
        grouped.setdefault(values.shape[0], {})[role] = values

    candidates: list[ScrfdCandidate] = []
    expected_strides: set[int] = set()
    for row_count, outputs in grouped.items():
        if set(outputs) != {"score", "bbox", "kps"}:
            continue
        stride = _stride_for_rows(row_count, network_width, network_height)
        expected_strides.add(stride)
        grid_width = network_width // stride
        grid_height = network_height // stride
        grid_y, grid_x = np.mgrid[0:grid_height, 0:grid_width]
        centers = np.stack((grid_x, grid_y), axis=-1).astype(np.float32)
        centers = (centers.reshape(-1, 2) * float(stride)).repeat(_SCRFD_ANCHORS, axis=0)
        scores = outputs["score"].reshape(-1)
        boxes = outputs["bbox"] * float(stride)
        keypoints = outputs["kps"].reshape(-1, 5, 2) * float(stride)
        for index in np.flatnonzero(np.isfinite(scores) & (scores >= threshold)):
            center_x, center_y = centers[index]
            left, top, right, bottom = boxes[index]
            if not np.all(np.isfinite((left, top, right, bottom))):
                continue
            x1 = min(max(float(center_x - left), 0.0), float(network_width))
            y1 = min(max(float(center_y - top), 0.0), float(network_height))
            x2 = min(max(float(center_x + right), 0.0), float(network_width))
            y2 = min(max(float(center_y + bottom), 0.0), float(network_height))
            if x2 <= x1 or y2 <= y1:
                continue
            points = keypoints[index] + centers[index]
            if not np.all(np.isfinite(points)):
                continue
            candidates.append(
                ScrfdCandidate(
                    bbox=BoundingBox(x1, y1, x2, y2),
                    score=min(1.0, max(0.0, float(scores[index]))),
                    landmarks=tuple((float(x), float(y)) for x, y in points),
                )
            )
    if expected_strides != set(_SCRFD_STRIDES):
        missing = sorted(set(_SCRFD_STRIDES) - expected_strides)
        raise ValueError(f"SCRFD tensor contract is incomplete; missing strides {missing}")
    return tuple(sorted(candidates, key=lambda item: item.score, reverse=True))


def _map_point(
    point: Point,
    parent: BoundingBox,
    *,
    network_width: int,
    network_height: int,
    maintain_aspect_ratio: bool,
    symmetric_padding: bool,
) -> Point:
    x, y = point
    if maintain_aspect_ratio:
        scale = min(network_width / parent.width, network_height / parent.height)
        pad_x = (network_width - parent.width * scale) / 2.0 if symmetric_padding else 0.0
        pad_y = (network_height - parent.height * scale) / 2.0 if symmetric_padding else 0.0
        return parent.x1 + (x - pad_x) / scale, parent.y1 + (y - pad_y) / scale
    return (
        parent.x1 + x * parent.width / network_width,
        parent.y1 + y * parent.height / network_height,
    )


def map_scrfd_candidate(
    candidate: ScrfdCandidate, parent: BoundingBox, result: ScrfdTensorResult
) -> ScrfdCandidate:
    top_left = _map_point(
        (candidate.bbox.x1, candidate.bbox.y1),
        parent,
        network_width=result.network_width,
        network_height=result.network_height,
        maintain_aspect_ratio=result.maintain_aspect_ratio,
        symmetric_padding=result.symmetric_padding,
    )
    bottom_right = _map_point(
        (candidate.bbox.x2, candidate.bbox.y2),
        parent,
        network_width=result.network_width,
        network_height=result.network_height,
        maintain_aspect_ratio=result.maintain_aspect_ratio,
        symmetric_padding=result.symmetric_padding,
    )
    mapped_box = BoundingBox(top_left[0], top_left[1], bottom_right[0], bottom_right[1])
    mapped_points = tuple(
        _map_point(
            point,
            parent,
            network_width=result.network_width,
            network_height=result.network_height,
            maintain_aspect_ratio=result.maintain_aspect_ratio,
            symmetric_padding=result.symmetric_padding,
        )
        for point in candidate.landmarks
    )
    return ScrfdCandidate(mapped_box, candidate.score, mapped_points)


def _iou(left: BoundingBox, right: BoundingBox) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    union = left.area + right.area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_scrfd_landmarks(
    face_bbox: BoundingBox,
    parent_bbox: BoundingBox,
    result: ScrfdTensorResult,
    *,
    minimum_iou: float = 0.35,
) -> tuple[Point, ...]:
    best_iou = 0.0
    best: ScrfdCandidate | None = None
    for candidate in result.candidates:
        try:
            mapped = map_scrfd_candidate(candidate, parent_bbox, result)
        except ValueError:
            continue
        overlap = _iou(face_bbox, mapped.bbox)
        if overlap > best_iou or (
            math.isclose(overlap, best_iou) and best and mapped.score > best.score
        ):
            best_iou, best = overlap, mapped
    if best is None or best_iou < minimum_iou:
        return ()
    return best.landmarks


def assign_scrfd_landmarks(
    face_bboxes: Sequence[BoundingBox],
    parent_bbox: BoundingBox,
    result: ScrfdTensorResult,
    *,
    minimum_iou: float = 0.35,
) -> tuple[tuple[Point, ...], ...]:
    """One-to-one match SCRFD proposals to face child boxes.

    A proposal is never reused for two faces under the same person.  Global
    IoU ordering makes the result deterministic and ambiguity fails closed.
    """

    mapped: list[tuple[int, ScrfdCandidate]] = []
    for candidate_index, candidate in enumerate(result.candidates):
        try:
            mapped.append((candidate_index, map_scrfd_candidate(candidate, parent_bbox, result)))
        except ValueError:
            continue
    pairs: list[tuple[float, float, int, int, tuple[Point, ...]]] = []
    for face_index, face_bbox in enumerate(face_bboxes):
        for candidate_index, candidate in mapped:
            overlap = _iou(face_bbox, candidate.bbox)
            if overlap >= minimum_iou:
                pairs.append(
                    (overlap, candidate.score, face_index, candidate_index, candidate.landmarks)
                )
    pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    assigned_faces: set[int] = set()
    assigned_candidates: set[int] = set()
    output: list[tuple[Point, ...]] = [() for _ in face_bboxes]
    for _overlap, _score, face_index, candidate_index, landmarks in pairs:
        if face_index in assigned_faces or candidate_index in assigned_candidates:
            continue
        output[face_index] = landmarks
        assigned_faces.add(face_index)
        assigned_candidates.add(candidate_index)
    return tuple(output)


def _tensor_layers(pyds: Any, tensor_meta: Any) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for index in range(int(tensor_meta.num_output_layers)):
        # NVIDIA's helper copies out_buf_ptrs_host[index] into layer.buffer;
        # output_layers_info() alone explicitly does not provide valid buffers.
        layer = pyds.get_nvds_LayerInfo(tensor_meta, index)
        dims = layer.inferDims
        num_dims = int(dims.numDims)
        if num_dims <= 0 or num_dims > _NVDSINFER_MAX_DIMS:
            raise ValueError(f"SCRFD output layer {index} has invalid numDims {num_dims}")
        shape = tuple(int(dims.d[item]) for item in range(num_dims))
        count = int(dims.numElements)
        if count <= 0 or any(dimension <= 0 for dimension in shape):
            raise ValueError(f"SCRFD output layer {index} has invalid dimensions {shape}")
        expected_count = math.prod(shape)
        if expected_count != count:
            raise ValueError(
                f"SCRFD output layer {index} shape {shape} contains "
                f"{expected_count} elements, metadata reports {count}"
            )
        dtype_id = int(layer.dataType)
        if dtype_id != 0:
            raise ValueError(f"SCRFD output layer {index} must use FP32, got dtype {dtype_id}")
        data_address = int(pyds.get_ptr(layer.buffer))
        if data_address == 0:
            raise ValueError(f"SCRFD output layer {index} has a null host buffer")
        raw = np.ctypeslib.as_array((ctypes.c_float * count).from_address(data_address)).copy()
        array = raw.reshape(shape)
        name = str(getattr(layer, "layerName", "") or f"layer-{index}")
        result[name] = array
    return result


def extract_scrfd_tensor_result(
    pyds: Any,
    parent_obj_meta: Any,
    *,
    unique_id: int,
    threshold: float = 0.65,
) -> ScrfdTensorResult | None:
    """Copy and decode SCRFD tensor meta attached to a secondary input object."""

    if parent_obj_meta is None:
        return None
    expected_meta_type = int(pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META)
    node = getattr(parent_obj_meta, "obj_user_meta_list", None)
    while node is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(node.data)
            if int(user_meta.base_meta.meta_type) == expected_meta_type:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                if int(tensor_meta.unique_id) == unique_id:
                    network = tensor_meta.network_info
                    layers = _tensor_layers(pyds, tensor_meta)
                    candidates = decode_scrfd_outputs(
                        layers,
                        network_width=int(network.width),
                        network_height=int(network.height),
                        threshold=threshold,
                    )
                    return ScrfdTensorResult(
                        candidates=candidates,
                        network_width=int(network.width),
                        network_height=int(network.height),
                        maintain_aspect_ratio=bool(tensor_meta.maintain_aspect_ratio),
                        symmetric_padding=bool(tensor_meta.symmetric_padding),
                    )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            LOGGER.exception("无法解码 SCRFD NvDsInferTensorMeta")
            return None
        try:
            node = node.next
        except StopIteration:
            break
    return None


def landmarks_from_scrfd_tensor(
    pyds: Any,
    face_obj_meta: Any,
    face_bbox: BoundingBox,
    *,
    unique_id: int,
    threshold: float = 0.65,
) -> tuple[Point, ...]:
    parent = getattr(face_obj_meta, "parent", None)
    parent_bbox = None if parent is None else _native_bbox(parent)
    if parent is None or parent_bbox is None:
        return ()
    result = extract_scrfd_tensor_result(
        pyds,
        parent,
        unique_id=unique_id,
        threshold=threshold,
    )
    if result is None:
        return ()
    return match_scrfd_landmarks(face_bbox, parent_bbox, result)


def _native_bbox(obj_meta: Any) -> BoundingBox | None:
    rect = getattr(obj_meta, "rect_params", None)
    if rect is None:
        return None
    try:
        return BoundingBox(
            float(rect.left),
            float(rect.top),
            float(rect.left + rect.width),
            float(rect.top + rect.height),
        )
    except (TypeError, ValueError):
        return None


__all__ = [
    "ScrfdCandidate",
    "ScrfdTensorResult",
    "assign_scrfd_landmarks",
    "decode_scrfd_outputs",
    "extract_scrfd_tensor_result",
    "landmarks_from_scrfd_tensor",
    "map_scrfd_candidate",
    "match_scrfd_landmarks",
]
