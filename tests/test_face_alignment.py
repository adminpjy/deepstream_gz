from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from deepstream_ai.domain import BoundingBox, FaceDetection
from deepstream_ai.face.alignment import (
    FivePointFaceAligner,
    canonical_landmarks,
    similarity_transform,
)
from deepstream_ai.face.errors import InvalidFaceInput


def test_similarity_transform_maps_landmarks_to_destination() -> None:
    source = canonical_landmarks((112, 112)) * 0.8 + np.asarray((7.0, 4.0))
    destination = canonical_landmarks((112, 112))

    matrix = similarity_transform(source, destination)
    mapped = source @ matrix[:, :2].T + matrix[:, 2]

    assert mapped == pytest.approx(destination, abs=1e-6)


def test_five_point_aligner_accepts_absolute_frame_landmarks() -> None:
    offset = np.asarray((100.0, 50.0))
    points = canonical_landmarks((112, 112)) + offset
    detection = FaceDetection(
        "cam",
        1,
        datetime.now(UTC),
        BoundingBox(100, 50, 212, 162),
        0.95,
        tuple(map(tuple, points)),
    )
    crop = np.zeros((112, 112, 4), dtype=np.uint8)
    crop[40:70, 30:80, 0] = 255

    aligned = FivePointFaceAligner().align(crop, detection)

    assert aligned.shape == (112, 112, 4)
    assert aligned.dtype == np.uint8


def test_similarity_transform_rejects_degenerate_landmarks() -> None:
    points = np.ones((5, 2), dtype=np.float64)

    with pytest.raises(InvalidFaceInput, match="degenerate"):
        similarity_transform(points, canonical_landmarks((112, 112)))
