from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from deepstream_ai.domain import BoundingBox, Track
from deepstream_ai.pipeline.metadata import (
    MetadataProbe,
    _face_landmarks,
    _hide_pgie_non_person_osd,
    _validate_gpu_surface_layout,
)


def test_nearest_track_prefers_smallest_containing_person() -> None:
    now = datetime.now(UTC)
    tracks = [
        Track("cam", 1, now, BoundingBox(0, 0, 100, 100)),
        Track("cam", 2, now, BoundingBox(20, 20, 80, 80)),
    ]

    match = MetadataProbe._nearest_track(BoundingBox(40, 40, 60, 60), tracks)

    assert match is not None
    assert match.track_id == 2


def test_nearest_track_rejects_face_outside_people() -> None:
    now = datetime.now(UTC)
    tracks = [Track("cam", 1, now, BoundingBox(0, 0, 100, 100))]

    assert MetadataProbe._nearest_track(BoundingBox(150, 150, 170, 170), tracks) is None


def _osd_object(*, component_id: int, class_id: int):
    return SimpleNamespace(
        unique_component_id=component_id,
        class_id=class_id,
        rect_params=SimpleNamespace(border_width=3, has_bg_color=True),
        text_params=SimpleNamespace(display_text="original", set_bg_clr=True),
    )


@pytest.mark.parametrize("class_id", [1, 2])
def test_pgie_non_person_classes_are_hidden_from_osd_without_removing_metadata(
    class_id: int,
) -> None:
    obj = _osd_object(component_id=1, class_id=class_id)

    hidden = _hide_pgie_non_person_osd(
        obj,
        pgie_unique_id=1,
        person_class_ids=(0,),
    )

    assert hidden
    assert obj.unique_component_id == 1
    assert obj.class_id == class_id
    assert obj.rect_params.border_width == 0
    assert obj.rect_params.has_bg_color is False
    assert obj.text_params.display_text == ""
    assert obj.text_params.set_bg_clr is False


@pytest.mark.parametrize(
    ("component_id", "class_id"),
    [
        (1, 0),  # PeopleNet Person remains visible.
        (2, 0),  # SCRFD face metadata and OSD remain untouched.
    ],
)
def test_person_and_scrfd_osd_are_not_changed(component_id: int, class_id: int) -> None:
    obj = _osd_object(component_id=component_id, class_id=class_id)

    hidden = _hide_pgie_non_person_osd(
        obj,
        pgie_unique_id=1,
        person_class_ids=(0,),
    )

    assert not hidden
    assert obj.rect_params.border_width == 3
    assert obj.rect_params.has_bg_color is True
    assert obj.text_params.display_text == "original"
    assert obj.text_params.set_bg_clr is True


def test_face_landmark_mask_can_decode_normalized_coordinates() -> None:
    class Mask:
        size = 10

        @staticmethod
        def get_mask_array():
            return [0.25, 0.2, 0.75, 0.2, 0.5, 0.5, 0.3, 0.8, 0.7, 0.8]

    class Object:
        mask_params = Mask()

    points = _face_landmarks(
        Object(),
        BoundingBox(100, 50, 200, 150),
        source="mask",
        coordinates="normalized",
        scale=1.0,
    )

    assert points[0] == pytest.approx((125.0, 70.0))
    assert points[4] == pytest.approx((170.0, 130.0))


def test_gpu_surface_layout_accepts_pitched_rgba() -> None:
    dtype, shape, strides, size = _validate_gpu_surface_layout(
        "uint8", (1080, 1920, 4), (8192, 4, 1), 1080 * 8192
    )

    assert dtype.name == "uint8"
    assert shape == (1080, 1920, 4)
    assert strides == (8192, 4, 1)
    assert size == 1080 * 8192


@pytest.mark.parametrize(
    ("dtype", "shape", "strides", "size"),
    [
        ("float32", (1080, 1920, 4), (7680, 4, 1), 1080 * 7680),
        ("uint8", (1080, 1920, 3), (5760, 3, 1), 1080 * 5760),
        ("uint8", (1080, 1920, 4), (7000, 4, 1), 1080 * 7000),
        ("uint8", (1080, 1920, 4), (7680, 4, 1), 1024),
        ("uint8", (1080, 1920, 4), (7680, 4, 1), 513 * 1024 * 1024),
    ],
)
def test_gpu_surface_layout_rejects_unsafe_contracts(
    dtype: str,
    shape: tuple[int, int, int],
    strides: tuple[int, int, int],
    size: int,
) -> None:
    with pytest.raises(ValueError, match="NvBufSurface"):
        _validate_gpu_surface_layout(dtype, shape, strides, size)
