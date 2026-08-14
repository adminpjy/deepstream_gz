from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from deepstream_ai.domain import BoundingBox, FaceDetection, Track
from deepstream_ai.face_anchored_continuity import FaceAnchoredTrackContinuityResolver
from deepstream_ai.pipeline.metadata import _TRACKER_REID_METADATA_KEY, FramePacket
from deepstream_ai.track_continuity import TrackContinuityConfig
from deepstream_ai.track_continuity_guard import EdgeBridgeConfig, SingleTargetBridgeConfig

NOW = datetime(2026, 8, 14, tzinfo=UTC)


def _embedding(cosine_to_first: float) -> np.ndarray:
    result = np.zeros(256, dtype=np.float32)
    result[0] = cosine_to_first
    result[1] = np.sqrt(max(0.0, 1.0 - cosine_to_first * cosine_to_first))
    result.setflags(write=False)
    return result


def _config() -> TrackContinuityConfig:
    return TrackContinuityConfig(
        enabled=True,
        max_gap_sec=12.0,
        min_iou=0.10,
        max_center_distance_ratio=0.75,
        min_area_ratio=0.40,
        max_area_ratio=2.50,
        min_match_score=0.55,
        ambiguity_margin=0.15,
        duplicate_iou=0.90,
        duplicate_iou_with_face=0.70,
        duplicate_face_iou=0.50,
        stale_retention_sec=30.0,
        reid_max_gap_sec=12.0,
        reid_match_min=0.85,
        reid_update_min=0.85,
        face_override_max_gap_sec=10.0,
        face_override_min_person_iou=0.50,
        face_override_min_person_containment=0.60,
        face_override_max_center_distance_ratio=0.30,
        face_override_min_area_ratio=0.55,
        face_override_max_area_ratio=1.60,
        face_override_min_face_iou=0.40,
        face_override_min_face_containment=0.55,
        face_override_max_face_center_distance_ratio=0.25,
        face_override_min_face_area_ratio=0.50,
        face_override_max_face_area_ratio=1.80,
        face_override_min_face_score=0.70,
        face_override_min_reid=0.80,
    )


def _resolver() -> FaceAnchoredTrackContinuityResolver:
    return FaceAnchoredTrackContinuityResolver(
        _config(),
        SingleTargetBridgeConfig(enabled=False),
        EdgeBridgeConfig(enabled=False),
    )


def _packet(
    frame: int,
    timestamp: datetime,
    raw_id: int,
    person_bbox: BoundingBox,
    face_bbox: BoundingBox,
    embedding: np.ndarray,
    *,
    face_score: float = 0.84,
) -> FramePacket:
    track = Track(
        "camera-01",
        raw_id,
        timestamp,
        person_bbox,
        0.8,
        metadata={_TRACKER_REID_METADATA_KEY: embedding},
    )
    face = FaceDetection(
        "camera-01",
        raw_id,
        timestamp,
        face_bbox,
        face_score,
    )
    return FramePacket(
        camera_id="camera-01",
        frame_number=frame,
        timestamp=timestamp,
        image=np.zeros((1080, 1920, 4), dtype=np.uint8),
        tracks=(track,),
        faces=(face,),
        behaviors=(),
        stream_time_ns=frame * 33_333_333,
    )


def test_task_1c2_pose_fragments_chain_to_first_business_id() -> None:
    """Reproduce the two observed 1c2 fragment boundaries with real geometry."""

    resolver = _resolver()
    first_embedding = _embedding(1.0)

    # Last trusted raw=1 observation before the first multi-second hole.
    first = resolver.resolve(
        _packet(
            1,
            NOW,
            0,
            BoundingBox(622.8451538, 186.4480438, 1437.6804199, 1055.2543182),
            BoundingBox(1027.2965088, 316.0397949, 1282.4134979, 659.0423889),
            first_embedding,
            face_score=0.83,
        )
    )
    assert first.tracks[0].track_id == 0

    # Task 1c2: body ReID fell to ~0.825, but person IoU stayed ~0.64 and the
    # real SCRFD face still overlapped strongly enough to prove continuity.
    second_at = NOW + timedelta(seconds=4.582295)
    second = resolver.resolve(
        _packet(
            2,
            second_at,
            2,
            BoundingBox(734.6405640, 253.8124847, 1647.1015625, 1061.5468597),
            BoundingBox(954.2686768, 373.1076050, 1219.2296143, 710.4863586),
            _embedding(0.825),
            face_score=0.835,
        )
    )
    assert second.tracks[0].track_id == 0
    assert second.tracks[0].metadata["raw_track_id"] == 2

    # A second fragment occurred ~9.2 s after the latest raw=2 face. The face
    # geometry is almost identical (IoU ~0.92). The face-trust refresh above is
    # what prevents this comparison from falling back to the much older raw=0 face.
    third_at = second_at + timedelta(seconds=9.219415)
    third = resolver.resolve(
        _packet(
            3,
            third_at,
            4,
            BoundingBox(420.6718445, 228.9374847, 1415.7109070, 1077.4140472),
            BoundingBox(963.6832275, 357.6934509, 1224.9965210, 690.5723572),
            _embedding(0.825),
            face_score=0.787,
        )
    )
    assert third.tracks[0].track_id == 0
    assert third.tracks[0].metadata["raw_track_id"] == 4


def test_face_geometry_cannot_override_low_different_person_reid() -> None:
    resolver = _resolver()
    resolver.resolve(
        _packet(
            1,
            NOW,
            0,
            BoundingBox(622.8451538, 186.4480438, 1437.6804199, 1055.2543182),
            BoundingBox(1027.2965088, 316.0397949, 1282.4134979, 659.0423889),
            _embedding(1.0),
        )
    )

    later = NOW + timedelta(seconds=4)
    other = resolver.resolve(
        _packet(
            2,
            later,
            9,
            BoundingBox(734.6405640, 253.8124847, 1647.1015625, 1061.5468597),
            BoundingBox(954.2686768, 373.1076050, 1219.2296143, 710.4863586),
            _embedding(0.759),
        )
    )

    assert other.tracks[0].track_id == 9
