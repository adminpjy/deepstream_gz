from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from deepstream_ai.domain import BoundingBox, FaceDetection, Track
from deepstream_ai.face.continuity_anchor import FaceContinuityAnchorRegistry, cosine_similarity
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.provisional_track_guard import BUSINESS_PROVISIONAL_KEY, ProvisionalTrackConfig, ProvisionalTrackGuard

NOW = datetime(2026, 8, 14, tzinfo=UTC)


def _packet(timestamp: datetime, *, with_face: bool = True) -> FramePacket:
    track = Track(
        camera_id="camera-a",
        track_id=7,
        timestamp=timestamp,
        bbox=BoundingBox(100, 100, 500, 900),
        confidence=0.95,
        metadata={"detector_confidence": 0.95, "raw_track_id": 7},
    )
    faces = (
        FaceDetection(
            camera_id="camera-a",
            track_id=7,
            timestamp=timestamp,
            bbox=BoundingBox(210, 150, 330, 300),
            score=0.90,
            landmarks=((220, 190), (300, 190), (260, 220), (230, 260), (290, 260)),
        ),
    ) if with_face else ()
    return FramePacket(
        camera_id="camera-a",
        frame_number=1,
        timestamp=timestamp,
        image=np.zeros((1080, 1920, 4), dtype=np.uint8),
        tracks=(track,),
        faces=faces,
        behaviors=(),
    )


def test_strong_new_track_is_analyzable_before_business_confirmation() -> None:
    guard = ProvisionalTrackGuard(
        ProvisionalTrackConfig(min_confirm_age_sec=0.6, confirm_after_sec=0.8)
    )
    analysis, visible = guard.partition(_packet(NOW))

    assert len(analysis.tracks) == 1
    assert analysis.tracks[0].metadata[BUSINESS_PROVISIONAL_KEY] is True
    assert visible.tracks == ()
    assert visible.faces == ()

    analysis, visible = guard.partition(_packet(NOW + timedelta(seconds=0.7)))
    assert analysis.tracks[0].metadata[BUSINESS_PROVISIONAL_KEY] is False
    assert len(visible.tracks) == 1
    assert len(visible.faces) == 1


def test_continuity_alias_skips_the_new_business_hold() -> None:
    guard = ProvisionalTrackGuard(ProvisionalTrackConfig())
    packet = _packet(NOW)
    aliased_track = Track(
        camera_id="camera-a",
        track_id=3,
        timestamp=NOW,
        bbox=packet.tracks[0].bbox,
        confidence=0.7,
        metadata={"raw_track_id": 7},
    )
    aliased_face = FaceDetection(
        camera_id="camera-a",
        track_id=3,
        timestamp=NOW,
        bbox=packet.faces[0].bbox,
        score=0.9,
        landmarks=packet.faces[0].landmarks,
    )
    packet = FramePacket(
        camera_id="camera-a",
        frame_number=1,
        timestamp=NOW,
        image=packet.image,
        tracks=(aliased_track,),
        faces=(aliased_face,),
        behaviors=(),
    )

    analysis, visible = guard.partition(packet)
    assert analysis.tracks[0].metadata[BUSINESS_PROVISIONAL_KEY] is False
    assert len(visible.tracks) == 1


def test_unknown_adaface_anchors_can_match_without_database_identity() -> None:
    registry = FaceContinuityAnchorRegistry(retention_sec=30)
    first = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    second = np.array([0.98, 0.15, 0.0], dtype=np.float32)
    a = registry.observe("camera-a", 1, first, timestamp=NOW, quality=0.8)
    b = registry.observe(
        "camera-a", 9, second, timestamp=NOW + timedelta(seconds=1), quality=0.75
    )

    assert cosine_similarity(a, b) > 0.98
    registry.alias("camera-a", 9, 1, now=NOW + timedelta(seconds=1))
    latest = registry.latest(
        "camera-a",
        1,
        now=NOW + timedelta(seconds=2),
        max_age_sec=10,
        min_quality=0.7,
    )
    assert latest is not None
    assert latest.timestamp == b.timestamp
