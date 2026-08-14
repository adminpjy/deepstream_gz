from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from deepstream_ai.domain import BoundingBox, FaceDetection, Track
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.weak_track_guard import WeakNewTrackConfig, WeakNewTrackGuard

NOW = datetime(2026, 8, 14, tzinfo=UTC)


def _guard() -> WeakNewTrackGuard:
    return WeakNewTrackGuard(
        WeakNewTrackConfig(
            enabled=True,
            instant_confirm_confidence=0.35,
            sustained_confirm_confidence=0.28,
            confirm_after_sec=0.8,
            suppress_log_after_sec=1.5,
            min_detector_observations=2,
            min_width_ratio=0.025,
            min_height_ratio=0.10,
            stale_retention_sec=30.0,
        )
    )


def _packet(
    timestamp: datetime,
    track: Track,
    *,
    face: FaceDetection | None = None,
    frame: int = 1,
) -> FramePacket:
    return FramePacket(
        camera_id="camera-a",
        frame_number=frame,
        timestamp=timestamp,
        image=np.zeros((720, 1280, 4), dtype=np.uint8),
        tracks=(track,),
        faces=(face,) if face is not None else (),
        behaviors=(),
        stream_time_ns=frame * 33_333_333,
    )


def _track(
    timestamp: datetime,
    confidence: float,
    *,
    detector_confidence: float | None,
    track_id: int | str = 1,
    bbox: BoundingBox | None = None,
) -> Track:
    metadata = {}
    if detector_confidence is not None:
        metadata["detector_confidence"] = detector_confidence
    return Track(
        "camera-a",
        track_id,
        timestamp,
        bbox or BoundingBox(400, 120, 900, 700),
        confidence,
        metadata=metadata,
    )


def test_weak_background_track_never_reaches_business_packet() -> None:
    guard = _guard()
    first = _track(NOW, 0.225, detector_confidence=0.225)
    assert guard.filter(_packet(NOW, first)).tracks == ()
    assert not guard.is_visible("camera-a", 1)

    # Tracker confidence may become much larger between detector frames. It must
    # not promote the false object because no new PeopleNet confidence exists.
    later = NOW + timedelta(seconds=2)
    tracker_only = _track(later, 0.70, detector_confidence=None)
    assert guard.filter(_packet(later, tracker_only, frame=60)).tracks == ()
    assert not guard.is_visible("camera-a", 1)


def test_strong_new_person_is_confirmed_immediately() -> None:
    guard = _guard()
    track = _track(NOW, 0.62, detector_confidence=0.62)

    result = guard.filter(_packet(NOW, track))

    assert result.tracks == (track,)
    assert guard.is_visible("camera-a", 1)


def test_repeated_detector_evidence_can_promote_real_low_confidence_person() -> None:
    guard = _guard()
    first = _track(NOW, 0.29, detector_confidence=0.29)
    assert guard.filter(_packet(NOW, first)).tracks == ()

    later = NOW + timedelta(seconds=1)
    second = _track(later, 0.31, detector_confidence=0.31)
    result = guard.filter(_packet(later, second, frame=30))

    assert result.tracks == (second,)
    assert guard.is_visible("camera-a", 1)


def test_real_scrfd_face_promotes_weak_person_without_raising_detector_threshold() -> None:
    guard = _guard()
    track = _track(NOW, 0.20, detector_confidence=0.20)
    face = FaceDetection(
        camera_id="camera-a",
        track_id=1,
        timestamp=NOW,
        bbox=BoundingBox(520, 150, 680, 330),
        score=0.8,
    )

    result = guard.filter(_packet(NOW, track, face=face))

    assert result.tracks == (track,)
    assert result.faces == (face,)
