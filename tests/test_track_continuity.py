from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from deepstream_ai.domain import BoundingBox, FaceDetection, Track
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.track_continuity import TrackContinuityConfig, TrackContinuityResolver

NOW = datetime(2026, 8, 12, tzinfo=UTC)


def _config() -> TrackContinuityConfig:
    return TrackContinuityConfig(
        enabled=True,
        max_gap_sec=8.0,
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
    )


def _packet(
    frame: int,
    timestamp: datetime,
    tracks: tuple[Track, ...],
    faces: tuple[FaceDetection, ...] = (),
) -> FramePacket:
    image = np.zeros((1080, 1920, 4), dtype=np.uint8)
    return FramePacket(
        camera_id="camera-a",
        frame_number=frame,
        timestamp=timestamp,
        image=image,
        tracks=tracks,
        faces=faces,
        behaviors=(),
        stream_time_ns=frame * 40_000_000,
    )


def _track(raw_id: int, timestamp: datetime, box: BoundingBox, confidence: float = 0.9) -> Track:
    return Track("camera-a", raw_id, timestamp, box, confidence)


def _face(raw_id: int, timestamp: datetime, box: BoundingBox) -> FaceDetection:
    return FaceDetection("camera-a", raw_id, timestamp, box, 0.9)


def test_recent_raw_id_switch_keeps_first_logical_id() -> None:
    resolver = TrackContinuityResolver(_config())
    box = BoundingBox(700, 200, 1100, 950)

    first = resolver.resolve(_packet(1, NOW, (_track(0, NOW, box),)))
    assert first.tracks[0].track_id == 0

    switched_at = NOW + timedelta(seconds=2)
    switched_box = BoundingBox(710, 205, 1110, 955)
    switched = resolver.resolve(
        _packet(
            50,
            switched_at,
            (_track(1, switched_at, switched_box),),
            (_face(1, switched_at, BoundingBox(830, 260, 940, 390)),),
        )
    )

    assert len(switched.tracks) == 1
    assert switched.tracks[0].track_id == 0
    assert switched.tracks[0].metadata["raw_track_id"] == 1
    assert switched.faces[0].track_id == 0
    assert resolver.logical_id("camera-a", 1) == 0


def test_two_distinct_people_in_same_frame_remain_separate() -> None:
    resolver = TrackContinuityResolver(_config())
    left = BoundingBox(100, 150, 450, 950)
    right = BoundingBox(1200, 150, 1550, 950)

    resolved = resolver.resolve(
        _packet(
            1,
            NOW,
            (_track(0, NOW, left), _track(1, NOW, right)),
        )
    )

    assert {track.track_id for track in resolved.tracks} == {0, 1}


def test_nearly_identical_same_frame_duplicate_collapses_to_one_logical_track() -> None:
    resolver = TrackContinuityResolver(_config())
    first_box = BoundingBox(700, 200, 1100, 950)
    resolver.resolve(_packet(1, NOW, (_track(0, NOW, first_box),)))

    timestamp = NOW + timedelta(milliseconds=200)
    duplicate_box = BoundingBox(705, 202, 1105, 952)
    resolved = resolver.resolve(
        _packet(
            2,
            timestamp,
            (
                _track(0, timestamp, first_box, 0.85),
                _track(2, timestamp, duplicate_box, 0.95),
            ),
        )
    )

    assert len(resolved.tracks) == 1
    assert resolved.tracks[0].track_id == 0
    assert resolver.logical_id("camera-a", 2) == 0


def test_overlapping_but_distinct_people_are_not_collapsed_without_face_corroboration() -> None:
    resolver = TrackContinuityResolver(_config())
    first_box = BoundingBox(500, 150, 900, 950)
    resolver.resolve(_packet(1, NOW, (_track(0, NOW, first_box),)))

    timestamp = NOW + timedelta(milliseconds=200)
    second_box = BoundingBox(600, 180, 1000, 950)
    resolved = resolver.resolve(
        _packet(
            2,
            timestamp,
            (_track(0, timestamp, first_box), _track(3, timestamp, second_box)),
        )
    )

    assert len(resolved.tracks) == 2
    assert {track.track_id for track in resolved.tracks} == {0, 3}
