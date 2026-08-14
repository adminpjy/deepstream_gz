from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from deepstream_ai.domain import BoundingBox, Track
from deepstream_ai.pipeline.metadata import _TRACKER_REID_METADATA_KEY, FramePacket
from deepstream_ai.track_continuity import TrackContinuityConfig
from deepstream_ai.track_continuity_guard import (
    EdgeBridgeConfig,
    GuardedTrackContinuityResolver,
    SingleTargetBridgeConfig,
)

NOW = datetime(2026, 8, 14, tzinfo=UTC)


def _continuity_config() -> TrackContinuityConfig:
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
    )


def _resolver() -> GuardedTrackContinuityResolver:
    return GuardedTrackContinuityResolver(
        _continuity_config(),
        SingleTargetBridgeConfig(
            enabled=True,
            max_gap_sec=10.0,
            min_iou=0.20,
            min_containment=0.45,
            max_center_distance_ratio=0.40,
            min_area_ratio=0.60,
            max_area_ratio=1.70,
        ),
        EdgeBridgeConfig(
            enabled=True,
            max_gap_sec=3.0,
            edge_margin_ratio=0.10,
            min_parallel_overlap=0.35,
            max_parallel_center_shift_ratio=0.60,
            min_area_ratio=0.15,
            max_area_ratio=4.00,
            ambiguity_margin=0.15,
        ),
    )


def _packet(frame: int, timestamp: datetime, tracks: tuple[Track, ...]) -> FramePacket:
    return FramePacket(
        camera_id="camera-a",
        frame_number=frame,
        timestamp=timestamp,
        image=np.zeros((1080, 1920, 4), dtype=np.uint8),
        tracks=tracks,
        faces=(),
        behaviors=(),
        stream_time_ns=frame * 40_000_000,
    )


def _track(raw_id: int, timestamp: datetime, box: BoundingBox, embedding=None) -> Track:
    metadata = {}
    if embedding is not None:
        metadata[_TRACKER_REID_METADATA_KEY] = np.asarray(embedding, dtype=np.float32)
    return Track("camera-a", raw_id, timestamp, box, 0.9, metadata=metadata)


def test_missing_reid_single_target_pose_change_keeps_first_business_id() -> None:
    resolver = _resolver()
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    first_box = BoundingBox(692.0, 233.0, 1522.0, 1047.0)
    first = resolver.resolve(_packet(1, NOW, (_track(0, NOW, first_box, person_a),)))
    assert first.tracks[0].track_id == 0

    later = NOW + timedelta(seconds=9.05)
    changed_box = BoundingBox(367.0, 328.0, 1018.0, 1078.0)
    resolved = resolver.resolve(_packet(227, later, (_track(1, later, changed_box),)))

    assert resolved.tracks[0].track_id == 0
    assert resolved.tracks[0].metadata["raw_track_id"] == 1
    assert resolver.presentation_track_id("camera-a", 1) == 0


def test_provisional_bridge_is_revoked_when_real_reid_conflicts() -> None:
    resolver = _resolver()
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    first_box = BoundingBox(692.0, 233.0, 1522.0, 1047.0)
    changed_box = BoundingBox(367.0, 328.0, 1018.0, 1078.0)
    resolver.resolve(_packet(1, NOW, (_track(0, NOW, first_box, person_a),)))

    bridged_at = NOW + timedelta(seconds=9.05)
    bridged = resolver.resolve(_packet(227, bridged_at, (_track(1, bridged_at, changed_box),)))
    assert bridged.tracks[0].track_id == 0

    verified_at = bridged_at + timedelta(seconds=0.20)
    rejected = resolver.resolve(
        _packet(232, verified_at, (_track(1, verified_at, changed_box, person_b),))
    )
    assert rejected.tracks[0].track_id == 1


def test_present_conflicting_reid_never_uses_geometry_bridge() -> None:
    resolver = _resolver()
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    box = BoundingBox(600.0, 250.0, 1450.0, 1080.0)
    resolver.resolve(_packet(1, NOW, (_track(0, NOW, box, person_a),)))

    later = NOW + timedelta(seconds=3)
    resolved = resolver.resolve(
        _packet(75, later, (_track(1, later, BoundingBox(610.0, 250.0, 1460.0, 1080.0), person_b),))
    )

    assert resolved.tracks[0].track_id == 1


def test_multi_person_scene_disables_missing_reid_geometry_bridge() -> None:
    resolver = _resolver()
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (_track(0, NOW, BoundingBox(500.0, 200.0, 1200.0, 1080.0), person_a),),
        )
    )

    later = NOW + timedelta(seconds=2)
    resolved = resolver.resolve(
        _packet(
            50,
            later,
            (
                _track(1, later, BoundingBox(520.0, 210.0, 1210.0, 1080.0)),
                _track(2, later, BoundingBox(1250.0, 210.0, 1800.0, 1080.0)),
            ),
        )
    )

    assert {track.track_id for track in resolved.tracks} == {1, 2}


def test_right_edge_partial_fragment_keeps_business_id_without_reid() -> None:
    resolver = _resolver()
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    full_edge = BoundingBox(1450.0, 180.0, 1920.0, 1040.0)
    first = resolver.resolve(_packet(1, NOW, (_track(0, NOW, full_edge, person_a),)))
    assert first.tracks[0].track_id == 0

    later = NOW + timedelta(seconds=1.0)
    partial_edge = BoundingBox(1765.0, 350.0, 1920.0, 900.0)
    resolved = resolver.resolve(_packet(26, later, (_track(1, later, partial_edge),)))

    assert resolved.tracks[0].track_id == 0
    assert resolved.tracks[0].metadata["raw_track_id"] == 1


def test_edge_bridge_is_revoked_by_later_conflicting_reid() -> None:
    resolver = _resolver()
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    full_edge = BoundingBox(1450.0, 180.0, 1920.0, 1040.0)
    partial_edge = BoundingBox(1765.0, 350.0, 1920.0, 900.0)
    resolver.resolve(_packet(1, NOW, (_track(0, NOW, full_edge, person_a),)))

    bridged_at = NOW + timedelta(seconds=1.0)
    bridged = resolver.resolve(_packet(26, bridged_at, (_track(1, bridged_at, partial_edge),)))
    assert bridged.tracks[0].track_id == 0

    verified_at = bridged_at + timedelta(milliseconds=200)
    rejected = resolver.resolve(
        _packet(31, verified_at, (_track(1, verified_at, partial_edge, person_b),))
    )
    assert rejected.tracks[0].track_id == 1


def test_edge_bridge_stops_when_another_person_competes_on_same_edge() -> None:
    resolver = _resolver()
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    old_a = BoundingBox(1500.0, 120.0, 1920.0, 650.0)
    active_b = BoundingBox(1500.0, 650.0, 1920.0, 1080.0)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (
                _track(0, NOW, old_a, person_a),
                _track(2, NOW, active_b, person_b),
            ),
        )
    )

    later = NOW + timedelta(seconds=1.0)
    ambiguous_fragment = BoundingBox(1740.0, 400.0, 1920.0, 850.0)
    resolved = resolver.resolve(
        _packet(
            26,
            later,
            (
                _track(1, later, ambiguous_fragment),
                _track(2, later, active_b, person_b),
            ),
        )
    )

    assert 1 in {track.track_id for track in resolved.tracks}


def test_same_person_raw_id_reuse_after_reconnect_recovers_old_business_id() -> None:
    resolver = _resolver()
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    box = BoundingBox(500.0, 200.0, 1200.0, 1080.0)
    first = resolver.resolve(_packet(1, NOW, (_track(0, NOW, box, person_a),)))
    assert first.tracks[0].track_id == 0

    resolver.begin_stream_generation("camera-a", 1)
    later = NOW + timedelta(seconds=2)
    recovered = resolver.resolve(
        _packet(2, later, (_track(0, later, BoundingBox(510, 205, 1210, 1080), person_a),))
    )

    assert recovered.tracks[0].track_id == 0
    assert recovered.tracks[0].metadata["raw_track_id"] == 0
    assert resolver.presentation_track_id("camera-a", 0) == 0


def test_different_person_reusing_raw_zero_after_reconnect_cannot_overwrite_business_zero() -> None:
    resolver = _resolver()
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    box = BoundingBox(500.0, 200.0, 1200.0, 1080.0)
    resolver.resolve(_packet(1, NOW, (_track(0, NOW, box, person_a),)))

    resolver.begin_stream_generation("camera-a", 1)
    later = NOW + timedelta(seconds=2)
    new_person = resolver.resolve(
        _packet(2, later, (_track(0, later, BoundingBox(1300, 200, 1850, 1080), person_b),))
    )

    assert new_person.tracks[0].track_id == "epoch-1:0"
    assert new_person.tracks[0].metadata["raw_track_id"] == 0
    assert resolver.presentation_track_id("camera-a", 0) == "epoch-1:0"
