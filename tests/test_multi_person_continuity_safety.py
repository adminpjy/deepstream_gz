from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from deepstream_ai.domain import BoundingBox, Track
from deepstream_ai.face.continuity_anchor import FACE_CONTINUITY_ANCHORS
from deepstream_ai.multi_person_continuity import (
    MultiPersonContinuitySafetyConfig,
    MultiPersonSafePoseAwareTrackContinuityResolver,
)
from deepstream_ai.pipeline.metadata import _TRACKER_REID_METADATA_KEY, FramePacket
from deepstream_ai.pose_aware_continuity import PoseFragmentConfig
from deepstream_ai.track_continuity import TrackContinuityConfig, _LogicalTrackState
from deepstream_ai.track_continuity_guard import EdgeBridgeConfig, SingleTargetBridgeConfig

NOW = datetime(2026, 8, 14, tzinfo=UTC)
CAMERA = "camera-04"


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
    )


def _resolver() -> MultiPersonSafePoseAwareTrackContinuityResolver:
    return MultiPersonSafePoseAwareTrackContinuityResolver(
        _config(),
        SingleTargetBridgeConfig(enabled=False),
        EdgeBridgeConfig(enabled=False),
        pose_config=PoseFragmentConfig(),
        safety_config=MultiPersonContinuitySafetyConfig(
            face_anchor_match_min=0.85,
            face_anchor_conflict_max=0.55,
        ),
    )


def _embedding(index: int) -> np.ndarray:
    value = np.zeros(256, dtype=np.float32)
    value[index] = 1.0
    return value


def _track(
    raw_id: int,
    at: datetime,
    box: BoundingBox,
    *,
    embedding: np.ndarray | None = None,
) -> Track:
    metadata = {}
    if embedding is not None:
        metadata[_TRACKER_REID_METADATA_KEY] = embedding
    return Track(CAMERA, raw_id, at, box, 0.9, metadata=metadata)


def _packet(at: datetime, *tracks: Track) -> FramePacket:
    return FramePacket(
        camera_id=CAMERA,
        frame_number=1,
        timestamp=at,
        image=np.zeros((1080, 1920, 4), dtype=np.uint8),
        tracks=tuple(tracks),
        faces=(),
        behaviors=(),
    )


def _state(
    logical: int,
    raw: int,
    at: datetime,
    box: BoundingBox,
    *,
    embedding: np.ndarray | None = None,
) -> _LogicalTrackState:
    return _LogicalTrackState(
        camera_id=CAMERA,
        logical_id=logical,
        current_raw_id=raw,
        last_bbox=box,
        last_seen=at,
        last_frame_number=1,
        raw_ids={raw},
        reid_embedding=embedding,
        trusted_bbox=box,
        trusted_seen=at,
    )


def test_geometry_cannot_steal_an_occupied_logical_id_in_multi_person_scene() -> None:
    resolver = _resolver()
    person_a = _embedding(0)
    a_box = BoundingBox(0, 537, 306, 1074)
    crossing_box = BoundingBox(15, 18, 403, 1041)

    resolver._states[(CAMERA, 0)] = _state(0, 0, NOW, a_box, embedding=person_a)
    resolver._states[(CAMERA, 12)] = _state(12, 12, NOW, crossing_box)
    resolver._raw_to_logical[(CAMERA, 0)] = 0
    resolver._raw_to_logical[(CAMERA, 12)] = 12
    resolver._self_started[(CAMERA, 12)] = NOW

    at = NOW + timedelta(milliseconds=100)
    resolver._late_rehome(
        _packet(
            at,
            _track(0, at, a_box, embedding=person_a),
            _track(12, at, crossing_box),
        )
    )

    assert resolver._raw_to_logical[(CAMERA, 12)] == 12
    assert (CAMERA, 12) not in resolver._provisional_raw


def test_identity_conflict_lock_prevents_missing_reid_remerge() -> None:
    resolver = _resolver()
    person_a = _embedding(0)
    person_b = _embedding(1)
    old_box = BoundingBox(20, 20, 420, 1040)
    fragment_box = BoundingBox(25, 25, 415, 1045)

    resolver._states[(CAMERA, 0)] = _state(0, 0, NOW, old_box, embedding=person_a)
    resolver._states[(CAMERA, 12)] = _state(12, 12, NOW, fragment_box)
    resolver._raw_to_logical[(CAMERA, 0)] = 0
    resolver._raw_to_logical[(CAMERA, 12)] = 12
    resolver._self_started[(CAMERA, 12)] = NOW

    alias_at = NOW + timedelta(milliseconds=100)
    resolver._late_rehome(_packet(alias_at, _track(12, alias_at, fragment_box)))
    assert resolver._raw_to_logical[(CAMERA, 12)] == 0
    assert resolver._provisional_raw[(CAMERA, 12)] == 0

    conflict_at = alias_at + timedelta(milliseconds=100)
    resolver._verify_provisional_assignments(
        _packet(conflict_at, _track(12, conflict_at, fragment_box, embedding=person_b))
    )
    assert (CAMERA, 12, 0) in resolver._identity_conflict_locks
    assert (CAMERA, 12) not in resolver._raw_to_logical

    # Simulate the base resolver recreating raw 12 as its own logical track on
    # the next sampled frame. Missing ReID must not reopen the rejected alias.
    resolver._states[(CAMERA, 12)] = _state(12, 12, conflict_at, fragment_box)
    resolver._raw_to_logical[(CAMERA, 12)] = 12
    resolver._self_started[(CAMERA, 12)] = conflict_at
    missing_at = conflict_at + timedelta(milliseconds=40)
    resolver._late_rehome(_packet(missing_at, _track(12, missing_at, fragment_box)))

    assert resolver._raw_to_logical[(CAMERA, 12)] == 12
    assert (CAMERA, 12) not in resolver._provisional_raw


def test_adaface_provisional_does_not_self_confirm_or_copy_anchor_early() -> None:
    resolver = _resolver()
    FACE_CONTINUITY_ANCHORS.clear_camera(CAMERA)
    try:
        box = BoundingBox(500, 100, 1100, 1000)
        old_face = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        first_new_face = np.array([0.97, 0.24, 0.0], dtype=np.float32)
        later_new_face = np.array([0.98, 0.20, 0.0], dtype=np.float32)

        resolver._states[(CAMERA, 0)] = _state(0, 0, NOW, box)
        resolver._states[(CAMERA, 13)] = _state(13, 13, NOW, box)
        resolver._raw_to_logical[(CAMERA, 0)] = 0
        resolver._raw_to_logical[(CAMERA, 13)] = 13
        resolver._self_started[(CAMERA, 13)] = NOW
        FACE_CONTINUITY_ANCHORS.observe(
            CAMERA, 0, old_face, timestamp=NOW, quality=0.9
        )

        alias_at = NOW + timedelta(milliseconds=100)
        FACE_CONTINUITY_ANCHORS.observe(
            CAMERA, 13, first_new_face, timestamp=alias_at, quality=0.9
        )
        resolver._late_rehome(_packet(alias_at, _track(13, alias_at, box)))
        assert resolver._provisional_mode[(CAMERA, 13)] == "adaface"

        logical_anchor = FACE_CONTINUITY_ANCHORS.latest(
            CAMERA, 0, now=alias_at, max_age_sec=12, min_quality=0.0
        )
        assert logical_anchor is not None
        assert logical_anchor.timestamp == NOW

        # Same initial raw anchor may not immediately prove its own alias.
        resolver._verify_provisional_assignments(
            _packet(alias_at, _track(13, alias_at, box))
        )
        assert resolver._provisional_raw[(CAMERA, 13)] == 0
        logical_anchor = FACE_CONTINUITY_ANCHORS.latest(
            CAMERA, 0, now=alias_at, max_age_sec=12, min_quality=0.0
        )
        assert logical_anchor is not None
        assert logical_anchor.timestamp == NOW

        confirm_at = alias_at + timedelta(milliseconds=200)
        FACE_CONTINUITY_ANCHORS.observe(
            CAMERA, 13, later_new_face, timestamp=confirm_at, quality=0.92
        )
        resolver._verify_provisional_assignments(
            _packet(confirm_at, _track(13, confirm_at, box))
        )
        assert (CAMERA, 13) not in resolver._provisional_raw
        logical_anchor = FACE_CONTINUITY_ANCHORS.latest(
            CAMERA, 0, now=confirm_at, max_age_sec=12, min_quality=0.0
        )
        assert logical_anchor is not None
        assert logical_anchor.timestamp == confirm_at
    finally:
        FACE_CONTINUITY_ANCHORS.clear_camera(CAMERA)
