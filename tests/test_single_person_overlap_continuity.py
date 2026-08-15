from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from deepstream_ai.domain import BoundingBox, Track
from deepstream_ai.multi_person_continuity import MultiPersonContinuitySafetyConfig
from deepstream_ai.pipeline.metadata import _TRACKER_REID_METADATA_KEY, FramePacket
from deepstream_ai.pose_aware_continuity import PoseFragmentConfig
from deepstream_ai.single_person_overlap_continuity import (
    SinglePersonOverlapConfig,
    SinglePersonOverlapContinuityResolver,
)
from deepstream_ai.track_continuity import TrackContinuityConfig, _LogicalTrackState
from deepstream_ai.track_continuity_guard import EdgeBridgeConfig, SingleTargetBridgeConfig

NOW = datetime(2026, 8, 15, tzinfo=UTC)
CAMERA = "camera-01"


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
        reid_match_min=0.85,
        reid_update_min=0.85,
    )


def _resolver() -> SinglePersonOverlapContinuityResolver:
    return SinglePersonOverlapContinuityResolver(
        _continuity_config(),
        SingleTargetBridgeConfig(enabled=False),
        EdgeBridgeConfig(enabled=False),
        pose_config=PoseFragmentConfig(),
        safety_config=MultiPersonContinuitySafetyConfig(),
        overlap_config=SinglePersonOverlapConfig(),
    )


def _embedding_pair(similarity: float) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros(256, dtype=np.float32)
    right = np.zeros(256, dtype=np.float32)
    left[0] = 1.0
    right[0] = float(similarity)
    right[1] = float(np.sqrt(max(0.0, 1.0 - similarity * similarity)))
    return left, right


def _track(raw_id: int, at: datetime, box: BoundingBox, embedding: np.ndarray) -> Track:
    return Track(
        CAMERA,
        raw_id,
        at,
        box,
        0.95,
        metadata={_TRACKER_REID_METADATA_KEY: embedding},
    )


def _state(
    logical: int,
    raw: int,
    at: datetime,
    box: BoundingBox,
    embedding: np.ndarray,
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


def test_single_person_head_pose_handoff_recovers_logical_id() -> None:
    """Regression for task 58b7...: IoU~0.67, containment~0.80, ReID~0.62."""

    resolver = _resolver()
    old_reid, new_reid = _embedding_pair(0.619)
    old_box = BoundingBox(100, 20, 500, 1020)
    new_box = BoundingBox(180, 20, 580, 1020)

    resolver._states[(CAMERA, 0)] = _state(0, 0, NOW, old_box, old_reid)
    resolver._states[(CAMERA, 1)] = _state(1, 1, NOW, new_box, new_reid)
    resolver._raw_to_logical[(CAMERA, 0)] = 0
    resolver._raw_to_logical[(CAMERA, 1)] = 1
    resolver._self_started[(CAMERA, 1)] = NOW

    at = NOW + timedelta(milliseconds=80)
    resolver._late_rehome(_packet(at, _track(1, at, new_box, new_reid)))

    assert resolver._raw_to_logical[(CAMERA, 1)] == 0
    assert resolver._provisional_raw[(CAMERA, 1)] == 0
    assert resolver._provisional_mode[(CAMERA, 1)] == "pose_fragment"


def test_same_overlap_is_not_rescued_while_two_raw_targets_are_visible() -> None:
    resolver = _resolver()
    old_reid, crossing_reid = _embedding_pair(0.619)
    old_box = BoundingBox(100, 20, 500, 1020)
    crossing_box = BoundingBox(180, 20, 580, 1020)

    resolver._states[(CAMERA, 0)] = _state(0, 0, NOW, old_box, old_reid)
    resolver._states[(CAMERA, 12)] = _state(12, 12, NOW, crossing_box, crossing_reid)
    resolver._raw_to_logical[(CAMERA, 0)] = 0
    resolver._raw_to_logical[(CAMERA, 12)] = 12
    resolver._self_started[(CAMERA, 12)] = NOW

    at = NOW + timedelta(milliseconds=80)
    resolver._late_rehome(
        _packet(
            at,
            _track(0, at, old_box, old_reid),
            _track(12, at, crossing_box, crossing_reid),
        )
    )

    assert resolver._raw_to_logical[(CAMERA, 12)] == 12
    assert (CAMERA, 12) not in resolver._provisional_raw


def test_recent_separated_multi_person_scene_blocks_followup_rescue() -> None:
    resolver = _resolver()
    old_reid, other_reid = _embedding_pair(0.70)
    old_box = BoundingBox(100, 20, 500, 1020)
    separated_box = BoundingBox(420, 20, 820, 1020)
    overlapping_box = BoundingBox(180, 20, 580, 1020)

    resolver._states[(CAMERA, 0)] = _state(0, 0, NOW, old_box, old_reid)
    resolver._states[(CAMERA, 12)] = _state(12, 12, NOW, separated_box, other_reid)
    resolver._raw_to_logical[(CAMERA, 0)] = 0
    resolver._raw_to_logical[(CAMERA, 12)] = 12
    resolver._self_started[(CAMERA, 12)] = NOW

    crowd_at = NOW + timedelta(milliseconds=80)
    resolver._late_rehome(
        _packet(
            crowd_at,
            _track(0, crowd_at, old_box, old_reid),
            _track(12, crowd_at, separated_box, other_reid),
        )
    )
    assert CAMERA in resolver._last_independent_multi_person

    # The crossing person moves into A's old position while A disappears.  The
    # recent crowd observation must prevent a geometry-only handoff to logical 0.
    resolver._states[(CAMERA, 12)].last_bbox = overlapping_box
    resolver._states[(CAMERA, 12)].last_seen = crowd_at
    single_at = crowd_at + timedelta(milliseconds=80)
    resolver._late_rehome(_packet(single_at, _track(12, single_at, overlapping_box, other_reid)))

    assert resolver._raw_to_logical[(CAMERA, 12)] == 12
    assert (CAMERA, 12) not in resolver._provisional_raw


def test_low_body_compatibility_is_not_rescued_even_in_single_person_view() -> None:
    resolver = _resolver()
    old_reid, new_reid = _embedding_pair(0.45)
    old_box = BoundingBox(100, 20, 500, 1020)
    new_box = BoundingBox(180, 20, 580, 1020)

    resolver._states[(CAMERA, 0)] = _state(0, 0, NOW, old_box, old_reid)
    resolver._states[(CAMERA, 1)] = _state(1, 1, NOW, new_box, new_reid)
    resolver._raw_to_logical[(CAMERA, 0)] = 0
    resolver._raw_to_logical[(CAMERA, 1)] = 1
    resolver._self_started[(CAMERA, 1)] = NOW

    at = NOW + timedelta(milliseconds=80)
    resolver._late_rehome(_packet(at, _track(1, at, new_box, new_reid)))

    assert resolver._raw_to_logical[(CAMERA, 1)] == 1
    assert (CAMERA, 1) not in resolver._provisional_raw
