from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from deepstream_ai.domain import BoundingBox, FaceDetection, Track
from deepstream_ai.pipeline.metadata import _TRACKER_REID_METADATA_KEY, FramePacket
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
        face_override_max_gap_sec=3.0,
        face_override_min_person_iou=0.25,
        face_override_min_person_containment=0.50,
        face_override_max_center_distance_ratio=0.33,
        face_override_min_area_ratio=0.60,
        face_override_max_area_ratio=1.45,
        face_override_min_face_iou=0.85,
        face_override_min_face_containment=0.95,
        face_override_max_face_center_distance_ratio=0.05,
        face_override_min_face_area_ratio=0.85,
        face_override_max_face_area_ratio=1.20,
        face_override_min_face_score=0.70,
        face_override_min_reid=0.80,
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


def _reid_track(
    raw_id: int,
    timestamp: datetime,
    box: BoundingBox,
    embedding: np.ndarray,
) -> Track:
    return Track(
        "camera-a",
        raw_id,
        timestamp,
        box,
        0.9,
        metadata={_TRACKER_REID_METADATA_KEY: embedding.astype(np.float32)},
    )


def _measured_7c_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic vectors with the cosine ordering measured from task 7c."""

    same_person = 0.873211
    old_to_other = 0.759243
    new_to_other = 0.755645
    person_a_old = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_a_new = np.zeros(256, dtype=np.float32)
    person_a_new[0] = same_person
    person_a_new[2] = np.sqrt(1.0 - same_person**2)
    person_b = np.zeros(256, dtype=np.float32)
    person_b[0] = old_to_other
    person_b[2] = (new_to_other - same_person * old_to_other) / person_a_new[2]
    person_b[1] = np.sqrt(1.0 - person_b[0] ** 2 - person_b[2] ** 2)
    return person_a_old, person_b, person_a_new


def test_measured_7c_fixture_reproduces_all_three_cosines() -> None:
    person_a_old, person_b, person_a_new = _measured_7c_embeddings()

    assert float(np.dot(person_a_old, person_a_new)) == pytest.approx(0.873211, abs=1e-6)
    assert float(np.dot(person_a_old, person_b)) == pytest.approx(0.759243, abs=1e-6)
    assert float(np.dot(person_a_new, person_b)) == pytest.approx(0.755645, abs=1e-6)


def test_reid_gallery_update_threshold_must_equal_match_threshold(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
track_continuity:
  reid_match_min: 0.85
  reid_update_min: 0.70
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must equal reid_match_min"):
        TrackContinuityConfig.from_file(config)


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


def test_no_vector_hijack_cannot_pollute_trusted_person_or_face_anchor() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a, person_b, person_a_new = _measured_7c_embeddings()
    a_box = BoundingBox(350, 225, 1285, 1080)
    b_box = BoundingBox(1300, 300, 1850, 1080)
    a_face = BoundingBox(850, 310, 1050, 580)
    b_face = BoundingBox(1450, 350, 1630, 590)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (
                _reid_track(0, NOW, a_box, person_a),
                _reid_track(1, NOW, b_box, person_b),
            ),
            (_face(0, NOW, a_face), _face(1, NOW, b_face)),
        )
    )

    no_vector_at = NOW + timedelta(milliseconds=40)
    resolver.resolve(
        _packet(
            2,
            no_vector_at,
            (_track(0, no_vector_at, b_box), _track(1, no_vector_at, b_box)),
            (_face(0, no_vector_at, b_face), _face(1, no_vector_at, b_face)),
        )
    )
    state_a = resolver._states[("camera-a", 0)]
    assert state_a.trusted_bbox == a_box
    assert state_a.trusted_face_bbox == a_face

    sampled_at = NOW + timedelta(milliseconds=80)
    resolver.resolve(
        _packet(
            3,
            sampled_at,
            (
                _reid_track(0, sampled_at, b_box, person_b),
                _reid_track(1, sampled_at, b_box, person_b),
            ),
            (_face(0, sampled_at, b_face), _face(1, sampled_at, b_face)),
        )
    )
    assert state_a.trusted_bbox == a_box
    assert state_a.trusted_face_bbox == a_face

    returned_at = NOW + timedelta(seconds=2)
    returned = resolver.resolve(
        _packet(
            50,
            returned_at,
            (_reid_track(2, returned_at, a_box, person_a_new),),
            (_face(2, returned_at, a_face),),
        )
    )
    assert [track.track_id for track in returned.tracks] == [0]


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


def test_7c_non_overlapping_reappearance_uses_reid_without_changing_raw_id() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    first_box = BoundingBox(1466, 353, 1920, 1080)
    resolver.resolve(_packet(1, NOW, (_reid_track(0, NOW, first_box, person_a),)))

    switched_at = NOW + timedelta(seconds=1.892)
    switched_box = BoundingBox(348, 245, 1288, 1079)
    resolved = resolver.resolve(
        _packet(49, switched_at, (_reid_track(2, switched_at, switched_box, person_a),))
    )

    assert resolved.tracks[0].track_id == 0
    assert resolved.tracks[0].metadata["raw_track_id"] == 2
    assert _TRACKER_REID_METADATA_KEY not in resolved.tracks[0].metadata


def test_distinct_reid_keeps_new_raw_track_separate() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    resolver.resolve(
        _packet(1, NOW, (_reid_track(0, NOW, BoundingBox(1466, 353, 1920, 1080), person_a),))
    )

    later = NOW + timedelta(seconds=1.892)
    resolved = resolver.resolve(
        _packet(49, later, (_reid_track(2, later, BoundingBox(348, 245, 1288, 1079), person_b),))
    )

    assert resolved.tracks[0].track_id == 2


def test_ambiguous_reid_candidates_do_not_merge() -> None:
    resolver = TrackContinuityResolver(_config())
    shared = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (
                _reid_track(10, NOW, BoundingBox(50, 100, 350, 900), shared),
                _reid_track(11, NOW, BoundingBox(1400, 100, 1700, 900), shared),
            ),
        )
    )

    later = NOW + timedelta(seconds=2)
    resolved = resolver.resolve(
        _packet(50, later, (_reid_track(12, later, BoundingBox(700, 100, 1000, 900), shared),))
    )

    assert resolved.tracks[0].track_id == 12


def test_cross_person_corroboration_prevents_reid_gallery_hijack() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    initial = resolver.resolve(
        _packet(
            1,
            NOW,
            (
                _reid_track(0, NOW, BoundingBox(600, 200, 1300, 1080), person_a),
                _reid_track(1, NOW, BoundingBox(1500, 250, 1900, 1080), person_b),
            ),
        )
    )
    assert {track.track_id for track in initial.tracks} == {0, 1}

    # NvDCF raw 0 jumps onto B while raw 1 independently corroborates B.
    hijacked_at = NOW + timedelta(seconds=1)
    held = resolver.resolve(
        _packet(
            25,
            hijacked_at,
            (
                _reid_track(0, hijacked_at, BoundingBox(1300, 250, 1800, 1080), person_b),
                _reid_track(1, hijacked_at, BoundingBox(1100, 250, 1500, 1080), person_b),
            ),
        )
    )
    assert len(held.tracks) == 1
    assert held.tracks[0].track_id == 1
    assert held.tracks[0].metadata["raw_track_id"] in {0, 1}
    assert resolver.presentation_track_id("camera-a", 0) == 1
    assert resolver.presentation_track_id("camera-a", 1) == 1

    # ReID is sampled, not emitted every frame. The hijacked raw ID must stay
    # isolated even on a subsequent frame without embedding/corroboration.
    no_reid_at = hijacked_at + timedelta(milliseconds=40)
    no_reid = resolver.resolve(
        _packet(
            26,
            no_reid_at,
            (
                _track(0, no_reid_at, BoundingBox(1320, 250, 1820, 1080)),
                _track(1, no_reid_at, BoundingBox(1120, 250, 1520, 1080)),
            ),
        )
    )
    assert len(no_reid.tracks) == 1
    assert no_reid.tracks[0].track_id == 1
    assert resolver.presentation_track_id("camera-a", 0) == 1

    returned_at = NOW + timedelta(seconds=3)
    returned = resolver.resolve(
        _packet(
            75,
            returned_at,
            (_reid_track(2, returned_at, BoundingBox(348, 245, 1288, 1079), person_a),),
        )
    )

    assert returned.tracks[0].track_id == 0
    assert returned.tracks[0].metadata["raw_track_id"] == 2
    assert resolver.presentation_track_id("camera-a", 2) == 0


def test_current_corroborator_must_still_match_its_own_gallery() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    person_c = np.eye(1, 256, 2, dtype=np.float32).reshape(-1)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (
                _reid_track(0, NOW, BoundingBox(100, 100, 500, 1000), person_a),
                _reid_track(1, NOW, BoundingBox(1300, 100, 1700, 1000), person_b),
            ),
        )
    )

    later = NOW + timedelta(seconds=1)
    resolved = resolver.resolve(
        _packet(
            25,
            later,
            (
                _reid_track(0, later, BoundingBox(900, 100, 1300, 1000), person_b),
                _reid_track(1, later, BoundingBox(1400, 100, 1800, 1000), person_c),
            ),
        )
    )

    assert {track.metadata["raw_track_id"] for track in resolved.tracks} == {0, 1}


def test_conflicting_reid_blocks_geometry_fallback() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    box = BoundingBox(700, 200, 1100, 950)
    resolver.resolve(_packet(1, NOW, (_reid_track(5, NOW, box, person_a),)))

    later = NOW + timedelta(seconds=2)
    resolved = resolver.resolve(
        _packet(
            50,
            later,
            (_reid_track(6, later, BoundingBox(705, 202, 1105, 952), person_b),),
        )
    )

    assert resolved.tracks[0].track_id == 6


def test_calibrated_7c_similarity_bridges_same_person_but_not_person_b() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a_old, person_b, person_a_new = _measured_7c_embeddings()
    resolver.resolve(
        _packet(
            1,
            NOW,
            (_reid_track(0, NOW, BoundingBox(350, 225, 1285, 1080), person_a_old),),
        )
    )

    b_at = NOW + timedelta(seconds=1)
    distinct = resolver.resolve(
        _packet(
            25,
            b_at,
            (_reid_track(1, b_at, BoundingBox(1450, 350, 1920, 1080), person_b),),
        )
    )
    assert distinct.tracks[0].track_id == 1

    a_returns_at = NOW + timedelta(seconds=2)
    returned = resolver.resolve(
        _packet(
            50,
            a_returns_at,
            (_reid_track(2, a_returns_at, BoundingBox(348, 245, 1288, 1079), person_a_new),),
        )
    )
    assert returned.tracks[0].track_id == 0
    assert returned.tracks[0].metadata["raw_track_id"] == 2


def test_measured_7c_hijack_redirects_b_and_preserves_a_gallery() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a_old, person_b, person_a_new = _measured_7c_embeddings()
    resolver.resolve(
        _packet(
            1,
            NOW,
            (
                _reid_track(0, NOW, BoundingBox(350, 225, 1285, 1080), person_a_old),
                _reid_track(1, NOW, BoundingBox(1450, 350, 1920, 1080), person_b),
            ),
        )
    )

    # Raw 0 is stolen by B. Repeated B vectors must neither appear as logical
    # A nor drift A's gallery toward B; they are redirected to logical B.
    for frame in range(2, 12):
        timestamp = NOW + timedelta(milliseconds=40 * frame)
        held = resolver.resolve(
            _packet(
                frame,
                timestamp,
                (
                    _reid_track(0, timestamp, BoundingBox(1200, 300, 1800, 1080), person_b),
                    _reid_track(1, timestamp, BoundingBox(1300, 300, 1850, 1080), person_b),
                ),
            )
        )
        assert {track.track_id for track in held.tracks} == {1}
        assert resolver.presentation_track_id("camera-a", 0) == 1

    returned_at = NOW + timedelta(seconds=2)
    returned = resolver.resolve(
        _packet(
            50,
            returned_at,
            (_reid_track(2, returned_at, BoundingBox(348, 245, 1288, 1079), person_a_new),),
        )
    )

    assert returned.tracks[0].track_id == 0
    assert returned.tracks[0].metadata["raw_track_id"] == 2


def test_stale_redirect_owner_cannot_delete_restored_raw_mapping() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a_old, person_b, person_a_new = _measured_7c_embeddings()
    person_x = np.eye(1, 256, 3, dtype=np.float32).reshape(-1)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (
                _reid_track(0, NOW, BoundingBox(350, 225, 1285, 1080), person_a_old),
                _reid_track(1, NOW, BoundingBox(1450, 350, 1920, 1080), person_b),
            ),
        )
    )
    # A is reacquired as raw 2 and linked to logical 0.
    resolver.resolve(
        _packet(
            25,
            NOW + timedelta(seconds=1),
            (
                _reid_track(
                    2,
                    NOW + timedelta(seconds=1),
                    BoundingBox(348, 245, 1288, 1079),
                    person_a_new,
                ),
                _reid_track(
                    1,
                    NOW + timedelta(seconds=1),
                    BoundingBox(1450, 350, 1920, 1080),
                    person_b,
                ),
            ),
        )
    )
    # Raw 2 is temporarily stolen by B, then produces an unrelated vector so
    # it has no current assignment before returning to A.
    resolver.resolve(
        _packet(
            50,
            NOW + timedelta(seconds=2),
            (
                _reid_track(
                    2,
                    NOW + timedelta(seconds=2),
                    BoundingBox(1200, 300, 1800, 1080),
                    person_b,
                ),
                _reid_track(
                    1,
                    NOW + timedelta(seconds=2),
                    BoundingBox(1300, 300, 1850, 1080),
                    person_b,
                ),
            ),
        )
    )
    resolver.resolve(
        _packet(
            51,
            NOW + timedelta(seconds=3),
            (
                _reid_track(
                    2,
                    NOW + timedelta(seconds=3),
                    BoundingBox(1200, 300, 1800, 1080),
                    person_x,
                ),
                _reid_track(
                    1,
                    NOW + timedelta(seconds=3),
                    BoundingBox(1300, 300, 1850, 1080),
                    person_b,
                ),
            ),
        )
    )
    restored = resolver.resolve(
        _packet(
            75,
            NOW + timedelta(seconds=4),
            (
                _reid_track(
                    2,
                    NOW + timedelta(seconds=4),
                    BoundingBox(348, 245, 1288, 1079),
                    person_a_new,
                ),
            ),
        )
    )
    assert restored.tracks[0].track_id == 0

    # Keep A fresh until logical B becomes stale. Purging B must not remove
    # the raw2->logical0 mapping now owned by A.
    resolver.resolve(
        _packet(
            500,
            NOW + timedelta(seconds=20),
            (
                _reid_track(
                    2,
                    NOW + timedelta(seconds=20),
                    BoundingBox(348, 245, 1288, 1079),
                    person_a_new,
                ),
            ),
        )
    )
    after_purge = resolver.resolve(
        _packet(
            875,
            NOW + timedelta(seconds=35),
            (
                _reid_track(
                    2,
                    NOW + timedelta(seconds=35),
                    BoundingBox(348, 245, 1288, 1079),
                    person_a_new,
                ),
            ),
        )
    )
    assert after_purge.tracks[0].track_id == 0
    assert resolver.presentation_track_id("camera-a", 2) == 0


def test_duplicate_geometry_does_not_override_conflicting_reid() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    box = BoundingBox(700, 200, 1100, 950)
    resolver.resolve(_packet(1, NOW, (_reid_track(0, NOW, box, person_a),)))

    later = NOW + timedelta(milliseconds=40)
    resolved = resolver.resolve(
        _packet(
            2,
            later,
            (
                _reid_track(0, later, box, person_a),
                _reid_track(2, later, BoundingBox(702, 201, 1102, 951), person_b),
            ),
        )
    )

    assert {track.track_id for track in resolved.tracks} == {0, 2}


def test_quarantine_keeps_original_owner_across_second_identity_jump() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    person_c = np.eye(1, 256, 2, dtype=np.float32).reshape(-1)
    a_box = BoundingBox(100, 100, 500, 1000)
    b_box = BoundingBox(700, 100, 1100, 1000)
    c_box = BoundingBox(1300, 100, 1700, 1000)

    resolver.resolve(
        _packet(
            1,
            NOW,
            (
                _reid_track(0, NOW, a_box, person_a),
                _reid_track(1, NOW, b_box, person_b),
                _reid_track(2, NOW, c_box, person_c),
            ),
        )
    )

    # Raw 0 first jumps from A to B and is redirected to B's logical track.
    first_jump_at = NOW + timedelta(seconds=1)
    first_jump = resolver.resolve(
        _packet(
            25,
            first_jump_at,
            (
                _reid_track(0, first_jump_at, b_box, person_b),
                _reid_track(1, first_jump_at, b_box, person_b),
                _reid_track(2, first_jump_at, c_box, person_c),
            ),
        )
    )
    assert {track.track_id for track in first_jump.tracks} == {1, 2}

    # It then jumps to C.  This must not replace the original A recovery
    # anchor; the conflicting raw is suppressed while C's trusted raw remains.
    second_jump_at = NOW + timedelta(seconds=2)
    second_jump = resolver.resolve(
        _packet(
            50,
            second_jump_at,
            (
                _reid_track(0, second_jump_at, c_box, person_c),
                _reid_track(2, second_jump_at, c_box, person_c),
            ),
        )
    )
    assert [track.track_id for track in second_jump.tracks] == [2]
    assert resolver.presentation_track_id("camera-a", 0) is None

    # When raw 0 sees A again it returns to logical A, not B or C.
    restored_at = NOW + timedelta(seconds=3)
    restored = resolver.resolve(
        _packet(
            75,
            restored_at,
            (_reid_track(0, restored_at, a_box, person_a),),
        )
    )
    assert [track.track_id for track in restored.tracks] == [0]
    assert resolver.presentation_track_id("camera-a", 0) == 0


def test_4e_face_overlap_overrides_borderline_reid_without_gallery_drift() -> None:
    resolver = TrackContinuityResolver(_config())
    gallery = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    incoming = np.zeros(256, dtype=np.float32)
    incoming[0] = 0.82
    incoming[1] = np.sqrt(1.0 - incoming[0] ** 2)
    old_person = BoundingBox(977.7, 297.9, 1522.3, 1059.7)
    new_person = BoundingBox(936.2, 310.4, 1556.3, 1056.3)
    old_face = BoundingBox(1005.89, 481.61, 1282.42, 780.93)
    new_face = BoundingBox(1006.71, 483.89, 1284.14, 774.52)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (_reid_track(0, NOW, old_person, gallery),),
            (_face(0, NOW, old_face),),
        )
    )

    later = NOW + timedelta(seconds=2.702)
    resolved = resolver.resolve(
        _packet(
            69,
            later,
            (_reid_track(1, later, new_person, incoming),),
            (_face(1, later, new_face),),
        )
    )

    assert [track.track_id for track in resolved.tracks] == [0]
    assert resolved.tracks[0].metadata["raw_track_id"] == 1
    state = resolver._states[("camera-a", 0)]
    assert np.dot(state.reid_embedding, gallery) == pytest.approx(1.0)


def test_face_overlap_override_rejects_below_floor_reid() -> None:
    resolver = TrackContinuityResolver(_config())
    gallery = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    incoming = np.zeros(256, dtype=np.float32)
    incoming[0] = 0.79
    incoming[1] = np.sqrt(1.0 - incoming[0] ** 2)
    old_person = BoundingBox(977.7, 297.9, 1522.3, 1059.7)
    new_person = BoundingBox(936.2, 310.4, 1556.3, 1056.3)
    old_face = BoundingBox(1005.89, 481.61, 1282.42, 780.93)
    new_face = BoundingBox(1006.71, 483.89, 1284.14, 774.52)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (_reid_track(0, NOW, old_person, gallery),),
            (_face(0, NOW, old_face),),
        )
    )

    later = NOW + timedelta(seconds=2.702)
    resolved = resolver.resolve(
        _packet(
            69,
            later,
            (_reid_track(1, later, new_person, incoming),),
            (_face(1, later, new_face),),
        )
    )

    assert [track.track_id for track in resolved.tracks] == [1]


def test_7c_distinct_faces_do_not_trigger_override() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a, person_b, _ = _measured_7c_embeddings()
    a_person = BoundingBox(352.6, 226.5, 1284.2, 1080.0)
    b_person = BoundingBox(972.6, 354.2, 1771.6, 1074.7)
    a_face = BoundingBox(848.5, 308.4, 1047.3, 581.2)
    b_face = BoundingBox(1769.6, 330.9, 1912.8, 519.7)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (_reid_track(0, NOW, a_person, person_a),),
            (_face(0, NOW, a_face),),
        )
    )

    later = NOW + timedelta(seconds=0.64)
    resolved = resolver.resolve(
        _packet(
            17,
            later,
            (_reid_track(1, later, b_person, person_b),),
            (_face(1, later, b_face),),
        )
    )

    assert [track.track_id for track in resolved.tracks] == [1]


def test_highly_overlapping_face_collapses_partial_person_duplicate() -> None:
    resolver = TrackContinuityResolver(_config())
    first_person = BoundingBox(487.6, 290.5, 1111.3, 1067.3)
    duplicate_person = BoundingBox(459.3, 284.2, 949.7, 1072.5)
    shared_face = BoundingBox(610.0, 370.0, 850.0, 670.0)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (_track(8, NOW, first_person),),
            (_face(8, NOW, shared_face),),
        )
    )
    later = NOW + timedelta(milliseconds=40)
    resolved = resolver.resolve(
        _packet(
            2,
            later,
            (
                _track(8, later, first_person, 0.95),
                _track(9, later, duplicate_person, 0.90),
            ),
            (
                _face(8, later, shared_face),
                _face(9, later, BoundingBox(611, 371, 851, 671)),
            ),
        )
    )

    assert [track.track_id for track in resolved.tracks] == [8]
    assert resolver.logical_id("camera-a", 9) == 8


def test_unrelated_reid_conflict_does_not_block_best_geometry_candidate() -> None:
    resolver = TrackContinuityResolver(_config())
    person_b = np.eye(1, 256, 1, dtype=np.float32).reshape(-1)
    near = BoundingBox(700, 200, 1100, 950)
    far = BoundingBox(50, 100, 350, 900)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (
                _track(0, NOW, near),
                _reid_track(1, NOW, far, person_b),
            ),
        )
    )

    later = NOW + timedelta(seconds=2)
    incoming = np.eye(1, 256, 2, dtype=np.float32).reshape(-1)
    resolved = resolver.resolve(
        _packet(
            50,
            later,
            (_reid_track(2, later, BoundingBox(705, 202, 1105, 952), incoming),),
        )
    )

    assert [track.track_id for track in resolved.tracks] == [0]


def test_missing_incoming_reid_cannot_geometry_merge_into_stable_gallery() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    box = BoundingBox(700, 200, 1100, 950)
    resolver.resolve(_packet(1, NOW, (_reid_track(0, NOW, box, person_a),)))

    later = NOW + timedelta(seconds=1)
    resolved = resolver.resolve(
        _packet(25, later, (_track(1, later, BoundingBox(705, 202, 1105, 952)),))
    )

    assert [track.track_id for track in resolved.tracks] == [1]


def test_measured_7c_other_person_cannot_override_even_at_same_face_location() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a, person_b, _ = _measured_7c_embeddings()
    person_box = BoundingBox(500, 200, 1200, 1050)
    face_box = BoundingBox(800, 300, 1050, 620)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (_reid_track(0, NOW, person_box, person_a),),
            (_face(0, NOW, face_box),),
        )
    )

    later = NOW + timedelta(seconds=1)
    resolved = resolver.resolve(
        _packet(
            25,
            later,
            (_reid_track(1, later, person_box, person_b),),
            (_face(1, later, face_box),),
        )
    )

    assert [track.track_id for track in resolved.tracks] == [1]


def test_active_duplicate_uses_stable_gallery_when_current_frame_has_no_old_vector() -> None:
    resolver = TrackContinuityResolver(_config())
    person = np.eye(1, 256, 0, dtype=np.float32).reshape(-1)
    first_box = BoundingBox(487.6, 290.5, 1111.3, 1067.3)
    resolver.resolve(_packet(1, NOW, (_reid_track(8, NOW, first_box, person),)))

    later = NOW + timedelta(milliseconds=40)
    duplicate_box = BoundingBox(459.3, 284.2, 949.7, 1072.5)
    resolved = resolver.resolve(
        _packet(
            2,
            later,
            (
                _track(8, later, first_box),
                _reid_track(9, later, duplicate_box, person),
            ),
        )
    )

    assert [track.track_id for track in resolved.tracks] == [8]
    assert resolver.logical_id("camera-a", 9) == 8


def test_active_duplicate_rejects_measured_7c_other_person_vector() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a, person_b, _ = _measured_7c_embeddings()
    a_box = BoundingBox(972.57806, 354.20309, 1771.5625, 1074.71091)
    b_box = BoundingBox(1133.40198, 310.12701, 1696.31604, 881.57037)
    resolver.resolve(_packet(1, NOW, (_reid_track(0, NOW, a_box, person_a),)))

    later = NOW + timedelta(milliseconds=136)
    resolved = resolver.resolve(
        _packet(
            4,
            later,
            (_track(0, later, a_box), _reid_track(1, later, b_box, person_b)),
        )
    )

    assert {track.track_id for track in resolved.tracks} == {0, 1}


def test_face_duplicate_uses_stable_gallery_to_reject_measured_7c_other_person() -> None:
    resolver = TrackContinuityResolver(_config())
    person_a, person_b, _ = _measured_7c_embeddings()
    person_box = BoundingBox(500, 200, 1200, 1050)
    face_box = BoundingBox(800, 300, 1050, 620)
    resolver.resolve(
        _packet(
            1,
            NOW,
            (_reid_track(0, NOW, person_box, person_a),),
            (_face(0, NOW, face_box),),
        )
    )

    later = NOW + timedelta(milliseconds=40)
    resolved = resolver.resolve(
        _packet(
            2,
            later,
            (
                _track(0, later, person_box),
                _reid_track(1, later, person_box, person_b),
            ),
            (_face(0, later, face_box), _face(1, later, face_box)),
        )
    )

    assert {track.track_id for track in resolved.tracks} == {0, 1}
