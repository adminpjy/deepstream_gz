from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from deepstream_ai.domain import (
    BehaviorDetection,
    BehaviorType,
    BoundingBox,
    FaceDetection,
    IdentityResult,
    Track,
)
from deepstream_ai.snapshot import EventSnapshotManager, SnapshotConfig, SnapshotKind

NOW = datetime(2026, 8, 10, tzinfo=UTC)


class RecordingEncoder:
    def __init__(self):
        self.images = []

    def encode(self, image, *, quality):
        self.images.append((image.copy(), quality))
        return b"encoded-image"


class RecordingStore:
    def __init__(self):
        self.writes = []

    def write(self, relative_path, payload):
        self.writes.append((Path(relative_path), payload))
        return Path("/snapshots") / relative_path


class FailingStore:
    def write(self, relative_path, payload):
        del relative_path, payload
        raise RuntimeError("simulated snapshot failure")


def track(track_id=1, *, timestamp=NOW) -> Track:
    return Track(
        camera_id="cam/unsafe",
        track_id=track_id,
        timestamp=timestamp,
        bbox=BoundingBox(2, 2, 8, 10),
        confidence=0.9,
    )


def face(track_id=1, *, timestamp=NOW) -> FaceDetection:
    return FaceDetection(
        camera_id="cam/unsafe",
        track_id=track_id,
        timestamp=timestamp,
        bbox=BoundingBox(3, 3, 7, 7),
        score=0.9,
    )


def manager(**config_values):
    encoder = RecordingEncoder()
    store = RecordingStore()
    config_values.setdefault("min_person_crop_width", 1)
    config_values.setdefault("min_person_crop_height", 1)
    config_values.setdefault("min_visible_ratio", 0.0)
    config = SnapshotConfig(**config_values)
    return EventSnapshotManager(config, encoder=encoder, store=store), encoder, store


def test_person_fallback_is_immediate_and_best_person_updates_once() -> None:
    snapshots, encoder, store = manager()
    low = np.full((12, 12, 3), 10, dtype=np.uint8)
    high = np.full((12, 12, 3), 200, dtype=np.uint8)

    assert snapshots.observe_person(low, track(), quality=0.2)
    assert snapshots.observe_person(
        high,
        track(timestamp=NOW + timedelta(milliseconds=40)),
        quality=0.9,
    )
    assert store.writes == []
    state = snapshots.state_for("cam/unsafe", 1)
    assert state is not None
    assert state.person_fallback is not None
    assert state.person_fallback.quality_score == 0.2
    assert state.best_person is not None
    assert state.best_person.quality_score == 0.9
    assert np.all(state.person_fallback.person_crop == 10)
    assert np.all(state.best_person.person_crop == 200)

    record = snapshots.finalize_track("cam/unsafe", 1)
    assert record.kind is SnapshotKind.PERSON
    assert record.quality == 0.9
    assert record.source == "best_person"
    assert store.writes[0][0].parent == Path("person")
    assert "/" not in store.writes[0][0].name
    assert store.writes[0][0].name.startswith("20260810T000000_040000_cam-unsafe-")
    assert store.writes[0][0].name.endswith("_track-1_unknown_sim-1.000_q0.900.jpg")
    assert np.all(encoder.images[0][0] == 200)
    assert snapshots.finalize_track("cam/unsafe", 1) is None


def test_face_is_deferred_and_saves_unannotated_upper_body_at_finalize() -> None:
    snapshots, encoder, store = manager(upper_body_fraction=0.5)
    frame = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    person = track()
    snapshots.observe_person(frame, person, quality=0.8)
    identity = IdentityResult("cam/unsafe", 1, NOW, "worker-7", 0.8, 0.7, 3)

    assert snapshots.observe_face(frame, person, face(), identity)
    assert store.writes == []
    state = snapshots.state_for("cam/unsafe", 1)
    assert state is not None
    assert state.person_fallback is not None
    assert state.face_fallback is not None
    assert state.identity_result is identity

    record = snapshots.finalize_track("cam/unsafe", 1)
    assert record.kind is SnapshotKind.FACE_KNOWN
    assert record.worker_id == "worker-7"
    assert record.source == "face_fallback"
    assert store.writes[0][0].parent == Path("face/know")
    assert store.writes[0][0].name.startswith("20260810T000000_000000_cam-unsafe-")
    assert store.writes[0][0].name.endswith("_track-1_worker-7_sim0.800_q0.900.jpg")
    expected = frame[0:7, 0:10]
    assert np.array_equal(encoder.images[0][0], expected)
    assert state.face_fallback.evidence_bbox.x1 <= face().bbox.x1
    assert state.face_fallback.evidence_bbox.y1 <= face().bbox.y1
    assert state.face_fallback.evidence_bbox.x2 >= face().bbox.x2
    assert state.face_fallback.evidence_bbox.y2 >= face().bbox.y2
    assert not state.face_fallback.full_frame_fallback
    assert not snapshots.observe_face(frame, person, face(), identity)
    assert snapshots.finalize_track("cam/unsafe", 1) is None


def test_unknown_face_uses_required_unknow_directory() -> None:
    snapshots, _, store = manager()
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    assert snapshots.observe_face(frame, track(2), face(2), quality=0.0)
    assert store.writes == []
    record = snapshots.finalize_track("cam/unsafe", 2)
    assert record.kind is SnapshotKind.FACE_UNKNOWN
    assert record.source == "face_fallback"
    assert store.writes[0][0].parent == Path("face/unknow")
    assert store.writes[0][0].name.startswith("20260810T000000_000000_cam-unsafe-")
    assert store.writes[0][0].name.endswith("_track-2_unknown_sim-1.000_q0.000.jpg")


def test_low_quality_face_fallback_survives_and_better_face_updates(
    caplog,
) -> None:
    snapshots, encoder, _ = manager(person_min_quality=1.0)
    first = np.full((12, 12, 3), 20, dtype=np.uint8)
    better = np.full((12, 12, 3), 220, dtype=np.uint8)
    person = track(3)

    with caplog.at_level(logging.INFO):
        assert snapshots.observe_person(first, person, quality=0.0)
        assert snapshots.observe_face(first, person, face(3), quality=0.0)
        later_track = track(3, timestamp=NOW + timedelta(milliseconds=40))
        later_face = face(3, timestamp=NOW + timedelta(milliseconds=40))
        assert snapshots.observe_face(better, later_track, later_face, quality=0.9)

    state = snapshots.state_for("cam/unsafe", 3)
    assert state is not None
    assert state.person_fallback is not None
    assert state.person_fallback.quality_score == 0.0
    assert state.face_fallback is not None
    assert state.face_fallback.quality_score == 0.0
    assert state.best_face is not None
    assert state.best_face.quality_score == 0.9
    assert state.best_face.timestamp == later_face.timestamp
    assert state.best_face.person_bbox == later_track.bbox
    assert state.best_face.face_bbox == later_face.bbox
    assert state.best_face.face_crop is not None
    assert state.best_face.person_crop.size < better.size
    assert state.identity_result is None

    identity = IdentityResult("cam/unsafe", 3, NOW, None, -1.0, 0.0)
    assert snapshots.observe_identity(identity)
    assert state.identity_result is identity
    assert state.best_face.timestamp == later_face.timestamp

    with caplog.at_level(logging.INFO):
        record = snapshots.finalize_track("cam/unsafe", 3)
    assert record.source == "best_face"
    assert record.kind is SnapshotKind.FACE_UNKNOWN
    assert np.all(encoder.images[0][0] == 220)
    assert "[TRACK_CREATE]" in caplog.text
    assert "[FACE_FALLBACK]" in caplog.text
    assert "[BEST_FACE_UPDATE]" in caplog.text
    assert "[TRACK_FINALIZE] camera=cam/unsafe track=3 source=best_face" in caplog.text
    assert "identity=unknown similarity=-1.000 quality=0.900 snapshot=" in caplog.text


def test_finalize_priority_and_all_created_tracks_have_evidence() -> None:
    snapshots, _, store = manager()
    low = np.full((12, 12, 3), 10, dtype=np.uint8)
    high = np.full((12, 12, 3), 200, dtype=np.uint8)

    snapshots.observe_person(low, track(10), quality=0.4)

    snapshots.observe_person(low, track(11), quality=0.2)
    snapshots.observe_person(
        high,
        track(11, timestamp=NOW + timedelta(milliseconds=40)),
        quality=0.8,
    )

    snapshots.observe_person(low, track(12), quality=0.2)
    snapshots.observe_face(low, track(12), face(12), quality=0.1)

    snapshots.observe_person(low, track(13), quality=0.2)
    snapshots.observe_face(low, track(13), face(13), quality=0.1)
    snapshots.observe_face(
        high,
        track(13, timestamp=NOW + timedelta(milliseconds=40)),
        face(13, timestamp=NOW + timedelta(milliseconds=40)),
        quality=0.9,
    )

    records = snapshots.finalize_all()
    assert {record.track_id: record.source for record in records} == {
        10: "person_fallback",
        11: "best_person",
        12: "face_fallback",
        13: "best_face",
    }
    assert snapshots.created_track_count == 4
    assert snapshots.finalized_track_count == 4
    assert snapshots.evidence_missing_count == 0
    assert len(store.writes) == 4
    assert snapshots.finalize_all() == ()
    assert len(store.writes) == 4
    summary = snapshots.log_summary()
    assert summary.person_tracks_created == 4
    assert summary.tracks_finalized == 4
    assert summary.best_face_evidence == 1
    assert summary.face_fallback_evidence == 1
    assert summary.best_person_evidence == 1
    assert summary.person_fallback_evidence == 1
    assert summary.know == 0
    assert summary.unknown == 2
    assert summary.snapshot_success == 4
    assert summary.snapshot_failed == 0
    assert summary.evidence_missing == 0


def test_missing_candidate_emits_evidence_missing(caplog) -> None:
    snapshots, _, _ = manager()
    with caplog.at_level(logging.ERROR):
        assert snapshots.finalize_track("camera-missing", 99) is None
    assert snapshots.evidence_missing_count == 1
    assert "[ERROR][EVIDENCE_MISSING]" in caplog.text
    assert "camera=camera-missing" in caplog.text
    assert "track=99" in caplog.text
    assert "reason=track_state_not_found" in caplog.text
    snapshots.clear_track("camera-missing", 99)
    assert snapshots.evidence_missing_count == 1


def test_behavior_snapshots_are_cooldown_limited() -> None:
    snapshots, _, store = manager(behavior_cooldown_seconds=5)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    def detection(at):
        return BehaviorDetection(
            camera_id="cam/unsafe",
            track_id=1,
            timestamp=at,
            behavior=BehaviorType.DRINKING,
            confidence=0.9,
            bbox=BoundingBox(1, 1, 15, 18),
        )

    first = snapshots.observe_behavior(frame, detection(NOW))
    assert first.kind is SnapshotKind.BEHAVIOR
    assert snapshots.observe_behavior(frame, detection(NOW + timedelta(seconds=2))) is None
    assert snapshots.observe_behavior(frame, detection(NOW + timedelta(seconds=6))) is not None
    assert len(store.writes) == 2
    assert all(path.parent == Path("behavior") for path, _ in store.writes)


def test_expired_face_free_tracks_are_finalized() -> None:
    snapshots, _, _ = manager(track_ttl_seconds=1)
    snapshots.observe_person(np.zeros((12, 12, 3), dtype=np.uint8), track(), quality=0.5)
    records = snapshots.expire_tracks(NOW + timedelta(seconds=2))
    assert len(records) == 1
    assert records[0].kind is SnapshotKind.PERSON


def test_person_quality_luminance_uses_deepstream_rgba_order() -> None:
    snapshots, encoder, _ = manager(sharpness_reference=100_000.0)
    pattern = (np.indices((12, 12)).sum(axis=0) % 2).astype(np.uint8) * 255
    red = np.zeros((12, 12, 4), dtype=np.uint8)
    blue = np.zeros_like(red)
    red[..., 0], red[..., 3] = pattern, 255
    blue[..., 2], blue[..., 3] = pattern, 255

    assert snapshots.observe_person(blue, track())
    assert snapshots.observe_person(
        red,
        track(timestamp=NOW + timedelta(milliseconds=40)),
    )
    snapshots.finalize_track("cam/unsafe", 1)

    assert np.count_nonzero(encoder.images[0][0][..., 0]) > 0
    assert np.count_nonzero(encoder.images[0][0][..., 2]) == 0


@pytest.mark.parametrize(
    "face_bbox",
    [
        BoundingBox(10, 10, 40, 35),
        BoundingBox(60, 10, 90, 35),
        BoundingBox(30, 2, 60, 30),
        BoundingBox(30, 55, 60, 80),
    ],
)
def test_upper_body_crop_expands_to_contain_the_full_face(face_bbox: BoundingBox) -> None:
    snapshots, encoder, _ = manager(
        padding_x_ratio=0.0,
        padding_top_ratio=0.0,
        upper_body_fraction=0.5,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    person = Track("camera-a", 20, NOW, BoundingBox(20, 20, 80, 90), 0.9)
    detected_face = FaceDetection("camera-a", 20, NOW, face_bbox, 0.8)

    snapshots.observe_person(frame, person, quality=0.2)
    assert snapshots.observe_face(frame, person, detected_face, quality=0.5)
    state = snapshots.state_for("camera-a", 20)
    candidate = state.face_fallback
    assert candidate.evidence_bbox.x1 <= face_bbox.x1
    assert candidate.evidence_bbox.y1 <= face_bbox.y1
    assert candidate.evidence_bbox.x2 >= face_bbox.x2
    assert candidate.evidence_bbox.y2 >= face_bbox.y2
    assert not candidate.full_frame_fallback

    snapshots.finalize_track("camera-a", 20)
    assert encoder.images[0][0].size < frame.size


def test_invalid_person_or_face_geometry_uses_full_frame_fallback() -> None:
    snapshots, encoder, _ = manager()
    frame = np.full((100, 100, 3), 77, dtype=np.uint8)
    outside_person = Track("camera-a", 30, NOW, BoundingBox(110, 110, 130, 160), 0.9)

    assert snapshots.observe_person(frame, outside_person)
    state = snapshots.state_for("camera-a", 30)
    assert state.person_fallback.full_frame_fallback
    assert state.person_fallback.evidence_bbox == BoundingBox(0, 0, 100, 100)
    snapshots.finalize_track("camera-a", 30)
    assert np.array_equal(encoder.images[0][0], frame)

    snapshots, encoder, _ = manager()
    person = Track("camera-a", 31, NOW, BoundingBox(20, 20, 80, 90), 0.9)
    outside_face = FaceDetection("camera-a", 31, NOW, BoundingBox(110, 10, 130, 30), 0.8)
    snapshots.observe_person(frame, person)
    assert snapshots.observe_face(frame, person, outside_face)
    state = snapshots.state_for("camera-a", 31)
    assert state.face_fallback.full_frame_fallback
    assert state.face_fallback.face_crop is None
    snapshots.finalize_track("camera-a", 31)
    assert np.array_equal(encoder.images[0][0], frame)


def test_face_and_person_timestamps_must_match() -> None:
    snapshots, _, _ = manager()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    person = Track("camera-a", 40, NOW, BoundingBox(20, 20, 80, 90), 0.9)
    detected_face = FaceDetection(
        "camera-a",
        40,
        NOW + timedelta(milliseconds=1),
        BoundingBox(30, 25, 60, 55),
        0.8,
    )

    snapshots.observe_person(frame, person)
    with pytest.raises(ValueError, match="same frame timestamp"):
        snapshots.observe_face(frame, person, detected_face)


def test_snapshot_write_failure_is_counted_and_never_silent(caplog) -> None:
    snapshots = EventSnapshotManager(
        SnapshotConfig(
            min_person_crop_width=1,
            min_person_crop_height=1,
            min_visible_ratio=0.0,
        ),
        encoder=RecordingEncoder(),
        store=FailingStore(),
    )
    snapshots.observe_person(
        np.zeros((12, 12, 3), dtype=np.uint8),
        track(50),
        quality=0.5,
    )

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(RuntimeError, match="simulated snapshot failure"),
    ):
        snapshots.finalize_track("cam/unsafe", 50)

    summary = snapshots.log_summary()
    assert summary.tracks_finalized == 1
    assert summary.snapshot_success == 0
    assert summary.snapshot_failed == 1
    assert summary.evidence_missing == 1
    assert "reason=snapshot_write_failed" in caplog.text
    assert "summary created=1 tracks_finalized=1 snapshot_success=0" in caplog.text


def test_summary_reports_created_track_that_was_not_finalized(caplog) -> None:
    snapshots, _, _ = manager()
    snapshots.observe_person(
        np.zeros((12, 12, 3), dtype=np.uint8),
        track(51),
        quality=0.5,
    )

    with caplog.at_level(logging.ERROR):
        summary = snapshots.log_summary()

    assert summary.person_tracks_created == 1
    assert summary.tracks_finalized == 0
    assert summary.snapshot_success == 0
    assert summary.evidence_missing == 1
    assert "summary created=1 tracks_finalized=0 snapshot_success=0" in caplog.text


def test_unicode_camera_and_worker_names_remain_distinct_in_filenames() -> None:
    snapshots, _, store = manager()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for camera_id, worker_id in (("车间一", "张三"), ("车间二", "李四")):
        person = Track(camera_id, 60, NOW, BoundingBox(20, 20, 80, 90), 0.9)
        detected_face = FaceDetection(
            camera_id,
            60,
            NOW,
            BoundingBox(30, 25, 60, 55),
            0.8,
        )
        identity = IdentityResult(camera_id, 60, NOW, worker_id, 0.8, 0.8)
        snapshots.observe_face(frame, person, detected_face, identity, quality=0.8)
        snapshots.finalize_track(camera_id, 60)

    names = [path.name for path, _payload in store.writes]
    assert len(names) == len(set(names)) == 2
    assert any("车间一" in name and "张三" in name for name in names)
    assert any("车间二" in name and "李四" in name for name in names)
