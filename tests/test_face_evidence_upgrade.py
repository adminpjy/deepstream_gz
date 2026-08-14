from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from deepstream_ai.domain import BoundingBox, FaceDetection, Track
from deepstream_ai.snapshot.alarm_manager import EventSnapshotManager
from deepstream_ai.snapshot.manager import SnapshotConfig

NOW = datetime(2026, 8, 14, tzinfo=UTC)


class _Encoder:
    def encode(self, image, *, quality):
        del quality
        return np.asarray(image).tobytes()


class _Store:
    def __init__(self) -> None:
        self.writes = []

    def write(self, relative_path, payload):
        self.writes.append((relative_path, payload))
        return relative_path


def _manager() -> EventSnapshotManager:
    return EventSnapshotManager(
        SnapshotConfig(
            min_person_crop_width=1,
            min_person_crop_height=1,
            min_visible_ratio=0.0,
            frame_color_space="rgba",
        ),
        encoder=_Encoder(),
        store=_Store(),
    )


def _track(at: datetime) -> Track:
    return Track(
        camera_id="camera-a",
        track_id=1,
        timestamp=at,
        bbox=BoundingBox(20, 20, 280, 295),
        confidence=0.95,
    )


def _face(
    at: datetime,
    *,
    score: float,
    blur: float,
    frontal: float,
) -> FaceDetection:
    bbox = BoundingBox(90, 45, 210, 165)
    return FaceDetection(
        camera_id="camera-a",
        track_id=1,
        timestamp=at,
        bbox=bbox,
        score=score,
        landmarks=(
            (122.0, 88.0),
            (178.0, 88.0),
            (150.0, 112.0),
            (130.0, 140.0),
            (170.0, 140.0),
        ),
        metadata={"blur_score": blur, "frontal_score": frontal},
    )


def test_real_scrfd_face_uses_unified_scorer_not_legacy_quality_hint() -> None:
    manager = _manager()
    frame = np.zeros((300, 300, 4), dtype=np.uint8)
    early = _face(NOW, score=0.90, blur=0.70, frontal=0.55)
    later = _face(
        NOW + timedelta(milliseconds=200),
        score=0.85,
        blur=0.85,
        frontal=0.95,
    )

    # A legacy caller hint of 1.0 must not lock in the early five-landmark face.
    assert manager.observe_face(frame, _track(NOW), early, quality=1.0)
    assert manager.observe_face(
        frame,
        _track(later.timestamp),
        later,
        quality=0.0,
    )

    state = manager.state_for("camera-a", 1)
    assert state is not None
    assert state.face_fallback is not None
    assert state.best_face is not None
    assert state.best_face.timestamp == later.timestamp
    assert state.best_face.quality_score > state.face_fallback.quality_score


def test_two_stable_frontal_faces_can_replace_slightly_higher_early_score() -> None:
    manager = _manager()
    frame = np.zeros((300, 300, 4), dtype=np.uint8)
    # The early face has a slightly higher total score but is not stable-quality
    # because blur is below the stable threshold. Two subsequent frontal faces
    # should therefore be allowed to replace it without lowering the global
    # identity/ReID thresholds.
    early = _face(NOW, score=0.95, blur=0.44, frontal=0.86)
    first_clear = _face(
        NOW + timedelta(milliseconds=200),
        score=0.70,
        blur=0.70,
        frontal=0.90,
    )
    second_clear = _face(
        NOW + timedelta(milliseconds=400),
        score=0.70,
        blur=0.70,
        frontal=0.90,
    )

    assert manager.observe_face(frame, _track(NOW), early)
    assert not manager.observe_face(frame, _track(first_clear.timestamp), first_clear)
    assert manager.observe_face(frame, _track(second_clear.timestamp), second_clear)

    state = manager.state_for("camera-a", 1)
    assert state is not None
    assert state.best_face is not None
    assert state.best_face.timestamp == second_clear.timestamp
