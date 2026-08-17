from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import cv2
import numpy as np

from deepstream_ai.domain import BehaviorDetection, BehaviorType, BoundingBox, FaceDetection, Track
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.production.contracts import RecognitionEvent
from deepstream_ai.production.session_reconcile import SessionFinalReconciler


def _frame(
    timestamp: datetime,
    *,
    track: Track | None = None,
    face: FaceDetection | None = None,
    behaviors: tuple[BehaviorDetection, ...] = (),
    sharp: bool = False,
) -> FramePacket:
    image = np.full((480, 640, 4), 110, dtype=np.uint8)
    image[:, :, 3] = 255
    if sharp:
        for offset in range(0, 480, 16):
            image[offset : offset + 4, :, :3] = 235
        for offset in range(0, 640, 16):
            image[:, offset : offset + 4, :3] = 20
    return FramePacket(
        camera_id="camera-01",
        frame_number=int(timestamp.timestamp() * 10),
        timestamp=timestamp,
        image=image,
        tracks=() if track is None else (track,),
        faces=() if face is None else (face,),
        behaviors=behaviors,
        stream_time_ns=int(timestamp.timestamp() * 1_000_000_000),
    )


def _track(timestamp: datetime, *, track_id: int = 7) -> Track:
    return Track(
        camera_id="camera-01",
        track_id=track_id,
        timestamp=timestamp,
        bbox=BoundingBox(180, 70, 430, 455),
        confidence=0.91,
    )


def _face(timestamp: datetime, *, track_id: int = 7) -> FaceDetection:
    return FaceDetection(
        camera_id="camera-01",
        track_id=track_id,
        timestamp=timestamp,
        bbox=BoundingBox(255, 92, 345, 190),
        score=0.88,
        landmarks=((275, 120), (325, 120), (300, 145), (280, 170), (320, 170)),
    )


def _event_documents(root):
    return [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("*/*/event.json")]


def test_person_is_pushed_once_and_better_evidence_replaces_files(tmp_path):
    now = datetime(2026, 8, 17, 2, 0, tzinfo=UTC)
    reconciler = SessionFinalReconciler(
        session_id="session-person",
        camera_id="camera-01",
        mock_root=tmp_path / "mock",
        identity_label=lambda _camera, _track: None,
    )
    reconciler.observe(_frame(now, track=_track(now)))
    reconciler.observe(
        _frame(
            now + timedelta(seconds=1),
            track=_track(now + timedelta(seconds=1)),
            sharp=True,
        )
    )
    reconciler.observe(
        _frame(
            now + timedelta(seconds=2),
            track=_track(now + timedelta(seconds=2)),
            sharp=True,
        )
    )
    reconciler.finalize()

    root = tmp_path / "mock" / "session-person"
    documents = _event_documents(root)
    person = [item for item in documents if item["eventType"] == "PERSON_APPEARED"]
    assert len(person) == 1
    event_dir = root / "PERSON_APPEARED" / person[0]["eventId"]
    assert (event_dir / "overview.jpg").is_file()
    assert (event_dir / "detail.jpg").is_file()
    assert person[0]["revision"] >= 1
    pushes = (root / "push-events.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum('"eventType":"PERSON_APPEARED"' in line and '"action":"CREATE"' in line for line in pushes) == 1


def test_face_and_employee_are_added_without_repeating_person(tmp_path):
    now = datetime(2026, 8, 17, 2, 10, tzinfo=UTC)
    identities: dict[int, str] = {}
    reconciler = SessionFinalReconciler(
        session_id="session-employee",
        camera_id="camera-01",
        mock_root=tmp_path / "mock",
        identity_label=lambda _camera, track_id: identities.get(int(track_id)),
    )
    reconciler.observe(_frame(now, track=_track(now)))
    second = now + timedelta(seconds=1)
    reconciler.observe(_frame(second, track=_track(second), face=_face(second), sharp=True))
    identities[7] = "worker001"
    third = now + timedelta(seconds=2)
    reconciler.observe(_frame(third, track=_track(third), face=_face(third), sharp=True))
    reconciler.finalize()

    documents = _event_documents(tmp_path / "mock" / "session-employee")
    counts = {name: sum(item["eventType"] == name for item in documents) for name in {
        "PERSON_APPEARED",
        "FACE_APPEARED",
        "EMPLOYEE_WORKING",
        "STRANGER",
    }}
    assert counts == {
        "PERSON_APPEARED": 1,
        "FACE_APPEARED": 1,
        "EMPLOYEE_WORKING": 1,
        "STRANGER": 0,
    }
    employee = next(item for item in documents if item["eventType"] == "EMPLOYEE_WORKING")
    assert employee["personId"] == "worker001"


def test_stranger_is_one_alarm_for_one_track(tmp_path):
    now = datetime(2026, 8, 17, 2, 20, tzinfo=UTC)
    reconciler = SessionFinalReconciler(
        session_id="session-stranger",
        camera_id="camera-01",
        mock_root=tmp_path / "mock",
        identity_label=lambda _camera, _track: None,
        stranger_grace_sec=1.0,
    )
    for seconds in (0, 2, 3, 4):
        timestamp = now + timedelta(seconds=seconds)
        reconciler.observe(
            _frame(timestamp, track=_track(timestamp), face=_face(timestamp), sharp=True)
        )
    reconciler.finalize()

    root = tmp_path / "mock" / "session-stranger"
    documents = _event_documents(root)
    assert sum(item["eventType"] == "STRANGER" for item in documents) == 1
    pushes = (root / "push-events.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum('"eventType":"STRANGER"' in line and '"action":"CREATE"' in line for line in pushes) == 1


def test_behavior_is_deduplicated_for_the_track(tmp_path):
    now = datetime(2026, 8, 17, 2, 30, tzinfo=UTC)
    reconciler = SessionFinalReconciler(
        session_id="session-smoking",
        camera_id="camera-01",
        mock_root=tmp_path / "mock",
        identity_label=lambda _camera, _track: None,
    )
    for index in range(6):
        timestamp = now + timedelta(seconds=index * 0.5)
        track = _track(timestamp)
        detection = BehaviorDetection(
            camera_id="camera-01",
            track_id=7,
            timestamp=timestamp,
            behavior=BehaviorType.SMOKING,
            confidence=0.62 + index * 0.03,
            bbox=BoundingBox(270, 130, 340, 250),
            model_name="smoking",
        )
        reconciler.observe(
            _frame(timestamp, track=track, behaviors=(detection,), sharp=index >= 3)
        )
    reconciler.finalize()

    root = tmp_path / "mock" / "session-smoking"
    documents = _event_documents(root)
    assert sum(item["eventType"] == "SMOKING" for item in documents) == 1
    pushes = (root / "push-events.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum('"eventType":"SMOKING"' in line and '"action":"CREATE"' in line for line in pushes) == 1


def test_no_person_session_keeps_bounded_recovery_candidates(tmp_path):
    now = datetime(2026, 8, 17, 2, 40, tzinfo=UTC)
    reconciler = SessionFinalReconciler(
        session_id="session-empty",
        camera_id="camera-01",
        mock_root=tmp_path / "mock",
        identity_label=lambda _camera, _track: None,
        max_no_person_candidates=3,
    )
    for index in range(6):
        timestamp = now + timedelta(seconds=index)
        reconciler.observe(_frame(timestamp))
    result = reconciler.finalize()

    assert result["noPersonDetected"] is True
    assert result["preRollRequiredForMissedPersonRecovery"] is True
    assert len(result["recoveryCandidateImages"]) == 3
    assert all((tmp_path / "mock" / "session-empty" / "reconcile-candidates" / f"scene-{index:02d}.jpg").is_file() for index in range(1, 4))


def test_left_object_scenario_is_mirrored_to_mock_push(tmp_path):
    now = datetime(2026, 8, 17, 2, 50, tzinfo=UTC)
    snapshot = tmp_path / "after.jpg"
    diff = tmp_path / "diff.jpg"
    assert cv2.imwrite(str(snapshot), np.full((720, 1280, 3), 120, dtype=np.uint8))
    mask = np.zeros((540, 960), dtype=np.uint8)
    mask[250:360, 400:520] = 255
    assert cv2.imwrite(str(diff), mask)

    reconciler = SessionFinalReconciler(
        session_id="session-left",
        camera_id="camera-01",
        mock_root=tmp_path / "mock",
        identity_label=lambda _camera, _track: None,
    )
    event = RecognitionEvent.create(
        session_id="session-left",
        camera_id="camera-01",
        event_type="LEFT_OBJECT",
        timestamp=now,
        confidence=0.87,
        snapshot=str(snapshot),
        extra={"boxes": [[400, 250, 120, 110]], "diffImage": str(diff)},
    )
    reconciler.observe_result(event)

    root = tmp_path / "mock" / "session-left"
    documents = _event_documents(root)
    assert sum(item["eventType"] == "LEFT_OBJECT" for item in documents) == 1
    document = next(item for item in documents if item["eventType"] == "LEFT_OBJECT")
    event_dir = root / "LEFT_OBJECT" / document["eventId"]
    assert (event_dir / "overview.jpg").is_file()
    assert (event_dir / "detail.jpg").is_file()
