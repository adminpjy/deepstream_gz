from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from deepstream_ai.alarm_dispatcher import AlarmAction, AlarmNotification, AlarmType, TrackAlarmManager
from deepstream_ai.domain import BoundingBox, FaceDetection, IdentityResult, Track

NOW = datetime(2026, 8, 15, tzinfo=UTC)


class CapturePublisher:
    def __init__(self) -> None:
        self.items: list[AlarmNotification] = []

    def publish(self, notification: AlarmNotification) -> None:
        self.items.append(notification)


def frame() -> np.ndarray:
    return np.zeros((720, 1280, 4), dtype=np.uint8)


def track(at: datetime = NOW, *, raw: int = 11) -> Track:
    return Track(
        camera_id="camera-01",
        track_id=7,
        timestamp=at,
        bbox=BoundingBox(300, 120, 700, 700),
        confidence=0.9,
        metadata={"raw_nvdcf_track_id": raw},
    )


def face(at: datetime) -> FaceDetection:
    return FaceDetection(
        camera_id="camera-01",
        track_id=7,
        timestamp=at,
        bbox=BoundingBox(410, 160, 560, 340),
        score=0.86,
    )


def identity(at: datetime, worker_id: str | None, similarity: float) -> IdentityResult:
    return IdentityResult(
        camera_id="camera-01",
        track_id=7,
        timestamp=at,
        worker_id=worker_id,
        similarity=similarity,
        confidence=max(0.0, similarity),
    )


def test_person_face_unknown_employee_is_one_track_alarm_lifecycle() -> None:
    publisher = CapturePublisher()
    alarms = TrackAlarmManager(publisher)
    image = frame()

    alarms.observe_person(image, track(), has_face=False)
    alarms.observe_face(image, track(NOW + timedelta(milliseconds=200)), face(NOW + timedelta(milliseconds=200)))
    alarms.observe_identity(identity(NOW + timedelta(seconds=1), None, 0.42), frame=image, track=track(NOW + timedelta(seconds=1)))
    alarms.observe_identity(identity(NOW + timedelta(seconds=2), "PJY", 0.82), frame=image, track=track(NOW + timedelta(seconds=2)))

    assert [(item.action, item.alarm_type) for item in publisher.items] == [
        (AlarmAction.RAISE, AlarmType.PERSON),
        (AlarmAction.UPDATE, AlarmType.STRANGER),
        (AlarmAction.RESOLVE, AlarmType.STRANGER),
        (AlarmAction.RECORD, AlarmType.EMPLOYEE),
    ]
    assert publisher.items[-1].worker_id == "PJY"
    assert publisher.items[-1].alarm_active is False
    assert publisher.items[0].tracker_id == publisher.items[-1].tracker_id == 7


def test_face_on_first_business_frame_raises_stranger_not_person() -> None:
    publisher = CapturePublisher()
    alarms = TrackAlarmManager(publisher)

    alarms.observe_person(frame(), track(), has_face=True)

    assert len(publisher.items) == 1
    assert publisher.items[0].action is AlarmAction.RAISE
    assert publisher.items[0].alarm_type is AlarmType.STRANGER


def test_raw_nvdcf_change_does_not_create_second_business_alarm() -> None:
    publisher = CapturePublisher()
    alarms = TrackAlarmManager(publisher)
    image = frame()

    alarms.observe_person(image, track(raw=11), has_face=False)
    alarms.observe_person(image, track(NOW + timedelta(seconds=1), raw=19), has_face=False)

    assert len(publisher.items) == 1
    assert publisher.items[0].tracker_id == 7
    assert publisher.items[0].raw_tracker_id == 11


def test_verified_employee_is_never_downgraded_by_later_unknown_result() -> None:
    publisher = CapturePublisher()
    alarms = TrackAlarmManager(publisher)
    image = frame()

    alarms.observe_person(image, track(), has_face=True)
    alarms.observe_identity(identity(NOW + timedelta(seconds=1), "PJY", 0.82), frame=image, track=track())
    count_after_employee = len(publisher.items)
    alarms.observe_identity(identity(NOW + timedelta(seconds=2), None, 0.50), frame=image, track=track())

    assert len(publisher.items) == count_after_employee
    assert publisher.items[-1].alarm_type is AlarmType.EMPLOYEE


def test_finalize_logs_unresolved_or_normal_state_once() -> None:
    publisher = CapturePublisher()
    alarms = TrackAlarmManager(publisher)
    image = frame()

    alarms.observe_person(image, track(), has_face=False)
    alarms.finalize_track("camera-01", 7, timestamp=NOW + timedelta(seconds=5))
    alarms.finalize_track("camera-01", 7, timestamp=NOW + timedelta(seconds=6))

    assert publisher.items[-1].action is AlarmAction.END
    assert publisher.items[-1].alarm_active is True
    assert publisher.items[-1].reason == "track_ended_unresolved"
