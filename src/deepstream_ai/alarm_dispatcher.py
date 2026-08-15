"""Track-scoped alarm lifecycle with a replaceable push boundary.

The current publisher only writes structured logs.  The state machine is kept
separate from the transport so a future HTTP/MQ alarm push can replace
``LoggingAlarmPublisher`` without changing person/face/identity orchestration.

Business semantics:
- person detected but no face: active alarm type ``人形``;
- face detected but no known identity: active alarm type ``陌生人``;
- known employee: resolve the active alarm for the same logical tracker id and
  emit one normal ``员工`` record with workid.

The key is the stable logical/business tracker id, not the transient raw NvDCF
id.  Raw NvDCF ids are carried only as diagnostics when metadata exposes them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Protocol

import numpy as np

from deepstream_ai.domain import BoundingBox, FaceDetection, IdentityResult, Track, TrackId

LOGGER = logging.getLogger(__name__)


class AlarmType(str, Enum):
    PERSON = "人形"
    STRANGER = "陌生人"
    EMPLOYEE = "员工"


class AlarmAction(str, Enum):
    RAISE = "raise"
    UPDATE = "update"
    RESOLVE = "resolve"
    RECORD = "record"
    END = "end"


@dataclass(frozen=True, slots=True)
class AlarmNotification:
    action: AlarmAction
    alarm_type: AlarmType
    camera_id: str
    tracker_id: TrackId
    timestamp: datetime
    image: np.ndarray | None
    worker_id: str | None = None
    previous_type: AlarmType | None = None
    raw_tracker_id: TrackId | None = None
    similarity: float | None = None
    alarm_active: bool = True
    reason: str = ""


class AlarmPublisher(Protocol):
    def publish(self, notification: AlarmNotification) -> None: ...


class LoggingAlarmPublisher:
    """Current alarm boundary; replace this method with real push later."""

    def publish(self, notification: AlarmNotification) -> None:
        image_shape = (
            "none"
            if notification.image is None
            else "x".join(str(int(value)) for value in notification.image.shape[:2][::-1])
        )
        LOGGER.info(
            "[ALARM_PUSH] action=%s type=%s camera=%s tracker=%s raw_tracker=%s "
            "workid=%s similarity=%s active=%s image=%s previous=%s reason=%s",
            notification.action.value,
            notification.alarm_type.value,
            notification.camera_id,
            notification.tracker_id,
            notification.raw_tracker_id if notification.raw_tracker_id is not None else "-",
            notification.worker_id or "-",
            f"{notification.similarity:.3f}" if notification.similarity is not None else "-",
            str(notification.alarm_active).lower(),
            image_shape,
            notification.previous_type.value if notification.previous_type else "-",
            notification.reason or "-",
        )


@dataclass(slots=True)
class _TrackAlarmState:
    camera_id: str
    tracker_id: TrackId
    current_type: AlarmType
    alarm_active: bool
    first_seen: datetime
    last_seen: datetime
    raw_tracker_id: TrackId | None = None
    worker_id: str | None = None
    latest_person_image: np.ndarray | None = None
    latest_face_image: np.ndarray | None = None

    def best_image(self) -> np.ndarray | None:
        return self.latest_person_image if self.latest_person_image is not None else self.latest_face_image


class TrackAlarmManager:
    """Deduplicate alarm transitions for one stable logical tracker id."""

    def __init__(self, publisher: AlarmPublisher | None = None) -> None:
        self.publisher = publisher or LoggingAlarmPublisher()
        self._states: dict[tuple[str, TrackId], _TrackAlarmState] = {}
        self._lock = RLock()

    def observe_person(self, frame: np.ndarray, track: Track, *, has_face: bool) -> None:
        image = _person_crop(frame, track.bbox)
        raw_id = _raw_tracker_id(track)
        key = track.key
        with self._lock:
            state = self._states.get(key)
            if state is None:
                alarm_type = AlarmType.STRANGER if has_face else AlarmType.PERSON
                state = _TrackAlarmState(
                    camera_id=track.camera_id,
                    tracker_id=track.track_id,
                    current_type=alarm_type,
                    alarm_active=True,
                    first_seen=track.timestamp,
                    last_seen=track.timestamp,
                    raw_tracker_id=raw_id,
                    latest_person_image=image,
                )
                self._states[key] = state
                self._publish_state(
                    state,
                    action=AlarmAction.RAISE,
                    timestamp=track.timestamp,
                    reason="face_present_on_first_observation" if has_face else "person_without_face",
                )
                return
            state.last_seen = track.timestamp
            state.raw_tracker_id = raw_id if raw_id is not None else state.raw_tracker_id
            if image is not None:
                state.latest_person_image = image
            if has_face and state.alarm_active and state.current_type is AlarmType.PERSON:
                self._transition_to_stranger(state, track.timestamp, reason="face_detected")

    def observe_face(self, frame: np.ndarray, track: Track, face: FaceDetection) -> None:
        key = track.key
        person_image = _person_crop(frame, track.bbox)
        face_image = _face_crop(frame, face.bbox)
        raw_id = _raw_tracker_id(track)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = _TrackAlarmState(
                    camera_id=track.camera_id,
                    tracker_id=track.track_id,
                    current_type=AlarmType.STRANGER,
                    alarm_active=True,
                    first_seen=face.timestamp,
                    last_seen=face.timestamp,
                    raw_tracker_id=raw_id,
                    latest_person_image=person_image,
                    latest_face_image=face_image,
                )
                self._states[key] = state
                self._publish_state(
                    state,
                    action=AlarmAction.RAISE,
                    timestamp=face.timestamp,
                    reason="face_without_known_identity",
                )
                return
            state.last_seen = face.timestamp
            state.raw_tracker_id = raw_id if raw_id is not None else state.raw_tracker_id
            if person_image is not None:
                state.latest_person_image = person_image
            if face_image is not None:
                state.latest_face_image = face_image
            if state.alarm_active and state.current_type is AlarmType.PERSON:
                self._transition_to_stranger(state, face.timestamp, reason="face_detected")

    def observe_identity(
        self,
        identity: IdentityResult,
        *,
        frame: np.ndarray | None = None,
        track: Track | None = None,
    ) -> None:
        key = (identity.camera_id, identity.track_id)
        person_image = _person_crop(frame, track.bbox) if frame is not None and track is not None else None
        raw_id = _raw_tracker_id(track) if track is not None else None
        with self._lock:
            state = self._states.get(key)
            if state is None:
                initial_type = AlarmType.EMPLOYEE if identity.known else AlarmType.STRANGER
                state = _TrackAlarmState(
                    camera_id=identity.camera_id,
                    tracker_id=identity.track_id,
                    current_type=initial_type,
                    alarm_active=not identity.known,
                    first_seen=identity.timestamp,
                    last_seen=identity.timestamp,
                    raw_tracker_id=raw_id,
                    worker_id=identity.worker_id if identity.known else None,
                    latest_person_image=person_image,
                )
                self._states[key] = state
                self._publish_state(
                    state,
                    action=AlarmAction.RECORD if identity.known else AlarmAction.RAISE,
                    timestamp=identity.timestamp,
                    similarity=identity.similarity,
                    reason="known_identity_without_prior_alarm" if identity.known else "unknown_identity",
                )
                return

            state.last_seen = identity.timestamp
            state.raw_tracker_id = raw_id if raw_id is not None else state.raw_tracker_id
            if person_image is not None:
                state.latest_person_image = person_image

            if identity.known:
                if state.current_type is AlarmType.EMPLOYEE and state.worker_id == identity.worker_id:
                    return
                previous = state.current_type
                if state.alarm_active:
                    self._publish_state(
                        state,
                        action=AlarmAction.RESOLVE,
                        timestamp=identity.timestamp,
                        previous_type=previous,
                        worker_id=identity.worker_id,
                        similarity=identity.similarity,
                        alarm_active=False,
                        reason="employee_identified_same_tracker",
                    )
                state.current_type = AlarmType.EMPLOYEE
                state.alarm_active = False
                state.worker_id = identity.worker_id
                self._publish_state(
                    state,
                    action=AlarmAction.RECORD,
                    timestamp=identity.timestamp,
                    previous_type=previous,
                    worker_id=identity.worker_id,
                    similarity=identity.similarity,
                    alarm_active=False,
                    reason="normal_employee_record",
                )
                return

            # Never downgrade an already verified employee. Unknown identity only
            # promotes an unverified person alarm to stranger.
            if state.current_type is AlarmType.EMPLOYEE:
                return
            if state.current_type is AlarmType.PERSON:
                self._transition_to_stranger(
                    state,
                    identity.timestamp,
                    reason="face_comparison_unknown",
                    similarity=identity.similarity,
                )

    def finalize_track(self, camera_id: str, tracker_id: TrackId, *, timestamp: datetime) -> None:
        key = (camera_id, tracker_id)
        with self._lock:
            state = self._states.pop(key, None)
            if state is None:
                return
            self._publish_state(
                state,
                action=AlarmAction.END,
                timestamp=timestamp,
                worker_id=state.worker_id,
                alarm_active=state.alarm_active,
                reason="track_ended_unresolved" if state.alarm_active else "track_ended_normal",
            )

    def _transition_to_stranger(
        self,
        state: _TrackAlarmState,
        timestamp: datetime,
        *,
        reason: str,
        similarity: float | None = None,
    ) -> None:
        previous = state.current_type
        state.current_type = AlarmType.STRANGER
        state.alarm_active = True
        self._publish_state(
            state,
            action=AlarmAction.UPDATE,
            timestamp=timestamp,
            previous_type=previous,
            similarity=similarity,
            reason=reason,
        )

    def _publish_state(
        self,
        state: _TrackAlarmState,
        *,
        action: AlarmAction,
        timestamp: datetime,
        previous_type: AlarmType | None = None,
        worker_id: str | None = None,
        similarity: float | None = None,
        alarm_active: bool | None = None,
        reason: str,
    ) -> None:
        self.publisher.publish(
            AlarmNotification(
                action=action,
                alarm_type=state.current_type,
                camera_id=state.camera_id,
                tracker_id=state.tracker_id,
                timestamp=timestamp,
                image=state.best_image(),
                worker_id=worker_id if worker_id is not None else state.worker_id,
                previous_type=previous_type,
                raw_tracker_id=state.raw_tracker_id,
                similarity=similarity,
                alarm_active=state.alarm_active if alarm_active is None else alarm_active,
                reason=reason,
            )
        )


def _raw_tracker_id(track: Track | None) -> TrackId | None:
    if track is None:
        return None
    metadata = track.metadata
    for key in ("raw_nvdcf_track_id", "raw_track_id", "tracker_raw_id"):
        if key in metadata:
            value = metadata[key]
            if value is not None and value != "":
                return value  # type: ignore[return-value]
    return None


def _person_crop(frame: np.ndarray | None, bbox: BoundingBox) -> np.ndarray | None:
    if frame is None:
        return None
    array = np.asarray(frame)
    if array.ndim not in (2, 3) or array.size == 0:
        return None
    height, width = array.shape[:2]
    x_pad = bbox.width * 0.12
    y_pad = bbox.height * 0.08
    try:
        expanded = BoundingBox(
            bbox.x1 - x_pad,
            bbox.y1 - y_pad,
            bbox.x2 + x_pad,
            bbox.y2 + y_pad,
        ).clipped(width, height)
    except ValueError:
        return None
    if expanded is None:
        return None
    rows, cols = expanded.integer_slices(width, height)
    crop = np.ascontiguousarray(array[rows, cols])
    return crop if crop.size else None


def _face_crop(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    array = np.asarray(frame)
    if array.ndim not in (2, 3) or array.size == 0:
        return None
    clipped = bbox.clipped(array.shape[1], array.shape[0])
    if clipped is None:
        return None
    rows, cols = clipped.integer_slices(array.shape[1], array.shape[0])
    crop = np.ascontiguousarray(array[rows, cols])
    return crop if crop.size else None


__all__ = [
    "AlarmAction",
    "AlarmNotification",
    "AlarmPublisher",
    "AlarmType",
    "LoggingAlarmPublisher",
    "TrackAlarmManager",
]
