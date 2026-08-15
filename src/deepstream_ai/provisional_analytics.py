"""Analytics dispatcher that lets provisional tracks build AdaFace anchors.

Brand-new NvDCF IDs are briefly hidden from business output by
:mod:`provisional_track_guard`. Their real SCRFD faces still pass through this
worker so AdaFace can corroborate a later continuity merge. Person/face/events,
snapshots, identity publication and alarm lifecycle start only after the
logical track is confirmed.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np

from deepstream_ai.alarm_dispatcher import AlarmPublisher, TrackAlarmManager
from deepstream_ai.analytics import AnalyticsDispatcher, _crop, _seconds_between
from deepstream_ai.domain import IdentityResult, Track
from deepstream_ai.events import AnalyticsEventType
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.provisional_track_guard import BUSINESS_PROVISIONAL_KEY

LOGGER = logging.getLogger(__name__)


class ProvisionalAwareAnalyticsDispatcher(AnalyticsDispatcher):
    def __init__(self, *args: Any, alarm_publisher: AlarmPublisher | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._provisional_last_seen: dict[tuple[str, object], datetime] = {}
        self._alarms = TrackAlarmManager(alarm_publisher)
        self._alarm_context_frame: np.ndarray | None = None
        self._alarm_context_tracks: dict[object, Track] = {}

    def _process(self, packet: FramePacket) -> None:
        tracks = {track.track_id: track for track in packet.tracks}
        faces_by_track = {face.track_id: face for face in packet.faces}
        provisional = {
            track.track_id
            for track in packet.tracks
            if bool(track.metadata.get(BUSINESS_PROVISIONAL_KEY, False))
        }
        # Keep one short-lived context only while this packet is processed.  It
        # lets identity transitions carry the current person image to the alarm
        # publisher without retaining full video frames between packets.
        self._alarm_context_frame = packet.image
        self._alarm_context_tracks = tracks
        try:
            for track in packet.tracks:
                key = track.key
                if track.track_id in provisional:
                    self._provisional_last_seen[key] = packet.timestamp
                    continue
                was_provisional = self._provisional_last_seen.pop(key, None) is not None
                self._last_seen[key] = packet.timestamp
                self._publish(AnalyticsEventType.PERSON, track, track)
                self._alarms.observe_person(
                    packet.image,
                    track,
                    has_face=track.track_id in faces_by_track,
                )
                if self._snapshots is not None:
                    self._snapshots.observe_person(
                        packet.image,
                        track,
                        has_face=track.track_id in faces_by_track,
                    )
                if was_provisional and self._recognizer is not None:
                    remembered = self._recognizer.result_for(track.camera_id, track.track_id)
                    if remembered is not None:
                        self._handle_identity(remembered, frame=packet.image, track=track)

            for face in packet.faces:
                track = tracks.get(face.track_id)
                if track is None:
                    continue
                is_provisional = face.track_id in provisional
                if not is_provisional:
                    self._publish(AnalyticsEventType.FACE, face, face)
                    self._remember_face(packet.image, track, face)
                    self._alarms.observe_face(packet.image, track, face)

                identity: IdentityResult | None = None
                if self._recognizer is not None:
                    try:
                        face_crop = _crop(packet.image, face.bbox)
                        identity = self._recognizer.observe(
                            face,
                            face_crop,
                            frame_shape=packet.image.shape,
                        )
                    except Exception:
                        LOGGER.exception(
                            "AdaFace inference failed camera_id=%s track=%s provisional=%s",
                            face.camera_id,
                            face.track_id,
                            is_provisional,
                        )
                elif not is_provisional:
                    identity = IdentityResult(
                        camera_id=face.camera_id,
                        track_id=face.track_id,
                        timestamp=face.timestamp,
                        worker_id=None,
                        similarity=-1.0,
                        confidence=0.0,
                    )
                if identity is not None and not is_provisional:
                    self._handle_identity(identity, frame=packet.image, track=track)

            if self._recognizer is not None:
                for identity in self._recognizer.recognize_due(packet.timestamp):
                    if (identity.camera_id, identity.track_id) in self._provisional_last_seen:
                        continue
                    self._handle_identity(
                        identity,
                        frame=packet.image,
                        track=tracks.get(identity.track_id),
                    )

            for behavior in packet.behaviors:
                if behavior.track_id in provisional:
                    continue
                self._publish(AnalyticsEventType.BEHAVIOR, behavior, behavior)
                if self._snapshots is not None:
                    self._publish_snapshot(self._snapshots.observe_behavior(packet.image, behavior))
            self._expire_missing(packet.timestamp)
        finally:
            self._alarm_context_frame = None
            self._alarm_context_tracks = {}

    def _handle_identity(
        self,
        identity: IdentityResult,
        *,
        frame: np.ndarray | None = None,
        track: Track | None = None,
    ) -> bool:
        changed = super()._handle_identity(identity)
        if not changed:
            return False
        if frame is None:
            frame = self._alarm_context_frame
        if track is None:
            track = self._alarm_context_tracks.get(identity.track_id)
        self._alarms.observe_identity(identity, frame=frame, track=track)
        return True

    def _finalize_track(self, camera_id: str, track_id: object, *, timestamp: datetime) -> None:
        # Base finalization may perform one final face comparison. Because this
        # class overrides _handle_identity, a last-moment employee match resolves
        # the active alarm before the track is closed.
        super()._finalize_track(camera_id, track_id, timestamp=timestamp)
        self._alarms.finalize_track(camera_id, track_id, timestamp=timestamp)

    def _expire_missing(self, now: datetime) -> None:
        super()._expire_missing(now)
        ttl = max(5.0, self.config.face_recognition.decision_timeout_sec * 2.0)
        stale = [
            key
            for key, last_seen in self._provisional_last_seen.items()
            if _seconds_between(now, last_seen) >= ttl
        ]
        for camera_id, track_id in stale:
            if self._recognizer is not None:
                self._recognizer.discard_track(camera_id, track_id)
            self._provisional_last_seen.pop((camera_id, track_id), None)
            LOGGER.info(
                "[PERSON_PROVISIONAL_EXPIRE] camera=%s track=%s business_created=false",
                camera_id,
                track_id,
            )


__all__ = ["ProvisionalAwareAnalyticsDispatcher"]
