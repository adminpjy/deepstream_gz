"""Analytics dispatcher that lets provisional tracks build AdaFace anchors.

Brand-new NvDCF IDs are briefly hidden from business output by
:mod:`provisional_track_guard`.  Their real SCRFD faces still pass through this
worker so AdaFace can corroborate a later continuity merge.  Person/face/events,
snapshots and identity publication start only after the logical track is
confirmed.
"""

from __future__ import annotations

import logging
from datetime import datetime

from deepstream_ai.analytics import AnalyticsDispatcher, _crop, _seconds_between
from deepstream_ai.domain import IdentityResult
from deepstream_ai.events import AnalyticsEventType
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.provisional_track_guard import BUSINESS_PROVISIONAL_KEY

LOGGER = logging.getLogger(__name__)


class ProvisionalAwareAnalyticsDispatcher(AnalyticsDispatcher):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._provisional_last_seen: dict[tuple[str, object], datetime] = {}

    def _process(self, packet: FramePacket) -> None:
        tracks = {track.track_id: track for track in packet.tracks}
        faces_by_track = {face.track_id: face for face in packet.faces}
        provisional = {
            track.track_id
            for track in packet.tracks
            if bool(track.metadata.get(BUSINESS_PROVISIONAL_KEY, False))
        }

        for track in packet.tracks:
            key = track.key
            if track.track_id in provisional:
                self._provisional_last_seen[key] = packet.timestamp
                continue
            was_provisional = self._provisional_last_seen.pop(key, None) is not None
            self._last_seen[key] = packet.timestamp
            self._publish(AnalyticsEventType.PERSON, track, track)
            if self._snapshots is not None:
                self._snapshots.observe_person(
                    packet.image,
                    track,
                    has_face=track.track_id in faces_by_track,
                )
            if was_provisional and self._recognizer is not None:
                remembered = self._recognizer.result_for(track.camera_id, track.track_id)
                if remembered is not None:
                    self._handle_identity(remembered)

        for face in packet.faces:
            track = tracks.get(face.track_id)
            if track is None:
                continue
            is_provisional = face.track_id in provisional
            if not is_provisional:
                self._publish(AnalyticsEventType.FACE, face, face)
                self._remember_face(packet.image, track, face)

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
                self._handle_identity(identity)

        if self._recognizer is not None:
            for identity in self._recognizer.recognize_due(packet.timestamp):
                if (identity.camera_id, identity.track_id) in self._provisional_last_seen:
                    continue
                self._handle_identity(identity)

        for behavior in packet.behaviors:
            if behavior.track_id in provisional:
                continue
            self._publish(AnalyticsEventType.BEHAVIOR, behavior, behavior)
            if self._snapshots is not None:
                self._publish_snapshot(self._snapshots.observe_behavior(packet.image, behavior))
        self._expire_missing(packet.timestamp)

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
