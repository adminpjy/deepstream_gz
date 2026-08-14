"""Lightweight per-task person activity tracking outside business inference."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, RLock
from typing import Protocol

from deepstream_ai.pipeline.metadata import FramePacket, FramePacketConsumer
from deepstream_ai.track_continuity import TrackContinuityResolver
from deepstream_ai.track_continuity_guard import GuardedTrackContinuityResolver

_NANOSECONDS = 1_000_000_000


@dataclass(frozen=True, slots=True)
class ActivitySnapshot:
    frames: int
    person_frames: int
    person_detections: int
    face_detections: int
    current_stream_time_ns: int | None
    last_person_stream_time_ns: int | None
    idle_seconds: float
    idle_triggered: bool


class PreviewSink(Protocol):
    def submit(self, packet: FramePacket) -> None: ...


class PersonActivityTracker:
    """Trigger once after a configured amount of person-free video time."""

    def __init__(self, idle_timeout_sec: float) -> None:
        if idle_timeout_sec <= 0:
            raise ValueError("idle_timeout_sec must be positive")
        self.idle_timeout_ns = round(float(idle_timeout_sec) * _NANOSECONDS)
        self.idle_event = Event()
        self._frames = 0
        self._person_frames = 0
        self._person_detections = 0
        self._face_detections = 0
        self._first_stream_time_ns: int | None = None
        self._current_stream_time_ns: int | None = None
        self._last_person_stream_time_ns: int | None = None
        self._lock = RLock()

    def observe(self, packet: FramePacket) -> bool:
        stream_time_ns = packet.stream_time_ns
        if stream_time_ns is None or stream_time_ns < 0:
            return False
        with self._lock:
            # RTSP reconnects and file seeks may move PTS backwards. Start a
            # fresh inactivity window instead of immediately expiring a task.
            if (
                self._current_stream_time_ns is not None
                and stream_time_ns < self._current_stream_time_ns
            ):
                self._first_stream_time_ns = stream_time_ns
                self._last_person_stream_time_ns = stream_time_ns if packet.tracks else None
            if self._first_stream_time_ns is None:
                self._first_stream_time_ns = stream_time_ns
            self._current_stream_time_ns = stream_time_ns
            self._frames += 1
            self._person_detections += len(packet.tracks)
            self._face_detections += len(packet.faces)
            if packet.tracks:
                self._person_frames += 1
                self._last_person_stream_time_ns = stream_time_ns
            baseline = self._last_person_stream_time_ns
            if baseline is None:
                baseline = self._first_stream_time_ns
            idle = max(0, stream_time_ns - baseline)
            triggered = idle >= self.idle_timeout_ns
            if triggered:
                self.idle_event.set()
            return triggered

    def snapshot(self) -> ActivitySnapshot:
        with self._lock:
            baseline = self._last_person_stream_time_ns
            if baseline is None:
                baseline = self._first_stream_time_ns
            idle_ns = (
                max(0, self._current_stream_time_ns - baseline)
                if self._current_stream_time_ns is not None and baseline is not None
                else 0
            )
            return ActivitySnapshot(
                frames=self._frames,
                person_frames=self._person_frames,
                person_detections=self._person_detections,
                face_detections=self._face_detections,
                current_stream_time_ns=self._current_stream_time_ns,
                last_person_stream_time_ns=self._last_person_stream_time_ns,
                idle_seconds=idle_ns / _NANOSECONDS,
                idle_triggered=self.idle_event.is_set(),
            )


class ActivityAwareConsumer:
    """Resolve short ID glitches, then fan out activity/preview/business work."""

    def __init__(
        self,
        delegate: FramePacketConsumer,
        activity: PersonActivityTracker,
        preview: PreviewSink | None = None,
        continuity: TrackContinuityResolver | None = None,
    ) -> None:
        self.delegate = delegate
        self.activity = activity
        self.preview = preview
        if continuity is None:
            app_config = getattr(delegate, "config", None)
            config_path = getattr(app_config, "config_path", None)
            if config_path is not None:
                continuity = GuardedTrackContinuityResolver.from_file(config_path)
        self.continuity = continuity

    def submit(self, packet: FramePacket) -> bool:
        if self.continuity is not None:
            packet = self.continuity.resolve(packet)
        self.activity.observe(packet)
        if self.preview is not None:
            self.preview.submit(packet)
        return self.delegate.submit(packet)

    def identity_label(self, camera_id: str, track_id: int | str) -> str | None:
        if self.continuity is not None:
            track_id = self.continuity.logical_id(camera_id, track_id)
        return self.delegate.identity_label(camera_id, track_id)

    def presentation_track_id(
        self,
        camera_id: str,
        raw_track_id: int | str,
    ) -> int | str | None:
        if self.continuity is None:
            return raw_track_id
        return self.continuity.presentation_track_id(camera_id, raw_track_id)


__all__ = [
    "ActivityAwareConsumer",
    "ActivitySnapshot",
    "PersonActivityTracker",
    "PreviewSink",
]
