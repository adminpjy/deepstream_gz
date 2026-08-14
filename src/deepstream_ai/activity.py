"""Lightweight per-task person activity tracking outside business inference."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, RLock
from typing import Protocol

from deepstream_ai.pipeline.metadata import FramePacket, FramePacketConsumer
from deepstream_ai.pose_aware_continuity import PoseAwareTrackContinuityResolver
from deepstream_ai.provisional_track_guard import ProvisionalTrackGuard
from deepstream_ai.stream_epoch import current_stream_generation
from deepstream_ai.track_continuity import TrackContinuityResolver

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
    """Resolve IDs, hold new business IDs briefly, then fan out business work."""

    def __init__(
        self,
        delegate: FramePacketConsumer,
        activity: PersonActivityTracker,
        preview: PreviewSink | None = None,
        continuity: TrackContinuityResolver | None = None,
        weak_tracks: ProvisionalTrackGuard | None = None,
    ) -> None:
        self.delegate = delegate
        self.activity = activity
        self.preview = preview
        app_config = getattr(delegate, "config", None)
        config_path = getattr(app_config, "config_path", None)
        if continuity is None and config_path is not None:
            continuity = PoseAwareTrackContinuityResolver.from_file(config_path)
        if weak_tracks is None and config_path is not None:
            weak_tracks = ProvisionalTrackGuard.from_file(config_path)
        self.continuity = continuity
        self.weak_tracks = weak_tracks

    def _sync_generation(self, camera_id: str) -> int:
        generation = current_stream_generation(camera_id)
        if self.continuity is not None:
            begin_generation = getattr(self.continuity, "begin_stream_generation", None)
            if callable(begin_generation):
                begin_generation(camera_id, generation)
        if self.weak_tracks is not None:
            self.weak_tracks.begin_stream_generation(camera_id, generation)
        return generation

    def submit(self, packet: FramePacket) -> bool:
        self._sync_generation(packet.camera_id)
        if self.continuity is not None:
            packet = self.continuity.resolve(packet)

        analysis_packet = packet
        visible_packet = packet
        if self.weak_tracks is not None:
            analysis_packet, visible_packet = self.weak_tracks.partition(packet)

        # Activity/preview must reflect only confirmed business people.  The
        # asynchronous analysis worker additionally receives provisional tracks
        # so real SCRFD faces can build AdaFace continuity anchors.
        self.activity.observe(visible_packet)
        if self.preview is not None:
            self.preview.submit(visible_packet)
        return self.delegate.submit(analysis_packet)

    def identity_label(self, camera_id: str, track_id: int | str) -> str | None:
        self._sync_generation(camera_id)
        if self.continuity is not None:
            track_id = self.continuity.logical_id(camera_id, track_id)
        return self.delegate.identity_label(camera_id, track_id)

    def presentation_track_id(
        self,
        camera_id: str,
        raw_track_id: int | str,
    ) -> int | str | None:
        self._sync_generation(camera_id)
        if self.continuity is None:
            logical_id: int | str = raw_track_id
        else:
            logical_id = self.continuity.presentation_track_id(camera_id, raw_track_id)
            if logical_id is None:
                return None
        if self.weak_tracks is not None and not self.weak_tracks.is_visible(camera_id, logical_id):
            return None
        return logical_id


__all__ = [
    "ActivityAwareConsumer",
    "ActivitySnapshot",
    "PersonActivityTracker",
    "PreviewSink",
]
