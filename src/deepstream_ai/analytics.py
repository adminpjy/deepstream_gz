"""Asynchronous business orchestration outside the DeepStream pipeline."""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from deepstream_ai.config import AppConfig
from deepstream_ai.domain import BoundingBox, FaceDetection, IdentityResult, Track, TrackId
from deepstream_ai.events import AnalyticsEvent, AnalyticsEventType, EventManager
from deepstream_ai.pipeline.metadata import FramePacket

LOGGER = logging.getLogger(__name__)
_STOP = object()


def _can_coalesce_frame(previous: FramePacket, current: FramePacket) -> bool:
    """Return true when ``current`` safely supersedes a queued tracking-only frame.

    Face/behavior packets are evidence-bearing and are never discarded. A queued
    packet containing a track that disappeared from the newer packet is also
    preserved so a short-lived person cannot vanish merely because analytics is
    temporarily slower than the video stream.
    """

    if previous.camera_id != current.camera_id:
        return False
    if previous.faces or previous.behaviors:
        return False
    if current.frame_number < previous.frame_number:
        return False
    previous_tracks = {track.track_id for track in previous.tracks}
    current_tracks = {track.track_id for track in current.tracks}
    return previous_tracks.issubset(current_tracks)


class _CoalescingFrameQueue(queue.Queue[FramePacket | object]):
    """Bounded queue that keeps evidence frames plus the freshest tracking state."""

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize=maxsize)
        self.coalesced = 0

    def put_latest(self, packet: FramePacket) -> bool:
        # queue.Queue exposes ``mutex``/``queue`` to subclasses. Replacing an
        # existing item leaves unfinished_tasks unchanged because exactly one
        # worker task still represents that pending slot.
        with self.not_full:
            for index in range(self._qsize() - 1, -1, -1):
                existing = self.queue[index]
                if isinstance(existing, FramePacket) and _can_coalesce_frame(existing, packet):
                    self.queue[index] = packet
                    self.coalesced += 1
                    self.not_empty.notify()
                    return True
            if self.maxsize > 0 and self._qsize() >= self.maxsize:
                return False
            self._put(packet)
            self.unfinished_tasks += 1
            self.not_empty.notify()
            return True


class JsonlEventJournal:
    """A backend-friendly durable stream of normalized analytics events."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def __call__(self, event: AnalyticsEvent) -> None:
        document = {
            "event_id": str(event.event_id),
            "event_type": event.event_type.value,
            "camera_id": event.camera_id,
            "track_id": str(event.track_id),
            "timestamp": event.timestamp.isoformat(),
            "payload": _jsonable(event.payload),
            "attributes": _jsonable(event.attributes),
        }
        encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._stream.write(encoded + "\n")

    def close(self) -> None:
        with self._lock:
            self._stream.flush()
            self._stream.close()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, UUID)):
        return str(value) if isinstance(value, UUID) else value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BoundingBox):
        return list(value.as_tuple())
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
            if item.name not in {"crop", "image"}
        }
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item) for key, item in value.items() if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


class AnalyticsDispatcher:
    """Bounded worker that keeps model/database/file I/O off streaming threads."""

    def __init__(self, config: AppConfig, *, queue_size: int = 8) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.config = config
        self.events = EventManager(strict=False)
        self._queue = _CoalescingFrameQueue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._initialization_error: BaseException | None = None
        self._fatal_error: BaseException | None = None
        self._accepting = False
        self._identity_lock = threading.RLock()
        self._identities: dict[tuple[str, TrackId], IdentityResult] = {}
        self._last_seen: dict[tuple[str, TrackId], datetime] = {}
        self._recognizer: Any | None = None
        self._snapshots: Any | None = None
        self._journal: JsonlEventJournal | None = None

    def start(self, timeout: float = 60.0) -> None:
        if self._thread is not None:
            raise RuntimeError("analytics dispatcher has already been started")
        self._thread = threading.Thread(
            target=self._worker,
            name="analytics-worker",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("分析工作线程初始化超时")
        if self._initialization_error is not None:
            raise RuntimeError("分析业务组件初始化失败") from self._initialization_error
        self._accepting = True

    def submit(self, packet: FramePacket) -> bool:
        if not self._accepting:
            return False
        return self._queue.put_latest(packet)

    def queue_metrics(self) -> dict[str, float | int]:
        size = self._queue.qsize()
        capacity = self._queue.maxsize
        ratio = size / capacity if capacity > 0 else 0.0
        return {
            "size": size,
            "capacity": capacity,
            "ratio": max(0.0, min(1.0, ratio)),
            "coalesced": self._queue.coalesced,
        }

    def identity_label(self, camera_id: str, track_id: TrackId) -> str | None:
        with self._identity_lock:
            identity = self._identities.get((camera_id, track_id))
        if identity is None:
            return None
        if identity.known:
            return f"id={identity.worker_id} sim={identity.similarity:.3f}"
        return f"unknown sim={identity.similarity:.3f}"

    def close(self, timeout: float = 60.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self._accepting = False
        # Evidence-bearing packets and the freshest coalesced tracking state are
        # already ordered in the queue; stop only after those accepted items.
        self._queue.put(_STOP)
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("分析工作线程停止超时")
        self._thread = None
        if self._fatal_error is not None:
            raise RuntimeError("分析工作线程异常退出") from self._fatal_error

    def _worker(self) -> None:
        try:
            self._initialize_components()
        except BaseException as exc:
            self._initialization_error = exc
            LOGGER.exception("分析业务组件初始化失败")
            self._ready.set()
            return
        self._ready.set()
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        break
                    assert isinstance(item, FramePacket)
                    try:
                        self._process(item)
                    except Exception:
                        LOGGER.exception(
                            "业务分析失败，跳过帧 camera_id=%s frame=%s",
                            item.camera_id,
                            item.frame_number,
                        )
                finally:
                    self._queue.task_done()
            self._finalize_all()
        except BaseException as exc:
            self._fatal_error = exc
            LOGGER.exception("分析工作线程异常退出")
        finally:
            if self._journal is not None:
                self._journal.close()

    def _initialize_components(self) -> None:
        if self.config.output.events_enabled:
            self._journal = JsonlEventJournal(
                self.config.resolve_path(self.config.output.events_path)
            )
            self.events.subscribe(self._journal)

        if self.config.output.snapshot.enabled:
            from deepstream_ai.snapshot import EventSnapshotManager
            from deepstream_ai.snapshot import SnapshotConfig as BusinessSnapshotConfig

            source = self.config.output.snapshot
            self._snapshots = EventSnapshotManager(
                BusinessSnapshotConfig(
                    root_dir=self.config.resolve_path(source.root),
                    jpeg_quality=source.jpeg_quality,
                    track_ttl_seconds=source.person_decision_delay_sec,
                    behavior_cooldown_seconds=source.behavior_cooldown_sec,
                    padding_x_ratio=source.padding_x_ratio,
                    padding_top_ratio=source.padding_top_ratio,
                    upper_body_fraction=source.upper_body_height_ratio,
                    min_person_crop_width=source.min_crop_width,
                    min_person_crop_height=source.min_crop_height,
                    min_visible_ratio=source.min_visible_ratio,
                )
            )

        if self.config.face_recognition.enabled:
            from deepstream_ai.database import PgVectorFaceRepository
            from deepstream_ai.face import (
                AdaFaceONNXAdapter,
                AdaFacePreprocessor,
                AdaFaceTensorRTAdapter,
                FaceFusionConfig,
                FaceRecognitionService,
                FivePointFaceAligner,
            )
            from deepstream_ai.face import FaceRecognitionConfig as BusinessFaceConfig

            database = PgVectorFaceRepository(self.config.database.dsn)
            database.ensure_schema()
            face = self.config.face_recognition
            preprocessor = AdaFacePreprocessor(
                (face.input_width, face.input_height),
                input_color="rgba",
            )
            model_path = self.config.resolve_path(face.model)
            if face.backend == "tensorrt":
                embedder = AdaFaceTensorRTAdapter(
                    engine_path=model_path,
                    preprocessor=preprocessor,
                )
            else:
                embedder = AdaFaceONNXAdapter(
                    model_path=model_path,
                    input_name=face.input_name,
                    output_name=face.output_name or None,
                    preprocessor=preprocessor,
                )
            fusion = FaceFusionConfig(
                min_candidates=face.min_candidates,
                max_candidates=face.max_candidates,
                max_track_age_seconds=face.decision_timeout_sec,
            )
            threshold = max(face.match_threshold, self.config.database.min_similarity)
            self._recognizer = FaceRecognitionService(
                embedder,
                database,
                BusinessFaceConfig(
                    similarity_threshold=threshold,
                    require_landmarks=face.require_landmarks,
                    fusion=fusion,
                ),
                aligner=FivePointFaceAligner((face.input_width, face.input_height)),
            )

    def _process(self, packet: FramePacket) -> None:
        tracks = {track.track_id: track for track in packet.tracks}
        faces_by_track = {face.track_id: face for face in packet.faces}
        for track in packet.tracks:
            key = track.key
            self._last_seen[key] = packet.timestamp
            self._publish(AnalyticsEventType.PERSON, track, track)
            if self._snapshots is not None:
                self._snapshots.observe_person(
                    packet.image,
                    track,
                    has_face=track.track_id in faces_by_track,
                )

        for face in packet.faces:
            track = tracks.get(face.track_id)
            if track is None:
                continue
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
                        "AdaFace 推理/比对失败 camera_id=%s track=%s",
                        face.camera_id,
                        face.track_id,
                    )
            else:
                identity = IdentityResult(
                    camera_id=face.camera_id,
                    track_id=face.track_id,
                    timestamp=face.timestamp,
                    worker_id=None,
                    similarity=-1.0,
                    confidence=0.0,
                )
            if identity is not None:
                self._handle_identity(identity)

        # A low-quality face may be the only face we ever see. Tick the
        # recognition policy on every processed video frame so the 1-2 second
        # fallback still fires even if SCRFD sees no subsequent face.
        if self._recognizer is not None:
            for identity in self._recognizer.recognize_due(packet.timestamp):
                self._handle_identity(identity)

        for behavior in packet.behaviors:
            self._publish(AnalyticsEventType.BEHAVIOR, behavior, behavior)
            if self._snapshots is not None:
                self._publish_snapshot(self._snapshots.observe_behavior(packet.image, behavior))
        self._expire_missing(packet.timestamp)

    def _remember_face(self, image: np.ndarray, track: Track, face: FaceDetection) -> None:
        if self._snapshots is None:
            return
        quality = 0.65 * face.score + 0.35 * min(1.0, face.bbox.area / (112 * 112))
        self._snapshots.observe_face(image, track, face, quality=quality)

    def _handle_identity(self, identity: IdentityResult) -> bool:
        changed = self._store_identity(identity)
        if not changed:
            return False
        if self._snapshots is not None:
            self._snapshots.observe_identity(identity)
        self._publish(AnalyticsEventType.IDENTITY, identity, identity)
        return True

    def _store_identity(self, identity: IdentityResult) -> bool:
        """Keep the strongest identity without ever downgrading known to unknown."""

        key = (identity.camera_id, identity.track_id)
        with self._identity_lock:
            previous = self._identities.get(key)
            if previous is None:
                self._identities[key] = identity
                return True
            if previous.known and not identity.known:
                return False
            if identity.known and not previous.known:
                self._identities[key] = identity
                return True
            if identity.similarity > previous.similarity:
                self._identities[key] = identity
                return True
            return False

    def _expire_missing(self, now: datetime) -> None:
        ttl = max(
            self.config.output.snapshot.person_decision_delay_sec,
            self.config.face_recognition.decision_timeout_sec,
        )
        stale = [
            key
            for key, last_seen in self._last_seen.items()
            if _seconds_between(now, last_seen) >= ttl
        ]
        for key in stale:
            self._finalize_track(*key, timestamp=now)

    def _finalize_track(self, camera_id: str, track_id: TrackId, *, timestamp: datetime) -> None:
        key = (camera_id, track_id)

        # Final no-miss identity fallback must happen before evidence is finalized
        # so a last successful match can route the saved face into know/ rather
        # than permanently finalizing it as unknown.
        if self._recognizer is not None:
            try:
                identity = self._recognizer.finalize_track(camera_id, track_id)
                if identity is not None:
                    self._handle_identity(identity)
            except Exception:
                LOGGER.exception(
                    "Track final face comparison failed camera_id=%s track=%s",
                    camera_id,
                    track_id,
                )

        if self._snapshots is not None:
            self._publish_snapshot(self._snapshots.finalize_track(camera_id, track_id))
        self._publish(
            AnalyticsEventType.TRACK_ENDED,
            Track(
                camera_id=camera_id,
                track_id=track_id,
                timestamp=timestamp,
                bbox=BoundingBox(0, 0, 1, 1),
            ),
            {"camera_id": camera_id, "track_id": str(track_id)},
        )
        if self._recognizer is not None:
            self._recognizer.discard_track(camera_id, track_id)
        if self._snapshots is not None:
            self._snapshots.clear_track(camera_id, track_id)
        self._last_seen.pop(key, None)
        with self._identity_lock:
            self._identities.pop(key, None)

    def _finalize_all(self) -> None:
        try:
            timestamp = max(self._last_seen.values(), default=datetime.now())
            for camera_id, track_id in tuple(self._last_seen):
                self._finalize_track(camera_id, track_id, timestamp=timestamp)
            if self._snapshots is not None:
                for record in self._snapshots.finalize_all():
                    self._publish_snapshot(record)
        finally:
            if self._snapshots is not None:
                self._snapshots.log_summary()

    def _publish(self, event_type: AnalyticsEventType, source: Any, payload: Any) -> None:
        self.events.publish(
            AnalyticsEvent(
                event_type=event_type,
                camera_id=source.camera_id,
                track_id=source.track_id,
                timestamp=source.timestamp,
                payload=payload,
            )
        )

    def _publish_snapshot(self, record: Any | None) -> None:
        if record is None:
            return
        self.events.publish(
            AnalyticsEvent(
                event_type=AnalyticsEventType.SNAPSHOT,
                camera_id=record.camera_id,
                track_id=record.track_id,
                timestamp=record.timestamp,
                payload=record,
            )
        )


def _crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    rows, columns = bbox.integer_slices(image.shape[1], image.shape[0])
    result = np.ascontiguousarray(image[rows, columns])
    if result.size == 0:
        raise ValueError("crop is empty")
    return result


def _localized_person_crop(
    image: np.ndarray,
    track: Track,
    face: FaceDetection,
) -> tuple[np.ndarray, Track, FaceDetection]:
    clipped = track.bbox.clipped(image.shape[1], image.shape[0])
    if clipped is None:
        raise ValueError("person bounding box is outside the frame")
    person = _crop(image, clipped).copy()
    height, width = person.shape[:2]
    local_track = Track(
        camera_id=track.camera_id,
        track_id=track.track_id,
        timestamp=track.timestamp,
        bbox=BoundingBox(0, 0, width, height),
        confidence=track.confidence,
        metadata=track.metadata,
    )
    left = max(0.0, face.bbox.x1 - clipped.x1)
    top = max(0.0, face.bbox.y1 - clipped.y1)
    right = min(float(width), max(left + 1.0, face.bbox.x2 - clipped.x1))
    bottom = min(float(height), max(top + 1.0, face.bbox.y2 - clipped.y1))
    if right <= left or bottom <= top:
        left, top, right, bottom = 0.0, 0.0, min(1.0, float(width)), min(1.0, float(height))
    local_face = FaceDetection(
        camera_id=face.camera_id,
        track_id=face.track_id,
        timestamp=face.timestamp,
        bbox=BoundingBox(left, top, right, bottom),
        score=face.score,
        landmarks=(),
        metadata=face.metadata,
    )
    person.setflags(write=False)
    return person, local_track, local_face


def _seconds_between(later: datetime, earlier: datetime) -> float:
    if later.tzinfo is None and earlier.tzinfo is not None:
        later = later.replace(tzinfo=earlier.tzinfo)
    elif later.tzinfo is not None and earlier.tzinfo is None:
        earlier = earlier.replace(tzinfo=later.tzinfo)
    return (later - earlier).total_seconds()


__all__ = ["AnalyticsDispatcher", "JsonlEventJournal"]
