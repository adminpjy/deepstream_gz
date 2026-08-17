"""Session-scoped business event aggregation and final reconciliation.

Everything in this module is downstream of the tuned DeepStream recognition
chain.  It observes normalized packets and scenario events but never changes
PeopleNet/NvDCF/SCRFD/AdaFace/behavior inference, thresholds, or metadata.

The final reconcile has two bounded recovery sources:
1. business-visible observations collected during the live session, used to
   select clearer evidence and upgrade person -> face -> identity/behavior;
2. provisional observations already retained by the existing weak-track guard
   for analytics.  These are never pushed live by this module.  Only at session
   end, and only under stricter recovery evidence, can they fill a short-person
   gap.  This does not relax the current real-time visibility guard.

If neither source ever saw a person, a few JPEG candidates are retained for the
future CPU pre-roll integration.  No second detector/model is loaded here.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from deepstream_ai.domain import BehaviorType, BoundingBox, FaceDetection, Track, TrackId
from deepstream_ai.pipeline.metadata import FramePacket, FramePacketConsumer
from deepstream_ai.production.contracts import RecognitionEvent
from deepstream_ai.production.publishers import ResultPublisher
from deepstream_ai.provisional_track_guard import BUSINESS_PROVISIONAL_KEY

LOGGER = logging.getLogger(__name__)


class MockBusinessEventType(str, Enum):
    PERSON_APPEARED = "PERSON_APPEARED"
    FACE_APPEARED = "FACE_APPEARED"
    STRANGER = "STRANGER"
    EMPLOYEE_WORKING = "EMPLOYEE_WORKING"
    SMOKING = "SMOKING"
    DRINKING = "DRINKING"
    EATING = "EATING"
    LEFT_OBJECT = "LEFT_OBJECT"
    LARGE_OBJECT_MOVING = "LARGE_OBJECT_MOVING"


_EVENT_LEVEL = {
    MockBusinessEventType.PERSON_APPEARED: "RECORD",
    MockBusinessEventType.FACE_APPEARED: "RECORD",
    MockBusinessEventType.STRANGER: "ALARM",
    MockBusinessEventType.EMPLOYEE_WORKING: "RECORD",
    MockBusinessEventType.SMOKING: "ALARM",
    MockBusinessEventType.DRINKING: "RECORD",
    MockBusinessEventType.EATING: "RECORD",
    MockBusinessEventType.LEFT_OBJECT: "ALARM",
    MockBusinessEventType.LARGE_OBJECT_MOVING: "ALARM",
}

_EVENT_LABEL = {
    MockBusinessEventType.PERSON_APPEARED: "PERSON",
    MockBusinessEventType.FACE_APPEARED: "FACE",
    MockBusinessEventType.STRANGER: "STRANGER",
    MockBusinessEventType.EMPLOYEE_WORKING: "EMPLOYEE",
    MockBusinessEventType.SMOKING: "SMOKING",
    MockBusinessEventType.DRINKING: "DRINKING",
    MockBusinessEventType.EATING: "EATING",
    MockBusinessEventType.LEFT_OBJECT: "LEFT OBJECT",
    MockBusinessEventType.LARGE_OBJECT_MOVING: "LARGE OBJECT MOVING",
}

# BGR.  Evidence labels stay ASCII because OpenCV's built-in font does not
# render Chinese reliably.
_EVENT_COLOR = {
    MockBusinessEventType.PERSON_APPEARED: (0, 215, 255),
    MockBusinessEventType.FACE_APPEARED: (255, 255, 0),
    MockBusinessEventType.STRANGER: (0, 0, 255),
    MockBusinessEventType.EMPLOYEE_WORKING: (0, 200, 0),
    MockBusinessEventType.SMOKING: (0, 128, 255),
    MockBusinessEventType.DRINKING: (255, 128, 0),
    MockBusinessEventType.EATING: (255, 0, 255),
    MockBusinessEventType.LEFT_OBJECT: (0, 0, 255),
    MockBusinessEventType.LARGE_OBJECT_MOVING: (180, 0, 180),
}

_BEHAVIOR_EVENT = {
    BehaviorType.SMOKING: MockBusinessEventType.SMOKING,
    BehaviorType.DRINKING: MockBusinessEventType.DRINKING,
    BehaviorType.EATING: MockBusinessEventType.EATING,
    BehaviorType.CARRYING: MockBusinessEventType.LARGE_OBJECT_MOVING,
}


def _cv2():
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV 不可用，无法生成模拟推送证据") from exc
    return cv2


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim != 3 or array.size == 0:
        raise ValueError("evidence image is empty")
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    return np.ascontiguousarray(array[:, :, :3])


def _expanded_detail_box(
    box: BoundingBox,
    width: int,
    height: int,
    *,
    minimum_width: int = 480,
) -> BoundingBox | None:
    """Keep a whole person/object and enough surrounding context."""

    clipped = box.clipped(width, height)
    if clipped is None:
        return None
    target_width = max(clipped.width * 1.50, float(min(minimum_width, width)))
    target_height = max(clipped.height * 1.30, target_width * 0.75)
    cx, cy = clipped.center
    x1 = cx - target_width / 2.0
    x2 = cx + target_width / 2.0
    y1 = cy - target_height * 0.48
    y2 = cy + target_height * 0.52
    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if x2 > width:
        x1 -= x2 - width
        x2 = float(width)
    if y2 > height:
        y1 -= y2 - height
        y2 = float(height)
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    try:
        return BoundingBox(
            x1,
            y1,
            max(x1 + 1.0, x2),
            max(y1 + 1.0, y2),
        ).clipped(width, height)
    except ValueError:
        return clipped


def _sharpness_score(image: np.ndarray) -> float:
    cv2 = _cv2()
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(1.0, max(0.0, variance / 400.0))


def _exposure_score(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    mean = float(np.asarray(image, dtype=np.float32).mean())
    return max(0.0, 1.0 - abs(mean - 128.0) / 128.0)


def _person_quality(frame: np.ndarray, track: Track) -> float:
    image = _to_bgr(frame)
    height, width = image.shape[:2]
    clipped = track.bbox.clipped(width, height)
    if clipped is None:
        return 0.0
    rows, cols = clipped.integer_slices(width, height)
    crop = image[rows, cols]
    area_ratio = clipped.area / float(max(1, width * height))
    size_score = min(1.0, math.sqrt(max(0.0, area_ratio) / 0.12))
    completeness = 1.0
    if (
        track.bbox.x1 < 1.0
        or track.bbox.y1 < 1.0
        or track.bbox.x2 > width - 1.0
        or track.bbox.y2 > height - 1.0
    ):
        completeness = 0.65
    return float(
        0.25 * track.confidence
        + 0.25 * size_score
        + 0.30 * _sharpness_score(crop)
        + 0.10 * _exposure_score(crop)
        + 0.10 * completeness
    )


def _face_quality(frame: np.ndarray, face: FaceDetection, track: Track) -> float:
    image = _to_bgr(frame)
    height, width = image.shape[:2]
    clipped = face.bbox.clipped(width, height)
    if clipped is None:
        return 0.0
    rows, cols = clipped.integer_slices(width, height)
    crop = image[rows, cols]
    pixels = max(clipped.width, clipped.height)
    size_score = min(1.0, pixels / 180.0)
    landmark_score = min(1.0, len(face.landmarks) / 5.0)
    person_score = _person_quality(frame, track)
    return float(
        0.28 * face.score
        + 0.22 * size_score
        + 0.25 * _sharpness_score(crop)
        + 0.10 * landmark_score
        + 0.15 * person_score
    )


def _detector_confidence(track: Track) -> float | None:
    value = track.metadata.get("detector_confidence")
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and 0.0 <= result <= 1.0 else None


@dataclass(slots=True)
class EvidenceFrame:
    timestamp: datetime
    image: np.ndarray
    person_box: BoundingBox | None
    primary_box: BoundingBox | None
    face_box: BoundingBox | None = None
    extra_boxes: tuple[BoundingBox, ...] = ()
    confidence: float | None = None
    quality: float = 0.0
    source: str = "LIVE"


@dataclass(slots=True)
class MockEventRecord:
    event_id: str
    event_type: MockBusinessEventType
    camera_id: str
    session_id: str
    track_id: str | None
    person_id: str | None
    start_time: str
    updated_at: str
    confidence: float | None
    source: str
    status: str = "ACTIVE"
    evidence_quality: float = 0.0
    revision: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "eventId": self.event_id,
            "sessionId": self.session_id,
            "cameraId": self.camera_id,
            "eventType": self.event_type.value,
            "eventLevel": _EVENT_LEVEL[self.event_type],
            "trackId": self.track_id,
            "personId": self.person_id,
            "startTime": self.start_time,
            "updatedAt": self.updated_at,
            "confidence": self.confidence,
            "source": self.source,
            "status": self.status,
            "revision": self.revision,
            "evidence": {"overview": "overview.jpg", "detail": "detail.jpg"},
        }


class MockPushStore:
    """File-backed stand-in for the future production HTTP/MQ publisher."""

    def __init__(self, root: str | Path, *, session_id: str, camera_id: str) -> None:
        self.root = Path(root).resolve() / session_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.camera_id = camera_id
        self.journal = self.root / "push-events.jsonl"
        self._records: dict[str, MockEventRecord] = {}
        self._lock = threading.RLock()

    def create(
        self,
        event_type: MockBusinessEventType,
        evidence: EvidenceFrame,
        *,
        track_id: TrackId | None = None,
        person_id: str | None = None,
        confidence: float | None = None,
        event_id: str | None = None,
    ) -> MockEventRecord:
        resolved_id = event_id or uuid4().hex
        now = evidence.timestamp.isoformat()
        record = MockEventRecord(
            event_id=resolved_id,
            event_type=event_type,
            camera_id=self.camera_id,
            session_id=self.session_id,
            track_id=None if track_id is None else str(track_id),
            person_id=person_id,
            start_time=now,
            updated_at=now,
            confidence=confidence,
            source=evidence.source,
            evidence_quality=float(evidence.quality),
        )
        with self._lock:
            if resolved_id in self._records:
                return self._records[resolved_id]
            self._records[resolved_id] = record
            self._write_record(record, evidence)
            self._append_push("CREATE", record)
        return record

    def improve(
        self,
        event_id: str,
        evidence: EvidenceFrame,
        *,
        person_id: str | None = None,
        confidence: float | None = None,
        minimum_gain: float = 0.05,
    ) -> bool:
        with self._lock:
            record = self._records.get(event_id)
            if record is None:
                return False
            identity_changed = person_id is not None and person_id != record.person_id
            confidence_changed = (
                confidence is not None
                and (record.confidence is None or confidence > record.confidence + 0.03)
            )
            evidence_changed = evidence.quality >= record.evidence_quality + minimum_gain
            if not (identity_changed or confidence_changed or evidence_changed):
                return False
            if identity_changed:
                record.person_id = person_id
            if confidence is not None and (
                record.confidence is None or confidence > record.confidence
            ):
                record.confidence = confidence
            if evidence_changed:
                record.evidence_quality = float(evidence.quality)
            record.updated_at = evidence.timestamp.isoformat()
            record.revision += 1
            self._write_record(record, evidence if evidence_changed else None)
            self._append_push("UPDATE", record)
            return True

    def resolve(self, event_id: str, *, timestamp: datetime, reason: str) -> bool:
        with self._lock:
            record = self._records.get(event_id)
            if record is None or record.status == "RESOLVED":
                return False
            record.status = "RESOLVED"
            record.updated_at = timestamp.isoformat()
            record.revision += 1
            self._write_record(record, None, extra={"resolveReason": reason})
            self._append_push("RESOLVE", record, extra={"reason": reason})
            return True

    def _event_dir(self, record: MockEventRecord) -> Path:
        return self.root / record.event_type.value / record.event_id

    def _write_record(
        self,
        record: MockEventRecord,
        evidence: EvidenceFrame | None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        directory = self._event_dir(record)
        directory.mkdir(parents=True, exist_ok=True)
        if evidence is not None:
            self._write_evidence(directory, record.event_type, evidence, person_id=record.person_id)
        document = record.as_dict()
        document["evidenceQuality"] = round(record.evidence_quality, 6)
        if extra:
            document.update(extra)
        _atomic_json(directory / "event.json", document)

    def _write_evidence(
        self,
        directory: Path,
        event_type: MockBusinessEventType,
        evidence: EvidenceFrame,
        *,
        person_id: str | None,
    ) -> None:
        cv2 = _cv2()
        image = _to_bgr(evidence.image)
        annotated = np.array(image, copy=True)
        color = _EVENT_COLOR[event_type]
        label = _EVENT_LABEL[event_type]
        if person_id:
            label = f"{label} {person_id}"
        if evidence.confidence is not None:
            label = f"{label} {evidence.confidence:.2f}"
        if evidence.person_box is not None:
            _draw_box(annotated, evidence.person_box, color, label)
        if evidence.face_box is not None:
            _draw_box(
                annotated,
                evidence.face_box,
                _EVENT_COLOR[MockBusinessEventType.FACE_APPEARED],
                "FACE",
            )
        if evidence.primary_box is not None and evidence.primary_box != evidence.person_box:
            _draw_box(annotated, evidence.primary_box, color, label)
        for extra_box in evidence.extra_boxes:
            if extra_box != evidence.primary_box:
                _draw_box(annotated, extra_box, color, label)
        cv2.imwrite(str(directory / "overview.jpg"), annotated)

        anchor = evidence.person_box or evidence.primary_box
        if anchor is None:
            detail = annotated
        else:
            height, width = annotated.shape[:2]
            detail_box = _expanded_detail_box(anchor, width, height)
            if detail_box is None:
                detail = annotated
            else:
                rows, cols = detail_box.integer_slices(width, height)
                detail = np.ascontiguousarray(annotated[rows, cols])
        cv2.imwrite(str(directory / "detail.jpg"), detail)

    def _append_push(
        self,
        action: str,
        record: MockEventRecord,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        document = {"action": action, **record.as_dict()}
        if extra:
            document.update(extra)
        line = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for record in self._records.values():
                counts[record.event_type.value] = counts.get(record.event_type.value, 0) + 1
            return {
                "root": str(self.root),
                "eventCount": len(self._records),
                "counts": counts,
            }


def _draw_box(image: np.ndarray, box: BoundingBox, color: tuple[int, int, int], label: str) -> None:
    cv2 = _cv2()
    height, width = image.shape[:2]
    clipped = box.clipped(width, height)
    if clipped is None:
        return
    x1, y1, x2, y2 = [int(round(value)) for value in clipped.as_tuple()]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        image,
        label,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


@dataclass(slots=True)
class _TrackEvidence:
    track_id: TrackId
    first_seen: datetime
    last_seen: datetime
    first_face_at: datetime | None = None
    person_id: str | None = None
    best_person: EvidenceFrame | None = None
    best_face: EvidenceFrame | None = None
    best_behavior: dict[BehaviorType, EvidenceFrame] = field(default_factory=dict)
    person_event: str | None = None
    face_event: str | None = None
    stranger_event: str | None = None
    employee_event: str | None = None
    behavior_events: dict[BehaviorType, str] = field(default_factory=dict)
    last_person_quality_sample: datetime | None = None
    last_face_quality_sample: datetime | None = None
    last_behavior_quality_sample: dict[BehaviorType, datetime] = field(default_factory=dict)


@dataclass(slots=True)
class _RecoveryTrack:
    track_id: TrackId
    first_seen: datetime
    last_seen: datetime
    observations: int = 0
    detector_observations: int = 0
    max_detector_confidence: float = 0.0
    best_person: EvidenceFrame | None = None
    best_face: EvidenceFrame | None = None
    last_quality_sample: datetime | None = None


class _ScenarioTapPublisher:
    """Forward current scenario events unchanged, then observe best-effort."""

    def __init__(self, delegate: ResultPublisher, observer: Callable[[RecognitionEvent], None]) -> None:
        self.delegate = delegate
        self.observer = observer

    def publish(self, event: RecognitionEvent) -> None:
        self.delegate.publish(event)
        try:
            self.observer(event)
        except Exception:
            LOGGER.exception("模拟业务事件观察失败 event=%s", event.event_id)

    def close(self) -> None:
        # Delegate is process-scoped and remains owned by the GPU worker.
        return


class ReconcileAnalysisDelegate:
    """Pass analysis packets through unchanged while retaining provisional evidence."""

    def __init__(self, delegate: FramePacketConsumer, reconciler: SessionFinalReconciler) -> None:
        self.delegate = delegate
        self.reconciler = reconciler
        self.config = getattr(delegate, "config", None)

    def submit(self, packet: FramePacket) -> bool:
        try:
            self.reconciler.observe_analysis(packet)
        except Exception:
            LOGGER.exception(
                "Session provisional 补偿证据收集失败 camera=%s frame=%s",
                packet.camera_id,
                packet.frame_number,
            )
        return bool(self.delegate.submit(packet))

    def identity_label(self, camera_id: str, track_id: TrackId) -> str | None:
        return self.delegate.identity_label(camera_id, track_id)


class SessionFinalReconciler:
    """Bounded evidence collector and idle-exit business reconciliation."""

    def __init__(
        self,
        *,
        session_id: str,
        camera_id: str,
        mock_root: str | Path,
        identity_label: Callable[[str, TrackId], str | None],
        stranger_grace_sec: float = 2.0,
        quality_sample_sec: float = 0.5,
        max_no_person_candidates: int = 8,
    ) -> None:
        self.session_id = session_id
        self.camera_id = camera_id
        self.identity_label = identity_label
        self.stranger_grace_sec = float(stranger_grace_sec)
        self.quality_sample_sec = float(quality_sample_sec)
        self.store = MockPushStore(mock_root, session_id=session_id, camera_id=camera_id)
        self._tracks: dict[TrackId, _TrackEvidence] = {}
        self._recovery_tracks: dict[TrackId, _RecoveryTrack] = {}
        self._recovered_track_count = 0
        self._no_person_candidates: deque[tuple[datetime, bytes]] = deque(
            maxlen=max(1, int(max_no_person_candidates))
        )
        self._last_no_person_sample: datetime | None = None
        self._finalized = False
        self._lock = threading.RLock()

    def scenario_publisher(self, delegate: ResultPublisher) -> ResultPublisher:
        return _ScenarioTapPublisher(delegate, self.observe_result)

    def observe(self, packet: FramePacket) -> None:
        with self._lock:
            if self._finalized:
                return
        if not packet.tracks:
            self._sample_no_person_candidate(packet)
            return

        faces = self._best_faces(packet)
        tracks = {track.track_id: track for track in packet.tracks}
        for track in packet.tracks:
            self._observe_track(packet, track, faces.get(track.track_id))
        for detection in packet.behaviors:
            track = tracks.get(detection.track_id)
            if track is None:
                continue
            event_type = _BEHAVIOR_EVENT.get(detection.behavior)
            if event_type is None:
                continue
            state = self._tracks.get(track.track_id)
            if state is None:
                continue
            best = state.best_behavior.get(detection.behavior)
            last_sample = state.last_behavior_quality_sample.get(detection.behavior)
            should_sample = best is None or last_sample is None or (
                packet.timestamp - last_sample
            ).total_seconds() >= self.quality_sample_sec
            if should_sample:
                quality = max(_person_quality(packet.image, track), detection.confidence)
                evidence = EvidenceFrame(
                    timestamp=packet.timestamp,
                    image=np.array(packet.image, copy=True),
                    person_box=track.bbox,
                    primary_box=detection.bbox,
                    face_box=faces.get(track.track_id).bbox if track.track_id in faces else None,
                    confidence=detection.confidence,
                    quality=quality,
                )
                state.last_behavior_quality_sample[detection.behavior] = packet.timestamp
                if best is None or evidence.quality > best.quality:
                    state.best_behavior[detection.behavior] = evidence
                    best = evidence
            if detection.behavior not in state.behavior_events:
                evidence = best
                if evidence is None:
                    continue
                record = self.store.create(
                    event_type,
                    evidence,
                    track_id=track.track_id,
                    person_id=state.person_id,
                    confidence=detection.confidence,
                )
                state.behavior_events[detection.behavior] = record.event_id

    def observe_analysis(self, packet: FramePacket) -> None:
        """Retain only provisional tracks; visible tracks are handled by ``observe``."""

        with self._lock:
            if self._finalized:
                return
        faces = self._best_faces(packet)
        for track in packet.tracks:
            if not bool(track.metadata.get(BUSINESS_PROVISIONAL_KEY, False)):
                continue
            state = self._recovery_tracks.get(track.track_id)
            if state is None:
                state = _RecoveryTrack(
                    track_id=track.track_id,
                    first_seen=track.timestamp,
                    last_seen=track.timestamp,
                )
                self._recovery_tracks[track.track_id] = state
            state.last_seen = track.timestamp
            state.observations += 1
            detector_confidence = _detector_confidence(track)
            if detector_confidence is not None:
                state.detector_observations += 1
                state.max_detector_confidence = max(
                    state.max_detector_confidence,
                    detector_confidence,
                )
            should_sample = state.last_quality_sample is None or (
                packet.timestamp - state.last_quality_sample
            ).total_seconds() >= self.quality_sample_sec
            face = faces.get(track.track_id)
            if state.best_person is None or should_sample:
                quality = _person_quality(packet.image, track)
                state.last_quality_sample = packet.timestamp
                evidence = EvidenceFrame(
                    timestamp=packet.timestamp,
                    image=np.array(packet.image, copy=True),
                    person_box=track.bbox,
                    primary_box=track.bbox,
                    face_box=face.bbox if face is not None else None,
                    confidence=detector_confidence or track.confidence,
                    quality=quality,
                    source="FINAL_RECOVERY",
                )
                if state.best_person is None or quality > state.best_person.quality:
                    state.best_person = evidence
            if face is not None:
                quality = _face_quality(packet.image, face, track)
                evidence = EvidenceFrame(
                    timestamp=packet.timestamp,
                    image=np.array(packet.image, copy=True),
                    person_box=track.bbox,
                    primary_box=track.bbox,
                    face_box=face.bbox,
                    confidence=face.score,
                    quality=quality,
                    source="FINAL_RECOVERY",
                )
                if state.best_face is None or quality > state.best_face.quality:
                    state.best_face = evidence

    @staticmethod
    def _best_faces(packet: FramePacket) -> dict[TrackId, FaceDetection]:
        faces: dict[TrackId, FaceDetection] = {}
        for face in packet.faces:
            previous = faces.get(face.track_id)
            if previous is None or face.score > previous.score:
                faces[face.track_id] = face
        return faces

    def _observe_track(
        self,
        packet: FramePacket,
        track: Track,
        face: FaceDetection | None,
    ) -> None:
        state = self._tracks.get(track.track_id)
        if state is None:
            evidence = EvidenceFrame(
                timestamp=packet.timestamp,
                image=np.array(packet.image, copy=True),
                person_box=track.bbox,
                primary_box=track.bbox,
                face_box=face.bbox if face is not None else None,
                confidence=track.confidence,
                quality=_person_quality(packet.image, track),
            )
            state = _TrackEvidence(
                track_id=track.track_id,
                first_seen=packet.timestamp,
                last_seen=packet.timestamp,
                best_person=evidence,
                last_person_quality_sample=packet.timestamp,
            )
            self._tracks[track.track_id] = state
            state.person_event = self.store.create(
                MockBusinessEventType.PERSON_APPEARED,
                evidence,
                track_id=track.track_id,
                confidence=track.confidence,
            ).event_id
        else:
            state.last_seen = packet.timestamp
            if self._elapsed(state.last_person_quality_sample, packet.timestamp):
                quality = _person_quality(packet.image, track)
                state.last_person_quality_sample = packet.timestamp
                if state.best_person is None or quality > state.best_person.quality:
                    state.best_person = EvidenceFrame(
                        timestamp=packet.timestamp,
                        image=np.array(packet.image, copy=True),
                        person_box=track.bbox,
                        primary_box=track.bbox,
                        face_box=face.bbox if face is not None else None,
                        confidence=track.confidence,
                        quality=quality,
                    )

        if face is not None and (
            state.best_face is None or self._elapsed(state.last_face_quality_sample, packet.timestamp)
        ):
            if state.first_face_at is None:
                state.first_face_at = face.timestamp
            face_quality = _face_quality(packet.image, face, track)
            state.last_face_quality_sample = packet.timestamp
            evidence = EvidenceFrame(
                timestamp=packet.timestamp,
                image=np.array(packet.image, copy=True),
                person_box=track.bbox,
                primary_box=track.bbox,
                face_box=face.bbox,
                confidence=face.score,
                quality=face_quality,
            )
            if state.best_face is None or face_quality > state.best_face.quality:
                state.best_face = evidence
            if state.face_event is None:
                state.face_event = self.store.create(
                    MockBusinessEventType.FACE_APPEARED,
                    evidence,
                    track_id=track.track_id,
                    confidence=face.score,
                ).event_id

        identity = self._safe_identity(track.track_id)
        if identity:
            self._ensure_employee(state, identity, packet.timestamp)
        elif (
            state.first_face_at is not None
            and (packet.timestamp - state.first_face_at).total_seconds() >= self.stranger_grace_sec
        ):
            self._ensure_stranger(state, packet.timestamp)

    def _elapsed(self, previous: datetime | None, current: datetime) -> bool:
        return previous is None or (current - previous).total_seconds() >= self.quality_sample_sec

    def _safe_identity(self, track_id: TrackId) -> str | None:
        try:
            value = self.identity_label(self.camera_id, track_id)
        except Exception:
            LOGGER.debug(
                "final reconcile identity lookup failed camera=%s track=%s",
                self.camera_id,
                track_id,
                exc_info=True,
            )
            return None
        return str(value).strip() if value is not None and str(value).strip() else None

    def _ensure_employee(self, state: _TrackEvidence, person_id: str, timestamp: datetime) -> None:
        state.person_id = person_id
        evidence = state.best_face or state.best_person
        if evidence is None:
            return
        if state.stranger_event is not None:
            self.store.resolve(
                state.stranger_event,
                timestamp=timestamp,
                reason="employee_identified_during_session_reconcile",
            )
        if state.employee_event is None:
            state.employee_event = self.store.create(
                MockBusinessEventType.EMPLOYEE_WORKING,
                evidence,
                track_id=state.track_id,
                person_id=person_id,
                confidence=evidence.confidence,
            ).event_id

    def _ensure_stranger(self, state: _TrackEvidence, timestamp: datetime) -> None:
        if state.person_id is not None or state.stranger_event is not None or state.best_face is None:
            return
        state.stranger_event = self.store.create(
            MockBusinessEventType.STRANGER,
            state.best_face,
            track_id=state.track_id,
            confidence=state.best_face.confidence,
        ).event_id

    def _sample_no_person_candidate(self, packet: FramePacket) -> None:
        previous = self._last_no_person_sample
        if previous is not None and (packet.timestamp - previous).total_seconds() < 0.75:
            return
        self._last_no_person_sample = packet.timestamp
        try:
            cv2 = _cv2()
            image = _to_bgr(packet.image)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                self._no_person_candidates.append((packet.timestamp, encoded.tobytes()))
        except Exception:
            LOGGER.debug("保存无人 Session 补偿候选帧失败", exc_info=True)

    def observe_result(self, event: RecognitionEvent) -> None:
        event_type = {
            "LEFT_OBJECT": MockBusinessEventType.LEFT_OBJECT,
            "LARGE_OBJECT_MOVING": MockBusinessEventType.LARGE_OBJECT_MOVING,
        }.get(str(event.event_type).strip().upper())
        if event_type is None or not event.snapshot:
            return
        cv2 = _cv2()
        image = cv2.imread(str(event.snapshot), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return
        boxes = self._scenario_boxes(event, image)
        evidence = EvidenceFrame(
            timestamp=event.timestamp,
            image=image,
            person_box=None,
            primary_box=boxes[0] if boxes else None,
            extra_boxes=tuple(boxes),
            confidence=event.confidence,
            quality=float(event.confidence or 0.75),
            source="FINAL_RECONCILE",
        )
        self.store.create(
            event_type,
            evidence,
            track_id=event.track_id,
            person_id=event.person_id,
            confidence=event.confidence,
            event_id=event.event_id,
        )

    @staticmethod
    def _scenario_boxes(event: RecognitionEvent, image: np.ndarray) -> list[BoundingBox]:
        raw = event.extra.get("boxes") if event.extra else None
        if not isinstance(raw, list):
            return []
        height, width = image.shape[:2]
        scale_x = 1.0
        scale_y = 1.0
        diff_path = event.extra.get("diffImage") if event.extra else None
        if diff_path:
            diff = _cv2().imread(str(diff_path), _cv2().IMREAD_GRAYSCALE)
            if diff is not None and diff.size:
                diff_h, diff_w = diff.shape[:2]
                scale_x = width / float(max(1, diff_w))
                scale_y = height / float(max(1, diff_h))
        boxes: list[BoundingBox] = []
        for values in raw:
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                continue
            try:
                x, y, box_width, box_height = [float(item) for item in values]
                box = BoundingBox(
                    x * scale_x,
                    y * scale_y,
                    (x + box_width) * scale_x,
                    (y + box_height) * scale_y,
                ).clipped(width, height)
            except (TypeError, ValueError):
                continue
            if box is not None:
                boxes.append(box)
        return boxes

    def finalize(self) -> dict[str, Any]:
        with self._lock:
            if self._finalized:
                return self.snapshot()
            self._finalized = True

        recovered = self._promote_provisional_recovery()
        for state in tuple(self._tracks.values()):
            identity = self._safe_identity(state.track_id)
            if identity:
                self._ensure_employee(state, identity, state.last_seen)
            elif state.best_face is not None:
                self._ensure_stranger(state, state.last_seen)
            self._improve_track_events(state)

        no_person = not self._tracks
        candidates = self._write_recovery_candidates() if no_person else []
        document = {
            "sessionId": self.session_id,
            "cameraId": self.camera_id,
            "finalizedAt": datetime.now().astimezone().isoformat(),
            "mode": "metadata_evidence_reconcile",
            "personTrackCount": len(self._tracks),
            "recoveredPersonTrackCount": recovered,
            "noPersonDetected": no_person,
            "preRollRequiredForMissedPersonRecovery": no_person,
            "recoveryCandidateImages": candidates,
            "mockPush": self.store.snapshot(),
        }
        _atomic_json(self.store.root / "final-reconcile.json", document)
        LOGGER.info(
            "[SESSION_FINAL_RECONCILE] session=%s camera=%s tracks=%d recovered=%d no_person=%s mock_events=%d",
            self.session_id,
            self.camera_id,
            len(self._tracks),
            recovered,
            str(no_person).lower(),
            self.store.snapshot()["eventCount"],
        )
        return document

    def _promote_provisional_recovery(self) -> int:
        recovered = 0
        for candidate in tuple(self._recovery_tracks.values()):
            if candidate.track_id in self._tracks or candidate.best_person is None:
                continue
            has_face = candidate.best_face is not None
            strong_multi = (
                candidate.detector_observations >= 2
                and candidate.max_detector_confidence >= 0.35
                and candidate.best_person.quality >= 0.45
            )
            very_strong_single = (
                candidate.detector_observations >= 1
                and candidate.max_detector_confidence >= 0.60
                and candidate.best_person.quality >= 0.58
            )
            if not (has_face or strong_multi or very_strong_single):
                continue
            state = _TrackEvidence(
                track_id=candidate.track_id,
                first_seen=candidate.first_seen,
                last_seen=candidate.last_seen,
                first_face_at=(
                    candidate.best_face.timestamp if candidate.best_face is not None else None
                ),
                best_person=candidate.best_person,
                best_face=candidate.best_face,
            )
            self._tracks[candidate.track_id] = state
            state.person_event = self.store.create(
                MockBusinessEventType.PERSON_APPEARED,
                candidate.best_person,
                track_id=candidate.track_id,
                confidence=(
                    candidate.max_detector_confidence
                    if candidate.max_detector_confidence > 0
                    else candidate.best_person.confidence
                ),
            ).event_id
            if candidate.best_face is not None:
                state.face_event = self.store.create(
                    MockBusinessEventType.FACE_APPEARED,
                    candidate.best_face,
                    track_id=candidate.track_id,
                    confidence=candidate.best_face.confidence,
                ).event_id
            identity = self._safe_identity(candidate.track_id)
            if identity:
                self._ensure_employee(state, identity, candidate.last_seen)
            elif candidate.best_face is not None:
                self._ensure_stranger(state, candidate.last_seen)
            recovered += 1
            LOGGER.warning(
                "[PERSON_FINAL_RECOVERY] session=%s camera=%s track=%s detector_obs=%d max_conf=%.3f face=%s quality=%.3f",
                self.session_id,
                self.camera_id,
                candidate.track_id,
                candidate.detector_observations,
                candidate.max_detector_confidence,
                str(has_face).lower(),
                candidate.best_person.quality,
            )
        self._recovered_track_count += recovered
        return recovered

    def _improve_track_events(self, state: _TrackEvidence) -> None:
        if state.person_event and state.best_person is not None:
            self.store.improve(
                state.person_event,
                state.best_person,
                person_id=state.person_id,
                confidence=state.best_person.confidence,
            )
        if state.face_event and state.best_face is not None:
            self.store.improve(
                state.face_event,
                state.best_face,
                person_id=state.person_id,
                confidence=state.best_face.confidence,
            )
        identity_evidence = state.best_face or state.best_person
        if state.employee_event and identity_evidence is not None:
            self.store.improve(
                state.employee_event,
                identity_evidence,
                person_id=state.person_id,
                confidence=identity_evidence.confidence,
            )
        if state.stranger_event and state.best_face is not None:
            self.store.improve(
                state.stranger_event,
                state.best_face,
                confidence=state.best_face.confidence,
            )
        for behavior, event_id in state.behavior_events.items():
            evidence = state.best_behavior.get(behavior)
            if evidence is not None:
                self.store.improve(
                    event_id,
                    evidence,
                    person_id=state.person_id,
                    confidence=evidence.confidence,
                )

    def _write_recovery_candidates(self) -> list[str]:
        if not self._no_person_candidates:
            return []
        directory = self.store.root / "reconcile-candidates"
        directory.mkdir(parents=True, exist_ok=True)
        values: list[str] = []
        for index, (timestamp, payload) in enumerate(self._no_person_candidates, start=1):
            path = directory / f"scene-{index:02d}.jpg"
            path.write_bytes(payload)
            values.append(str(path))
            _atomic_json(
                directory / f"scene-{index:02d}.json",
                {"timestamp": timestamp.isoformat(), "source": "LIVE_NO_PERSON"},
            )
        return values

    def snapshot(self) -> dict[str, Any]:
        return {
            "mockPush": self.store.snapshot(),
            "reconcileFinalized": self._finalized,
            "reconcileTrackCount": len(self._tracks),
            "reconcileRecoveryCandidateCount": len(self._recovery_tracks),
            "reconcileRecoveredTrackCount": self._recovered_track_count,
        }

    def close(self) -> None:
        self._no_person_candidates.clear()
        self._recovery_tracks.clear()


__all__ = [
    "EvidenceFrame",
    "MockBusinessEventType",
    "MockPushStore",
    "ReconcileAnalysisDelegate",
    "SessionFinalReconciler",
]
