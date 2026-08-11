"""Independent, overlay-free event snapshot management."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Protocol

import numpy as np

from deepstream_ai.domain import (
    BehaviorDetection,
    BoundingBox,
    FaceDetection,
    IdentityResult,
    Track,
    TrackId,
)

LOGGER = logging.getLogger(__name__)
_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class SnapshotError(RuntimeError):
    """Base snapshot failure."""


class SnapshotEncodingError(SnapshotError):
    """An image could not be safely encoded."""


class SnapshotWriteError(SnapshotError):
    """An encoded snapshot could not be persisted atomically."""


class SnapshotKind(str, Enum):
    PERSON = "person"
    FACE_KNOWN = "face_known"
    FACE_UNKNOWN = "face_unknown"
    BEHAVIOR = "behavior"


@dataclass(frozen=True, slots=True)
class SnapshotConfig:
    root_dir: Path = Path("output/snapshot")
    jpeg_quality: int = 92
    person_min_quality: float = 0.0
    upper_body_fraction: float = 0.75
    padding_x_ratio: float = 0.20
    padding_top_ratio: float = 0.20
    min_person_crop_width: int = 16
    min_person_crop_height: int = 32
    min_visible_ratio: float = 0.50
    track_ttl_seconds: float = 5.0
    behavior_cooldown_seconds: float = 10.0
    sharpness_reference: float = 120.0
    frame_color_space: str = "rgba"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_dir", Path(self.root_dir))
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not 0.0 <= self.person_min_quality <= 1.0:
            raise ValueError("person_min_quality must be between 0 and 1")
        if not 0.1 <= self.upper_body_fraction <= 1.0:
            raise ValueError("upper_body_fraction must be between 0.1 and 1")
        if not 0.0 <= self.padding_x_ratio <= 1.0:
            raise ValueError("padding_x_ratio must be between 0 and 1")
        if not 0.0 <= self.padding_top_ratio <= 1.0:
            raise ValueError("padding_top_ratio must be between 0 and 1")
        if self.min_person_crop_width < 1 or self.min_person_crop_height < 1:
            raise ValueError("minimum person crop dimensions must be positive")
        if not 0.0 <= self.min_visible_ratio <= 1.0:
            raise ValueError("min_visible_ratio must be between 0 and 1")
        if self.track_ttl_seconds <= 0 or self.behavior_cooldown_seconds < 0:
            raise ValueError("snapshot timing values are invalid")
        if self.sharpness_reference <= 0:
            raise ValueError("sharpness_reference must be positive")
        color = self.frame_color_space.strip().lower()
        if color not in {"rgb", "rgba", "bgr", "bgra"}:
            raise ValueError("frame_color_space must be rgb, rgba, bgr, or bgra")
        object.__setattr__(self, "frame_color_space", color)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> SnapshotConfig:
        if not values:
            return cls()
        if isinstance(values.get("snapshots"), Mapping):
            values = values["snapshots"]  # type: ignore[assignment]
        elif isinstance(values.get("snapshot"), Mapping):
            values = values["snapshot"]  # type: ignore[assignment]
        raw_person_crop = values.get("person_crop", {})
        person_crop = raw_person_crop if isinstance(raw_person_crop, Mapping) else {}
        return cls(
            root_dir=Path(
                str(
                    values.get(
                        "root_dir", values.get("root", values.get("path", "output/snapshot"))
                    )
                )
            ),
            jpeg_quality=int(values.get("jpeg_quality", 92)),
            person_min_quality=float(values.get("person_min_quality", 0.0)),
            upper_body_fraction=float(
                person_crop.get(
                    "upper_body_height_ratio",
                    values.get("upper_body_fraction", values.get("upper_body_scale", 0.75)),
                )
            ),
            padding_x_ratio=float(person_crop.get("padding_x_ratio", 0.20)),
            padding_top_ratio=float(person_crop.get("padding_top_ratio", 0.20)),
            min_person_crop_width=int(person_crop.get("min_crop_width", 16)),
            min_person_crop_height=int(person_crop.get("min_crop_height", 32)),
            min_visible_ratio=float(person_crop.get("min_visible_ratio", 0.50)),
            track_ttl_seconds=float(
                values.get("track_ttl_seconds", values.get("person_decision_delay_sec", 5.0))
            ),
            behavior_cooldown_seconds=float(
                values.get("behavior_cooldown_seconds", values.get("behavior_cooldown_sec", 10.0))
            ),
            sharpness_reference=float(values.get("sharpness_reference", 120.0)),
            frame_color_space=str(values.get("frame_color_space", "rgba")),
        )

    @classmethod
    def from_runtime_config(cls, config: object) -> SnapshotConfig:
        return cls(
            root_dir=Path(
                str(getattr(config, "root", getattr(config, "root_dir", "output/snapshot")))
            ),
            jpeg_quality=int(getattr(config, "jpeg_quality", 92)),
            upper_body_fraction=float(getattr(config, "upper_body_height_ratio", 0.75)),
            padding_x_ratio=float(getattr(config, "padding_x_ratio", 0.20)),
            padding_top_ratio=float(getattr(config, "padding_top_ratio", 0.20)),
            min_person_crop_width=int(getattr(config, "min_crop_width", 16)),
            min_person_crop_height=int(getattr(config, "min_crop_height", 32)),
            min_visible_ratio=float(getattr(config, "min_visible_ratio", 0.50)),
            track_ttl_seconds=float(
                getattr(
                    config, "person_decision_delay_sec", getattr(config, "track_ttl_seconds", 5.0)
                )
            ),
            behavior_cooldown_seconds=float(
                getattr(
                    config,
                    "behavior_cooldown_sec",
                    getattr(config, "behavior_cooldown_seconds", 10.0),
                )
            ),
            frame_color_space=str(getattr(config, "frame_color_space", "rgba")),
        )


class ImageEncoder(Protocol):
    def encode(self, image: np.ndarray, *, quality: int) -> bytes: ...


class SnapshotStore(Protocol):
    def write(self, relative_path: Path, payload: bytes) -> Path: ...


class JpegImageEncoder:
    """JPEG encoding with lazy OpenCV import and a Pillow fallback."""

    def __init__(self, input_color: str = "rgba") -> None:
        input_color = input_color.strip().lower()
        if input_color not in {"rgb", "rgba", "bgr", "bgra"}:
            raise ValueError("input_color must be rgb, rgba, bgr, or bgra")
        self.input_color = input_color

    def encode(self, image: np.ndarray, *, quality: int) -> bytes:
        array = _validated_image(image)
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:
            return self._encode_pillow(array, quality)
        try:
            success, encoded = cv2.imencode(
                ".jpg",
                self._opencv_pixels(array),
                [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
            )
        except Exception as exc:
            raise SnapshotEncodingError("OpenCV failed to encode JPEG") from exc
        if not success or encoded is None or encoded.size == 0:
            raise SnapshotEncodingError("OpenCV returned an empty JPEG")
        return encoded.tobytes()

    def _encode_pillow(self, array: np.ndarray, quality: int) -> bytes:
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SnapshotEncodingError(
                "snapshot JPEG encoding requires opencv-python or Pillow"
            ) from exc
        import io

        if self.input_color.startswith("bgr") and array.ndim == 3:
            if array.shape[2] == 3:
                array = array[..., ::-1]
            elif array.shape[2] == 4:
                array = array[..., [2, 1, 0, 3]]
        try:
            output = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(array)).convert("RGB").save(
                output,
                format="JPEG",
                quality=quality,
            )
            result = output.getvalue()
        except Exception as exc:
            raise SnapshotEncodingError("Pillow failed to encode JPEG") from exc
        if not result:
            raise SnapshotEncodingError("Pillow returned an empty JPEG")
        return result

    def _opencv_pixels(self, array: np.ndarray) -> np.ndarray:
        if self.input_color.startswith("rgb") and array.ndim == 3:
            if array.shape[2] == 3:
                array = array[..., ::-1]
            elif array.shape[2] == 4:
                array = array[..., [2, 1, 0, 3]]
        return np.ascontiguousarray(array)


class FilesystemSnapshotStore:
    """Write beneath one root using fsync + atomic replace."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).resolve()

    def write(self, relative_path: Path, payload: bytes) -> Path:
        relative_path = Path(relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise SnapshotWriteError("snapshot path must stay below the configured root")
        destination = (self.root_dir / relative_path).resolve()
        try:
            destination.relative_to(self.root_dir)
        except ValueError as exc:
            raise SnapshotWriteError("snapshot path escapes the configured root") from exc
        if not payload:
            raise SnapshotWriteError("refusing to write an empty snapshot")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
        except Exception as exc:
            if temporary_name:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning("Could not remove temporary snapshot %s", temporary_name)
            raise SnapshotWriteError(f"failed to write snapshot: {destination}") from exc
        return destination


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    kind: SnapshotKind
    camera_id: str
    track_id: TrackId
    timestamp: datetime
    path: Path
    quality: float | None = None
    worker_id: str | None = None
    similarity: float = -1.0
    behavior: str | None = None
    source: str | None = None
    person_bbox: BoundingBox | None = None
    face_bbox: BoundingBox | None = None
    evidence_bbox: BoundingBox | None = None
    full_frame_fallback: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """One bounded in-memory candidate; it never retains the full video frame."""

    camera_id: str
    track_id: TrackId
    timestamp: datetime
    person_bbox: BoundingBox
    face_bbox: BoundingBox | None
    evidence_bbox: BoundingBox
    person_crop: np.ndarray
    face_crop: np.ndarray | None
    full_frame_fallback: bool
    worker_id: str | None
    similarity: float
    quality_score: float

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        if self.track_id == "" or self.track_id is None:
            raise ValueError("track_id cannot be empty")
        object.__setattr__(self, "quality_score", _quality(self.quality_score))
        similarity = float(self.similarity)
        if not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0:
            raise ValueError("similarity must be between -1 and 1")
        object.__setattr__(self, "similarity", similarity)
        person_crop = np.ascontiguousarray(self.person_crop)
        person_crop.setflags(write=False)
        object.__setattr__(self, "person_crop", person_crop)
        if self.face_crop is not None:
            face_crop = np.ascontiguousarray(self.face_crop)
            face_crop.setflags(write=False)
            object.__setattr__(self, "face_crop", face_crop)


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    person_tracks_created: int
    tracks_finalized: int
    best_face_evidence: int
    face_fallback_evidence: int
    best_person_evidence: int
    person_fallback_evidence: int
    know: int
    unknown: int
    snapshot_success: int
    snapshot_failed: int
    evidence_missing: int


@dataclass(slots=True)
class TrackEvidenceState:
    """The four evidence slots and identity state for one tracker lifecycle."""

    camera_id: str
    track_id: TrackId
    last_seen: datetime
    person_fallback: EvidenceCandidate | None = None
    best_person: EvidenceCandidate | None = None
    face_fallback: EvidenceCandidate | None = None
    best_face: EvidenceCandidate | None = None
    identity_result: IdentityResult | None = None

    def selected(self) -> tuple[str, EvidenceCandidate] | None:
        for source, candidate in (
            ("best_face", self.best_face),
            ("face_fallback", self.face_fallback),
            ("best_person", self.best_person),
            ("person_fallback", self.person_fallback),
        ):
            if candidate is not None:
                return source, candidate
        return None


class EventSnapshotManager:
    """Own bounded per-track evidence state and one-shot snapshot persistence."""

    def __init__(
        self,
        config: SnapshotConfig | None = None,
        *,
        encoder: ImageEncoder | None = None,
        store: SnapshotStore | None = None,
    ) -> None:
        self.config = config or SnapshotConfig()
        self.encoder = encoder or JpegImageEncoder(self.config.frame_color_space)
        self.store = store or FilesystemSnapshotStore(self.config.root_dir)
        self._states: dict[tuple[str, TrackId], TrackEvidenceState] = {}
        self._finalized_tracks: set[tuple[str, TrackId]] = set()
        self._missing_tracks: set[tuple[str, TrackId]] = set()
        self._behavior_last: dict[tuple[str, TrackId, str], datetime] = {}
        self._created_count = 0
        self._finalized_count = 0
        self._missing_count = 0
        self._snapshot_success_count = 0
        self._snapshot_failed_count = 0
        self._source_counts = {
            "best_face": 0,
            "face_fallback": 0,
            "best_person": 0,
            "person_fallback": 0,
        }
        self._known_count = 0
        self._unknown_count = 0
        self._summary_emitted = False
        self._lock = RLock()

    def observe_person(
        self,
        frame: np.ndarray,
        track: Track,
        *,
        has_face: bool = False,
        quality: float | None = None,
    ) -> bool:
        """Create the mandatory fallback, then retain only a strictly better person."""

        del has_face  # A face must never delete the mandatory person fallback.
        key = track.key
        crop, evidence_bbox, full_frame_fallback = _evidence_crop(
            frame,
            track.bbox,
            self.config,
        )
        if quality is not None:
            candidate_quality = _quality(quality)
        elif full_frame_fallback:
            candidate_quality = 0.0
        else:
            candidate_quality = self._person_quality(crop, frame, track)
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                state.last_seen = track.timestamp
                baseline = max(
                    state.person_fallback.quality_score
                    if state.person_fallback is not None
                    else -1.0,
                    state.best_person.quality_score if state.best_person is not None else -1.0,
                )
                if (
                    candidate_quality <= baseline
                    or candidate_quality < self.config.person_min_quality
                ):
                    return False
            elif key in self._finalized_tracks:
                return False

        candidate = EvidenceCandidate(
            camera_id=track.camera_id,
            track_id=track.track_id,
            timestamp=track.timestamp,
            person_bbox=track.bbox,
            face_bbox=None,
            evidence_bbox=evidence_bbox,
            person_crop=crop,
            face_crop=None,
            full_frame_fallback=full_frame_fallback,
            worker_id=None,
            similarity=-1.0,
            quality_score=candidate_quality,
        )
        with self._lock:
            state = self._states.get(key)
            if state is None:
                if key in self._finalized_tracks:
                    return False
                self._states[key] = TrackEvidenceState(
                    camera_id=track.camera_id,
                    track_id=track.track_id,
                    last_seen=track.timestamp,
                    person_fallback=candidate,
                )
                self._created_count += 1
                LOGGER.info(
                    "[TRACK_CREATE] camera=%s track=%s person_conf=%.3f",
                    track.camera_id,
                    track.track_id,
                    track.confidence,
                )
                return True
            baseline = max(
                state.person_fallback.quality_score if state.person_fallback is not None else -1.0,
                state.best_person.quality_score if state.best_person is not None else -1.0,
            )
            if candidate_quality > baseline:
                state.best_person = candidate
                state.last_seen = track.timestamp
                return True
        return False

    def observe_face(
        self,
        frame: np.ndarray,
        track: Track,
        face: FaceDetection,
        identity: IdentityResult | None = None,
        *,
        known: bool | None = None,
        quality: float | None = None,
    ) -> bool:
        """Create an unconditional first-face fallback, then retain a better face."""

        if face.key != track.key:
            raise ValueError("face and person track keys do not match")
        if face.timestamp != track.timestamp:
            raise ValueError("face and person evidence must come from the same frame timestamp")
        del known  # Directory selection is deferred and comes from identity_result.
        key = track.key
        with self._lock:
            state_exists = key in self._states
        if not state_exists:
            self.observe_person(frame, track)

        candidate_quality = _quality(face.score if quality is None else quality)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return False
            state.last_seen = face.timestamp
            first_face = state.face_fallback is None
            baseline = max(
                state.face_fallback.quality_score if state.face_fallback is not None else -1.0,
                state.best_face.quality_score if state.best_face is not None else -1.0,
            )
            if not first_face and candidate_quality <= baseline:
                if identity is not None:
                    state.identity_result = identity
                return False

        person_crop, evidence_bbox, full_frame_fallback = _evidence_crop(
            frame,
            track.bbox,
            self.config,
            face_bbox=face.bbox,
            upper_body=True,
        )
        face_crop = _optional_crop(frame, face.bbox)
        candidate = EvidenceCandidate(
            camera_id=track.camera_id,
            track_id=track.track_id,
            timestamp=face.timestamp,
            person_bbox=track.bbox,
            face_bbox=face.bbox,
            evidence_bbox=evidence_bbox,
            person_crop=person_crop,
            face_crop=face_crop,
            full_frame_fallback=full_frame_fallback,
            worker_id=identity.worker_id if identity is not None else None,
            similarity=identity.similarity if identity is not None else -1.0,
            quality_score=candidate_quality,
        )
        with self._lock:
            state = self._states[key]
            if identity is not None:
                state.identity_result = identity
            if state.face_fallback is None:
                state.face_fallback = candidate
                LOGGER.info(
                    "[FACE_FALLBACK] camera=%s track=%s face_conf=%.3f quality=%.3f",
                    face.camera_id,
                    face.track_id,
                    face.score,
                    candidate_quality,
                )
                return True
            old_quality = max(
                state.face_fallback.quality_score,
                state.best_face.quality_score if state.best_face is not None else -1.0,
            )
            if candidate_quality > old_quality:
                state.best_face = candidate
                LOGGER.info(
                    "[BEST_FACE_UPDATE] camera=%s track=%s old_quality=%.3f new_quality=%.3f",
                    face.camera_id,
                    face.track_id,
                    old_quality,
                    candidate_quality,
                )
                return True
        return False

    def observe_identity(self, identity: IdentityResult) -> bool:
        """Update identity independently from the selected evidence frame."""

        key = (identity.camera_id, identity.track_id)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return False
            state.identity_result = identity
            return True

    def observe_behavior(
        self,
        frame: np.ndarray,
        detection: BehaviorDetection,
        *,
        crop_bbox: BoundingBox | None = None,
    ) -> SnapshotRecord | None:
        key = (detection.camera_id, detection.track_id, detection.behavior.value)
        with self._lock:
            last_saved = self._behavior_last.get(key)
            if (
                last_saved is not None
                and _seconds_between(detection.timestamp, last_saved)
                < self.config.behavior_cooldown_seconds
            ):
                return None
        crop = _crop(frame, crop_bbox or detection.bbox).copy()
        relative = Path("behavior") / self._filename(
            detection.camera_id,
            detection.track_id,
            detection.timestamp,
            detection.behavior.value,
        )
        path = self._write(relative, crop)
        with self._lock:
            self._behavior_last[key] = detection.timestamp
        LOGGER.info(
            "Saved behavior snapshot camera=%s track=%s behavior=%s path=%s",
            detection.camera_id,
            detection.track_id,
            detection.behavior.value,
            path,
        )
        return SnapshotRecord(
            kind=SnapshotKind.BEHAVIOR,
            camera_id=detection.camera_id,
            track_id=detection.track_id,
            timestamp=detection.timestamp,
            path=path,
            behavior=detection.behavior.value,
        )

    def finalize_track(self, camera_id: str, track_id: TrackId) -> SnapshotRecord | None:
        """Persist exactly one candidate using the required evidence priority."""

        key = (camera_id, track_id)
        with self._lock:
            if key in self._finalized_tracks or key in self._missing_tracks:
                return None
            state = self._states.get(key)
            selected = state.selected() if state is not None else None
            self._finalized_count += 1
        if state is None or selected is None:
            with self._lock:
                if key not in self._missing_tracks:
                    self._missing_count += 1
                    self._snapshot_failed_count += 1
                self._missing_tracks.add(key)
            LOGGER.error(
                "[ERROR][EVIDENCE_MISSING] camera=%s track=%s reason=%s",
                camera_id,
                track_id,
                "track_state_not_found" if state is None else "no_candidate",
            )
            return None

        source, candidate = selected
        identity = state.identity_result
        known = identity.known if identity is not None else False
        worker_id = identity.worker_id if known and identity is not None else None
        similarity = identity.similarity if identity is not None else -1.0
        if source in {"best_face", "face_fallback"}:
            category = "know" if known else "unknow"
            relative = (
                Path("face")
                / category
                / self._evidence_filename(
                    candidate.camera_id,
                    candidate.track_id,
                    candidate.timestamp,
                    worker_id,
                    similarity,
                    candidate.quality_score,
                )
            )
            kind = SnapshotKind.FACE_KNOWN if known else SnapshotKind.FACE_UNKNOWN
        else:
            relative = Path("person") / self._evidence_filename(
                candidate.camera_id,
                candidate.track_id,
                candidate.timestamp,
                worker_id,
                similarity,
                candidate.quality_score,
            )
            kind = SnapshotKind.PERSON
        try:
            path = self._write(relative, candidate.person_crop)
        except Exception:
            with self._lock:
                if key not in self._missing_tracks:
                    self._missing_count += 1
                    self._snapshot_failed_count += 1
                self._missing_tracks.add(key)
            LOGGER.exception(
                "[ERROR][EVIDENCE_MISSING] camera=%s track=%s reason=snapshot_write_failed",
                camera_id,
                track_id,
            )
            raise
        with self._lock:
            self._states.pop(key, None)
            self._finalized_tracks.add(key)
            self._missing_tracks.discard(key)
            self._snapshot_success_count += 1
            self._source_counts[source] += 1
            if source in {"best_face", "face_fallback"}:
                if known:
                    self._known_count += 1
                else:
                    self._unknown_count += 1
        LOGGER.info(
            "[TRACK_FINALIZE] camera=%s track=%s source=%s identity=%s "
            "similarity=%.3f quality=%.3f snapshot=%s",
            camera_id,
            track_id,
            source,
            worker_id or "unknown",
            similarity,
            candidate.quality_score,
            path,
        )
        return SnapshotRecord(
            kind=kind,
            camera_id=camera_id,
            track_id=track_id,
            timestamp=candidate.timestamp,
            path=path,
            quality=candidate.quality_score,
            worker_id=worker_id,
            similarity=similarity,
            source=source,
            person_bbox=candidate.person_bbox,
            face_bbox=candidate.face_bbox,
            evidence_bbox=candidate.evidence_bbox,
            full_frame_fallback=candidate.full_frame_fallback,
        )

    def expire_tracks(self, now: datetime | None = None) -> tuple[SnapshotRecord, ...]:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            expired = [
                key
                for key, state in self._states.items()
                if _seconds_between(now, state.last_seen) >= self.config.track_ttl_seconds
            ]
        records: list[SnapshotRecord] = []
        for camera_id, track_id in expired:
            record = self.finalize_track(camera_id, track_id)
            if record is not None:
                records.append(record)
        return tuple(records)

    def finalize_all(self) -> tuple[SnapshotRecord, ...]:
        with self._lock:
            keys = tuple(self._states)
        records: list[SnapshotRecord] = []
        for camera_id, track_id in keys:
            record = self.finalize_track(camera_id, track_id)
            if record is not None:
                records.append(record)
        return tuple(records)

    def clear_track(self, camera_id: str, track_id: TrackId) -> None:
        """Release dedupe state after a tracker declares an ID permanently gone."""

        key = (camera_id, track_id)
        with self._lock:
            self._states.pop(key, None)
            self._finalized_tracks.discard(key)
            self._missing_tracks.discard(key)
            self._behavior_last = {
                saved: timestamp
                for saved, timestamp in self._behavior_last.items()
                if saved[:2] != key
            }

    @property
    def pending_track_count(self) -> int:
        with self._lock:
            return len(self._states)

    @property
    def created_track_count(self) -> int:
        with self._lock:
            return self._created_count

    @property
    def finalized_track_count(self) -> int:
        with self._lock:
            return self._finalized_count

    @property
    def evidence_missing_count(self) -> int:
        with self._lock:
            return self._missing_count

    def state_for(self, camera_id: str, track_id: TrackId) -> TrackEvidenceState | None:
        """Expose one state for diagnostics and deterministic tests."""

        with self._lock:
            return self._states.get((camera_id, track_id))

    def summary(self) -> EvidenceSummary:
        with self._lock:
            # At the only production emission point all tracks must already be
            # terminal.  Include any still-unfinalized created lifecycle in the
            # missing count so it can never disappear silently from the ledger.
            evidence_missing = max(
                self._missing_count,
                self._created_count - self._snapshot_success_count,
            )
            return EvidenceSummary(
                person_tracks_created=self._created_count,
                tracks_finalized=self._finalized_count,
                best_face_evidence=self._source_counts["best_face"],
                face_fallback_evidence=self._source_counts["face_fallback"],
                best_person_evidence=self._source_counts["best_person"],
                person_fallback_evidence=self._source_counts["person_fallback"],
                know=self._known_count,
                unknown=self._unknown_count,
                snapshot_success=self._snapshot_success_count,
                snapshot_failed=self._snapshot_failed_count,
                evidence_missing=evidence_missing,
            )

    def log_summary(self) -> EvidenceSummary:
        with self._lock:
            if self._summary_emitted:
                return self.summary()
            self._summary_emitted = True
        summary = self.summary()
        LOGGER.info(
            "\n========== Evidence Summary ==========\n"
            "Person Tracks Created:       %d\n"
            "Tracks Finalized:            %d\n"
            "Best Face Evidence:          %d\n"
            "Face Fallback Evidence:      %d\n"
            "Best Person Evidence:        %d\n"
            "Person Fallback Evidence:    %d\n"
            "Know:                        %d\n"
            "Unknown:                     %d\n"
            "Snapshot Success:            %d\n"
            "Snapshot Failed:             %d\n"
            "Evidence Missing:            %d\n"
            "======================================",
            summary.person_tracks_created,
            summary.tracks_finalized,
            summary.best_face_evidence,
            summary.face_fallback_evidence,
            summary.best_person_evidence,
            summary.person_fallback_evidence,
            summary.know,
            summary.unknown,
            summary.snapshot_success,
            summary.snapshot_failed,
            summary.evidence_missing,
        )
        source_total = (
            summary.best_face_evidence
            + summary.face_fallback_evidence
            + summary.best_person_evidence
            + summary.person_fallback_evidence
        )
        face_total = summary.best_face_evidence + summary.face_fallback_evidence
        if (
            summary.person_tracks_created != summary.tracks_finalized
            or summary.tracks_finalized != summary.snapshot_success
            or source_total != summary.snapshot_success
            or summary.know + summary.unknown != face_total
            or summary.evidence_missing != 0
        ):
            LOGGER.error(
                "[ERROR][EVIDENCE_MISSING] summary created=%d tracks_finalized=%d "
                "snapshot_success=%d source_total=%d face_total=%d identities=%d "
                "snapshot_failed=%d evidence_missing=%d",
                summary.person_tracks_created,
                summary.tracks_finalized,
                summary.snapshot_success,
                source_total,
                face_total,
                summary.know + summary.unknown,
                summary.snapshot_failed,
                summary.evidence_missing,
            )
        return summary

    def _person_quality(self, crop: np.ndarray, frame: np.ndarray, track: Track) -> float:
        gray = _gray(crop, self.config.frame_color_space)
        if min(gray.shape) >= 3:
            center = gray[1:-1, 1:-1]
            laplacian = (
                gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
            )
            sharpness = min(1.0, float(np.var(laplacian)) / self.config.sharpness_reference)
        else:
            sharpness = 0.0
        frame_area = max(1, int(frame.shape[0]) * int(frame.shape[1]))
        size = min(1.0, crop.shape[0] * crop.shape[1] / (frame_area * 0.25))
        return _quality(0.45 * sharpness + 0.35 * size + 0.20 * track.confidence)

    def _write(self, relative: Path, image: np.ndarray) -> Path:
        payload = self.encoder.encode(image, quality=self.config.jpeg_quality)
        if not isinstance(payload, bytes) or not payload:
            raise SnapshotEncodingError("image encoder must return non-empty bytes")
        return self.store.write(relative, payload)

    @staticmethod
    def _filename(
        camera_id: str,
        track_id: TrackId,
        timestamp: datetime,
        event: str,
        worker_id: str | None = None,
    ) -> str:
        parts = [
            _safe_component(camera_id),
            f"track-{_safe_component(str(track_id))}",
            timestamp.strftime("%Y%m%dT%H%M%S_%f"),
            _safe_component(event),
        ]
        if worker_id:
            parts.append(_safe_component(worker_id))
        return "_".join(parts) + ".jpg"

    @staticmethod
    def _evidence_filename(
        camera_id: str,
        track_id: TrackId,
        timestamp: datetime,
        worker_id: str | None,
        similarity: float,
        quality: float,
    ) -> str:
        parts = [
            timestamp.strftime("%Y%m%dT%H%M%S_%f"),
            _safe_component(camera_id),
            f"track-{_safe_component(str(track_id))}",
            _safe_component(worker_id or "unknown"),
            f"sim{float(similarity):.3f}",
            f"q{_quality(quality):.3f}",
        ]
        return "_".join(parts) + ".jpg"


def _validated_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.size == 0:
        raise SnapshotEncodingError("snapshot image is empty")
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (1, 3, 4)):
        raise SnapshotEncodingError("snapshot must be HxW, HxWx1, HxWx3, or HxWx4")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise SnapshotEncodingError("snapshot contains invalid pixels")
    if array.dtype != np.uint8:
        working = array.astype(np.float32)
        if np.issubdtype(array.dtype, np.floating) and working.max(initial=0.0) <= 1.0:
            working *= 255.0
        array = np.clip(working, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _crop(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    array = _frame_array(frame)
    rows, columns = bbox.integer_slices(array.shape[1], array.shape[0])
    crop = array[rows, columns]
    if crop.size == 0:
        raise SnapshotEncodingError("snapshot crop is empty")
    return crop


def _frame_array(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.size == 0:
        raise SnapshotEncodingError("snapshot image is empty")
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (1, 3, 4)):
        raise SnapshotEncodingError("snapshot must be HxW, HxWx1, HxWx3, or HxWx4")
    if not np.issubdtype(array.dtype, np.number):
        raise SnapshotEncodingError("snapshot pixels must be numeric")
    return array


def _evidence_crop(
    frame: np.ndarray,
    person_bbox: BoundingBox,
    config: SnapshotConfig,
    *,
    face_bbox: BoundingBox | None = None,
    upper_body: bool = False,
) -> tuple[np.ndarray, BoundingBox, bool]:
    """Create one owned ROI, falling back to the full frame on unsafe geometry."""

    array = _frame_array(frame)
    height, width = array.shape[:2]
    frame_bbox = BoundingBox(0, 0, width, height)
    person = person_bbox.clipped(width, height)
    if person is None:
        return np.array(array, copy=True, order="C"), frame_bbox, True
    visible_ratio = person.area / person_bbox.area
    if (
        visible_ratio < config.min_visible_ratio
        or person.width < config.min_person_crop_width
        or person.height < config.min_person_crop_height
    ):
        return np.array(array, copy=True, order="C"), frame_bbox, True

    evidence_bbox = person
    if upper_body:
        left = person.x1 - person.width * config.padding_x_ratio
        top = person.y1 - person.height * config.padding_top_ratio
        right = person.x2 + person.width * config.padding_x_ratio
        bottom = person.y1 + person.height * config.upper_body_fraction
        if face_bbox is not None:
            # A partially/outside-frame face cannot be guaranteed inside an ROI.
            if not _contains(frame_bbox, face_bbox):
                return np.array(array, copy=True, order="C"), frame_bbox, True
            left = min(left, face_bbox.x1)
            top = min(top, face_bbox.y1)
            right = max(right, face_bbox.x2)
            bottom = max(bottom, face_bbox.y2)
        try:
            proposed = BoundingBox(left, top, right, bottom)
        except ValueError:
            return np.array(array, copy=True, order="C"), frame_bbox, True
        clipped = proposed.clipped(width, height)
        if clipped is None:
            return np.array(array, copy=True, order="C"), frame_bbox, True
        evidence_bbox = clipped

    if (
        evidence_bbox.width < config.min_person_crop_width
        or evidence_bbox.height < config.min_person_crop_height
        or (face_bbox is not None and not _contains(evidence_bbox, face_bbox))
    ):
        return np.array(array, copy=True, order="C"), frame_bbox, True
    return np.ascontiguousarray(_crop(array, evidence_bbox)), evidence_bbox, False


def _optional_crop(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    array = _frame_array(frame)
    clipped = bbox.clipped(array.shape[1], array.shape[0])
    if clipped is None:
        return None
    return np.ascontiguousarray(_crop(array, clipped))


def _contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    return (
        outer.x1 <= inner.x1
        and outer.y1 <= inner.y1
        and outer.x2 >= inner.x2
        and outer.y2 >= inner.y2
    )


def _gray(image: np.ndarray, color_space: str = "rgba") -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    if image.shape[2] == 1:
        return image[..., 0].astype(np.float32, copy=False)
    pixels = image[..., :3].astype(np.float32, copy=False)
    if color_space.startswith("bgr"):
        return 0.114 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.299 * pixels[..., 2]
    return 0.299 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.114 * pixels[..., 2]


def _quality(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("quality must be between 0 and 1")
    return value


def _seconds_between(later: datetime, earlier: datetime) -> float:
    if later.tzinfo is None and earlier.tzinfo is not None:
        later = later.replace(tzinfo=earlier.tzinfo)
    elif later.tzinfo is not None and earlier.tzinfo is None:
        earlier = earlier.replace(tzinfo=later.tzinfo)
    return (later - earlier).total_seconds()


def _safe_component(value: str) -> str:
    original = value.strip()
    cleaned = _UNSAFE_FILENAME.sub("-", original).strip(" .")
    reserved = cleaned.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
    changed = cleaned != original or len(cleaned) > 80 or reserved or not cleaned
    if reserved:
        cleaned = f"_{cleaned}"
    if not cleaned:
        cleaned = "unknown"
    if changed:
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
        cleaned = cleaned[:71].rstrip(" .-_") or "unknown"
        cleaned = f"{cleaned}-{digest}"
    return cleaned[:80]


__all__ = [
    "EvidenceCandidate",
    "EvidenceSummary",
    "EventSnapshotManager",
    "FilesystemSnapshotStore",
    "ImageEncoder",
    "JpegImageEncoder",
    "SnapshotConfig",
    "SnapshotEncodingError",
    "SnapshotError",
    "SnapshotKind",
    "SnapshotRecord",
    "SnapshotStore",
    "SnapshotWriteError",
    "TrackEvidenceState",
]
