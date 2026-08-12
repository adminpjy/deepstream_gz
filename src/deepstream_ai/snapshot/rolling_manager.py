"""Rolling per-track evidence persistence.

The policy is intentionally simple:
- persist a person snapshot as soon as a valid tracked person is observed;
- keep exactly one stable person file per (camera_id, track_id);
- when a face is observed, replace the person file with the same-frame upper-body crop
  and keep exactly one stable face file for that track;
- later, clearer faces overwrite both files together;
- identity and evidence selection remain independent.

This makes snapshots visible while a track is still alive and prevents a track from
finishing without evidence merely because no high-quality face was found.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import RLock

import numpy as np

from deepstream_ai.domain import BehaviorDetection, BoundingBox, FaceDetection, IdentityResult, Track, TrackId
from deepstream_ai.snapshot.manager import (
    EvidenceCandidate,
    EvidenceSummary,
    FilesystemSnapshotStore,
    ImageEncoder,
    JpegImageEncoder,
    SnapshotConfig,
    SnapshotKind,
    SnapshotRecord,
    SnapshotStore,
    TrackEvidenceState,
)

LOGGER = logging.getLogger(__name__)
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class EventSnapshotManager:
    """Keep one rolling person/face evidence pair for every active track."""

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
        self._finalized: set[tuple[str, TrackId]] = set()
        self._behavior_last: dict[tuple[str, TrackId, str], datetime] = {}
        self._created = 0
        self._finalized_count = 0
        self._snapshot_success = 0
        self._snapshot_failed = 0
        self._missing = 0
        self._source_counts = {
            "best_face": 0,
            "face_fallback": 0,
            "best_person": 0,
            "person_fallback": 0,
        }
        self._known = 0
        self._unknown = 0
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
        del has_face
        key = track.key
        crop_result = _person_crop(frame, track.bbox, self.config)
        if crop_result is None:
            LOGGER.warning(
                "[PERSON_SNAPSHOT_SKIP] camera=%s track=%s reason=invalid_bbox bbox=%s",
                track.camera_id,
                track.track_id,
                track.bbox.as_tuple(),
            )
            return False
        crop, evidence_bbox = crop_result
        candidate_quality = _clamp01(quality) if quality is not None else _person_quality(
            crop, frame, track, self.config
        )
        candidate = EvidenceCandidate(
            camera_id=track.camera_id,
            track_id=track.track_id,
            timestamp=track.timestamp,
            person_bbox=track.bbox,
            face_bbox=None,
            evidence_bbox=evidence_bbox,
            person_crop=crop,
            face_crop=None,
            full_frame_fallback=False,
            worker_id=None,
            similarity=-1.0,
            quality_score=candidate_quality,
        )

        with self._lock:
            state = self._states.get(key)
            if state is None:
                if key in self._finalized:
                    return False
                state = TrackEvidenceState(
                    camera_id=track.camera_id,
                    track_id=track.track_id,
                    last_seen=track.timestamp,
                    person_fallback=candidate,
                )
                self._states[key] = state
                self._created += 1
                should_write = True
                source = "person_fallback"
                LOGGER.info(
                    "[TRACK_CREATE] camera=%s track=%s person_conf=%.3f quality=%.3f",
                    track.camera_id,
                    track.track_id,
                    track.confidence,
                    candidate_quality,
                )
            else:
                state.last_seen = track.timestamp
                # Once a face exists, person evidence must stay synchronized to
                # the selected face frame; later person-only frames must not overwrite it.
                if state.face_fallback is not None or state.best_face is not None:
                    return False
                baseline = max(
                    state.person_fallback.quality_score if state.person_fallback else -1.0,
                    state.best_person.quality_score if state.best_person else -1.0,
                )
                if candidate_quality <= baseline:
                    return False
                state.best_person = candidate
                should_write = True
                source = "best_person"

        if should_write:
            self._persist_person(candidate)
            LOGGER.info(
                "[PERSON_EVIDENCE_UPDATE] camera=%s track=%s source=%s quality=%.3f",
                track.camera_id,
                track.track_id,
                source,
                candidate_quality,
            )
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
        del known
        if face.key != track.key:
            raise ValueError("face and person track keys do not match")
        key = track.key
        with self._lock:
            state = self._states.get(key)
        if state is None:
            self.observe_person(frame, track)
            with self._lock:
                state = self._states.get(key)
            if state is None:
                return False

        pair = _face_pair_crop(frame, track.bbox, face.bbox, self.config)
        if pair is None:
            LOGGER.warning(
                "[FACE_SNAPSHOT_SKIP] camera=%s track=%s reason=invalid_geometry face=%s person=%s",
                face.camera_id,
                face.track_id,
                face.bbox.as_tuple(),
                track.bbox.as_tuple(),
            )
            return False
        person_crop, face_crop, evidence_bbox = pair
        computed_quality = _face_quality(face_crop, face, frame)
        # The caller's quality remains a useful hint, but never replaces image
        # sharpness. Blend them so a clear later frame can beat an early detector-only score.
        if quality is not None:
            computed_quality = _clamp01(0.75 * computed_quality + 0.25 * _clamp01(quality))

        current_identity = identity or state.identity_result
        candidate = EvidenceCandidate(
            camera_id=track.camera_id,
            track_id=track.track_id,
            timestamp=face.timestamp,
            person_bbox=track.bbox,
            face_bbox=face.bbox,
            evidence_bbox=evidence_bbox,
            person_crop=person_crop,
            face_crop=face_crop,
            full_frame_fallback=False,
            worker_id=current_identity.worker_id if current_identity else None,
            similarity=current_identity.similarity if current_identity else -1.0,
            quality_score=computed_quality,
        )

        with self._lock:
            state = self._states[key]
            state.last_seen = face.timestamp
            if identity is not None:
                state.identity_result = identity
            baseline = max(
                state.face_fallback.quality_score if state.face_fallback else -1.0,
                state.best_face.quality_score if state.best_face else -1.0,
            )
            first_face = state.face_fallback is None
            if not first_face and computed_quality <= baseline:
                return False
            if first_face:
                state.face_fallback = candidate
                source = "face_fallback"
                LOGGER.info(
                    "[FACE_FALLBACK] camera=%s track=%s face_conf=%.3f quality=%.3f",
                    face.camera_id,
                    face.track_id,
                    face.score,
                    computed_quality,
                )
            else:
                state.best_face = candidate
                source = "best_face"
                LOGGER.info(
                    "[BEST_FACE_UPDATE] camera=%s track=%s old_quality=%.3f new_quality=%.3f",
                    face.camera_id,
                    face.track_id,
                    baseline,
                    computed_quality,
                )

        # The pair is always written together from the same frame.
        self._persist_person(candidate)
        self._persist_face(candidate, current_identity)
        LOGGER.info(
            "[FACE_EVIDENCE_UPDATE] camera=%s track=%s source=%s quality=%.3f",
            face.camera_id,
            face.track_id,
            source,
            computed_quality,
        )
        return True

    def observe_identity(self, identity: IdentityResult) -> bool:
        key = (identity.camera_id, identity.track_id)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return False
            previous = state.identity_result
            state.identity_result = identity
            selected = state.best_face or state.face_fallback
        if selected is not None and (
            previous is None
            or previous.worker_id != identity.worker_id
            or abs(previous.similarity - identity.similarity) > 1e-6
        ):
            self._persist_face(selected, identity)
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
            previous = self._behavior_last.get(key)
            if previous is not None and (detection.timestamp - previous).total_seconds() < self.config.behavior_cooldown_seconds:
                return None
        crop = _crop_box(frame, crop_bbox or detection.bbox)
        if crop is None:
            return None
        relative = Path("behavior") / f"{_base_name(detection.camera_id, detection.track_id)}_{_safe(detection.behavior.value)}.jpg"
        path = self._write(relative, crop)
        with self._lock:
            self._behavior_last[key] = detection.timestamp
        return SnapshotRecord(
            kind=SnapshotKind.BEHAVIOR,
            camera_id=detection.camera_id,
            track_id=detection.track_id,
            timestamp=detection.timestamp,
            path=path,
            behavior=detection.behavior.value,
        )

    def finalize_track(self, camera_id: str, track_id: TrackId) -> SnapshotRecord | None:
        key = (camera_id, track_id)
        with self._lock:
            if key in self._finalized:
                return None
            state = self._states.get(key)
            self._finalized_count += 1
            if state is None:
                self._missing += 1
                self._snapshot_failed += 1
                LOGGER.error("[ERROR][EVIDENCE_MISSING] camera=%s track=%s reason=no_state", camera_id, track_id)
                return None
            selected = state.selected()
            if selected is None:
                self._missing += 1
                self._snapshot_failed += 1
                LOGGER.error("[ERROR][EVIDENCE_MISSING] camera=%s track=%s reason=no_candidate", camera_id, track_id)
                return None
            source, candidate = selected
            identity = state.identity_result
            self._finalized.add(key)
            self._snapshot_success += 1
            self._source_counts[source] += 1
            if source in {"best_face", "face_fallback"}:
                if identity is not None and identity.known:
                    self._known += 1
                else:
                    self._unknown += 1

        # Ensure the latest identity category is reflected on disk at finalization.
        if source in {"best_face", "face_fallback"}:
            self._persist_person(candidate)
            self._persist_face(candidate, identity)
            kind = SnapshotKind.FACE_KNOWN if identity is not None and identity.known else SnapshotKind.FACE_UNKNOWN
            path = self._face_path(candidate, identity)
        else:
            self._persist_person(candidate)
            kind = SnapshotKind.PERSON
            path = self._person_path(candidate)

        LOGGER.info(
            "[TRACK_FINALIZE] camera=%s track=%s source=%s identity=%s similarity=%.3f quality=%.3f",
            camera_id,
            track_id,
            source,
            identity.worker_id if identity and identity.known else "unknown",
            identity.similarity if identity else -1.0,
            candidate.quality_score,
        )
        return SnapshotRecord(
            kind=kind,
            camera_id=camera_id,
            track_id=track_id,
            timestamp=candidate.timestamp,
            path=path,
            quality=candidate.quality_score,
            worker_id=identity.worker_id if identity and identity.known else None,
            similarity=identity.similarity if identity else -1.0,
            source=source,
            person_bbox=candidate.person_bbox,
            face_bbox=candidate.face_bbox,
            evidence_bbox=candidate.evidence_bbox,
            full_frame_fallback=False,
        )

    def expire_tracks(self, now: datetime) -> tuple[SnapshotRecord, ...]:
        with self._lock:
            keys = [
                key for key, state in self._states.items()
                if (now - state.last_seen).total_seconds() >= self.config.track_ttl_seconds
            ]
        return tuple(filter(None, (self.finalize_track(*key) for key in keys)))

    def finalize_all(self) -> tuple[SnapshotRecord, ...]:
        with self._lock:
            keys = tuple(self._states)
        return tuple(filter(None, (self.finalize_track(*key) for key in keys)))

    def clear_track(self, camera_id: str, track_id: TrackId) -> None:
        key = (camera_id, track_id)
        with self._lock:
            self._states.pop(key, None)
            self._finalized.discard(key)
            self._behavior_last = {k: v for k, v in self._behavior_last.items() if k[:2] != key}

    def state_for(self, camera_id: str, track_id: TrackId) -> TrackEvidenceState | None:
        with self._lock:
            return self._states.get((camera_id, track_id))

    @property
    def pending_track_count(self) -> int:
        with self._lock:
            return len(self._states)

    def summary(self) -> EvidenceSummary:
        with self._lock:
            return EvidenceSummary(
                person_tracks_created=self._created,
                tracks_finalized=self._finalized_count,
                best_face_evidence=self._source_counts["best_face"],
                face_fallback_evidence=self._source_counts["face_fallback"],
                best_person_evidence=self._source_counts["best_person"],
                person_fallback_evidence=self._source_counts["person_fallback"],
                know=self._known,
                unknown=self._unknown,
                snapshot_success=self._snapshot_success,
                snapshot_failed=self._snapshot_failed,
                evidence_missing=self._missing,
            )

    def log_summary(self) -> EvidenceSummary:
        with self._lock:
            if self._summary_emitted:
                return self.summary()
            self._summary_emitted = True
        summary = self.summary()
        LOGGER.info(
            "Evidence Summary tracks=%d finalized=%d success=%d failed=%d missing=%d best_face=%d face_fallback=%d best_person=%d person_fallback=%d know=%d unknown=%d",
            summary.person_tracks_created,
            summary.tracks_finalized,
            summary.snapshot_success,
            summary.snapshot_failed,
            summary.evidence_missing,
            summary.best_face_evidence,
            summary.face_fallback_evidence,
            summary.best_person_evidence,
            summary.person_fallback_evidence,
            summary.know,
            summary.unknown,
        )
        return summary

    def _persist_person(self, candidate: EvidenceCandidate) -> Path:
        return self._write(self._person_relative(candidate), candidate.person_crop)

    def _persist_face(self, candidate: EvidenceCandidate, identity: IdentityResult | None) -> Path:
        if candidate.face_crop is None:
            raise ValueError("face candidate has no face crop")
        target = self._face_relative(candidate, identity)
        path = self._write(target, candidate.face_crop)
        self._remove_other_face_category(candidate, target)
        return path

    def _remove_other_face_category(self, candidate: EvidenceCandidate, keep: Path) -> None:
        root = getattr(self.store, "root_dir", None)
        if root is None:
            return
        for category in ("know", "unknow"):
            relative = Path("face") / category / f"{_base_name(candidate.camera_id, candidate.track_id)}.jpg"
            if relative == keep:
                continue
            try:
                (Path(root) / relative).unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Unable to remove stale face evidence: %s", relative)

    def _person_relative(self, candidate: EvidenceCandidate) -> Path:
        return Path("person") / f"{_base_name(candidate.camera_id, candidate.track_id)}.jpg"

    def _face_relative(self, candidate: EvidenceCandidate, identity: IdentityResult | None) -> Path:
        category = "know" if identity is not None and identity.known else "unknow"
        return Path("face") / category / f"{_base_name(candidate.camera_id, candidate.track_id)}.jpg"

    def _person_path(self, candidate: EvidenceCandidate) -> Path:
        root = getattr(self.store, "root_dir", self.config.root_dir)
        return Path(root) / self._person_relative(candidate)

    def _face_path(self, candidate: EvidenceCandidate, identity: IdentityResult | None) -> Path:
        root = getattr(self.store, "root_dir", self.config.root_dir)
        return Path(root) / self._face_relative(candidate, identity)

    def _write(self, relative: Path, image: np.ndarray) -> Path:
        payload = self.encoder.encode(image, quality=self.config.jpeg_quality)
        return self.store.write(relative, payload)


def _base_name(camera_id: str, track_id: TrackId) -> str:
    return f"{_safe(camera_id)}_track-{_safe(str(track_id))}"


def _safe(value: str) -> str:
    cleaned = _SAFE.sub("-", value.strip()).strip(".-_")
    return (cleaned or "unknown")[:80]


def _clamp01(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _crop_box(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    array = np.asarray(frame)
    if array.ndim not in (2, 3) or array.size == 0:
        return None
    clipped = bbox.clipped(array.shape[1], array.shape[0])
    if clipped is None:
        return None
    rows, cols = clipped.integer_slices(array.shape[1], array.shape[0])
    crop = np.ascontiguousarray(array[rows, cols])
    return crop if crop.size else None


def _person_crop(
    frame: np.ndarray,
    bbox: BoundingBox,
    config: SnapshotConfig,
) -> tuple[np.ndarray, BoundingBox] | None:
    array = np.asarray(frame)
    if array.size == 0:
        return None
    clipped = bbox.clipped(array.shape[1], array.shape[0])
    if clipped is None:
        return None
    # Never fall back to the entire frame for a Person snapshot. The previous
    # behavior could save a rack/cabinet scene when geometry was marginal.
    if clipped.width < config.min_person_crop_width or clipped.height < config.min_person_crop_height:
        return None
    crop = _crop_box(array, clipped)
    if crop is None:
        return None
    return crop, clipped


def _face_pair_crop(
    frame: np.ndarray,
    person_bbox: BoundingBox,
    face_bbox: BoundingBox,
    config: SnapshotConfig,
) -> tuple[np.ndarray, np.ndarray, BoundingBox] | None:
    array = np.asarray(frame)
    if array.size == 0:
        return None
    height, width = array.shape[:2]
    person = person_bbox.clipped(width, height)
    face = face_bbox.clipped(width, height)
    if person is None or face is None:
        return None
    if face.width < 8 or face.height < 8:
        return None

    left = min(person.x1 - person.width * config.padding_x_ratio, face.x1)
    top = min(person.y1 - person.height * config.padding_top_ratio, face.y1)
    right = max(person.x2 + person.width * config.padding_x_ratio, face.x2)
    bottom = max(person.y1 + person.height * config.upper_body_fraction, face.y2)
    try:
        evidence = BoundingBox(left, top, right, bottom).clipped(width, height)
    except ValueError:
        return None
    if evidence is None:
        return None
    person_crop = _crop_box(array, evidence)
    face_crop = _crop_box(array, face)
    if person_crop is None or face_crop is None:
        return None
    return person_crop, face_crop, evidence


def _sharpness(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    pixels = image[..., :3].astype(np.float32, copy=False) if image.ndim == 3 else image.astype(np.float32, copy=False)
    if pixels.ndim == 3:
        gray = 0.299 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.114 * pixels[..., 2]
    else:
        gray = pixels
    if min(gray.shape[:2]) < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    # A soft reference prevents moderate CCTV faces from saturating too early.
    return _clamp01(float(np.var(lap)) / 160.0)


def _frontal_score(face: FaceDetection) -> float | None:
    if len(face.landmarks) < 5:
        return None
    left_eye, right_eye, nose, left_mouth, right_mouth = face.landmarks[:5]
    eye_dx = max(1e-6, abs(right_eye[0] - left_eye[0]))
    eye_level = 1.0 - min(1.0, abs(right_eye[1] - left_eye[1]) / (eye_dx * 0.35))
    eye_center_x = (left_eye[0] + right_eye[0]) * 0.5
    nose_center = 1.0 - min(1.0, abs(nose[0] - eye_center_x) / (eye_dx * 0.45))
    mouth_center_x = (left_mouth[0] + right_mouth[0]) * 0.5
    mouth_center = 1.0 - min(1.0, abs(mouth_center_x - eye_center_x) / (eye_dx * 0.55))
    return _clamp01((eye_level + nose_center + mouth_center) / 3.0)


def _face_quality(face_crop: np.ndarray, face: FaceDetection, frame: np.ndarray) -> float:
    sharp = _sharpness(face_crop)
    frame_area = max(1.0, float(frame.shape[0] * frame.shape[1]))
    size = _clamp01(face.bbox.area / (frame_area * 0.018))
    frontal = _frontal_score(face)
    if frontal is None:
        return _clamp01(0.45 * sharp + 0.40 * face.score + 0.15 * size)
    return _clamp01(0.40 * sharp + 0.30 * face.score + 0.20 * frontal + 0.10 * size)


def _person_quality(crop: np.ndarray, frame: np.ndarray, track: Track, config: SnapshotConfig) -> float:
    sharp = _sharpness(crop)
    frame_area = max(1.0, float(frame.shape[0] * frame.shape[1]))
    size = _clamp01(track.bbox.area / (frame_area * 0.22))
    return _clamp01(0.45 * sharp + 0.35 * size + 0.20 * track.confidence)


__all__ = ["EventSnapshotManager"]
