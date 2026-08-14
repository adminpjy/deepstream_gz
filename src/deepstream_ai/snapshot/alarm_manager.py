"""Alarm-friendly rolling snapshot presentation.

AdaFace and snapshot evidence use the same face-quality scorer on the raw SCRFD
ROI, while the persisted face image includes surrounding head/neck/shoulder
context. Person and face evidence always come from the same winning frame.

Person-only evidence is anchored to the first valid tracked-person bbox so an
NvDCF shadow box cannot gradually overwrite a human with a sharp rack/cabinet.
When a face exists, the user-facing person evidence is anchored around that face
rather than trusting the full current tracker bbox.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np

from deepstream_ai.domain import BoundingBox, FaceDetection, IdentityResult, Track
from deepstream_ai.face.quality import FaceFusionConfig, FaceQualityScorer
from deepstream_ai.snapshot.manager import EvidenceCandidate
from deepstream_ai.snapshot.rolling_manager import EventSnapshotManager as RollingEventSnapshotManager
from deepstream_ai.snapshot import rolling_manager as rolling

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceEvidencePolicy:
    """Small evidence-only policy; it never changes detection or tracking IDs."""

    quality_upgrade_margin: float = 0.005
    frontal_override_delta: float = 0.10
    frontal_quality_tolerance: float = 0.06
    stable_frontal_min: float = 0.82
    stable_blur_min: float = 0.45
    stable_detector_min: float = 0.65
    stable_required: int = 2
    stable_quality_tolerance: float = 0.04
    stable_max_gap_sec: float = 0.80
    hold_log_interval_sec: float = 2.0


@dataclass(frozen=True, slots=True)
class _FaceEvidenceMetrics:
    quality: float
    detector: float
    size: float
    blur: float
    frontal: float


@dataclass(slots=True)
class _FrontalStability:
    last_timestamp: datetime
    last_bbox: BoundingBox
    streak: int


class EventSnapshotManager(RollingEventSnapshotManager):
    """Rolling manager with stable person evidence and quality-first face upgrades."""

    def __init__(self, *args, face_policy: FaceEvidencePolicy | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.face_policy = face_policy or FaceEvidencePolicy()
        # Match FaceRecognitionService's default FaceQualityScorer exactly in
        # production: score + size + blur + frontal, with the same references
        # and frame color space. The old snapshot-only score is intentionally
        # no longer used for real five-landmark SCRFD detections.
        self._face_scorer = FaceQualityScorer(
            FaceFusionConfig(frame_color_space=self.config.frame_color_space)
        )
        self._selected_face_metrics: dict[tuple[str, object], _FaceEvidenceMetrics] = {}
        self._frontal_stability: dict[tuple[str, object], _FrontalStability] = {}
        self._last_hold_log: dict[tuple[str, object], datetime] = {}

    def observe_person(
        self,
        frame: np.ndarray,
        track: Track,
        *,
        has_face: bool = False,
        quality: float | None = None,
    ) -> bool:
        """Persist the first person immediately, but do not let NvDCF drift overwrite it."""

        key = track.key
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                state.last_seen = track.timestamp
                # Once face evidence exists, only a better face may replace the
                # synchronized person/face pair.
                if state.face_fallback is not None or state.best_face is not None:
                    return False
                anchor = state.person_fallback
            else:
                anchor = None

        # First observation for this track: preserve the zero-miss rule and
        # write it immediately.
        if state is None or anchor is None:
            return super().observe_person(
                frame,
                track,
                has_face=has_face,
                quality=quality,
            )

        # Subsequent person-only frames may improve the original evidence, but
        # must stay close in scale/location to the first valid person bbox.
        if not _person_geometry_consistent(anchor.person_bbox, track.bbox):
            LOGGER.info(
                "[PERSON_EVIDENCE_HOLD] camera=%s track=%s reason=geometry_drift "
                "anchor=%s current=%s",
                track.camera_id,
                track.track_id,
                anchor.person_bbox.as_tuple(),
                track.bbox.as_tuple(),
            )
            return False

        return super().observe_person(
            frame,
            track,
            has_face=has_face,
            quality=quality,
        )

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

        pair = _alarm_face_pair_crop(frame, track.bbox, face.bbox, self.config)
        if pair is None:
            LOGGER.warning(
                "[FACE_SNAPSHOT_SKIP] camera=%s track=%s reason=invalid_geometry face=%s person=%s",
                face.camera_id,
                face.track_id,
                face.bbox.as_tuple(),
                track.bbox.as_tuple(),
            )
            return False

        person_crop, display_face_crop, raw_face_crop, evidence_bbox = pair
        scored = self._face_scorer.score(
            face,
            crop=raw_face_crop,
            frame_shape=frame.shape,
        )
        metrics = _FaceEvidenceMetrics(
            quality=scored.quality,
            detector=face.score,
            size=scored.size_score,
            blur=scored.blur_score,
            frontal=scored.frontal_score,
        )
        # Preserve the old unit-test/legacy hook only for detections that do not
        # carry the five SCRFD landmarks. Real production SCRFD faces always use
        # the unified scorer above, so the old caller-side ad-hoc score cannot
        # lock in an early obscured face anymore.
        if quality is not None and len(face.landmarks) < 5:
            metrics = replace(metrics, quality=rolling._clamp01(quality))

        stable_count = self._update_frontal_stability(key, face, metrics)
        current_identity = identity or state.identity_result
        candidate = EvidenceCandidate(
            camera_id=track.camera_id,
            track_id=track.track_id,
            timestamp=face.timestamp,
            person_bbox=track.bbox,
            face_bbox=face.bbox,
            evidence_bbox=evidence_bbox,
            person_crop=person_crop,
            # Persist the expanded alarm-display crop, not the tiny model ROI.
            face_crop=display_face_crop,
            full_frame_fallback=False,
            worker_id=current_identity.worker_id if current_identity else None,
            similarity=current_identity.similarity if current_identity else -1.0,
            quality_score=metrics.quality,
        )

        with self._lock:
            state = self._states[key]
            state.last_seen = face.timestamp
            if identity is not None:
                state.identity_result = identity
            baseline_candidate = state.best_face or state.face_fallback
            baseline_metrics = self._selected_face_metrics.get(key)
            first_face = state.face_fallback is None

            reason: str | None
            if first_face:
                reason = "first_face"
            else:
                if baseline_metrics is None and baseline_candidate is not None:
                    baseline_metrics = _FaceEvidenceMetrics(
                        quality=baseline_candidate.quality_score,
                        detector=0.0,
                        size=0.0,
                        blur=0.0,
                        frontal=0.5,
                    )
                reason = _upgrade_reason(
                    baseline_metrics,
                    metrics,
                    stable_count=stable_count,
                    policy=self.face_policy,
                )

            if reason is None:
                self._log_face_hold(
                    key,
                    face,
                    baseline_metrics,
                    metrics,
                    stable_count,
                )
                return False

            old_quality = baseline_candidate.quality_score if baseline_candidate is not None else -1.0
            if first_face:
                state.face_fallback = candidate
                source = "face_fallback"
                LOGGER.info(
                    "[FACE_FALLBACK] camera=%s track=%s face_conf=%.3f quality=%.3f "
                    "frontal=%.3f blur=%.3f size=%.3f person_display=%sx%s face_display=%sx%s",
                    face.camera_id,
                    face.track_id,
                    face.score,
                    metrics.quality,
                    metrics.frontal,
                    metrics.blur,
                    metrics.size,
                    person_crop.shape[1],
                    person_crop.shape[0],
                    display_face_crop.shape[1],
                    display_face_crop.shape[0],
                )
            else:
                state.best_face = candidate
                source = "best_face"
                LOGGER.info(
                    "[BEST_FACE_UPDATE] camera=%s track=%s reason=%s old_quality=%.3f new_quality=%.3f "
                    "new_frontal=%.3f new_blur=%.3f new_size=%.3f new_detector=%.3f stable=%d "
                    "person_display=%sx%s face_display=%sx%s",
                    face.camera_id,
                    face.track_id,
                    reason,
                    old_quality,
                    metrics.quality,
                    metrics.frontal,
                    metrics.blur,
                    metrics.size,
                    metrics.detector,
                    stable_count,
                    person_crop.shape[1],
                    person_crop.shape[0],
                    display_face_crop.shape[1],
                    display_face_crop.shape[0],
                )
            self._selected_face_metrics[key] = metrics

        # The pair is always written together from the same winning frame.
        self._persist_person(candidate)
        self._persist_face(candidate, current_identity)
        LOGGER.info(
            "[FACE_EVIDENCE_UPDATE] camera=%s track=%s source=%s quality=%.3f frontal=%.3f "
            "blur=%.3f evidence=%s",
            face.camera_id,
            face.track_id,
            source,
            metrics.quality,
            metrics.frontal,
            metrics.blur,
            evidence_bbox.as_tuple(),
        )
        return True

    def clear_track(self, camera_id, track_id) -> None:
        key = (camera_id, track_id)
        super().clear_track(camera_id, track_id)
        with self._lock:
            self._selected_face_metrics.pop(key, None)
            self._frontal_stability.pop(key, None)
            self._last_hold_log.pop(key, None)

    def _update_frontal_stability(
        self,
        key: tuple[str, object],
        face: FaceDetection,
        metrics: _FaceEvidenceMetrics,
    ) -> int:
        policy = self.face_policy
        qualifies = (
            metrics.frontal >= policy.stable_frontal_min
            and metrics.blur >= policy.stable_blur_min
            and metrics.detector >= policy.stable_detector_min
        )
        previous = self._frontal_stability.get(key)
        streak = 1 if qualifies else 0
        if qualifies and previous is not None:
            gap = max(0.0, (face.timestamp - previous.last_timestamp).total_seconds())
            if gap <= policy.stable_max_gap_sec and _face_geometry_stable(previous.last_bbox, face.bbox):
                streak = previous.streak + 1
        self._frontal_stability[key] = _FrontalStability(
            last_timestamp=face.timestamp,
            last_bbox=face.bbox,
            streak=streak,
        )
        return streak

    def _log_face_hold(
        self,
        key: tuple[str, object],
        face: FaceDetection,
        old: _FaceEvidenceMetrics | None,
        new: _FaceEvidenceMetrics,
        stable_count: int,
    ) -> None:
        if old is None:
            return
        previous = self._last_hold_log.get(key)
        if previous is not None:
            elapsed = max(0.0, (face.timestamp - previous).total_seconds())
            if elapsed < self.face_policy.hold_log_interval_sec:
                return
        self._last_hold_log[key] = face.timestamp
        LOGGER.info(
            "[FACE_EVIDENCE_HOLD] camera=%s track=%s old_q=%.3f old_front=%.3f old_blur=%.3f "
            "new_q=%.3f new_front=%.3f new_blur=%.3f new_det=%.3f stable=%d",
            face.camera_id,
            face.track_id,
            old.quality,
            old.frontal,
            old.blur,
            new.quality,
            new.frontal,
            new.blur,
            new.detector,
            stable_count,
        )


def _upgrade_reason(
    old: _FaceEvidenceMetrics | None,
    new: _FaceEvidenceMetrics,
    *,
    stable_count: int,
    policy: FaceEvidencePolicy,
) -> str | None:
    if old is None:
        return "quality_upgrade"
    if new.quality >= old.quality + policy.quality_upgrade_margin:
        return "quality_upgrade"
    if (
        new.frontal >= old.frontal + policy.frontal_override_delta
        and new.quality >= old.quality - policy.frontal_quality_tolerance
        and new.detector >= policy.stable_detector_min
    ):
        return "frontal_override"
    if (
        stable_count >= policy.stable_required
        and new.frontal >= policy.stable_frontal_min
        and new.blur >= policy.stable_blur_min
        and new.detector >= policy.stable_detector_min
        and new.quality >= old.quality - policy.stable_quality_tolerance
        and (
            new.frontal >= old.frontal + 0.03
            or new.blur >= old.blur + 0.08
            or new.detector >= old.detector + 0.08
        )
    ):
        return "stable_frontal_override"
    return None


def _face_geometry_stable(previous: BoundingBox, current: BoundingBox) -> bool:
    area_ratio = current.area / max(previous.area, 1.0)
    if not 0.55 <= area_ratio <= 1.80:
        return False
    previous_x, previous_y = previous.center
    current_x, current_y = current.center
    if abs(current_x - previous_x) > 0.35 * max(previous.width, current.width):
        return False
    if abs(current_y - previous_y) > 0.35 * max(previous.height, current.height):
        return False
    return True


def _person_geometry_consistent(anchor: BoundingBox, current: BoundingBox) -> bool:
    """Conservative evidence-only drift guard; it never changes tracker IDs."""

    area_ratio = current.area / max(anchor.area, 1.0)
    if not 0.45 <= area_ratio <= 2.25:
        return False

    anchor_x, anchor_y = anchor.center
    current_x, current_y = current.center
    max_dx = 0.90 * max(anchor.width, current.width)
    max_dy = 0.90 * max(anchor.height, current.height)
    if abs(current_x - anchor_x) > max_dx:
        return False
    if abs(current_y - anchor_y) > max_dy:
        return False
    return True


def _alarm_face_pair_crop(
    frame: np.ndarray,
    person_bbox: BoundingBox,
    face_bbox: BoundingBox,
    config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, BoundingBox] | None:
    """Return face-anchored person crop, expanded display face and raw face ROI.

    The person crop deliberately does not span the full tracker bbox. A drifted
    NvDCF box can cover a rack or cabinet, while a valid SCRFD face remains a far
    stronger spatial cue for where the human actually is in the frame.
    """

    array = np.asarray(frame)
    if array.size == 0:
        return None
    height, width = array.shape[:2]
    person = person_bbox.clipped(width, height)
    face = face_bbox.clipped(width, height)
    if person is None or face is None or face.width < 8 or face.height < 8:
        return None

    face_cx, _face_cy = face.center
    half_width = face.width * (2.0 + config.padding_x_ratio)
    top_extension = face.height * (0.75 + config.padding_top_ratio)
    bottom_extension = face.height * (3.0 + config.upper_body_fraction)
    left = face_cx - half_width
    right = face_cx + half_width
    top = face.y1 - top_extension
    bottom = face.y2 + bottom_extension
    try:
        evidence = BoundingBox(left, top, right, bottom).clipped(width, height)
    except ValueError:
        return None
    if evidence is None:
        return None

    person_face_ratio = person.area / max(face.area, 1.0)
    if 4.0 <= person_face_ratio <= 80.0:
        sane_left = max(0.0, person.x1 - person.width * config.padding_x_ratio)
        sane_right = min(float(width), person.x2 + person.width * config.padding_x_ratio)
        if sane_left <= face.x1 and sane_right >= face.x2:
            narrowed_left = max(evidence.x1, sane_left)
            narrowed_right = min(evidence.x2, sane_right)
            if narrowed_right > narrowed_left:
                evidence = BoundingBox(
                    narrowed_left,
                    evidence.y1,
                    narrowed_right,
                    evidence.y2,
                )

    person_crop = rolling._crop_box(array, evidence)
    raw_face_crop = rolling._crop_box(array, face)
    if person_crop is None or raw_face_crop is None:
        return None

    display_left = face.x1 - face.width * config.padding_x_ratio
    display_top = face.y1 - face.height * config.padding_top_ratio
    display_right = face.x2 + face.width * config.padding_x_ratio
    display_bottom = face.y2 + face.height * config.upper_body_fraction
    try:
        display_box = BoundingBox(
            display_left,
            display_top,
            display_right,
            display_bottom,
        ).clipped(width, height)
    except ValueError:
        return None
    if display_box is None:
        return None
    display_face_crop = rolling._crop_box(array, display_box)
    if display_face_crop is None:
        return None

    return person_crop, display_face_crop, raw_face_crop, evidence


__all__ = ["EventSnapshotManager", "FaceEvidencePolicy"]
