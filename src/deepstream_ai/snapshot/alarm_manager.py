"""Alarm-friendly rolling snapshot presentation.

This keeps the existing rolling evidence lifecycle intact, but changes the face
snapshot that is written for users: AdaFace quality is still calculated from the
raw face ROI, while the persisted face evidence includes surrounding head/neck/
shoulder context. Person and face evidence always come from the same frame.

Person-only evidence is deliberately anchored to the first valid tracked-person
bbox. NvDCF shadow tracking can briefly drift onto a textured background after a
person becomes occluded or leaves the frame. Such tracker-only geometry must not
replace an already valid person snapshot merely because the rack/cabinet crop is
larger and has a higher Laplacian sharpness score.
"""

from __future__ import annotations

import logging

import numpy as np

from deepstream_ai.domain import BoundingBox, FaceDetection, IdentityResult, Track
from deepstream_ai.snapshot.manager import EvidenceCandidate
from deepstream_ai.snapshot.rolling_manager import EventSnapshotManager as RollingEventSnapshotManager
from deepstream_ai.snapshot import rolling_manager as rolling

LOGGER = logging.getLogger(__name__)


class EventSnapshotManager(RollingEventSnapshotManager):
    """Rolling manager with stable person evidence and larger face display crops."""

    def observe_person(
        self,
        frame: np.ndarray,
        track: Track,
        *,
        has_face: bool = False,
        quality: float | None = None,
    ) -> bool:
        """Persist the first person immediately, but do not let NvDCF drift overwrite it.

        The base rolling manager continuously replaces ``best_person`` using a
        score that rewards image sharpness and bbox area. In a server-room scene
        a cabinet/rack is both very sharp and very large, so a shadow-tracked box
        that drifts from the person onto the rack can incorrectly become the
        highest-quality person snapshot.

        Keep zero-miss behavior for a new track by delegating the first frame to
        the base manager. Later person-only updates are allowed only while their
        bbox remains geometrically consistent with the original fallback bbox.
        As soon as a face is detected, ``observe_face`` is authoritative and may
        replace both person and face evidence from that same winning frame.
        """

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

        # First observation for this track: preserve the existing zero-miss rule
        # and write it immediately.
        if state is None or anchor is None:
            return super().observe_person(
                frame,
                track,
                has_face=has_face,
                quality=quality,
            )

        # Subsequent person-only frames may improve the original evidence, but
        # must stay close in scale/location to the first valid person bbox. This
        # prevents gradual shadow-tracker drift from replacing a human with a
        # rack/cabinet image while still allowing normal small movements.
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
        computed_quality = rolling._face_quality(raw_face_crop, face, frame)
        if quality is not None:
            computed_quality = rolling._clamp01(
                0.75 * computed_quality + 0.25 * rolling._clamp01(quality)
            )

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
                    "[FACE_FALLBACK] camera=%s track=%s face_conf=%.3f quality=%.3f display=%sx%s",
                    face.camera_id,
                    face.track_id,
                    face.score,
                    computed_quality,
                    display_face_crop.shape[1],
                    display_face_crop.shape[0],
                )
            else:
                state.best_face = candidate
                source = "best_face"
                LOGGER.info(
                    "[BEST_FACE_UPDATE] camera=%s track=%s old_quality=%.3f new_quality=%.3f display=%sx%s",
                    face.camera_id,
                    face.track_id,
                    baseline,
                    computed_quality,
                    display_face_crop.shape[1],
                    display_face_crop.shape[0],
                )

        # Keep the user-facing person and face pictures synchronized to exactly
        # the same winning frame.
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


def _person_geometry_consistent(anchor: BoundingBox, current: BoundingBox) -> bool:
    """Conservative evidence-only drift guard; it never changes tracker IDs.

    The guard compares every person-only replacement to the *first* valid bbox,
    not to the previous replacement, so a shadow target cannot slowly ratchet
    its way across a rack. Face evidence bypasses this guard because a detected
    face provides the stronger same-frame human cue and replaces the pair in
    ``observe_face``.
    """

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
    """Return same-frame person crop, expanded display face, raw face and evidence box."""

    array = np.asarray(frame)
    if array.size == 0:
        return None
    height, width = array.shape[:2]
    person = person_bbox.clipped(width, height)
    face = face_bbox.clipped(width, height)
    if person is None or face is None or face.width < 8 or face.height < 8:
        return None

    # Person evidence: upper body around the same winning face frame.
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

    person_crop = rolling._crop_box(array, evidence)
    raw_face_crop = rolling._crop_box(array, face)
    if person_crop is None or raw_face_crop is None:
        return None

    # User-facing face evidence: enlarge around the detector ROI. Reuse the
    # configured crop ratios so the presentation remains config-driven.
    display_left = max(evidence.x1, face.x1 - face.width * config.padding_x_ratio)
    display_top = max(evidence.y1, face.y1 - face.height * config.padding_top_ratio)
    display_right = min(evidence.x2, face.x2 + face.width * config.padding_x_ratio)
    display_bottom = min(
        evidence.y2,
        face.y2 + face.height * config.upper_body_fraction,
    )
    try:
        display_box = BoundingBox(
            display_left,
            display_top,
            display_right,
            display_bottom,
        )
    except ValueError:
        return None
    display_face_crop = rolling._crop_box(array, display_box)
    if display_face_crop is None:
        return None

    return person_crop, display_face_crop, raw_face_crop, evidence


__all__ = ["EventSnapshotManager"]
