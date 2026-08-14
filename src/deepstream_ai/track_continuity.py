"""Business-layer continuity for short NvDCF ID switches.

NvDCF remains the authoritative tracker. This resolver preserves the first raw
object ID as the stable logical/business ID when a recently fragmented target
can be recovered conservatively.

Normal recovery uses the tracker ReID gallery. A borderline ReID result may be
accepted only when both the person geometry and a real SCRFD face are nearly
continuous with trusted observations from the old logical track. Borderline
observations never update the trusted ReID gallery.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from threading import RLock

import numpy as np
import yaml

from deepstream_ai.domain import BoundingBox, TrackId
from deepstream_ai.pipeline.metadata import _TRACKER_REID_METADATA_KEY, FramePacket

LOGGER = logging.getLogger(__name__)
_REID_EMA_ALPHA = 0.10


@dataclass(frozen=True, slots=True)
class TrackContinuityConfig:
    enabled: bool
    max_gap_sec: float
    min_iou: float
    max_center_distance_ratio: float
    min_area_ratio: float
    max_area_ratio: float
    min_match_score: float
    ambiguity_margin: float
    duplicate_iou: float
    duplicate_iou_with_face: float
    duplicate_face_iou: float
    stale_retention_sec: float
    reid_max_gap_sec: float = 10.0
    reid_match_min: float = 0.85
    reid_update_min: float = 0.85
    reid_hijack_margin: float = 0.15
    reid_ambiguity_margin: float = 0.08
    fragment_min_reid: float = 0.82
    fragment_max_gap_sec: float = 0.12
    fragment_min_iou: float = 0.65
    fragment_min_containment: float = 0.85
    fragment_max_center_distance_ratio: float = 0.13
    fragment_min_area_ratio: float = 0.70
    fragment_max_area_ratio: float = 1.35
    face_override_max_gap_sec: float = 3.0
    face_override_min_person_iou: float = 0.25
    face_override_min_person_containment: float = 0.50
    face_override_max_center_distance_ratio: float = 0.33
    face_override_min_area_ratio: float = 0.60
    face_override_max_area_ratio: float = 1.45
    face_override_min_face_iou: float = 0.85
    face_override_min_face_containment: float = 0.95
    face_override_max_face_center_distance_ratio: float = 0.05
    face_override_min_face_area_ratio: float = 0.85
    face_override_max_face_area_ratio: float = 1.20
    face_override_min_face_score: float = 0.70
    face_override_min_reid: float = 0.80

    @classmethod
    def from_file(cls, config_path: str | Path) -> "TrackContinuityConfig":
        path = Path(config_path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        section = raw.get("track_continuity", {}) if isinstance(raw, dict) else {}
        if not isinstance(section, dict):
            section = {}
        result = cls(
            enabled=bool(section.get("enabled", True)),
            max_gap_sec=float(section.get("max_gap_sec", 8.0)),
            min_iou=float(section.get("min_iou", 0.10)),
            max_center_distance_ratio=float(section.get("max_center_distance_ratio", 0.75)),
            min_area_ratio=float(section.get("min_area_ratio", 0.40)),
            max_area_ratio=float(section.get("max_area_ratio", 2.50)),
            min_match_score=float(section.get("min_match_score", 0.55)),
            ambiguity_margin=float(section.get("ambiguity_margin", 0.15)),
            duplicate_iou=float(section.get("duplicate_iou", 0.90)),
            duplicate_iou_with_face=float(section.get("duplicate_iou_with_face", 0.70)),
            duplicate_face_iou=float(section.get("duplicate_face_iou", 0.50)),
            stale_retention_sec=float(section.get("stale_retention_sec", 30.0)),
            reid_max_gap_sec=float(section.get("reid_max_gap_sec", 10.0)),
            reid_match_min=float(section.get("reid_match_min", 0.85)),
            reid_update_min=float(section.get("reid_update_min", 0.85)),
            reid_hijack_margin=float(section.get("reid_hijack_margin", 0.15)),
            reid_ambiguity_margin=float(section.get("reid_ambiguity_margin", 0.08)),
            fragment_min_reid=float(section.get("fragment_min_reid", 0.82)),
            fragment_max_gap_sec=float(section.get("fragment_max_gap_sec", 0.12)),
            fragment_min_iou=float(section.get("fragment_min_iou", 0.65)),
            fragment_min_containment=float(section.get("fragment_min_containment", 0.85)),
            fragment_max_center_distance_ratio=float(
                section.get("fragment_max_center_distance_ratio", 0.13)
            ),
            fragment_min_area_ratio=float(section.get("fragment_min_area_ratio", 0.70)),
            fragment_max_area_ratio=float(section.get("fragment_max_area_ratio", 1.35)),
            face_override_max_gap_sec=float(section.get("face_override_max_gap_sec", 3.0)),
            face_override_min_person_iou=float(
                section.get("face_override_min_person_iou", 0.25)
            ),
            face_override_min_person_containment=float(
                section.get("face_override_min_person_containment", 0.50)
            ),
            face_override_max_center_distance_ratio=float(
                section.get("face_override_max_center_distance_ratio", 0.33)
            ),
            face_override_min_area_ratio=float(
                section.get("face_override_min_area_ratio", 0.60)
            ),
            face_override_max_area_ratio=float(
                section.get("face_override_max_area_ratio", 1.45)
            ),
            face_override_min_face_iou=float(
                section.get("face_override_min_face_iou", 0.85)
            ),
            face_override_min_face_containment=float(
                section.get("face_override_min_face_containment", 0.95)
            ),
            face_override_max_face_center_distance_ratio=float(
                section.get("face_override_max_face_center_distance_ratio", 0.05)
            ),
            face_override_min_face_area_ratio=float(
                section.get("face_override_min_face_area_ratio", 0.85)
            ),
            face_override_max_face_area_ratio=float(
                section.get("face_override_max_face_area_ratio", 1.20)
            ),
            face_override_min_face_score=float(
                section.get("face_override_min_face_score", 0.70)
            ),
            face_override_min_reid=float(section.get("face_override_min_reid", 0.80)),
        )
        if (
            result.max_gap_sec <= 0
            or result.reid_max_gap_sec < result.max_gap_sec
            or result.stale_retention_sec < result.reid_max_gap_sec
            or result.fragment_max_gap_sec <= 0
            or result.fragment_max_gap_sec > result.max_gap_sec
            or result.face_override_max_gap_sec <= 0
            or result.face_override_max_gap_sec > result.max_gap_sec
        ):
            raise ValueError("track_continuity timing values are invalid")
        for name in (
            "min_iou",
            "max_center_distance_ratio",
            "min_match_score",
            "ambiguity_margin",
            "duplicate_iou",
            "duplicate_iou_with_face",
            "duplicate_face_iou",
            "reid_match_min",
            "reid_update_min",
            "reid_hijack_margin",
            "reid_ambiguity_margin",
            "fragment_min_reid",
            "fragment_min_iou",
            "fragment_min_containment",
            "fragment_max_center_distance_ratio",
            "face_override_min_person_iou",
            "face_override_min_person_containment",
            "face_override_max_center_distance_ratio",
            "face_override_min_face_iou",
            "face_override_min_face_containment",
            "face_override_max_face_center_distance_ratio",
            "face_override_min_face_score",
            "face_override_min_reid",
        ):
            value = getattr(result, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"track_continuity.{name} must be between 0 and 1")
        if result.duplicate_iou_with_face > result.duplicate_iou:
            raise ValueError("duplicate_iou_with_face must not exceed duplicate_iou")
        if not math.isclose(result.reid_update_min, result.reid_match_min, abs_tol=1e-9):
            raise ValueError(
                "track_continuity reid_update_min must equal reid_match_min "
                "to prevent gallery identity drift"
            )
        if not 0.0 < result.fragment_min_area_ratio <= 1.0 <= result.fragment_max_area_ratio:
            raise ValueError("track_continuity fragment area ratios are invalid")
        if not 0.0 < result.face_override_min_area_ratio <= 1.0 <= result.face_override_max_area_ratio:
            raise ValueError("track_continuity face override person area ratios are invalid")
        if not 0.0 < result.face_override_min_face_area_ratio <= 1.0 <= result.face_override_max_face_area_ratio:
            raise ValueError("track_continuity face override face area ratios are invalid")
        if result.fragment_min_reid >= result.reid_match_min:
            raise ValueError("track_continuity fragment_min_reid must be below reid_match_min")
        if result.face_override_min_reid >= result.reid_match_min:
            raise ValueError("track_continuity face_override_min_reid must be below reid_match_min")
        if not 0.0 < result.min_area_ratio <= 1.0 <= result.max_area_ratio:
            raise ValueError("track_continuity area ratios are invalid")
        return result


@dataclass(slots=True)
class _LogicalTrackState:
    camera_id: str
    logical_id: TrackId
    current_raw_id: TrackId | None
    last_bbox: BoundingBox
    last_seen: datetime
    last_frame_number: int
    raw_ids: set[TrackId] = field(default_factory=set)
    reid_embedding: np.ndarray | None = None
    trusted_bbox: BoundingBox | None = None
    trusted_seen: datetime | None = None
    trusted_face_bbox: BoundingBox | None = None
    trusted_face_seen: datetime | None = None


@dataclass(frozen=True, slots=True)
class _QuarantinedRaw:
    original_logical: TrackId
    redirect_logical: TrackId | None


@dataclass(frozen=True, slots=True)
class _GeometryCandidate:
    score: float
    state: _LogicalTrackState
    iou: float
    containment: float
    center_ratio: float
    area_ratio: float
    reid_similarity: float | None
    face_iou: float = 0.0
    fragment_override: bool = False
    face_override: bool = False


class TrackContinuityResolver:
    """Map short-lived raw NvDCF fragments onto a stable logical track ID."""

    def __init__(self, config: TrackContinuityConfig) -> None:
        self.config = config
        self._raw_to_logical: dict[tuple[str, TrackId], TrackId] = {}
        self._states: dict[tuple[str, TrackId], _LogicalTrackState] = {}
        self._quarantined_raw: dict[tuple[str, TrackId], _QuarantinedRaw] = {}
        self._lock = RLock()

    def resolve(self, packet: FramePacket) -> FramePacket:
        if not self.config.enabled or not packet.tracks:
            return packet

        with self._lock:
            self._purge(packet.timestamp)
            current_raw_ids = {track.track_id for track in packet.tracks}
            track_by_raw = {track.track_id: track for track in packet.tracks}
            faces_by_raw: dict[TrackId, list] = {}
            for face in packet.faces:
                faces_by_raw.setdefault(face.track_id, []).append(face)

            assignments: dict[TrackId, TrackId] = {}
            occupied_logical: set[TrackId] = set()

            for track in packet.tracks:
                logical = self._raw_to_logical.get((packet.camera_id, track.track_id))
                if logical is None:
                    continue
                assignments[track.track_id] = logical
                occupied_logical.add(logical)

            newly_quarantined = self._reid_hijacks(packet, assignments, track_by_raw)
            for raw_id, redirect_logical in newly_quarantined.items():
                logical = assignments.pop(raw_id)
                occupied_logical.discard(logical)
                self._raw_to_logical.pop((packet.camera_id, raw_id), None)
                self._quarantined_raw[(packet.camera_id, raw_id)] = _QuarantinedRaw(
                    original_logical=logical,
                    redirect_logical=redirect_logical,
                )
                state = self._states.get((packet.camera_id, logical))
                if state is not None:
                    state.raw_ids.discard(raw_id)
                    if state.current_raw_id == raw_id:
                        state.current_raw_id = None

            quarantined_raw_ids: set[TrackId] = set()
            for track in packet.tracks:
                raw_id = track.track_id
                quarantine = self._quarantined_raw.get((packet.camera_id, raw_id))
                if quarantine is None:
                    continue
                original_logical = quarantine.original_logical
                state = self._states.get((packet.camera_id, original_logical))
                incoming = _track_reid_embedding(track)
                similarity = (
                    _cosine(incoming, state.reid_embedding)
                    if incoming is not None and state is not None and state.reid_embedding is not None
                    else -1.0
                )
                if similarity >= self.config.reid_match_min:
                    self._quarantined_raw.pop((packet.camera_id, raw_id), None)
                    previous_logical = assignments.pop(raw_id, None)
                    redirect_logical = quarantine.redirect_logical
                    redirect_state = (
                        self._states.get((packet.camera_id, redirect_logical))
                        if redirect_logical is not None
                        else None
                    )
                    if redirect_state is not None:
                        redirect_state.raw_ids.discard(raw_id)
                        if redirect_state.current_raw_id == raw_id:
                            redirect_state.current_raw_id = None
                    elif previous_logical is not None:
                        previous_state = self._states.get((packet.camera_id, previous_logical))
                        if previous_state is not None:
                            previous_state.raw_ids.discard(raw_id)
                            if previous_state.current_raw_id == raw_id:
                                previous_state.current_raw_id = None
                    self._raw_to_logical[(packet.camera_id, raw_id)] = original_logical
                    assignments[raw_id] = original_logical
                    occupied_logical.add(original_logical)
                    LOGGER.info(
                        "[TRACK_CONTINUITY_REID_RESTORE] camera=%s raw=%s logical=%s similarity=%.3f",
                        packet.camera_id,
                        raw_id,
                        original_logical,
                        similarity,
                    )
                    continue

                redirect_logical = quarantine.redirect_logical
                redirect_state = (
                    self._states.get((packet.camera_id, redirect_logical))
                    if redirect_logical is not None
                    else None
                )
                redirect_similarity = (
                    _cosine(incoming, redirect_state.reid_embedding)
                    if incoming is not None
                    and redirect_state is not None
                    and redirect_state.reid_embedding is not None
                    else None
                )
                if redirect_state is not None and (
                    incoming is None
                    or redirect_similarity is not None
                    and redirect_similarity >= self.config.reid_update_min
                ):
                    assignments[raw_id] = redirect_logical
                    self._raw_to_logical[(packet.camera_id, raw_id)] = redirect_logical
                    redirect_state.raw_ids.add(raw_id)
                    occupied_logical.add(redirect_logical)
                    continue

                assignments.pop(raw_id, None)
                self._raw_to_logical.pop((packet.camera_id, raw_id), None)
                quarantined_raw_ids.add(raw_id)

            occupied_logical = set(assignments.values())

            for track in packet.tracks:
                raw_id = track.track_id
                if raw_id in assignments or raw_id in quarantined_raw_ids:
                    continue

                # Same-frame duplicate fragments can be collapsed using strong
                # face overlap even when the current frame has no ReID vector.
                duplicate_candidates: list[tuple[float, TrackId, TrackId, float]] = []
                for existing_raw, logical in assignments.items():
                    if existing_raw in quarantined_raw_ids:
                        continue
                    existing = track_by_raw.get(existing_raw)
                    if existing is None:
                        continue
                    person_iou = _iou(existing.bbox, track.bbox)
                    person_containment = _containment(existing.bbox, track.bbox)
                    face_iou = _max_face_iou(
                        faces_by_raw.get(existing_raw, ()), faces_by_raw.get(raw_id, ())
                    )
                    incoming_reid = _track_reid_embedding(track)
                    existing_reid = _track_reid_embedding(existing)
                    existing_state = self._states.get((packet.camera_id, logical))
                    existing_identity = (
                        existing_reid
                        if existing_reid is not None
                        else existing_state.reid_embedding if existing_state is not None else None
                    )
                    reid_similarity = (
                        _cosine(incoming_reid, existing_identity)
                        if incoming_reid is not None and existing_identity is not None
                        else None
                    )
                    face_corroborated = face_iou >= self.config.face_override_min_face_iou
                    if reid_similarity is not None and reid_similarity < self.config.reid_update_min:
                        if not (
                            reid_similarity >= self.config.face_override_min_reid
                            and face_corroborated
                        ):
                            continue
                    if (
                        existing_state is not None
                        and existing_state.reid_embedding is not None
                        and incoming_reid is None
                        and not face_corroborated
                    ):
                        continue
                    duplicate = (
                        person_iou >= self.config.duplicate_iou
                        or (
                            person_iou >= self.config.duplicate_iou_with_face
                            and face_iou >= self.config.duplicate_face_iou
                        )
                        or (
                            face_corroborated
                            and person_containment
                            >= self.config.face_override_min_person_containment
                        )
                    )
                    if duplicate:
                        duplicate_candidates.append(
                            (person_iou + 0.25 * face_iou, logical, existing_raw, face_iou)
                        )

                duplicate_candidates.sort(key=lambda item: item[0], reverse=True)
                if duplicate_candidates:
                    best = duplicate_candidates[0]
                    second = duplicate_candidates[1][0] if len(duplicate_candidates) > 1 else -1.0
                    if best[0] - second >= self.config.ambiguity_margin:
                        logical = best[1]
                        assignments[raw_id] = logical
                        self._raw_to_logical[(packet.camera_id, raw_id)] = logical
                        state = self._states.get((packet.camera_id, logical))
                        if state is not None:
                            state.raw_ids.add(raw_id)
                        LOGGER.info(
                            "[TRACK_CONTINUITY_DUPLICATE] camera=%s raw=%s logical=%s "
                            "existing_raw=%s person_iou=%.3f face_iou=%.3f",
                            packet.camera_id,
                            raw_id,
                            logical,
                            best[2],
                            best[0] - 0.25 * best[3],
                            best[3],
                        )
                        continue

                active_duplicate_candidates: list[tuple[float, TrackId, TrackId, float, str]] = []
                incoming_reid = _track_reid_embedding(track)
                if incoming_reid is not None and not faces_by_raw.get(raw_id):
                    for existing_raw, logical in assignments.items():
                        existing = track_by_raw.get(existing_raw)
                        state = self._states.get((packet.camera_id, logical))
                        if existing is None or state is None or state.reid_embedding is None:
                            continue
                        similarity = _cosine(incoming_reid, state.reid_embedding)
                        if similarity < self.config.fragment_min_reid:
                            continue
                        iou, containment, center_ratio, area_ratio = _box_metrics(
                            existing.bbox, track.bbox
                        )
                        strict_identity = similarity >= self.config.reid_match_min
                        geometry_matches = (
                            iou >= (0.20 if strict_identity else self.config.fragment_min_iou)
                            and containment
                            >= (0.40 if strict_identity else self.config.fragment_min_containment)
                            and center_ratio
                            <= (
                                0.35
                                if strict_identity
                                else self.config.fragment_max_center_distance_ratio
                            )
                            and (0.60 if strict_identity else self.config.fragment_min_area_ratio)
                            <= area_ratio
                            <= (1.70 if strict_identity else self.config.fragment_max_area_ratio)
                        )
                        if geometry_matches:
                            active_duplicate_candidates.append(
                                (
                                    similarity + 0.10 * iou + 0.025 * containment,
                                    logical,
                                    existing_raw,
                                    similarity,
                                    "strict" if strict_identity else "fragment",
                                )
                            )
                active_duplicate_candidates.sort(reverse=True, key=lambda item: item[0])
                if active_duplicate_candidates:
                    best_duplicate = active_duplicate_candidates[0]
                    second_duplicate = (
                        active_duplicate_candidates[1][0]
                        if len(active_duplicate_candidates) > 1
                        else -1.0
                    )
                    if best_duplicate[0] - second_duplicate >= self.config.ambiguity_margin:
                        logical = best_duplicate[1]
                        assignments[raw_id] = logical
                        occupied_logical.add(logical)
                        self._raw_to_logical[(packet.camera_id, raw_id)] = logical
                        state = self._states.get((packet.camera_id, logical))
                        if state is not None:
                            state.raw_ids.add(raw_id)
                        LOGGER.info(
                            "[TRACK_CONTINUITY_DUPLICATE_REID] camera=%s raw=%s "
                            "logical=%s existing_raw=%s mode=%s similarity=%.3f",
                            packet.camera_id,
                            raw_id,
                            logical,
                            best_duplicate[2],
                            best_duplicate[4],
                            best_duplicate[3],
                        )
                        continue

                geometry_candidates: list[_GeometryCandidate] = []
                reid_candidates: list[tuple[float, _LogicalTrackState, float]] = []
                incoming_reid = _track_reid_embedding(track)
                incoming_faces = faces_by_raw.get(raw_id, ())
                best_incoming_face = _best_face(incoming_faces)
                occupied_reid_match = any(
                    state.camera_id == packet.camera_id
                    and state.logical_id in occupied_logical
                    and state.reid_embedding is not None
                    and incoming_reid is not None
                    and _cosine(incoming_reid, state.reid_embedding) >= self.config.reid_match_min
                    for state in self._states.values()
                )
                for state in self._states.values():
                    if state.camera_id != packet.camera_id or state.logical_id in occupied_logical:
                        continue
                    if state.current_raw_id in current_raw_ids:
                        continue
                    gap = _seconds_between(track.timestamp, state.last_seen)
                    if gap < 0 or gap > self.config.reid_max_gap_sec:
                        continue
                    state_bbox = state.trusted_bbox or state.last_bbox
                    area_ratio = track.bbox.area / max(state_bbox.area, 1.0)
                    similarity: float | None = None
                    if incoming_reid is not None and state.reid_embedding is not None:
                        similarity = _cosine(incoming_reid, state.reid_embedding)
                        if similarity >= self.config.reid_match_min:
                            reid_candidates.append((similarity, state, area_ratio))
                    if gap > self.config.max_gap_sec:
                        continue
                    iou, containment, center_ratio, area_ratio = _box_metrics(
                        state_bbox, track.bbox
                    )
                    metrics = self._match_metrics(state_bbox, track.bbox)
                    score = metrics[0] if metrics is not None else 0.0
                    last_iou, last_containment, last_center, last_area = _box_metrics(
                        state.last_bbox, track.bbox
                    )
                    fragment_override = (
                        similarity is not None
                        and self.config.fragment_min_reid <= similarity < self.config.reid_match_min
                        and not occupied_reid_match
                        and 0.0 <= gap <= self.config.fragment_max_gap_sec
                        and last_iou >= self.config.fragment_min_iou
                        and last_containment >= self.config.fragment_min_containment
                        and last_center <= self.config.fragment_max_center_distance_ratio
                        and self.config.fragment_min_area_ratio
                        <= last_area
                        <= self.config.fragment_max_area_ratio
                    )

                    face_iou = 0.0
                    face_override = False
                    if (
                        best_incoming_face is not None
                        and best_incoming_face.score >= self.config.face_override_min_face_score
                        and state.trusted_face_bbox is not None
                        and state.trusted_face_seen is not None
                        and similarity is not None
                        and self.config.face_override_min_reid
                        <= similarity
                        < self.config.reid_match_min
                        and not occupied_reid_match
                    ):
                        face_gap = _seconds_between(
                            best_incoming_face.timestamp, state.trusted_face_seen
                        )
                        face_iou, face_containment, face_center, face_area = _box_metrics(
                            state.trusted_face_bbox, best_incoming_face.bbox
                        )
                        face_override = (
                            0.0 <= face_gap <= self.config.face_override_max_gap_sec
                            and iou >= self.config.face_override_min_person_iou
                            and containment
                            >= self.config.face_override_min_person_containment
                            and center_ratio
                            <= self.config.face_override_max_center_distance_ratio
                            and self.config.face_override_min_area_ratio
                            <= area_ratio
                            <= self.config.face_override_max_area_ratio
                            and face_iou >= self.config.face_override_min_face_iou
                            and face_containment
                            >= self.config.face_override_min_face_containment
                            and face_center
                            <= self.config.face_override_max_face_center_distance_ratio
                            and self.config.face_override_min_face_area_ratio
                            <= face_area
                            <= self.config.face_override_max_face_area_ratio
                        )

                    if (
                        (metrics is None or score < self.config.min_match_score)
                        and not fragment_override
                        and not face_override
                    ):
                        continue
                    if fragment_override:
                        iou = last_iou
                        containment = last_containment
                        center_ratio = last_center
                        area_ratio = last_area
                        fragment_metrics = self._match_metrics(state.last_bbox, track.bbox)
                        if fragment_metrics is not None:
                            score = fragment_metrics[0]
                    if face_override:
                        # Put a strongly face-corroborated candidate ahead of a
                        # geometry-only candidate without altering identity data.
                        score = max(score, 0.90 + 0.05 * face_iou)
                    geometry_candidates.append(
                        _GeometryCandidate(
                            score=score,
                            state=state,
                            iou=iou,
                            containment=containment,
                            center_ratio=center_ratio,
                            area_ratio=area_ratio,
                            reid_similarity=similarity,
                            face_iou=face_iou,
                            fragment_override=fragment_override,
                            face_override=face_override,
                        )
                    )

                reid_candidates.sort(key=lambda item: item[0], reverse=True)
                reid_ambiguous = False
                if reid_candidates:
                    best_reid = reid_candidates[0]
                    second_reid = reid_candidates[1][0] if len(reid_candidates) > 1 else -1.0
                    if best_reid[0] - second_reid >= self.config.reid_ambiguity_margin:
                        state = best_reid[1]
                        logical = state.logical_id
                        assignments[raw_id] = logical
                        occupied_logical.add(logical)
                        self._raw_to_logical[(packet.camera_id, raw_id)] = logical
                        state.current_raw_id = raw_id
                        state.raw_ids.add(raw_id)
                        LOGGER.info(
                            "[TRACK_CONTINUITY_MERGE] camera=%s raw=%s logical=%s gap=%.3f "
                            "mode=reid similarity=%.3f area_ratio=%.3f",
                            packet.camera_id,
                            raw_id,
                            logical,
                            _seconds_between(track.timestamp, state.last_seen),
                            best_reid[0],
                            best_reid[2],
                        )
                    else:
                        reid_ambiguous = True

                geometry_candidates.sort(key=lambda item: item.score, reverse=True)
                if raw_id not in assignments and geometry_candidates and not reid_ambiguous:
                    best = geometry_candidates[0]
                    second_score = (
                        geometry_candidates[1].score if len(geometry_candidates) > 1 else -1.0
                    )
                    own_conflict = (
                        best.reid_similarity is not None
                        and best.reid_similarity < self.config.reid_match_min
                    )
                    missing_identity = incoming_reid is None and best.state.reid_embedding is not None
                    continuity_override = best.fragment_override or best.face_override
                    if best.score - second_score >= self.config.ambiguity_margin and (
                        (not own_conflict and not missing_identity) or continuity_override
                    ):
                        state = best.state
                        logical = state.logical_id
                        assignments[raw_id] = logical
                        occupied_logical.add(logical)
                        self._raw_to_logical[(packet.camera_id, raw_id)] = logical
                        state.current_raw_id = raw_id
                        state.raw_ids.add(raw_id)
                        mode = (
                            "face_continuity"
                            if best.face_override
                            else "short_gap_fragment"
                            if best.fragment_override
                            else "geometry"
                        )
                        LOGGER.info(
                            "[TRACK_CONTINUITY_MERGE] camera=%s raw=%s logical=%s gap=%.3f "
                            "mode=%s score=%.3f iou=%.3f face_iou=%.3f "
                            "reid_similarity=%s center_ratio=%.3f area_ratio=%.3f",
                            packet.camera_id,
                            raw_id,
                            logical,
                            _seconds_between(track.timestamp, state.last_seen),
                            mode,
                            best.score,
                            best.iou,
                            best.face_iou,
                            (
                                f"{best.reid_similarity:.3f}"
                                if best.reid_similarity is not None
                                else "missing"
                            ),
                            best.center_ratio,
                            best.area_ratio,
                        )
                    elif own_conflict:
                        LOGGER.info(
                            "[TRACK_CONTINUITY_REID_REJECT] camera=%s raw=%s "
                            "candidate_logical=%s similarity=%.3f score=%.3f iou=%.3f "
                            "face_iou=%.3f",
                            packet.camera_id,
                            raw_id,
                            best.state.logical_id,
                            best.reid_similarity,
                            best.score,
                            best.iou,
                            best.face_iou,
                        )

                if raw_id not in assignments:
                    logical = raw_id
                    assignments[raw_id] = logical
                    occupied_logical.add(logical)
                    self._raw_to_logical[(packet.camera_id, raw_id)] = logical
                    best_face = _best_face(faces_by_raw.get(raw_id, ()))
                    self._states[(packet.camera_id, logical)] = _LogicalTrackState(
                        camera_id=packet.camera_id,
                        logical_id=logical,
                        current_raw_id=raw_id,
                        last_bbox=track.bbox,
                        last_seen=track.timestamp,
                        last_frame_number=packet.frame_number,
                        raw_ids={raw_id},
                        reid_embedding=_track_reid_embedding(track),
                        trusted_bbox=track.bbox,
                        trusted_seen=track.timestamp,
                        trusted_face_bbox=best_face.bbox if best_face is not None else None,
                        trusted_face_seen=best_face.timestamp if best_face is not None else None,
                    )

            resolved_by_logical = {}
            selected_raw_by_logical: dict[TrackId, TrackId] = {}
            for track in packet.tracks:
                if track.track_id in quarantined_raw_ids:
                    continue
                logical = assignments[track.track_id]
                metadata = {
                    key: value
                    for key, value in track.metadata.items()
                    if not str(key).startswith("_")
                }
                metadata["raw_track_id"] = track.track_id
                resolved = replace(track, track_id=logical, metadata=metadata)
                current = resolved_by_logical.get(logical)
                current_raw = selected_raw_by_logical.get(logical)
                new_rank = (1 if faces_by_raw.get(track.track_id) else 0, track.confidence)
                old_rank = (
                    1 if current_raw is not None and faces_by_raw.get(current_raw) else 0,
                    current.confidence if current is not None else -1.0,
                )
                if current is None or new_rank > old_rank:
                    resolved_by_logical[logical] = resolved
                    selected_raw_by_logical[logical] = track.track_id

            for logical, resolved in resolved_by_logical.items():
                raw_id = selected_raw_by_logical[logical]
                best_face = _best_face(faces_by_raw.get(raw_id, ()))
                state = self._states.get((packet.camera_id, logical))
                if state is None:
                    state = _LogicalTrackState(
                        camera_id=packet.camera_id,
                        logical_id=logical,
                        current_raw_id=raw_id,
                        last_bbox=resolved.bbox,
                        last_seen=resolved.timestamp,
                        last_frame_number=packet.frame_number,
                        raw_ids={raw_id},
                        reid_embedding=_track_reid_embedding(track_by_raw[raw_id]),
                        trusted_bbox=resolved.bbox,
                        trusted_seen=resolved.timestamp,
                        trusted_face_bbox=best_face.bbox if best_face is not None else None,
                        trusted_face_seen=best_face.timestamp if best_face is not None else None,
                    )
                    self._states[(packet.camera_id, logical)] = state
                else:
                    state.current_raw_id = raw_id
                    state.last_seen = resolved.timestamp
                    state.last_frame_number = packet.frame_number
                    state.raw_ids.add(raw_id)
                    state.last_bbox = resolved.bbox
                    incoming = _track_reid_embedding(track_by_raw[raw_id])
                    incoming_similarity = (
                        _cosine(incoming, state.reid_embedding)
                        if incoming is not None and state.reid_embedding is not None
                        else None
                    )
                    trusted = (
                        state.reid_embedding is None
                        or incoming_similarity is not None
                        and incoming_similarity >= self.config.reid_update_min
                    )
                    self._update_reid(state, incoming)
                    if trusted:
                        state.trusted_bbox = resolved.bbox
                        state.trusted_seen = resolved.timestamp
                        if best_face is not None:
                            state.trusted_face_bbox = best_face.bbox
                            state.trusted_face_seen = best_face.timestamp

            def logical_for(raw_id: TrackId) -> TrackId:
                return assignments.get(
                    raw_id, self._raw_to_logical.get((packet.camera_id, raw_id), raw_id)
                )

            best_face_by_logical = {}
            for face in packet.faces:
                if face.track_id in quarantined_raw_ids:
                    continue
                logical = logical_for(face.track_id)
                resolved = replace(face, track_id=logical)
                previous = best_face_by_logical.get(logical)
                if previous is None or resolved.score > previous.score:
                    best_face_by_logical[logical] = resolved

            best_behavior = {}
            for behavior in packet.behaviors:
                if behavior.track_id in quarantined_raw_ids:
                    continue
                logical = logical_for(behavior.track_id)
                resolved = replace(behavior, track_id=logical)
                key = (logical, behavior.behavior, behavior.model_name)
                previous = best_behavior.get(key)
                if previous is None or resolved.confidence > previous.confidence:
                    best_behavior[key] = resolved

            return replace(
                packet,
                tracks=tuple(resolved_by_logical.values()),
                faces=tuple(best_face_by_logical.values()),
                behaviors=tuple(best_behavior.values()),
            )

    def logical_id(self, camera_id: str, raw_track_id: TrackId) -> TrackId:
        with self._lock:
            return self._raw_to_logical.get((camera_id, raw_track_id), raw_track_id)

    def presentation_track_id(
        self, camera_id: str, raw_track_id: TrackId
    ) -> TrackId | None:
        with self._lock:
            key = (camera_id, raw_track_id)
            if key in self._quarantined_raw and key not in self._raw_to_logical:
                return None
            return self._raw_to_logical.get(key, raw_track_id)

    def _reid_hijacks(
        self,
        packet: FramePacket,
        assignments: dict[TrackId, TrackId],
        track_by_raw: dict[TrackId, object],
    ) -> dict[TrackId, TrackId | None]:
        quarantined: dict[TrackId, TrackId | None] = {}
        for raw_id, logical in assignments.items():
            if (packet.camera_id, raw_id) in self._quarantined_raw:
                continue
            track = track_by_raw[raw_id]
            incoming = _track_reid_embedding(track)
            state = self._states.get((packet.camera_id, logical))
            if incoming is None or state is None or state.reid_embedding is None:
                continue
            own_similarity = _cosine(incoming, state.reid_embedding)
            if own_similarity >= self.config.reid_update_min:
                continue

            corroboration: list[tuple[float, TrackId]] = []
            for other_raw, other_logical in assignments.items():
                if other_raw == raw_id or other_logical == logical:
                    continue
                other_state = self._states.get((packet.camera_id, other_logical))
                if other_state is None or other_state.reid_embedding is None:
                    continue
                other_incoming = _track_reid_embedding(track_by_raw[other_raw])
                if other_incoming is None:
                    continue
                if _cosine(other_incoming, other_state.reid_embedding) < self.config.reid_update_min:
                    continue
                similarity = _cosine(incoming, other_incoming)
                corroboration.append((similarity, other_logical))
            corroboration.sort(reverse=True, key=lambda item: item[0])
            if not corroboration:
                continue
            other_similarity, other_logical = corroboration[0]
            second_similarity = corroboration[1][0] if len(corroboration) > 1 else -1.0
            if (
                other_similarity >= self.config.reid_match_min
                and other_similarity - own_similarity >= self.config.reid_hijack_margin
            ):
                redirect_logical = (
                    other_logical
                    if other_similarity - second_similarity >= self.config.reid_ambiguity_margin
                    else None
                )
                quarantined[raw_id] = redirect_logical
                LOGGER.info(
                    "[TRACK_CONTINUITY_REID_HOLD] camera=%s raw=%s logical=%s "
                    "own_similarity=%.3f corroborating_logical=%s other_similarity=%.3f redirect=%s",
                    packet.camera_id,
                    raw_id,
                    logical,
                    own_similarity,
                    other_logical,
                    other_similarity,
                    redirect_logical,
                )
        return quarantined

    def _update_reid(self, state: _LogicalTrackState, incoming: np.ndarray | None) -> None:
        if incoming is None:
            return
        if state.reid_embedding is None:
            state.reid_embedding = incoming
            return
        if _cosine(incoming, state.reid_embedding) < self.config.reid_update_min:
            return
        updated = (1.0 - _REID_EMA_ALPHA) * state.reid_embedding + _REID_EMA_ALPHA * incoming
        norm = float(np.linalg.norm(updated))
        if not np.isfinite(norm) or norm <= 1e-12:
            return
        result = np.asarray(updated / norm, dtype=np.float32)
        result.setflags(write=False)
        state.reid_embedding = result

    def _match_metrics(
        self, previous: BoundingBox, current: BoundingBox
    ) -> tuple[float, float, float, float] | None:
        area_ratio = current.area / max(previous.area, 1.0)
        if not self.config.min_area_ratio <= area_ratio <= self.config.max_area_ratio:
            return None
        iou = _iou(previous, current)
        px, py = previous.center
        cx, cy = current.center
        center_distance = math.hypot(cx - px, cy - py)
        scale = max(
            1.0,
            math.hypot(previous.width, previous.height),
            math.hypot(current.width, current.height),
        )
        center_ratio = center_distance / scale
        if iou < self.config.min_iou and center_ratio > self.config.max_center_distance_ratio:
            return None
        center_score = max(
            0.0,
            1.0 - center_ratio / max(self.config.max_center_distance_ratio, 1e-6),
        )
        size_score = min(area_ratio, 1.0 / max(area_ratio, 1e-6))
        score = 0.55 * iou + 0.30 * center_score + 0.15 * size_score
        return score, iou, center_ratio, area_ratio

    def _purge(self, now: datetime) -> None:
        stale = [
            key
            for key, state in self._states.items()
            if _seconds_between(now, state.last_seen) > self.config.stale_retention_sec
        ]
        for key in stale:
            state = self._states.pop(key)
            for raw_id in state.raw_ids:
                raw_key = (state.camera_id, raw_id)
                if self._raw_to_logical.get(raw_key) == state.logical_id:
                    self._raw_to_logical.pop(raw_key, None)
            for raw_key, quarantine in tuple(self._quarantined_raw.items()):
                if raw_key[0] == state.camera_id and quarantine.original_logical == state.logical_id:
                    self._quarantined_raw.pop(raw_key, None)
                elif raw_key[0] == state.camera_id and quarantine.redirect_logical == state.logical_id:
                    self._quarantined_raw[raw_key] = _QuarantinedRaw(
                        original_logical=quarantine.original_logical,
                        redirect_logical=None,
                    )


def _seconds_between(later: datetime, earlier: datetime) -> float:
    if later.tzinfo is None and earlier.tzinfo is not None:
        later = later.replace(tzinfo=earlier.tzinfo)
    elif later.tzinfo is not None and earlier.tzinfo is None:
        earlier = earlier.replace(tzinfo=later.tzinfo)
    return (later - earlier).total_seconds()


def _track_reid_embedding(track) -> np.ndarray | None:
    value = track.metadata.get(_TRACKER_REID_METADATA_KEY)
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.size == 0:
        return None
    if value.dtype != np.float32 or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    if abs(norm - 1.0) <= 1e-5:
        return value
    result = np.asarray(value / norm, dtype=np.float32)
    result.setflags(write=False)
    return result


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return -1.0
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def _containment(first: BoundingBox, second: BoundingBox) -> float:
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    return intersection / max(min(first.area, second.area), 1.0)


def _box_metrics(
    first: BoundingBox, second: BoundingBox
) -> tuple[float, float, float, float]:
    iou = _iou(first, second)
    containment = _containment(first, second)
    first_center = first.center
    second_center = second.center
    center_distance = math.dist(first_center, second_center)
    scale = max(
        1.0,
        math.hypot(first.width, first.height),
        math.hypot(second.width, second.height),
    )
    return iou, containment, center_distance / scale, second.area / max(first.area, 1.0)


def _best_face(faces):
    return max(faces, key=lambda face: face.score, default=None)


def _max_face_iou(first_faces, second_faces) -> float:
    best = 0.0
    for first in first_faces:
        for second in second_faces:
            best = max(best, _iou(first.bbox, second.bbox))
    return best


__all__ = ["TrackContinuityConfig", "TrackContinuityResolver"]
