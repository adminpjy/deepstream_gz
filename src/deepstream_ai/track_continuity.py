"""Small business-layer bridge for short NvDCF ID switches.

NvDCF remains the authoritative tracker. This module only aliases a newly
created raw object_id back to the most plausible existing logical track in the
same camera. The first raw ID is preserved as the logical/business ID, so normal
alarm semantics do not change.

Two cases are handled conservatively:
1. a recently-lost raw ID is replaced after a brief stream/association glitch;
2. NvDCF briefly emits two almost-identical raw tracks for the same person.

For multi-person safety, ordinary simultaneously visible tracks are never
merged. Same-frame merging requires nearly identical person boxes, or strongly
overlapping person boxes corroborated by overlapping face detections.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from threading import RLock

import yaml

from deepstream_ai.domain import BoundingBox, TrackId
from deepstream_ai.pipeline.metadata import FramePacket

LOGGER = logging.getLogger(__name__)


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
        )
        if result.max_gap_sec <= 0 or result.stale_retention_sec < result.max_gap_sec:
            raise ValueError("track_continuity timing values are invalid")
        for name in (
            "min_iou",
            "max_center_distance_ratio",
            "min_match_score",
            "ambiguity_margin",
            "duplicate_iou",
            "duplicate_iou_with_face",
            "duplicate_face_iou",
        ):
            value = getattr(result, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"track_continuity.{name} must be between 0 and 1")
        if result.duplicate_iou_with_face > result.duplicate_iou:
            raise ValueError("duplicate_iou_with_face must not exceed duplicate_iou")
        if not 0.0 < result.min_area_ratio <= 1.0 <= result.max_area_ratio:
            raise ValueError("track_continuity area ratios are invalid")
        return result


@dataclass(slots=True)
class _LogicalTrackState:
    camera_id: str
    logical_id: TrackId
    current_raw_id: TrackId
    last_bbox: BoundingBox
    last_seen: datetime
    last_frame_number: int
    raw_ids: set[TrackId] = field(default_factory=set)


class TrackContinuityResolver:
    """Map short-lived raw NvDCF fragments onto a stable logical track ID."""

    def __init__(self, config: TrackContinuityConfig) -> None:
        self.config = config
        self._raw_to_logical: dict[tuple[str, TrackId], TrackId] = {}
        self._states: dict[tuple[str, TrackId], _LogicalTrackState] = {}
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

            # Existing raw IDs keep their established logical IDs.
            for track in packet.tracks:
                logical = self._raw_to_logical.get((packet.camera_id, track.track_id))
                if logical is None:
                    continue
                assignments[track.track_id] = logical
                occupied_logical.add(logical)

            for track in packet.tracks:
                raw_id = track.track_id
                if raw_id in assignments:
                    continue

                # First handle the rare NvDCF duplicate-target case. A new raw
                # track can share an existing logical ID only when the current
                # boxes are almost identical, or when strong box overlap is
                # corroborated by overlapping SCRFD face boxes.
                duplicate_candidates: list[tuple[float, TrackId, TrackId, float]] = []
                for existing_raw, logical in assignments.items():
                    existing = track_by_raw.get(existing_raw)
                    if existing is None:
                        continue
                    person_iou = _iou(existing.bbox, track.bbox)
                    face_iou = _max_face_iou(
                        faces_by_raw.get(existing_raw, ()),
                        faces_by_raw.get(raw_id, ()),
                    )
                    duplicate = person_iou >= self.config.duplicate_iou or (
                        person_iou >= self.config.duplicate_iou_with_face
                        and face_iou >= self.config.duplicate_face_iou
                    )
                    if duplicate:
                        duplicate_candidates.append((person_iou + 0.25 * face_iou, logical, existing_raw, face_iou))

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

                # Otherwise look only at recently missing logical tracks. This
                # bridges a short RTSP flash or one failed NvDCF association.
                candidates: list[tuple[float, _LogicalTrackState, float, float, float]] = []
                for state in self._states.values():
                    if state.camera_id != packet.camera_id or state.logical_id in occupied_logical:
                        continue
                    if state.current_raw_id in current_raw_ids:
                        continue
                    gap = _seconds_between(track.timestamp, state.last_seen)
                    if gap < 0 or gap > self.config.max_gap_sec:
                        continue
                    metrics = self._match_metrics(state.last_bbox, track.bbox)
                    if metrics is None:
                        continue
                    score, iou, center_ratio, area_ratio = metrics
                    if score < self.config.min_match_score:
                        continue
                    candidates.append((score, state, iou, center_ratio, area_ratio))

                candidates.sort(key=lambda item: item[0], reverse=True)
                if candidates:
                    best = candidates[0]
                    second_score = candidates[1][0] if len(candidates) > 1 else -1.0
                    if best[0] - second_score >= self.config.ambiguity_margin:
                        state = best[1]
                        logical = state.logical_id
                        assignments[raw_id] = logical
                        occupied_logical.add(logical)
                        self._raw_to_logical[(packet.camera_id, raw_id)] = logical
                        state.current_raw_id = raw_id
                        state.raw_ids.add(raw_id)
                        LOGGER.info(
                            "[TRACK_CONTINUITY_MERGE] camera=%s raw=%s logical=%s gap=%.3f "
                            "score=%.3f iou=%.3f center_ratio=%.3f area_ratio=%.3f",
                            packet.camera_id,
                            raw_id,
                            logical,
                            _seconds_between(track.timestamp, state.last_seen),
                            best[0],
                            best[2],
                            best[3],
                            best[4],
                        )

                if raw_id not in assignments:
                    logical = raw_id
                    assignments[raw_id] = logical
                    occupied_logical.add(logical)
                    self._raw_to_logical[(packet.camera_id, raw_id)] = logical
                    self._states[(packet.camera_id, logical)] = _LogicalTrackState(
                        camera_id=packet.camera_id,
                        logical_id=logical,
                        current_raw_id=raw_id,
                        last_bbox=track.bbox,
                        last_seen=track.timestamp,
                        last_frame_number=packet.frame_number,
                        raw_ids={raw_id},
                    )

            # Deduplicate raw tracker fragments that now share one logical ID.
            # Prefer a track carrying a face this frame, then detector confidence.
            resolved_by_logical = {}
            selected_raw_by_logical: dict[TrackId, TrackId] = {}
            for track in packet.tracks:
                logical = assignments[track.track_id]
                metadata = dict(track.metadata)
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
                    )
                    self._states[(packet.camera_id, logical)] = state
                else:
                    state.current_raw_id = raw_id
                    state.last_bbox = resolved.bbox
                    state.last_seen = resolved.timestamp
                    state.last_frame_number = packet.frame_number
                    state.raw_ids.add(raw_id)

            def logical_for(raw_id: TrackId) -> TrackId:
                return assignments.get(raw_id, self._raw_to_logical.get((packet.camera_id, raw_id), raw_id))

            best_face_by_logical = {}
            for face in packet.faces:
                logical = logical_for(face.track_id)
                resolved = replace(face, track_id=logical)
                previous = best_face_by_logical.get(logical)
                if previous is None or resolved.score > previous.score:
                    best_face_by_logical[logical] = resolved

            best_behavior = {}
            for behavior in packet.behaviors:
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

    def _match_metrics(
        self,
        previous: BoundingBox,
        current: BoundingBox,
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
                self._raw_to_logical.pop((state.camera_id, raw_id), None)


def _seconds_between(later: datetime, earlier: datetime) -> float:
    if later.tzinfo is None and earlier.tzinfo is not None:
        later = later.replace(tzinfo=earlier.tzinfo)
    elif later.tzinfo is not None and earlier.tzinfo is None:
        earlier = earlier.replace(tzinfo=later.tzinfo)
    return (later - earlier).total_seconds()


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


def _max_face_iou(first_faces, second_faces) -> float:
    best = 0.0
    for first in first_faces:
        for second in second_faces:
            best = max(best, _iou(first.bbox, second.bbox))
    return best


__all__ = ["TrackContinuityConfig", "TrackContinuityResolver"]
