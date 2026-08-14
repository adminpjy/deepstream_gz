"""Pose-aware business continuity layered on NvDCF.

This layer addresses a failure mode seen in live tests: one visible person changes
pose, PeopleNet/NvDCF briefly creates a narrow/tall or wide/short fragment, and
body ReID drops even though the physical person never left the frame.

Two recovery signals are used without lowering the global 0.85 body-ReID rule:
1. very strong same-person fragment geometry during a short provisional window;
2. a real AdaFace embedding produced asynchronously from SCRFD faces, including
   unknown people that do not exist in the worker database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from deepstream_ai.face.continuity_anchor import (
    FACE_CONTINUITY_ANCHORS,
    cosine_similarity,
)
from deepstream_ai.face_anchored_continuity import FaceAnchoredTrackContinuityResolver
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.track_continuity import (
    _box_metrics,
    _cosine,
    _max_face_iou,
    _seconds_between,
    _track_reid_embedding,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PoseFragmentConfig:
    enabled: bool = True
    provisional_window_sec: float = 1.20
    geometry_max_gap_sec: float = 1.20
    geometry_min_iou: float = 0.15
    geometry_min_containment: float = 0.82
    geometry_max_center_ratio: float = 0.32
    geometry_min_area_ratio: float = 0.10
    geometry_max_area_ratio: float = 4.50
    geometry_body_conflict_min: float = 0.55
    geometry_confirm_after_sec: float = 0.90
    face_anchor_max_gap_sec: float = 12.0
    face_anchor_min_quality: float = 0.62
    face_anchor_match_min: float = 0.68
    face_anchor_ambiguity_margin: float = 0.08
    face_anchor_min_iou: float = 0.05
    face_anchor_min_containment: float = 0.25
    face_anchor_max_center_ratio: float = 0.55

    @classmethod
    def from_file(cls, config_path: str | Path) -> "PoseFragmentConfig":
        try:
            raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        continuity = raw.get("track_continuity", {}) if isinstance(raw, dict) else {}
        section = continuity.get("pose_fragment_bridge", {}) if isinstance(continuity, dict) else {}
        if not isinstance(section, dict):
            section = {}
        return cls(
            enabled=bool(section.get("enabled", True)),
            provisional_window_sec=float(section.get("provisional_window_sec", 1.20)),
            geometry_max_gap_sec=float(section.get("geometry_max_gap_sec", 1.20)),
            geometry_min_iou=float(section.get("geometry_min_iou", 0.15)),
            geometry_min_containment=float(section.get("geometry_min_containment", 0.82)),
            geometry_max_center_ratio=float(section.get("geometry_max_center_ratio", 0.32)),
            geometry_min_area_ratio=float(section.get("geometry_min_area_ratio", 0.10)),
            geometry_max_area_ratio=float(section.get("geometry_max_area_ratio", 4.50)),
            geometry_body_conflict_min=float(section.get("geometry_body_conflict_min", 0.55)),
            geometry_confirm_after_sec=float(section.get("geometry_confirm_after_sec", 0.90)),
            face_anchor_max_gap_sec=float(section.get("face_anchor_max_gap_sec", 12.0)),
            face_anchor_min_quality=float(section.get("face_anchor_min_quality", 0.62)),
            face_anchor_match_min=float(section.get("face_anchor_match_min", 0.68)),
            face_anchor_ambiguity_margin=float(section.get("face_anchor_ambiguity_margin", 0.08)),
            face_anchor_min_iou=float(section.get("face_anchor_min_iou", 0.05)),
            face_anchor_min_containment=float(section.get("face_anchor_min_containment", 0.25)),
            face_anchor_max_center_ratio=float(section.get("face_anchor_max_center_ratio", 0.55)),
        )


class PoseAwareTrackContinuityResolver(FaceAnchoredTrackContinuityResolver):
    """Re-home young self IDs before they become permanent business tracks."""

    def __init__(self, *args, pose_config: PoseFragmentConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pose_config = pose_config
        self._self_started: dict[tuple[str, object], datetime] = {}
        self._pose_alias_started: dict[tuple[str, object], datetime] = {}

    @classmethod
    def from_file(cls, config_path: str | Path) -> "PoseAwareTrackContinuityResolver":
        from deepstream_ai.track_continuity import TrackContinuityConfig
        from deepstream_ai.track_continuity_guard import EdgeBridgeConfig, SingleTargetBridgeConfig

        return cls(
            TrackContinuityConfig.from_file(config_path),
            SingleTargetBridgeConfig.from_file(config_path),
            EdgeBridgeConfig.from_file(config_path),
            pose_config=PoseFragmentConfig.from_file(config_path),
        )

    def begin_stream_generation(self, camera_id: str, generation: int) -> None:
        previous = self._stream_generations.get(camera_id)
        super().begin_stream_generation(camera_id, generation)
        if previous is not None and previous != int(generation):
            for key in [key for key in self._self_started if key[0] == camera_id]:
                self._self_started.pop(key, None)
            for key in [key for key in self._pose_alias_started if key[0] == camera_id]:
                self._pose_alias_started.pop(key, None)

    def resolve(self, packet: FramePacket) -> FramePacket:
        if self.pose_config.enabled and packet.tracks:
            with self._lock:
                self._late_rehome(packet)
        resolved = super().resolve(packet)
        if self.pose_config.enabled and packet.tracks:
            with self._lock:
                for track in packet.tracks:
                    token = self._epoch_raw_id(packet.camera_id, track.track_id)
                    key = (packet.camera_id, token)
                    logical = self._raw_to_logical.get(key)
                    if logical == token:
                        self._self_started.setdefault(key, track.timestamp)
                    else:
                        self._self_started.pop(key, None)
        return resolved

    def _late_rehome(self, packet: FramePacket) -> None:
        config = self.pose_config
        current_tokens = {
            self._epoch_raw_id(packet.camera_id, track.track_id): track
            for track in packet.tracks
        }
        faces_by_token = {}
        for face in packet.faces:
            token = self._epoch_raw_id(packet.camera_id, face.track_id)
            faces_by_token.setdefault(token, []).append(face)

        for token, track in current_tokens.items():
            key = (packet.camera_id, token)
            if self._raw_to_logical.get(key) != token:
                continue
            started = self._self_started.get(key)
            if started is None:
                continue
            age = _seconds_between(track.timestamp, started)
            if age < 0.0 or age > config.provisional_window_sec:
                continue
            self_state = self._states.get((packet.camera_id, token))
            incoming_reid = _track_reid_embedding(track)
            incoming_anchor = FACE_CONTINUITY_ANCHORS.latest(
                packet.camera_id,
                token,
                now=track.timestamp,
                max_age_sec=config.face_anchor_max_gap_sec,
                min_quality=config.face_anchor_min_quality,
            )

            candidates = []
            for state in self._states.values():
                if state.camera_id != packet.camera_id or state.logical_id == token:
                    continue
                gap = _seconds_between(track.timestamp, state.last_seen)
                if gap < 0.0 or gap > config.face_anchor_max_gap_sec:
                    continue
                iou, containment, center_ratio, area_ratio = _box_metrics(
                    state.last_bbox, track.bbox
                )
                if not (
                    config.geometry_min_area_ratio
                    <= area_ratio
                    <= config.geometry_max_area_ratio
                ):
                    continue

                body_similarity = None
                if incoming_reid is not None and state.reid_embedding is not None:
                    body_similarity = _cosine(incoming_reid, state.reid_embedding)

                anchor_similarity = None
                candidate_anchor = FACE_CONTINUITY_ANCHORS.latest(
                    packet.camera_id,
                    state.logical_id,
                    now=track.timestamp,
                    max_age_sec=config.face_anchor_max_gap_sec,
                    min_quality=config.face_anchor_min_quality,
                )
                if incoming_anchor is not None and candidate_anchor is not None:
                    anchor_similarity = cosine_similarity(incoming_anchor, candidate_anchor)

                face_conflict = False
                candidate_faces = ()
                if state.current_raw_id in current_tokens:
                    candidate_faces = faces_by_token.get(state.current_raw_id, ())
                incoming_faces = faces_by_token.get(token, ())
                if incoming_faces and candidate_faces:
                    face_conflict = _max_face_iou(incoming_faces, candidate_faces) < 0.05

                anchor_ok = (
                    anchor_similarity is not None
                    and anchor_similarity >= config.face_anchor_match_min
                    and iou >= config.face_anchor_min_iou
                    and containment >= config.face_anchor_min_containment
                    and center_ratio <= config.face_anchor_max_center_ratio
                )
                geometry_ok = (
                    gap <= config.geometry_max_gap_sec
                    and containment >= config.geometry_min_containment
                    and center_ratio <= config.geometry_max_center_ratio
                    and (iou >= config.geometry_min_iou or containment >= 0.92)
                    and not face_conflict
                    and (
                        body_similarity is None
                        or body_similarity >= config.geometry_body_conflict_min
                    )
                )
                if not anchor_ok and not geometry_ok:
                    continue
                score = (
                    2.0 + float(anchor_similarity)
                    if anchor_ok and anchor_similarity is not None
                    else containment + 0.25 * iou - 0.20 * center_ratio
                )
                candidates.append(
                    (
                        score,
                        state,
                        "adaface" if anchor_ok else "pose_fragment",
                        iou,
                        containment,
                        center_ratio,
                        area_ratio,
                        body_similarity,
                        anchor_similarity,
                    )
                )

            candidates.sort(key=lambda item: item[0], reverse=True)
            if not candidates:
                continue
            best = candidates[0]
            second = candidates[1][0] if len(candidates) > 1 else -1.0
            required_margin = (
                config.face_anchor_ambiguity_margin if best[2] == "adaface" else 0.15
            )
            if best[0] - second < required_margin:
                continue

            state = best[1]
            if self_state is not None:
                self._states.pop((packet.camera_id, token), None)
            self._raw_to_logical.pop(key, None)
            self._set_provisional_alias(packet.camera_id, token, state, mode=best[2])
            self._pose_alias_started[key] = track.timestamp
            self._self_started.pop(key, None)
            FACE_CONTINUITY_ANCHORS.alias(
                packet.camera_id,
                token,
                state.logical_id,
                now=track.timestamp,
            )
            LOGGER.info(
                "[TRACK_CONTINUITY_%s_MERGE] camera=%s raw=%s logical=%s age=%.3f gap=%.3f "
                "iou=%.3f containment=%.3f center=%.3f area_ratio=%.3f body_reid=%s face_similarity=%s status=provisional",
                "ADAFACE" if best[2] == "adaface" else "POSE_FRAGMENT",
                packet.camera_id,
                track.track_id,
                state.logical_id,
                age,
                _seconds_between(track.timestamp, state.last_seen),
                best[3],
                best[4],
                best[5],
                best[6],
                f"{best[7]:.3f}" if best[7] is not None else "missing",
                f"{best[8]:.3f}" if best[8] is not None else "missing",
            )

    def _verify_provisional_assignments(self, packet: FramePacket) -> None:
        """Keep pose/AdaFace aliases unless identity evidence strongly conflicts."""

        config = self.pose_config
        for track in packet.tracks:
            raw_key = (packet.camera_id, track.track_id)
            logical = self._provisional_raw.get(raw_key)
            if logical is None:
                continue
            mode = self._provisional_mode.get(raw_key, "geometry")
            if mode not in {"pose_fragment", "adaface"}:
                continue
            state = self._states.get((packet.camera_id, logical))
            if state is None or self._raw_to_logical.get(raw_key) != logical:
                self._provisional_raw.pop(raw_key, None)
                self._provisional_mode.pop(raw_key, None)
                self._pose_alias_started.pop(raw_key, None)
                continue

            incoming = _track_reid_embedding(track)
            body_similarity = (
                _cosine(incoming, state.reid_embedding)
                if incoming is not None and state.reid_embedding is not None
                else None
            )
            incoming_anchor = FACE_CONTINUITY_ANCHORS.latest(
                packet.camera_id,
                track.track_id,
                now=track.timestamp,
                max_age_sec=config.face_anchor_max_gap_sec,
                min_quality=config.face_anchor_min_quality,
            )
            logical_anchor = FACE_CONTINUITY_ANCHORS.latest(
                packet.camera_id,
                logical,
                now=track.timestamp,
                max_age_sec=config.face_anchor_max_gap_sec,
                min_quality=config.face_anchor_min_quality,
            )
            face_similarity = (
                cosine_similarity(incoming_anchor, logical_anchor)
                if incoming_anchor is not None and logical_anchor is not None
                else None
            )
            started = self._pose_alias_started.get(raw_key, track.timestamp)
            age = max(0.0, _seconds_between(track.timestamp, started))

            confirmed = (
                body_similarity is not None and body_similarity >= self.config.reid_match_min
            ) or (
                face_similarity is not None
                and face_similarity >= config.face_anchor_match_min
            ) or age >= config.geometry_confirm_after_sec
            strong_conflict = (
                body_similarity is not None
                and body_similarity < config.geometry_body_conflict_min
                and (
                    face_similarity is None
                    or face_similarity < config.face_anchor_match_min
                )
            )
            if confirmed and not strong_conflict:
                self._provisional_raw.pop(raw_key, None)
                self._provisional_mode.pop(raw_key, None)
                self._pose_alias_started.pop(raw_key, None)
                FACE_CONTINUITY_ANCHORS.alias(
                    packet.camera_id,
                    track.track_id,
                    logical,
                    now=track.timestamp,
                )
                LOGGER.info(
                    "[TRACK_CONTINUITY_POSE_CONFIRM] camera=%s raw=%s logical=%s mode=%s body_reid=%s face_similarity=%s age=%.3f",
                    packet.camera_id,
                    track.track_id,
                    logical,
                    mode,
                    f"{body_similarity:.3f}" if body_similarity is not None else "missing",
                    f"{face_similarity:.3f}" if face_similarity is not None else "missing",
                    age,
                )
                continue
            if strong_conflict:
                self._provisional_raw.pop(raw_key, None)
                self._provisional_mode.pop(raw_key, None)
                self._pose_alias_started.pop(raw_key, None)
                self._raw_to_logical.pop(raw_key, None)
                state.raw_ids.discard(track.track_id)
                if state.current_raw_id == track.track_id:
                    state.current_raw_id = None
                LOGGER.info(
                    "[TRACK_CONTINUITY_POSE_REJECT] camera=%s raw=%s candidate_logical=%s mode=%s body_reid=%.3f face_similarity=%s",
                    packet.camera_id,
                    track.track_id,
                    logical,
                    mode,
                    body_similarity,
                    f"{face_similarity:.3f}" if face_similarity is not None else "missing",
                )

        # Let the parent apply its strict verification to single-target/edge aliases.
        saved = {
            key: (self._provisional_raw[key], self._provisional_mode.get(key, "geometry"))
            for key in list(self._provisional_raw)
            if self._provisional_mode.get(key) in {"pose_fragment", "adaface"}
        }
        for key in saved:
            self._provisional_raw.pop(key, None)
            self._provisional_mode.pop(key, None)
        try:
            super()._verify_provisional_assignments(packet)
        finally:
            for key, (logical, mode) in saved.items():
                if self._raw_to_logical.get(key) == logical:
                    self._provisional_raw[key] = logical
                    self._provisional_mode[key] = mode


__all__ = ["PoseAwareTrackContinuityResolver", "PoseFragmentConfig"]
