"""Multi-person safety policy for pose/AdaFace business-track continuity.

The pose bridge is intentionally aggressive in a true single-person scene, but
that policy is unsafe when a second person overlaps or crosses an existing
track. This layer keeps the single-person recovery behavior while requiring
independent identity evidence in crowded scenes.

Safety invariants:
* geometry alone never lets a new raw ID steal an occupied logical ID;
* a strong body/face conflict locks that raw/logical pair for the raw lifetime;
* time-only pose confirmation is allowed only while the alias stayed single-person;
* AdaFace anchors are copied to the logical ID only after confirmation, never at
  provisional assignment time, so an anchor cannot confirm itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from deepstream_ai.face.continuity_anchor import FACE_CONTINUITY_ANCHORS, cosine_similarity
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.pose_aware_continuity import PoseAwareTrackContinuityResolver, PoseFragmentConfig
from deepstream_ai.track_continuity import (
    TrackContinuityConfig,
    _box_metrics,
    _cosine,
    _max_face_iou,
    _seconds_between,
    _track_reid_embedding,
)
from deepstream_ai.track_continuity_guard import (
    EdgeBridgeConfig,
    GuardedTrackContinuityResolver,
    SingleTargetBridgeConfig,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MultiPersonContinuitySafetyConfig:
    """Identity gates used only when two independent raw targets may compete."""

    face_anchor_match_min: float = 0.85
    face_anchor_conflict_max: float = 0.55

    @classmethod
    def from_file(cls, config_path: str | Path) -> "MultiPersonContinuitySafetyConfig":
        try:
            raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        continuity = raw.get("track_continuity", {}) if isinstance(raw, dict) else {}
        section = continuity.get("pose_fragment_bridge", {}) if isinstance(continuity, dict) else {}
        if not isinstance(section, dict):
            section = {}
        result = cls(
            face_anchor_match_min=float(
                section.get("multi_person_face_anchor_match_min", 0.85)
            ),
            face_anchor_conflict_max=float(section.get("face_anchor_conflict_max", 0.55)),
        )
        if not 0.0 <= result.face_anchor_conflict_max < result.face_anchor_match_min <= 1.0:
            raise ValueError(
                "track_continuity.pose_fragment_bridge multi-person face thresholds are invalid"
            )
        return result


class MultiPersonSafePoseAwareTrackContinuityResolver(PoseAwareTrackContinuityResolver):
    """Pose-aware continuity with crossing/occlusion identity isolation."""

    def __init__(
        self,
        config: TrackContinuityConfig,
        bridge_config: SingleTargetBridgeConfig,
        edge_config: EdgeBridgeConfig | None = None,
        *,
        pose_config: PoseFragmentConfig,
        safety_config: MultiPersonContinuitySafetyConfig | None = None,
    ) -> None:
        super().__init__(
            config,
            bridge_config,
            edge_config,
            pose_config=pose_config,
        )
        self.safety_config = safety_config or MultiPersonContinuitySafetyConfig()
        self._identity_conflict_locks: set[tuple[str, object, object]] = set()
        self._pose_alias_multi_person: dict[tuple[str, object], bool] = {}
        self._pose_alias_initial_face_timestamp: dict[tuple[str, object], datetime] = {}

    @classmethod
    def from_file(cls, config_path: str | Path) -> "MultiPersonSafePoseAwareTrackContinuityResolver":
        return cls(
            TrackContinuityConfig.from_file(config_path),
            SingleTargetBridgeConfig.from_file(config_path),
            EdgeBridgeConfig.from_file(config_path),
            pose_config=PoseFragmentConfig.from_file(config_path),
            safety_config=MultiPersonContinuitySafetyConfig.from_file(config_path),
        )

    def begin_stream_generation(self, camera_id: str, generation: int) -> None:
        previous = self._stream_generations.get(camera_id)
        super().begin_stream_generation(camera_id, generation)
        if previous is not None and previous != int(generation):
            self._identity_conflict_locks = {
                item for item in self._identity_conflict_locks if item[0] != camera_id
            }
            for mapping in (
                self._pose_alias_multi_person,
                self._pose_alias_initial_face_timestamp,
            ):
                for key in [key for key in mapping if key[0] == camera_id]:
                    mapping.pop(key, None)

    def _late_rehome(self, packet: FramePacket) -> None:
        """Re-home young raw IDs, but make identity authoritative in crowds."""

        config = self.pose_config
        current_tokens = {
            self._epoch_raw_id(packet.camera_id, track.track_id): track
            for track in packet.tracks
        }
        multi_person_scene = len(current_tokens) > 1
        faces_by_token: dict[object, list] = {}
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
                if (packet.camera_id, token, state.logical_id) in self._identity_conflict_locks:
                    continue

                gap = _seconds_between(track.timestamp, state.last_seen)
                if gap < 0.0 or gap > config.face_anchor_max_gap_sec:
                    continue
                iou, containment, center_ratio, area_ratio = _box_metrics(
                    state.last_bbox, track.bbox
                )
                if not config.geometry_min_area_ratio <= area_ratio <= config.geometry_max_area_ratio:
                    continue

                body_similarity = None
                if incoming_reid is not None and state.reid_embedding is not None:
                    body_similarity = _cosine(incoming_reid, state.reid_embedding)
                body_strict = (
                    body_similarity is not None
                    and body_similarity >= self.config.reid_match_min
                )

                candidate_anchor = FACE_CONTINUITY_ANCHORS.latest(
                    packet.camera_id,
                    state.logical_id,
                    now=track.timestamp,
                    max_age_sec=config.face_anchor_max_gap_sec,
                    min_quality=config.face_anchor_min_quality,
                )
                anchor_similarity = (
                    cosine_similarity(incoming_anchor, candidate_anchor)
                    if incoming_anchor is not None and candidate_anchor is not None
                    else None
                )

                candidate_active = (
                    state.current_raw_id is not None
                    and state.current_raw_id != token
                    and state.current_raw_id in current_tokens
                )
                identity_required = multi_person_scene or candidate_active
                required_face_similarity = (
                    self.safety_config.face_anchor_match_min
                    if identity_required
                    else config.face_anchor_match_min
                )

                incoming_faces = faces_by_token.get(token, ())
                candidate_faces = (
                    faces_by_token.get(state.current_raw_id, ())
                    if state.current_raw_id in current_tokens
                    else ()
                )
                face_conflict = bool(
                    incoming_faces
                    and candidate_faces
                    and _max_face_iou(incoming_faces, candidate_faces) < 0.05
                )

                anchor_ok = (
                    anchor_similarity is not None
                    and anchor_similarity >= required_face_similarity
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
                    and (not identity_required or body_strict)
                )
                if not anchor_ok and not geometry_ok:
                    continue

                mode = "adaface" if anchor_ok else "pose_fragment"
                score = (
                    2.0 + float(anchor_similarity)
                    if anchor_ok and anchor_similarity is not None
                    else containment + 0.25 * iou - 0.20 * center_ratio
                )
                candidates.append(
                    (
                        score,
                        state,
                        mode,
                        iou,
                        containment,
                        center_ratio,
                        area_ratio,
                        body_similarity,
                        anchor_similarity,
                        identity_required,
                        candidate_active,
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
            self._pose_alias_multi_person[key] = bool(best[9])
            if best[2] == "adaface" and incoming_anchor is not None:
                self._pose_alias_initial_face_timestamp[key] = incoming_anchor.timestamp
            else:
                self._pose_alias_initial_face_timestamp.pop(key, None)
            self._self_started.pop(key, None)

            # Raw/logical face anchors intentionally stay independent until confirmation.
            LOGGER.info(
                "[TRACK_CONTINUITY_%s_MERGE] camera=%s raw=%s logical=%s age=%.3f gap=%.3f "
                "iou=%.3f containment=%.3f center=%.3f area_ratio=%.3f body_reid=%s "
                "face_similarity=%s status=provisional multi_person=%s occupied=%s",
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
                str(bool(best[9])).lower(),
                str(bool(best[10])).lower(),
            )

    def _verify_provisional_assignments(self, packet: FramePacket) -> None:
        """Verify pose/AdaFace aliases without geometry self-confirmation."""

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
                self._clear_pose_alias_bookkeeping(raw_key)
                self._provisional_raw.pop(raw_key, None)
                self._provisional_mode.pop(raw_key, None)
                continue

            if len(packet.tracks) > 1:
                self._pose_alias_multi_person[raw_key] = True
            multi_person = self._pose_alias_multi_person.get(raw_key, False)
            required_face_similarity = (
                self.safety_config.face_anchor_match_min
                if multi_person
                else config.face_anchor_match_min
            )

            incoming = _track_reid_embedding(track)
            body_similarity = (
                _cosine(incoming, state.reid_embedding)
                if incoming is not None and state.reid_embedding is not None
                else None
            )
            body_confirm = (
                body_similarity is not None
                and body_similarity >= self.config.reid_match_min
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

            initial_face_timestamp = self._pose_alias_initial_face_timestamp.get(raw_key)
            later_independent_face = (
                incoming_anchor is not None
                and face_similarity is not None
                and face_similarity >= required_face_similarity
                and (
                    mode != "adaface"
                    or initial_face_timestamp is None
                    or incoming_anchor.timestamp > initial_face_timestamp
                )
            )

            started = self._pose_alias_started.get(raw_key, track.timestamp)
            age = max(0.0, _seconds_between(track.timestamp, started))
            time_confirm = (
                mode == "pose_fragment"
                and not multi_person
                and age >= config.geometry_confirm_after_sec
            )

            strong_body_conflict = (
                body_similarity is not None
                and body_similarity < config.geometry_body_conflict_min
                and not later_independent_face
            )
            strong_face_conflict = (
                face_similarity is not None
                and face_similarity < self.safety_config.face_anchor_conflict_max
                and not body_confirm
            )
            strong_conflict = strong_body_conflict or strong_face_conflict

            if strong_conflict:
                self._identity_conflict_locks.add((packet.camera_id, track.track_id, logical))
                self._provisional_raw.pop(raw_key, None)
                self._provisional_mode.pop(raw_key, None)
                self._raw_to_logical.pop(raw_key, None)
                self._clear_pose_alias_bookkeeping(raw_key)
                state.raw_ids.discard(track.track_id)
                if state.current_raw_id == track.track_id:
                    state.current_raw_id = None
                LOGGER.warning(
                    "[TRACK_CONTINUITY_IDENTITY_CONFLICT_LOCK] camera=%s raw=%s logical=%s "
                    "mode=%s body_reid=%s face_similarity=%s",
                    packet.camera_id,
                    track.track_id,
                    logical,
                    mode,
                    f"{body_similarity:.3f}" if body_similarity is not None else "missing",
                    f"{face_similarity:.3f}" if face_similarity is not None else "missing",
                )
                LOGGER.info(
                    "[TRACK_CONTINUITY_POSE_REJECT] camera=%s raw=%s candidate_logical=%s "
                    "mode=%s body_reid=%s face_similarity=%s conflict_locked=true",
                    packet.camera_id,
                    track.track_id,
                    logical,
                    mode,
                    f"{body_similarity:.3f}" if body_similarity is not None else "missing",
                    f"{face_similarity:.3f}" if face_similarity is not None else "missing",
                )
                continue

            confirmed = body_confirm or later_independent_face or time_confirm
            if confirmed:
                reason = (
                    "body_reid"
                    if body_confirm
                    else "later_adaface"
                    if later_independent_face
                    else "single_person_time"
                )
                self._provisional_raw.pop(raw_key, None)
                self._provisional_mode.pop(raw_key, None)
                self._clear_pose_alias_bookkeeping(raw_key)
                FACE_CONTINUITY_ANCHORS.alias(
                    packet.camera_id,
                    track.track_id,
                    logical,
                    now=track.timestamp,
                )
                LOGGER.info(
                    "[TRACK_CONTINUITY_POSE_CONFIRM] camera=%s raw=%s logical=%s mode=%s "
                    "reason=%s body_reid=%s face_similarity=%s age=%.3f multi_person=%s",
                    packet.camera_id,
                    track.track_id,
                    logical,
                    mode,
                    reason,
                    f"{body_similarity:.3f}" if body_similarity is not None else "missing",
                    f"{face_similarity:.3f}" if face_similarity is not None else "missing",
                    age,
                    str(multi_person).lower(),
                )

        saved = {
            key: (self._provisional_raw[key], self._provisional_mode.get(key, "geometry"))
            for key in list(self._provisional_raw)
            if self._provisional_mode.get(key) in {"pose_fragment", "adaface"}
        }
        for key in saved:
            self._provisional_raw.pop(key, None)
            self._provisional_mode.pop(key, None)
        try:
            GuardedTrackContinuityResolver._verify_provisional_assignments(self, packet)
        finally:
            for key, (logical, mode) in saved.items():
                if self._raw_to_logical.get(key) == logical:
                    self._provisional_raw[key] = logical
                    self._provisional_mode[key] = mode
                else:
                    self._clear_pose_alias_bookkeeping(key)

    def _clear_pose_alias_bookkeeping(self, raw_key: tuple[str, object]) -> None:
        self._pose_alias_started.pop(raw_key, None)
        self._pose_alias_multi_person.pop(raw_key, None)
        self._pose_alias_initial_face_timestamp.pop(raw_key, None)


__all__ = [
    "MultiPersonContinuitySafetyConfig",
    "MultiPersonSafePoseAwareTrackContinuityResolver",
]
