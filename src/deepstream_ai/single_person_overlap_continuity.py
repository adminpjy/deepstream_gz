"""Recover a very short NvDCF raw-ID handoff without weakening crowd safety.

Live task 58b7d5b6281a4551 exposed a narrow failure mode: one physical person stayed
in view, changed head/upper-body pose, NvDCF retired raw 0 and immediately
created raw 1.  The boxes still overlapped strongly (IoU about 0.66) but body
ReID dropped to about 0.62, so the strict 0.85 business ReID rule correctly
refused a normal merge.  The new raw ID then survived long enough to become a
second business track.

This layer only repairs that immediate one-person handoff.  It runs *after* the
existing multi-person-safe resolver and never lowers the global ReID threshold.
A fallback merge is allowed only when:
- exactly one raw target is currently visible;
- no independently separated multi-person scene was seen very recently;
- the old logical target disappeared only a few hundred milliseconds ago;
- old/new person boxes still overlap strongly and remain geometrically close;
- body ReID is at least weakly compatible (only a conflict gate, not identity
  proof).

The merge stays provisional and is verified by the existing pose/AdaFace logic.
If later evidence conflicts, the existing identity-conflict lock still rejects
and separates the raw track.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from deepstream_ai.multi_person_continuity import (
    MultiPersonContinuitySafetyConfig,
    MultiPersonSafePoseAwareTrackContinuityResolver,
)
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.pose_aware_continuity import PoseFragmentConfig
from deepstream_ai.track_continuity import (
    TrackContinuityConfig,
    _box_metrics,
    _cosine,
    _seconds_between,
    _track_reid_embedding,
)
from deepstream_ai.track_continuity_guard import EdgeBridgeConfig, SingleTargetBridgeConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SinglePersonOverlapConfig:
    """Conservative geometry gate for an immediate raw-ID handoff."""

    enabled: bool = True
    max_gap_sec: float = 0.35
    min_iou: float = 0.60
    min_containment: float = 0.65
    max_center_ratio: float = 0.35
    min_area_ratio: float = 0.45
    max_area_ratio: float = 2.20
    min_body_reid: float = 0.58
    ambiguity_margin: float = 0.10
    independent_pair_max_iou: float = 0.50
    crowd_cooldown_sec: float = 2.0

    @classmethod
    def from_file(cls, config_path: str | Path) -> "SinglePersonOverlapConfig":
        try:
            raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        continuity = raw.get("track_continuity", {}) if isinstance(raw, dict) else {}
        section = continuity.get("pose_fragment_bridge", {}) if isinstance(continuity, dict) else {}
        if not isinstance(section, dict):
            section = {}
        result = cls(
            enabled=bool(section.get("single_person_overlap_enabled", True)),
            max_gap_sec=float(section.get("single_person_overlap_max_gap_sec", 0.35)),
            min_iou=float(section.get("single_person_overlap_min_iou", 0.60)),
            min_containment=float(section.get("single_person_overlap_min_containment", 0.65)),
            max_center_ratio=float(section.get("single_person_overlap_max_center_ratio", 0.35)),
            min_area_ratio=float(section.get("single_person_overlap_min_area_ratio", 0.45)),
            max_area_ratio=float(section.get("single_person_overlap_max_area_ratio", 2.20)),
            min_body_reid=float(section.get("single_person_overlap_min_body_reid", 0.58)),
            ambiguity_margin=float(section.get("single_person_overlap_ambiguity_margin", 0.10)),
            independent_pair_max_iou=float(
                section.get("single_person_overlap_independent_pair_max_iou", 0.50)
            ),
            crowd_cooldown_sec=float(
                section.get("single_person_overlap_crowd_cooldown_sec", 2.0)
            ),
        )
        if not 0.0 <= result.min_body_reid <= 1.0:
            raise ValueError("single-person overlap min_body_reid must be in [0, 1]")
        if not 0.0 <= result.min_iou <= 1.0 or not 0.0 <= result.min_containment <= 1.0:
            raise ValueError("single-person overlap geometry thresholds must be in [0, 1]")
        if result.max_gap_sec <= 0.0 or result.crowd_cooldown_sec < 0.0:
            raise ValueError("single-person overlap timing must be positive")
        if result.min_area_ratio <= 0.0 or result.max_area_ratio < result.min_area_ratio:
            raise ValueError("single-person overlap area ratio range is invalid")
        return result


class SinglePersonOverlapContinuityResolver(MultiPersonSafePoseAwareTrackContinuityResolver):
    """Add a narrow one-person raw-ID handoff rescue to the crowd-safe resolver."""

    def __init__(
        self,
        config: TrackContinuityConfig,
        bridge_config: SingleTargetBridgeConfig,
        edge_config: EdgeBridgeConfig | None = None,
        *,
        pose_config: PoseFragmentConfig,
        safety_config: MultiPersonContinuitySafetyConfig | None = None,
        overlap_config: SinglePersonOverlapConfig | None = None,
    ) -> None:
        super().__init__(
            config,
            bridge_config,
            edge_config,
            pose_config=pose_config,
            safety_config=safety_config,
        )
        self.overlap_config = overlap_config or SinglePersonOverlapConfig()
        self._last_independent_multi_person: dict[str, datetime] = {}

    @classmethod
    def from_file(cls, config_path: str | Path) -> "SinglePersonOverlapContinuityResolver":
        return cls(
            TrackContinuityConfig.from_file(config_path),
            SingleTargetBridgeConfig.from_file(config_path),
            EdgeBridgeConfig.from_file(config_path),
            pose_config=PoseFragmentConfig.from_file(config_path),
            safety_config=MultiPersonContinuitySafetyConfig.from_file(config_path),
            overlap_config=SinglePersonOverlapConfig.from_file(config_path),
        )

    def begin_stream_generation(self, camera_id: str, generation: int) -> None:
        previous = self._stream_generations.get(camera_id)
        super().begin_stream_generation(camera_id, generation)
        if previous is not None and previous != int(generation):
            self._last_independent_multi_person.pop(camera_id, None)

    def _late_rehome(self, packet: FramePacket) -> None:
        # Keep all existing multi-person / AdaFace protections authoritative.
        super()._late_rehome(packet)

        config = self.overlap_config
        if not config.enabled or not packet.tracks:
            return

        current_tokens = {
            self._epoch_raw_id(packet.camera_id, track.track_id): track
            for track in packet.tracks
        }
        if self._independent_multi_person_scene(current_tokens):
            self._last_independent_multi_person[packet.camera_id] = packet.timestamp
            return

        # This rescue is intentionally unavailable while two raw targets are
        # visible.  It is only for the next-frame handoff after the old raw ID
        # has disappeared from downstream object metadata.
        if len(current_tokens) != 1:
            return

        crowded_at = self._last_independent_multi_person.get(packet.camera_id)
        if (
            crowded_at is not None
            and _seconds_between(packet.timestamp, crowded_at) <= config.crowd_cooldown_sec
        ):
            return

        token, track = next(iter(current_tokens.items()))
        key = (packet.camera_id, token)
        if self._raw_to_logical.get(key) != token:
            return
        started = self._self_started.get(key)
        if started is None:
            return
        age = _seconds_between(track.timestamp, started)
        if age < 0.0 or age > self.pose_config.provisional_window_sec:
            return

        incoming_reid = _track_reid_embedding(track)
        if incoming_reid is None:
            return

        self_state = self._states.get((packet.camera_id, token))
        candidates: list[tuple[float, object, float, float, float, float, float, float]] = []
        for state in self._states.values():
            if state.camera_id != packet.camera_id or state.logical_id == token:
                continue
            if (packet.camera_id, token, state.logical_id) in self._identity_conflict_locks:
                continue
            # An occupied logical ID remains under the strict multi-person path.
            if state.current_raw_id in current_tokens:
                continue

            gap = _seconds_between(track.timestamp, state.last_seen)
            if gap < 0.0 or gap > config.max_gap_sec:
                continue
            if state.reid_embedding is None:
                continue
            body_similarity = _cosine(incoming_reid, state.reid_embedding)
            if body_similarity < config.min_body_reid:
                continue

            iou, containment, center_ratio, area_ratio = _box_metrics(state.last_bbox, track.bbox)
            if iou < config.min_iou:
                continue
            if containment < config.min_containment:
                continue
            if center_ratio > config.max_center_ratio:
                continue
            if not config.min_area_ratio <= area_ratio <= config.max_area_ratio:
                continue

            score = (
                1.20 * iou
                + 0.55 * containment
                + 0.45 * body_similarity
                - 0.25 * center_ratio
                - 0.10 * min(1.0, gap / max(config.max_gap_sec, 1e-6))
            )
            candidates.append(
                (
                    score,
                    state,
                    iou,
                    containment,
                    center_ratio,
                    area_ratio,
                    body_similarity,
                    gap,
                )
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            return
        best = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else -1.0
        if best[0] - second_score < config.ambiguity_margin:
            LOGGER.info(
                "[TRACK_CONTINUITY_SINGLE_OVERLAP_HOLD] camera=%s raw=%s candidates=%d margin=%.3f",
                packet.camera_id,
                track.track_id,
                len(candidates),
                best[0] - second_score,
            )
            return

        state = best[1]
        if self_state is not None:
            self._states.pop((packet.camera_id, token), None)
        self._raw_to_logical.pop(key, None)
        # Reuse pose_fragment verification: the alias is not final.  Later body
        # ReID / AdaFace may confirm it, and strong conflict still tears it down.
        self._set_provisional_alias(packet.camera_id, token, state, mode="pose_fragment")
        self._pose_alias_started[key] = track.timestamp
        self._pose_alias_multi_person[key] = False
        self._pose_alias_initial_face_timestamp.pop(key, None)
        self._self_started.pop(key, None)

        LOGGER.info(
            "[TRACK_CONTINUITY_SINGLE_PERSON_OVERLAP_MERGE] camera=%s raw=%s logical=%s "
            "age=%.3f gap=%.3f iou=%.3f containment=%.3f center=%.3f area_ratio=%.3f "
            "body_reid=%.3f status=provisional",
            packet.camera_id,
            track.track_id,
            state.logical_id,
            age,
            best[7],
            best[2],
            best[3],
            best[4],
            best[5],
            best[6],
        )

    def _independent_multi_person_scene(self, current_tokens: dict[object, object]) -> bool:
        """Return true only for clearly separated simultaneous raw targets.

        A same-person NvDCF split can briefly expose two heavily overlapping raw
        boxes.  Treating every two-raw frame as a crowd caused task 58b7... to
        lose the single-person fallback.  Clearly separated pairs still arm a
        short cooldown so a real crossing person cannot be merged immediately
        after one of the two tracks disappears.
        """

        if len(current_tokens) <= 1:
            return False
        tracks = list(current_tokens.values())
        for index, left in enumerate(tracks):
            for right in tracks[index + 1 :]:
                iou, _containment, _center_ratio, _area_ratio = _box_metrics(
                    left.bbox, right.bbox
                )
                if iou < self.overlap_config.independent_pair_max_iou:
                    return True
        return False


__all__ = [
    "SinglePersonOverlapConfig",
    "SinglePersonOverlapContinuityResolver",
]
