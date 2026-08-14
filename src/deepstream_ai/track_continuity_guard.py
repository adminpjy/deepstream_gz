"""Conservative missing-ReID recovery layered on business track continuity.

The normal resolver remains authoritative for ReID, face-backed continuity and
multi-person scenes. This guard covers one narrow failure mode seen in live
camera testing: exactly one visible person changes pose, NvDCF emits a new raw
ID, and that first new object has no exported ReID vector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.track_continuity import (
    TrackContinuityConfig,
    TrackContinuityResolver,
    _box_metrics,
    _seconds_between,
    _track_reid_embedding,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SingleTargetBridgeConfig:
    enabled: bool = True
    max_gap_sec: float = 10.0
    min_iou: float = 0.20
    min_containment: float = 0.45
    max_center_distance_ratio: float = 0.40
    min_area_ratio: float = 0.60
    max_area_ratio: float = 1.70

    @classmethod
    def from_file(cls, config_path: str | Path) -> "SingleTargetBridgeConfig":
        path = Path(config_path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        continuity = raw.get("track_continuity", {}) if isinstance(raw, dict) else {}
        bridge = continuity.get("single_target_bridge", {}) if isinstance(continuity, dict) else {}
        if not isinstance(bridge, dict):
            bridge = {}
        result = cls(
            enabled=bool(bridge.get("enabled", True)),
            max_gap_sec=float(bridge.get("max_gap_sec", 10.0)),
            min_iou=float(bridge.get("min_iou", 0.20)),
            min_containment=float(bridge.get("min_containment", 0.45)),
            max_center_distance_ratio=float(bridge.get("max_center_distance_ratio", 0.40)),
            min_area_ratio=float(bridge.get("min_area_ratio", 0.60)),
            max_area_ratio=float(bridge.get("max_area_ratio", 1.70)),
        )
        if result.max_gap_sec <= 0:
            raise ValueError("track_continuity.single_target_bridge.max_gap_sec must be positive")
        for name in ("min_iou", "min_containment", "max_center_distance_ratio"):
            value = getattr(result, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"track_continuity.single_target_bridge.{name} must be between 0 and 1"
                )
        if not 0.0 < result.min_area_ratio <= 1.0 <= result.max_area_ratio:
            raise ValueError("track_continuity.single_target_bridge area ratios are invalid")
        return result


class GuardedTrackContinuityResolver(TrackContinuityResolver):
    """Add a single-visible-target provisional bridge before normal resolution."""

    def __init__(
        self,
        config: TrackContinuityConfig,
        bridge_config: SingleTargetBridgeConfig,
    ) -> None:
        super().__init__(config)
        self.bridge_config = bridge_config

    @classmethod
    def from_file(cls, config_path: str | Path) -> "GuardedTrackContinuityResolver":
        return cls(
            TrackContinuityConfig.from_file(config_path),
            SingleTargetBridgeConfig.from_file(config_path),
        )

    def resolve(self, packet: FramePacket) -> FramePacket:
        bridge = self.bridge_config
        # Multi-person/crossing scenes must stay on strict ReID/face continuity.
        # Likewise, if the new raw object already carries ReID, let the base
        # resolver make the identity decision instead of bypassing the 0.85 gate.
        if bridge.enabled and self.config.enabled and len(packet.tracks) == 1:
            track = packet.tracks[0]
            raw_key = (packet.camera_id, track.track_id)
            incoming_reid = _track_reid_embedding(track)
            if incoming_reid is None:
                with self._lock:
                    self._purge(packet.timestamp)
                    if raw_key not in self._raw_to_logical and raw_key not in self._quarantined_raw:
                        current_raw_ids = {candidate.track_id for candidate in packet.tracks}
                        candidates = []
                        for state in self._states.values():
                            if state.camera_id != packet.camera_id:
                                continue
                            if state.logical_id == track.track_id:
                                continue
                            if state.current_raw_id in current_raw_ids:
                                continue
                            gap = _seconds_between(track.timestamp, state.last_seen)
                            if gap < 0.0 or gap > bridge.max_gap_sec:
                                continue
                            # Use the last emitted body box here, not the trusted
                            # ReID anchor: repeated pose fragments may legitimately
                            # evolve while no new vector is exported.
                            iou, containment, center_ratio, area_ratio = _box_metrics(
                                state.last_bbox, track.bbox
                            )
                            if (
                                iou >= bridge.min_iou
                                and containment >= bridge.min_containment
                                and center_ratio <= bridge.max_center_distance_ratio
                                and bridge.min_area_ratio <= area_ratio <= bridge.max_area_ratio
                            ):
                                candidates.append(
                                    (
                                        state,
                                        gap,
                                        iou,
                                        containment,
                                        center_ratio,
                                        area_ratio,
                                    )
                                )

                        # Ambiguity is a hard stop. A geometry-only fallback may
                        # never choose between two plausible people.
                        if len(candidates) == 1:
                            state, gap, iou, containment, center_ratio, area_ratio = candidates[0]
                            self._raw_to_logical[raw_key] = state.logical_id
                            state.raw_ids.add(track.track_id)
                            state.current_raw_id = track.track_id
                            LOGGER.info(
                                "[TRACK_CONTINUITY_SINGLE_TARGET_BRIDGE] camera=%s raw=%s "
                                "logical=%s gap=%.3f iou=%.3f containment=%.3f "
                                "center_ratio=%.3f area_ratio=%.3f reid=missing",
                                packet.camera_id,
                                track.track_id,
                                state.logical_id,
                                gap,
                                iou,
                                containment,
                                center_ratio,
                                area_ratio,
                            )

        return super().resolve(packet)


__all__ = ["GuardedTrackContinuityResolver", "SingleTargetBridgeConfig"]