"""Conservative missing-ReID recovery layered on business track continuity.

The normal resolver remains authoritative for strict ReID and face-backed
continuity. Two narrow geometry fallbacks cover repeatable live-camera failure
modes while preserving the first NvDCF raw ID as the business ID:

* a single visible person changes pose and a replacement raw ID has no ReID;
* a person is clipped by the left/right image edge and NvDCF briefly fragments
  while part of the body is still visible.

Both fallbacks are provisional. They require an already trusted ReID anchor and
are confirmed or revoked as soon as a real tracker ReID vector arrives. The
normal 0.85 business ReID threshold is never lowered.

RTSP reconnects are also generation-aware. NvDCF can restart raw IDs from zero;
new-generation raw IDs are internally namespaced so they can never overwrite an
older business ID merely because the numeric tracker ID was reused.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.track_continuity import (
    TrackContinuityConfig,
    TrackContinuityResolver,
    _box_metrics,
    _cosine,
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


@dataclass(frozen=True, slots=True)
class EdgeBridgeConfig:
    """Short missing-ReID continuity for a person clipped by a lateral edge."""

    enabled: bool = True
    max_gap_sec: float = 3.0
    edge_margin_ratio: float = 0.10
    min_parallel_overlap: float = 0.35
    max_parallel_center_shift_ratio: float = 0.60
    min_area_ratio: float = 0.15
    max_area_ratio: float = 4.00
    ambiguity_margin: float = 0.15

    @classmethod
    def from_file(cls, config_path: str | Path) -> "EdgeBridgeConfig":
        path = Path(config_path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        continuity = raw.get("track_continuity", {}) if isinstance(raw, dict) else {}
        bridge = continuity.get("edge_bridge", {}) if isinstance(continuity, dict) else {}
        if not isinstance(bridge, dict):
            bridge = {}
        result = cls(
            enabled=bool(bridge.get("enabled", True)),
            max_gap_sec=float(bridge.get("max_gap_sec", 3.0)),
            edge_margin_ratio=float(bridge.get("edge_margin_ratio", 0.10)),
            min_parallel_overlap=float(bridge.get("min_parallel_overlap", 0.35)),
            max_parallel_center_shift_ratio=float(
                bridge.get("max_parallel_center_shift_ratio", 0.60)
            ),
            min_area_ratio=float(bridge.get("min_area_ratio", 0.15)),
            max_area_ratio=float(bridge.get("max_area_ratio", 4.00)),
            ambiguity_margin=float(bridge.get("ambiguity_margin", 0.15)),
        )
        if result.max_gap_sec <= 0:
            raise ValueError("track_continuity.edge_bridge.max_gap_sec must be positive")
        for name in (
            "edge_margin_ratio",
            "min_parallel_overlap",
            "max_parallel_center_shift_ratio",
            "ambiguity_margin",
        ):
            value = getattr(result, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"track_continuity.edge_bridge.{name} must be between 0 and 1")
        if not 0.0 < result.min_area_ratio <= 1.0 <= result.max_area_ratio:
            raise ValueError("track_continuity.edge_bridge area ratios are invalid")
        return result


class GuardedTrackContinuityResolver(TrackContinuityResolver):
    """Add conservative missing-ReID recovery and reconnect-safe raw generations."""

    def __init__(
        self,
        config: TrackContinuityConfig,
        bridge_config: SingleTargetBridgeConfig,
        edge_config: EdgeBridgeConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.bridge_config = bridge_config
        self.edge_config = edge_config or EdgeBridgeConfig()
        # Geometry-only aliases are provisional until a later exported ReID
        # vector reaches the normal 0.85 identity threshold.
        self._provisional_raw: dict[tuple[str, object], object] = {}
        self._provisional_mode: dict[tuple[str, object], str] = {}
        self._stream_generations: dict[str, int] = {}

    @classmethod
    def from_file(cls, config_path: str | Path) -> "GuardedTrackContinuityResolver":
        return cls(
            TrackContinuityConfig.from_file(config_path),
            SingleTargetBridgeConfig.from_file(config_path),
            EdgeBridgeConfig.from_file(config_path),
        )

    def begin_stream_generation(self, camera_id: str, generation: int) -> None:
        generation = max(0, int(generation))
        with self._lock:
            previous = self._stream_generations.get(camera_id)
            if previous is None:
                self._stream_generations[camera_id] = generation
                return
            if previous == generation:
                return
            self._stream_generations[camera_id] = generation
            for key in [key for key in self._raw_to_logical if key[0] == camera_id]:
                self._raw_to_logical.pop(key, None)
            for key in [key for key in self._quarantined_raw if key[0] == camera_id]:
                self._quarantined_raw.pop(key, None)
            for key in [key for key in self._provisional_raw if key[0] == camera_id]:
                self._provisional_raw.pop(key, None)
                self._provisional_mode.pop(key, None)
            for state in self._states.values():
                if state.camera_id != camera_id:
                    continue
                state.current_raw_id = None
                state.raw_ids.clear()
            LOGGER.warning(
                "[TRACK_CONTINUITY_EPOCH] camera=%s generation=%d previous=%d "
                "raw_aliases_cleared=true trusted_state_preserved=true",
                camera_id,
                generation,
                previous,
            )

    def resolve(self, packet: FramePacket) -> FramePacket:
        packet, native_raw_ids = self._namespace_packet(packet)
        bridge = self.bridge_config
        if self.config.enabled:
            with self._lock:
                self._purge(packet.timestamp)
                self._verify_provisional_assignments(packet)
                if self.edge_config.enabled:
                    self._bridge_edge_fragments(packet, native_raw_ids)

                # Multi-person/crossing scenes must stay on strict ReID/face
                # continuity. If the new raw object already carries ReID, the
                # base resolver also remains authoritative from its first frame.
                if bridge.enabled and len(packet.tracks) == 1:
                    track = packet.tracks[0]
                    raw_key = (packet.camera_id, track.track_id)
                    incoming_reid = _track_reid_embedding(track)
                    if (
                        incoming_reid is None
                        and raw_key not in self._raw_to_logical
                        and raw_key not in self._quarantined_raw
                    ):
                        current_raw_ids = {candidate.track_id for candidate in packet.tracks}
                        candidates = []
                        for state in self._states.values():
                            if state.camera_id != packet.camera_id:
                                continue
                            if state.logical_id == track.track_id:
                                continue
                            if state.current_raw_id in current_raw_ids:
                                continue
                            # A geometry-only bridge is allowed only when the old
                            # logical track already has a trusted identity anchor.
                            if state.reid_embedding is None:
                                continue
                            gap = _seconds_between(track.timestamp, state.last_seen)
                            if gap < 0.0 or gap > bridge.max_gap_sec:
                                continue
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
                            self._set_provisional_alias(
                                packet.camera_id,
                                track.track_id,
                                state,
                                mode="single_target",
                            )
                            LOGGER.info(
                                "[TRACK_CONTINUITY_SINGLE_TARGET_BRIDGE] camera=%s raw=%s "
                                "logical=%s gap=%.3f iou=%.3f containment=%.3f "
                                "center_ratio=%.3f area_ratio=%.3f reid=missing status=provisional",
                                packet.camera_id,
                                native_raw_ids.get(track.track_id, track.track_id),
                                state.logical_id,
                                gap,
                                iou,
                                containment,
                                center_ratio,
                                area_ratio,
                            )

        resolved = super().resolve(packet)
        return self._restore_native_raw_metadata(resolved, native_raw_ids)

    def logical_id(self, camera_id: str, raw_track_id):
        return super().logical_id(camera_id, self._epoch_raw_id(camera_id, raw_track_id))

    def presentation_track_id(self, camera_id: str, raw_track_id):
        return super().presentation_track_id(
            camera_id,
            self._epoch_raw_id(camera_id, raw_track_id),
        )

    def _bridge_edge_fragments(
        self,
        packet: FramePacket,
        native_raw_ids: dict[object, object],
    ) -> None:
        """Provisionally reconnect missing-ReID fragments at the left/right edge.

        Bottom-edge matching is intentionally excluded: people commonly touch the
        floor/image bottom and it is not a useful identity constraint. A bridge
        is attempted only for a new raw ID with no exported ReID, a trusted old
        ReID anchor, short same-edge geometry, and no plausible competing target.
        """

        if not packet.tracks or packet.image.ndim < 2:
            return
        height, width = packet.image.shape[:2]
        if width <= 0 or height <= 0:
            return
        config = self.edge_config
        current_raw_ids = {track.track_id for track in packet.tracks}
        occupied_logical = {
            logical
            for raw_id in current_raw_ids
            if (logical := self._raw_to_logical.get((packet.camera_id, raw_id))) is not None
        }

        for track in packet.tracks:
            raw_key = (packet.camera_id, track.track_id)
            if raw_key in self._raw_to_logical or raw_key in self._quarantined_raw:
                continue
            # Once a real ReID vector exists the base resolver, with its strict
            # 0.85 threshold, is authoritative. Edge geometry never lowers it.
            if _track_reid_embedding(track) is not None:
                continue
            incoming_edges = _lateral_edges(track.bbox, width, config.edge_margin_ratio)
            if not incoming_edges:
                continue

            candidates: list[tuple[float, object, str, float, float, float, float]] = []
            for state in self._states.values():
                if state.camera_id != packet.camera_id:
                    continue
                if state.logical_id in occupied_logical or state.logical_id == track.track_id:
                    continue
                if state.current_raw_id in current_raw_ids:
                    continue
                if state.reid_embedding is None:
                    continue
                gap = _seconds_between(track.timestamp, state.last_seen)
                if gap < 0.0 or gap > config.max_gap_sec:
                    continue
                shared_edges = incoming_edges.intersection(
                    _lateral_edges(state.last_bbox, width, config.edge_margin_ratio)
                )
                if not shared_edges:
                    continue
                area_ratio = track.bbox.area / max(state.last_bbox.area, 1.0)
                if not config.min_area_ratio <= area_ratio <= config.max_area_ratio:
                    continue
                for edge in shared_edges:
                    if _has_edge_competitor(
                        packet,
                        track.track_id,
                        track.bbox,
                        edge,
                        width,
                        config,
                    ):
                        continue
                    overlap, center_shift = _parallel_edge_metrics(state.last_bbox, track.bbox)
                    if overlap < config.min_parallel_overlap:
                        continue
                    if center_shift > config.max_parallel_center_shift_ratio:
                        continue
                    size_score = min(area_ratio, 1.0 / max(area_ratio, 1e-6))
                    score = overlap + 0.10 * (1.0 - center_shift) + 0.05 * size_score
                    candidates.append(
                        (score, state, edge, gap, overlap, center_shift, area_ratio)
                    )

            # Keep one best edge per logical target before ambiguity checking.
            best_by_logical = {}
            for candidate in candidates:
                logical = candidate[1].logical_id
                previous = best_by_logical.get(logical)
                if previous is None or candidate[0] > previous[0]:
                    best_by_logical[logical] = candidate
            ranked = sorted(best_by_logical.values(), key=lambda item: item[0], reverse=True)
            if not ranked:
                continue
            second_score = ranked[1][0] if len(ranked) > 1 else -1.0
            best = ranked[0]
            if best[0] - second_score < config.ambiguity_margin:
                LOGGER.info(
                    "[TRACK_CONTINUITY_EDGE_AMBIGUOUS] camera=%s raw=%s candidates=%d "
                    "best_score=%.3f second_score=%.3f",
                    packet.camera_id,
                    native_raw_ids.get(track.track_id, track.track_id),
                    len(ranked),
                    best[0],
                    second_score,
                )
                continue

            _, state, edge, gap, overlap, center_shift, area_ratio = best
            self._set_provisional_alias(
                packet.camera_id,
                track.track_id,
                state,
                mode=f"edge_{edge}",
            )
            occupied_logical.add(state.logical_id)
            LOGGER.info(
                "[TRACK_CONTINUITY_EDGE_BRIDGE] camera=%s raw=%s logical=%s edge=%s "
                "gap=%.3f overlap=%.3f center_shift=%.3f area_ratio=%.3f "
                "reid=missing status=provisional shadow_source=unavailable_safe_fallback",
                packet.camera_id,
                native_raw_ids.get(track.track_id, track.track_id),
                state.logical_id,
                edge,
                gap,
                overlap,
                center_shift,
                area_ratio,
            )

    def _set_provisional_alias(self, camera_id, raw_id, state, *, mode: str) -> None:
        raw_key = (camera_id, raw_id)
        self._raw_to_logical[raw_key] = state.logical_id
        self._provisional_raw[raw_key] = state.logical_id
        self._provisional_mode[raw_key] = mode
        state.raw_ids.add(raw_id)
        state.current_raw_id = raw_id

    def _epoch_raw_id(self, camera_id: str, raw_track_id):
        generation = self._stream_generations.get(camera_id, 0)
        if generation <= 0:
            return raw_track_id
        prefix = f"epoch-{generation}:"
        if isinstance(raw_track_id, str) and raw_track_id.startswith(prefix):
            return raw_track_id
        return f"{prefix}{raw_track_id}"

    def _namespace_packet(self, packet: FramePacket) -> tuple[FramePacket, dict[object, object]]:
        generation = self._stream_generations.get(packet.camera_id, 0)
        if generation <= 0 or not packet.tracks:
            return packet, {}
        token_by_raw = {
            track.track_id: self._epoch_raw_id(packet.camera_id, track.track_id)
            for track in packet.tracks
        }
        native_by_token = {token: raw for raw, token in token_by_raw.items()}
        tracks = tuple(
            replace(track, track_id=token_by_raw[track.track_id]) for track in packet.tracks
        )
        faces = tuple(
            replace(face, track_id=token_by_raw.get(face.track_id, face.track_id))
            for face in packet.faces
        )
        behaviors = tuple(
            replace(
                behavior,
                track_id=token_by_raw.get(behavior.track_id, behavior.track_id),
            )
            for behavior in packet.behaviors
        )
        return replace(packet, tracks=tracks, faces=faces, behaviors=behaviors), native_by_token

    @staticmethod
    def _restore_native_raw_metadata(
        packet: FramePacket,
        native_raw_ids: dict[object, object],
    ) -> FramePacket:
        if not native_raw_ids:
            return packet
        tracks = []
        for track in packet.tracks:
            metadata = dict(track.metadata)
            raw_id = metadata.get("raw_track_id")
            if raw_id in native_raw_ids:
                metadata["raw_track_id"] = native_raw_ids[raw_id]
            tracks.append(replace(track, metadata=metadata))
        return replace(packet, tracks=tuple(tracks))

    def _verify_provisional_assignments(self, packet: FramePacket) -> None:
        """Confirm or revoke a geometry alias as soon as real ReID arrives."""

        for track in packet.tracks:
            raw_key = (packet.camera_id, track.track_id)
            logical = self._provisional_raw.get(raw_key)
            if logical is None:
                continue
            mode = self._provisional_mode.get(raw_key, "geometry")
            state = self._states.get((packet.camera_id, logical))
            if state is None or self._raw_to_logical.get(raw_key) != logical:
                self._provisional_raw.pop(raw_key, None)
                self._provisional_mode.pop(raw_key, None)
                continue
            incoming = _track_reid_embedding(track)
            if incoming is None or state.reid_embedding is None:
                continue
            similarity = _cosine(incoming, state.reid_embedding)
            if similarity >= self.config.reid_match_min:
                self._provisional_raw.pop(raw_key, None)
                self._provisional_mode.pop(raw_key, None)
                LOGGER.info(
                    "[TRACK_CONTINUITY_SINGLE_TARGET_CONFIRM] camera=%s raw=%s "
                    "logical=%s similarity=%.3f mode=%s",
                    packet.camera_id,
                    track.track_id,
                    logical,
                    similarity,
                    mode,
                )
                continue

            # A real vector below the strict identity threshold has precedence
            # over the earlier geometry-only guess. Remove the alias before the
            # base resolver sees this packet; it can still recover through its
            # existing face-backed/borderline rules when evidence supports it.
            self._provisional_raw.pop(raw_key, None)
            self._provisional_mode.pop(raw_key, None)
            self._raw_to_logical.pop(raw_key, None)
            state.raw_ids.discard(track.track_id)
            if state.current_raw_id == track.track_id:
                state.current_raw_id = None
            LOGGER.info(
                "[TRACK_CONTINUITY_SINGLE_TARGET_REJECT] camera=%s raw=%s "
                "candidate_logical=%s similarity=%.3f threshold=%.3f mode=%s",
                packet.camera_id,
                track.track_id,
                logical,
                similarity,
                self.config.reid_match_min,
                mode,
            )


def _lateral_edges(bbox, frame_width: int, margin_ratio: float) -> set[str]:
    margin = max(1.0, float(frame_width) * margin_ratio)
    edges: set[str] = set()
    if bbox.x1 <= margin:
        edges.add("left")
    if bbox.x2 >= float(frame_width) - margin:
        edges.add("right")
    return edges


def _parallel_edge_metrics(previous, current) -> tuple[float, float]:
    intersection = max(0.0, min(previous.y2, current.y2) - max(previous.y1, current.y1))
    shorter = max(1.0, min(previous.height, current.height))
    overlap = intersection / shorter
    previous_center = (previous.y1 + previous.y2) * 0.5
    current_center = (current.y1 + current.y2) * 0.5
    center_shift = abs(current_center - previous_center) / max(
        1.0,
        previous.height,
        current.height,
    )
    return min(1.0, overlap), center_shift


def _has_edge_competitor(
    packet: FramePacket,
    raw_id,
    bbox,
    edge: str,
    frame_width: int,
    config: EdgeBridgeConfig,
) -> bool:
    for other in packet.tracks:
        if other.track_id == raw_id:
            continue
        if edge not in _lateral_edges(other.bbox, frame_width, config.edge_margin_ratio):
            continue
        overlap, _ = _parallel_edge_metrics(other.bbox, bbox)
        if overlap >= max(0.20, config.min_parallel_overlap * 0.75):
            return True
    return False


__all__ = [
    "EdgeBridgeConfig",
    "GuardedTrackContinuityResolver",
    "SingleTargetBridgeConfig",
]
