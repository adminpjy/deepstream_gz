"""Short business-ID hold for brand-new NvDCF logical tracks.

A new raw/logical ID is kept out of preview/business events for a short window,
but the analysis packet still carries it with private metadata so SCRFD/AdaFace
can build a continuity anchor. Existing logical IDs recovered by continuity are
visible immediately.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from threading import RLock

import yaml

from deepstream_ai.pipeline.metadata import FramePacket

LOGGER = logging.getLogger(__name__)
BUSINESS_PROVISIONAL_KEY = "_business_provisional"


@dataclass(frozen=True, slots=True)
class ProvisionalTrackConfig:
    enabled: bool = True
    min_confirm_age_sec: float = 0.60
    instant_confirm_confidence: float = 0.35
    sustained_confirm_confidence: float = 0.28
    confirm_after_sec: float = 0.80
    suppress_log_after_sec: float = 1.50
    min_detector_observations: int = 2
    min_width_ratio: float = 0.025
    min_height_ratio: float = 0.10
    stale_retention_sec: float = 30.0

    @classmethod
    def from_file(cls, config_path: str | Path) -> "ProvisionalTrackConfig":
        try:
            raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        section = raw.get("weak_new_track", {}) if isinstance(raw, dict) else {}
        if not isinstance(section, dict):
            section = {}
        result = cls(
            enabled=bool(section.get("enabled", True)),
            min_confirm_age_sec=float(section.get("min_confirm_age_sec", 0.60)),
            instant_confirm_confidence=float(section.get("instant_confirm_confidence", 0.35)),
            sustained_confirm_confidence=float(section.get("sustained_confirm_confidence", 0.28)),
            confirm_after_sec=float(section.get("confirm_after_sec", 0.80)),
            suppress_log_after_sec=float(section.get("suppress_log_after_sec", 1.50)),
            min_detector_observations=int(section.get("min_detector_observations", 2)),
            min_width_ratio=float(section.get("min_width_ratio", 0.025)),
            min_height_ratio=float(section.get("min_height_ratio", 0.10)),
            stale_retention_sec=float(section.get("stale_retention_sec", 30.0)),
        )
        if result.min_confirm_age_sec < 0 or result.confirm_after_sec < result.min_confirm_age_sec:
            raise ValueError("weak_new_track provisional timing is invalid")
        if result.suppress_log_after_sec < result.confirm_after_sec:
            raise ValueError("weak_new_track suppress_log_after_sec is invalid")
        return result


@dataclass(slots=True)
class _State:
    first_seen: datetime
    last_seen: datetime
    max_detector_confidence: float
    detector_observations: int
    confirmed: bool = False
    suppression_logged: bool = False


class ProvisionalTrackGuard:
    """Return separate analysis and visible views of a packet."""

    def __init__(self, config: ProvisionalTrackConfig) -> None:
        self.config = config
        self._states: dict[tuple[str, object], _State] = {}
        self._generations: dict[str, int] = {}
        self._lock = RLock()

    @classmethod
    def from_file(cls, config_path: str | Path) -> "ProvisionalTrackGuard":
        return cls(ProvisionalTrackConfig.from_file(config_path))

    def begin_stream_generation(self, camera_id: str, generation: int) -> None:
        with self._lock:
            previous = self._generations.get(camera_id)
            generation = int(generation)
            self._generations[camera_id] = generation
            if previous is None or previous == generation:
                return
            for key in list(self._states):
                if key[0] == camera_id and not self._states[key].confirmed:
                    self._states.pop(key, None)

    def partition(self, packet: FramePacket) -> tuple[FramePacket, FramePacket]:
        if not self.config.enabled or not packet.tracks:
            self._purge(packet.timestamp)
            return packet, packet

        with self._lock:
            self._purge(packet.timestamp)
            face_ids = {face.track_id for face in packet.faces}
            image_height, image_width = packet.image.shape[:2]
            visible_ids: set[object] = set()
            marked_tracks = []

            for track in packet.tracks:
                key = (track.camera_id, track.track_id)
                state = self._states.get(key)
                raw_value = track.metadata.get("detector_confidence")
                try:
                    detector_confidence = float(raw_value) if raw_value is not None else None
                except (TypeError, ValueError):
                    detector_confidence = None
                if detector_confidence is not None and not math.isfinite(detector_confidence):
                    detector_confidence = None
                if state is None and detector_confidence is None:
                    detector_confidence = track.confidence
                detector_observation = detector_confidence is not None
                raw_id = track.metadata.get("raw_track_id", track.track_id)
                continuity_alias = raw_id != track.track_id

                if state is None:
                    state = _State(
                        first_seen=track.timestamp,
                        last_seen=track.timestamp,
                        max_detector_confidence=float(detector_confidence or 0.0),
                        detector_observations=1 if detector_observation else 0,
                        confirmed=continuity_alias,
                    )
                    self._states[key] = state
                    LOGGER.info(
                        "[PERSON_PROVISIONAL] camera=%s track=%s raw=%s conf=%.3f alias=%s",
                        track.camera_id,
                        track.track_id,
                        raw_id,
                        track.confidence,
                        continuity_alias,
                    )
                else:
                    state.last_seen = track.timestamp
                    if detector_observation and detector_confidence is not None:
                        state.detector_observations += 1
                        state.max_detector_confidence = max(
                            state.max_detector_confidence, detector_confidence
                        )
                    if continuity_alias:
                        state.confirmed = True

                age = max(0.0, (track.timestamp - state.first_seen).total_seconds())
                width_ratio = track.bbox.width / max(1.0, float(image_width))
                height_ratio = track.bbox.height / max(1.0, float(image_height))
                reason: str | None = None
                if not state.confirmed and age >= self.config.min_confirm_age_sec:
                    if track.track_id in face_ids:
                        reason = "real_face"
                    elif state.max_detector_confidence >= self.config.instant_confirm_confidence:
                        reason = "detector_confidence"
                    elif (
                        age >= self.config.confirm_after_sec
                        and state.detector_observations >= self.config.min_detector_observations
                        and state.max_detector_confidence >= self.config.sustained_confirm_confidence
                        and width_ratio >= self.config.min_width_ratio
                        and height_ratio >= self.config.min_height_ratio
                    ):
                        reason = "sustained_detector"
                if reason is not None:
                    state.confirmed = True
                    LOGGER.info(
                        "[PERSON_CONFIRMED] camera=%s track=%s reason=%s age=%.3f max_conf=%.3f observations=%d",
                        track.camera_id,
                        track.track_id,
                        reason,
                        age,
                        state.max_detector_confidence,
                        state.detector_observations,
                    )
                elif (
                    not state.confirmed
                    and age >= self.config.suppress_log_after_sec
                    and not state.suppression_logged
                ):
                    state.suppression_logged = True
                    LOGGER.warning(
                        "[PERSON_FALSE_POSITIVE_SUPPRESSED] camera=%s track=%s age=%.3f max_conf=%.3f observations=%d bbox_ratio=%.4fx%.4f",
                        track.camera_id,
                        track.track_id,
                        age,
                        state.max_detector_confidence,
                        state.detector_observations,
                        width_ratio,
                        height_ratio,
                    )

                metadata = dict(track.metadata)
                metadata[BUSINESS_PROVISIONAL_KEY] = not state.confirmed
                marked_tracks.append(replace(track, metadata=metadata))
                if state.confirmed:
                    visible_ids.add(track.track_id)

            analysis_packet = replace(packet, tracks=tuple(marked_tracks))
            if len(visible_ids) == len(packet.tracks):
                return analysis_packet, analysis_packet
            visible_packet = replace(
                packet,
                tracks=tuple(track for track in marked_tracks if track.track_id in visible_ids),
                faces=tuple(face for face in packet.faces if face.track_id in visible_ids),
                behaviors=tuple(
                    behavior for behavior in packet.behaviors if behavior.track_id in visible_ids
                ),
            )
            return analysis_packet, visible_packet

    def is_visible(self, camera_id: str, track_id: object) -> bool:
        if not self.config.enabled:
            return True
        with self._lock:
            state = self._states.get((camera_id, track_id))
            return True if state is None else state.confirmed

    def _purge(self, now: datetime) -> None:
        for key, state in list(self._states.items()):
            if (now - state.last_seen).total_seconds() > self.config.stale_retention_sec:
                self._states.pop(key, None)


__all__ = ["BUSINESS_PROVISIONAL_KEY", "ProvisionalTrackConfig", "ProvisionalTrackGuard"]
