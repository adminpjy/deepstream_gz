"""Business-layer confirmation for weak new PeopleNet/NvDCF tracks.

Low-confidence PeopleNet proposals remain available to NvDCF so an established
person can survive crouching, partial occlusion, and pose changes.  A brand-new
logical track is different: if it begins weak, it stays provisional until later
evidence proves it is a real person.  Provisional tracks are hidden from OSD,
preview, snapshots, face recognition and downstream business logic.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

from deepstream_ai.pipeline.metadata import _TRACKER_REID_METADATA_KEY, FramePacket

LOGGER = logging.getLogger(__name__)
_SECTION = "weak_new_track"


@dataclass(frozen=True, slots=True)
class WeakNewTrackConfig:
    enabled: bool = True
    instant_confirm_confidence: float = 0.35
    sustained_confirm_confidence: float = 0.28
    confirm_after_sec: float = 0.8
    suppress_log_after_sec: float = 1.5
    min_detector_observations: int = 2
    min_width_ratio: float = 0.025
    min_height_ratio: float = 0.10
    stale_retention_sec: float = 30.0

    @classmethod
    def from_file(cls, config_path: str | Path) -> "WeakNewTrackConfig":
        try:
            raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        section = raw.get(_SECTION, {}) if isinstance(raw, dict) else {}
        if not isinstance(section, dict):
            section = {}
        result = cls(
            enabled=bool(section.get("enabled", True)),
            instant_confirm_confidence=float(
                section.get("instant_confirm_confidence", 0.35)
            ),
            sustained_confirm_confidence=float(
                section.get("sustained_confirm_confidence", 0.28)
            ),
            confirm_after_sec=float(section.get("confirm_after_sec", 0.8)),
            suppress_log_after_sec=float(section.get("suppress_log_after_sec", 1.5)),
            min_detector_observations=int(section.get("min_detector_observations", 2)),
            min_width_ratio=float(section.get("min_width_ratio", 0.025)),
            min_height_ratio=float(section.get("min_height_ratio", 0.10)),
            stale_retention_sec=float(section.get("stale_retention_sec", 30.0)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        for name in ("instant_confirm_confidence", "sustained_confirm_confidence"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{_SECTION}.{name} must be between 0 and 1")
        if self.sustained_confirm_confidence > self.instant_confirm_confidence:
            raise ValueError(
                f"{_SECTION}.sustained_confirm_confidence must not exceed instant threshold"
            )
        if self.confirm_after_sec < 0 or self.suppress_log_after_sec < self.confirm_after_sec:
            raise ValueError(f"{_SECTION} timing values are invalid")
        if self.min_detector_observations < 1:
            raise ValueError(f"{_SECTION}.min_detector_observations must be positive")
        if not 0.0 < self.min_width_ratio <= 1.0 or not 0.0 < self.min_height_ratio <= 1.0:
            raise ValueError(f"{_SECTION} size ratios must be between 0 and 1")
        if self.stale_retention_sec <= 0:
            raise ValueError(f"{_SECTION}.stale_retention_sec must be positive")


@dataclass(slots=True)
class _WeakTrackState:
    first_seen: datetime
    last_seen: datetime
    max_detector_confidence: float
    detector_observations: int
    confirmed: bool = False
    suppression_logged: bool = False


class WeakNewTrackGuard:
    """Hide weak brand-new logical tracks until corroborating evidence arrives."""

    def __init__(self, config: WeakNewTrackConfig) -> None:
        self.config = config
        self._states: dict[tuple[str, object], _WeakTrackState] = {}
        self._generations: dict[str, int] = {}
        self._lock = RLock()

    @classmethod
    def from_file(cls, config_path: str | Path) -> "WeakNewTrackGuard":
        return cls(WeakNewTrackConfig.from_file(config_path))

    def begin_stream_generation(self, camera_id: str, generation: int) -> None:
        with self._lock:
            previous = self._generations.get(camera_id)
            if previous is None:
                self._generations[camera_id] = int(generation)
                return
            if previous == int(generation):
                return
            self._generations[camera_id] = int(generation)
            # Confirmed logical identities may legitimately survive a reconnect
            # through the continuity resolver. Only discard unconfirmed junk.
            for key in list(self._states):
                if key[0] == camera_id and not self._states[key].confirmed:
                    self._states.pop(key, None)

    def filter(self, packet: FramePacket) -> FramePacket:
        if not self.config.enabled or not packet.tracks:
            self._purge(packet.timestamp)
            return packet

        with self._lock:
            self._purge(packet.timestamp)
            face_ids = {face.track_id for face in packet.faces}
            visible_ids: set[object] = set()
            image_height, image_width = packet.image.shape[:2]

            for track in packet.tracks:
                key = (track.camera_id, track.track_id)
                state = self._states.get(key)
                is_new = state is None
                has_reid = _TRACKER_REID_METADATA_KEY in track.metadata
                # The first effective NvDCF observation is detector-backed. On
                # later skipped PGIE frames Track.confidence may be tracker
                # confidence, so only treat later ReID-bearing observations as
                # fresh detector evidence.
                detector_observation = is_new or has_reid
                detector_confidence = track.confidence if detector_observation else None
                if state is None:
                    state = _WeakTrackState(
                        first_seen=track.timestamp,
                        last_seen=track.timestamp,
                        max_detector_confidence=float(detector_confidence or 0.0),
                        detector_observations=1 if detector_observation else 0,
                    )
                    self._states[key] = state
                    LOGGER.info(
                        "[PERSON_PROVISIONAL] camera=%s track=%s conf=%.3f bbox=%.0fx%.0f",
                        track.camera_id,
                        track.track_id,
                        track.confidence,
                        track.bbox.width,
                        track.bbox.height,
                    )
                else:
                    state.last_seen = track.timestamp
                    if detector_observation and detector_confidence is not None:
                        state.detector_observations += 1
                        state.max_detector_confidence = max(
                            state.max_detector_confidence,
                            float(detector_confidence),
                        )

                if state.confirmed:
                    visible_ids.add(track.track_id)
                    continue

                age = max(0.0, (track.timestamp - state.first_seen).total_seconds())
                width_ratio = track.bbox.width / max(1.0, float(image_width))
                height_ratio = track.bbox.height / max(1.0, float(image_height))
                reason: str | None = None
                if track.track_id in face_ids:
                    reason = "real_face"
                elif state.max_detector_confidence >= self.config.instant_confirm_confidence:
                    reason = "detector_confidence"
                elif (
                    age >= self.config.confirm_after_sec
                    and state.detector_observations >= self.config.min_detector_observations
                    and state.max_detector_confidence
                    >= self.config.sustained_confirm_confidence
                    and width_ratio >= self.config.min_width_ratio
                    and height_ratio >= self.config.min_height_ratio
                ):
                    reason = "sustained_detector"

                if reason is not None:
                    state.confirmed = True
                    visible_ids.add(track.track_id)
                    LOGGER.info(
                        "[PERSON_CONFIRMED] camera=%s track=%s reason=%s age=%.3f "
                        "max_conf=%.3f detector_observations=%d bbox_ratio=%.4fx%.4f",
                        track.camera_id,
                        track.track_id,
                        reason,
                        age,
                        state.max_detector_confidence,
                        state.detector_observations,
                        width_ratio,
                        height_ratio,
                    )
                    continue

                if age >= self.config.suppress_log_after_sec and not state.suppression_logged:
                    state.suppression_logged = True
                    LOGGER.warning(
                        "[PERSON_FALSE_POSITIVE_SUPPRESSED] camera=%s track=%s age=%.3f "
                        "max_conf=%.3f detector_observations=%d bbox_ratio=%.4fx%.4f",
                        track.camera_id,
                        track.track_id,
                        age,
                        state.max_detector_confidence,
                        state.detector_observations,
                        width_ratio,
                        height_ratio,
                    )

            if len(visible_ids) == len(packet.tracks):
                return packet
            tracks = tuple(track for track in packet.tracks if track.track_id in visible_ids)
            faces = tuple(face for face in packet.faces if face.track_id in visible_ids)
            behaviors = tuple(
                behavior for behavior in packet.behaviors if behavior.track_id in visible_ids
            )
            return replace(packet, tracks=tracks, faces=faces, behaviors=behaviors)

    def is_visible(self, camera_id: str, track_id: object) -> bool:
        if not self.config.enabled:
            return True
        with self._lock:
            state = self._states.get((camera_id, track_id))
            return True if state is None else state.confirmed

    def _purge(self, now: datetime) -> None:
        cutoff = self.config.stale_retention_sec
        for key, state in list(self._states.items()):
            if (now - state.last_seen).total_seconds() > cutoff:
                self._states.pop(key, None)


__all__ = ["WeakNewTrackConfig", "WeakNewTrackGuard"]
