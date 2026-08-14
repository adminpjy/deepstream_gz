"""Thread-safe short-lived AdaFace anchors for cross-track continuity.

The recognition worker owns AdaFace inference.  Streaming/tracking code must not
run AdaFace synchronously, so this registry is the hand-off point between the two
threads: every real face embedding produced by the recognition service is kept
for a short time and may corroborate a later NvDCF raw-ID fragment.

Anchors are deliberately independent from database identity.  Two faces can be
compared even when both are unknown workers.  The registry stores only normalized
embeddings plus quality/timestamp metadata; it never publishes business events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import RLock

import numpy as np

from deepstream_ai.domain import TrackId


@dataclass(frozen=True, slots=True)
class FaceContinuityAnchor:
    camera_id: str
    track_id: TrackId
    timestamp: datetime
    embedding: np.ndarray
    quality: float


class FaceContinuityAnchorRegistry:
    """Keep the newest high-quality normalized face anchor per track."""

    def __init__(self, *, retention_sec: float = 30.0) -> None:
        if retention_sec <= 0:
            raise ValueError("retention_sec must be positive")
        self.retention_sec = float(retention_sec)
        self._anchors: dict[tuple[str, TrackId], FaceContinuityAnchor] = {}
        self._lock = RLock()

    def observe(
        self,
        camera_id: str,
        track_id: TrackId,
        embedding: np.ndarray,
        *,
        timestamp: datetime,
        quality: float,
    ) -> FaceContinuityAnchor:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("face continuity embedding must be a finite 1-D vector")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("face continuity embedding norm is invalid")
        normalized = np.ascontiguousarray(vector / norm, dtype=np.float32)
        normalized.setflags(write=False)
        anchor = FaceContinuityAnchor(
            camera_id=camera_id,
            track_id=track_id,
            timestamp=timestamp,
            embedding=normalized,
            quality=max(0.0, min(1.0, float(quality))),
        )
        key = (camera_id, track_id)
        with self._lock:
            self._purge_locked(timestamp)
            previous = self._anchors.get(key)
            # Prefer fresher evidence.  For an identical timestamp, retain the
            # clearer face rather than replacing it with a weaker retry sample.
            if previous is None or timestamp > previous.timestamp or (
                timestamp == previous.timestamp and anchor.quality > previous.quality
            ):
                self._anchors[key] = anchor
                return anchor
            return previous

    def latest(
        self,
        camera_id: str,
        track_id: TrackId,
        *,
        now: datetime,
        max_age_sec: float,
        min_quality: float = 0.0,
    ) -> FaceContinuityAnchor | None:
        if max_age_sec <= 0:
            return None
        with self._lock:
            self._purge_locked(now)
            anchor = self._anchors.get((camera_id, track_id))
            if anchor is None or anchor.quality < min_quality:
                return None
            age = _seconds_between(now, anchor.timestamp)
            if age < 0.0 or age > max_age_sec:
                return None
            return anchor

    def alias(
        self,
        camera_id: str,
        source_track_id: TrackId,
        target_track_id: TrackId,
        *,
        now: datetime,
    ) -> None:
        """Copy a fresh provisional raw anchor onto its accepted logical ID."""

        with self._lock:
            self._purge_locked(now)
            source = self._anchors.get((camera_id, source_track_id))
            if source is None:
                return
            target_key = (camera_id, target_track_id)
            target = self._anchors.get(target_key)
            if target is None or source.timestamp > target.timestamp or (
                source.timestamp == target.timestamp and source.quality > target.quality
            ):
                self._anchors[target_key] = FaceContinuityAnchor(
                    camera_id=camera_id,
                    track_id=target_track_id,
                    timestamp=source.timestamp,
                    embedding=source.embedding,
                    quality=source.quality,
                )

    def clear_camera(self, camera_id: str) -> None:
        with self._lock:
            for key in [key for key in self._anchors if key[0] == camera_id]:
                self._anchors.pop(key, None)

    def _purge_locked(self, now: datetime) -> None:
        for key, anchor in list(self._anchors.items()):
            if _seconds_between(now, anchor.timestamp) > self.retention_sec:
                self._anchors.pop(key, None)


def cosine_similarity(first: FaceContinuityAnchor, second: FaceContinuityAnchor) -> float:
    if first.embedding.shape != second.embedding.shape:
        return -1.0
    return float(np.clip(np.dot(first.embedding, second.embedding), -1.0, 1.0))


def _seconds_between(later: datetime, earlier: datetime) -> float:
    if later.tzinfo is None and earlier.tzinfo is not None:
        later = later.replace(tzinfo=earlier.tzinfo)
    elif later.tzinfo is not None and earlier.tzinfo is None:
        earlier = earlier.replace(tzinfo=later.tzinfo)
    return (later - earlier).total_seconds()


FACE_CONTINUITY_ANCHORS = FaceContinuityAnchorRegistry()


__all__ = [
    "FACE_CONTINUITY_ANCHORS",
    "FaceContinuityAnchor",
    "FaceContinuityAnchorRegistry",
    "cosine_similarity",
]
