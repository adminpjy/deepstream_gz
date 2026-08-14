"""High-level quality-first AdaFace recognition service.

SCRFD is responsible for finding faces.  This service keeps a small per-track
candidate window and decides *when* AdaFace/database comparison is worth doing:
- a clear face is compared immediately, even if it is the first usable frame;
- if no clear face appears, the best real SCRFD candidate is compared after a
  short timeout so a track cannot wait forever;
- unknown tracks keep collecting faces and are retried at a bounded cadence;
- a materially better later face may upgrade an earlier result;
- track finalization performs one last comparison when real face evidence was
  captured but no known identity was obtained.

The comparison policy never fabricates a face.  Every candidate comes from a
real SCRFD detection with the configured landmark requirement.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from deepstream_ai.database import FaceVectorRepository
from deepstream_ai.domain import FaceDetection, IdentityResult, TrackId
from deepstream_ai.face.alignment import FivePointFaceAligner
from deepstream_ai.face.embedding import FaceEmbedder
from deepstream_ai.face.errors import InvalidFaceInput
from deepstream_ai.face.quality import FaceFusionConfig, MultiFrameFaceFusion, normalize_embedding

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceRecognitionConfig:
    similarity_threshold: float = 0.55
    recognize_once_per_track: bool = False
    require_landmarks: bool = True
    fusion: FaceFusionConfig = field(default_factory=FaceFusionConfig)
    # Quality is the existing weighted detector/size/sharpness/frontal score.
    # Around 0.72 has proven to represent a genuinely useful CCTV face without
    # requiring a perfect frontal portrait.
    high_quality_threshold: float = 0.72
    # A later face must normally improve this much before re-comparing a known
    # track. Unknown tracks additionally get the bounded retry path below.
    retry_quality_improvement: float = 0.06
    # Unknown identities get another chance even when the scalar quality score
    # is similar, because a different pose can still produce a better embedding.
    unknown_retry_sec: float = 2.0
    # Fuse only the strongest recent faces for one comparison. Weak side/blurred
    # candidates should not dilute a later good embedding.
    max_compare_candidates: int = 3

    def __post_init__(self) -> None:
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if not 0.0 <= self.high_quality_threshold <= 1.0:
            raise ValueError("high_quality_threshold must be between 0 and 1")
        if not 0.0 <= self.retry_quality_improvement <= 1.0:
            raise ValueError("retry_quality_improvement must be between 0 and 1")
        if self.unknown_retry_sec <= 0:
            raise ValueError("unknown_retry_sec must be positive")
        if self.max_compare_candidates < 1:
            raise ValueError("max_compare_candidates must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "FaceRecognitionConfig":
        if not values:
            return cls()
        if isinstance(values.get("face_recognition"), Mapping):
            values = values["face_recognition"]  # type: ignore[assignment]
        return cls(
            similarity_threshold=float(
                values.get("similarity_threshold", values.get("match_threshold", 0.55))
            ),
            recognize_once_per_track=bool(values.get("recognize_once_per_track", False)),
            require_landmarks=bool(values.get("require_landmarks", True)),
            fusion=FaceFusionConfig.from_mapping(values),
            high_quality_threshold=float(values.get("high_quality_threshold", 0.72)),
            retry_quality_improvement=float(values.get("retry_quality_improvement", 0.06)),
            unknown_retry_sec=float(values.get("unknown_retry_sec", 2.0)),
            max_compare_candidates=int(values.get("max_compare_candidates", 3)),
        )

    @classmethod
    def from_runtime_config(cls, config: object) -> "FaceRecognitionConfig":
        """Adapt :mod:`deepstream_ai.config`'s slots dataclass by attributes."""

        return cls(
            similarity_threshold=float(
                getattr(config, "match_threshold", getattr(config, "similarity_threshold", 0.55))
            ),
            recognize_once_per_track=bool(getattr(config, "recognize_once_per_track", False)),
            require_landmarks=bool(getattr(config, "require_landmarks", True)),
            fusion=FaceFusionConfig.from_runtime_config(config),
            high_quality_threshold=float(getattr(config, "high_quality_threshold", 0.72)),
            retry_quality_improvement=float(
                getattr(config, "retry_quality_improvement", 0.06)
            ),
            unknown_retry_sec=float(getattr(config, "unknown_retry_sec", 2.0)),
            max_compare_candidates=int(getattr(config, "max_compare_candidates", 3)),
        )


@dataclass(frozen=True, slots=True)
class _PendingFace:
    detection: FaceDetection
    crop: np.ndarray = field(repr=False, compare=False)
    quality: float


class FaceRecognitionService:
    """Collect real SCRFD faces and compare them using a zero-miss-first policy."""

    def __init__(
        self,
        embedder: FaceEmbedder,
        repository: FaceVectorRepository,
        config: FaceRecognitionConfig | None = None,
        *,
        fusion: MultiFrameFaceFusion | None = None,
        aligner: FivePointFaceAligner | None = None,
    ) -> None:
        if not isinstance(embedder, FaceEmbedder):
            raise TypeError("embedder must provide embed(face_crop)")
        if not isinstance(repository, FaceVectorRepository):
            raise TypeError("repository does not implement the face-vector contract")
        self.embedder = embedder
        self.repository = repository
        self.config = config or FaceRecognitionConfig()
        # Keep the existing scorer/config contract, but delay expensive AdaFace
        # embedding until the policy decides a comparison is due.
        self.fusion = fusion or MultiFrameFaceFusion(self.config.fusion)
        self.aligner = aligner or FivePointFaceAligner()
        self._pending: dict[tuple[str, TrackId], deque[_PendingFace]] = {}
        self._window_started: dict[tuple[str, TrackId], datetime] = {}
        self._recognized: dict[tuple[str, TrackId], IdentityResult] = {}
        self._last_recognition_at: dict[tuple[str, TrackId], datetime] = {}
        self._last_recognition_quality: dict[tuple[str, TrackId], float] = {}

    def observe(
        self,
        detection: FaceDetection,
        face_crop: np.ndarray | None = None,
        *,
        frame_shape: Sequence[int] | None = None,
    ) -> IdentityResult | None:
        key = detection.key
        if self.config.recognize_once_per_track and key in self._recognized:
            return self._recognized[key]
        crop = face_crop if face_crop is not None else detection.crop
        if crop is None:
            raise ValueError("face_crop is required for AdaFace inference")
        raw_crop = np.asarray(crop)
        if raw_crop.size == 0:
            raise InvalidFaceInput("face crop is empty")
        if len(detection.landmarks) < 5 and self.config.require_landmarks:
            raise InvalidFaceInput(
                "face detector did not provide five landmarks; refusing unaligned AdaFace inference"
            )

        candidate = self.fusion.scorer.score(
            detection,
            crop=raw_crop,
            frame_shape=frame_shape,
            embedding=None,
        )
        if candidate.quality < self.config.fusion.min_quality:
            return None

        pending = self._pending.setdefault(
            key,
            deque(maxlen=self.config.fusion.max_candidates),
        )
        pending.append(
            _PendingFace(
                detection=detection,
                crop=np.ascontiguousarray(raw_crop).copy(),
                quality=candidate.quality,
            )
        )
        self._window_started.setdefault(key, detection.timestamp)

        reason = self._comparison_reason(key, detection.timestamp)
        if reason is None:
            return None
        return self.recognize_track(
            *key,
            allow_single=True,
            reason=reason,
        )

    def _comparison_reason(
        self,
        key: tuple[str, TrackId],
        now: datetime,
    ) -> str | None:
        pending = self._pending.get(key)
        if not pending:
            return None
        best_quality = max(item.quality for item in pending)
        previous = self._recognized.get(key)
        last_quality = self._last_recognition_quality.get(key)
        materially_better = (
            last_quality is None
            or best_quality >= last_quality + self.config.retry_quality_improvement
        )

        if best_quality >= self.config.high_quality_threshold and (
            previous is None or materially_better
        ):
            return "high_quality" if previous is None else "quality_upgrade"

        started = self._window_started.get(key, now)
        window_age = _seconds_between(now, started)
        if previous is None and window_age >= self.config.fusion.max_track_age_seconds:
            return "timeout_fallback"

        if previous is not None and not previous.known:
            last_at = self._last_recognition_at.get(key)
            if (
                last_at is not None
                and _seconds_between(now, last_at) >= self.config.unknown_retry_sec
                and len(pending) >= self.config.fusion.min_candidates
            ):
                return "unknown_retry"
        return None

    def recognize_track(
        self,
        camera_id: str,
        track_id: TrackId,
        *,
        allow_single: bool = False,
        reason: str = "manual",
    ) -> IdentityResult:
        key = (camera_id, track_id)
        pending = tuple(self._pending.get(key, ()))
        minimum = 1 if allow_single else self.config.fusion.min_candidates
        if len(pending) < minimum:
            raise ValueError("track does not yet have enough face candidates")

        ordered = sorted(pending, key=lambda item: item.quality, reverse=True)
        compare_count = min(len(ordered), self.config.max_compare_candidates)
        if not allow_single:
            compare_count = max(self.config.fusion.min_candidates, compare_count)
        selected = ordered[:compare_count]
        embeddings: list[np.ndarray] = []
        weights: list[float] = []
        for item in selected:
            if len(item.detection.landmarks) >= 5:
                embedding_crop = self.aligner.align(item.crop, item.detection)
            elif self.config.require_landmarks:
                continue
            else:
                embedding_crop = item.crop
            embeddings.append(self.embedder.embed(embedding_crop))
            weights.append(max(item.quality, 1e-6))
        if not embeddings:
            raise InvalidFaceInput("no valid face candidate could be embedded")

        matrix = np.stack(embeddings, axis=0)
        fused = np.average(matrix, axis=0, weights=np.asarray(weights, dtype=np.float32))
        embedding = normalize_embedding(fused)
        match = self.repository.find_nearest(embedding)
        best = selected[0]
        average_quality = float(np.mean([item.quality for item in selected]))
        similarity = -1.0 if match is None else match.similarity
        known = match is not None and similarity >= self.config.similarity_threshold
        result = IdentityResult(
            camera_id=camera_id,
            track_id=track_id,
            timestamp=best.detection.timestamp,
            worker_id=match.worker_id if known else None,
            similarity=similarity,
            confidence=max(0.0, min(1.0, average_quality * max(0.0, similarity))),
            sample_count=len(embeddings),
        )

        preferred = self._prefer_result(self._recognized.get(key), result)
        self._recognized[key] = preferred
        self._last_recognition_at[key] = best.detection.timestamp
        self._last_recognition_quality[key] = max(
            best.quality,
            self._last_recognition_quality.get(key, -1.0),
        )
        self._pending.pop(key, None)
        self._window_started.pop(key, None)
        self.fusion.consume(camera_id, track_id)

        LOGGER.info(
            "[FACE_IDENTITY] camera=%s track=%s reason=%s candidate_known=%s "
            "candidate_similarity=%.3f retained_known=%s retained_similarity=%.3f "
            "samples=%d best_quality=%.3f",
            camera_id,
            track_id,
            reason,
            result.known,
            result.similarity,
            preferred.known,
            preferred.similarity,
            result.sample_count,
            best.quality,
        )
        return preferred

    def finalize_track(self, camera_id: str, track_id: TrackId) -> IdentityResult | None:
        """Run the final no-miss fallback for a track that still owns real faces."""

        key = (camera_id, track_id)
        pending = self._pending.get(key)
        if not pending:
            return None
        previous = self._recognized.get(key)
        if previous is not None and previous.known:
            return None
        return self.recognize_track(
            camera_id,
            track_id,
            allow_single=True,
            reason="finalize_fallback",
        )

    @staticmethod
    def _prefer_result(
        previous: IdentityResult | None,
        incoming: IdentityResult,
    ) -> IdentityResult:
        if previous is None:
            return incoming
        # Never downgrade a known identity to unknown. Among known results, keep
        # the stronger similarity; an unknown result may always be upgraded.
        if previous.known and not incoming.known:
            return previous
        if incoming.known and not previous.known:
            return incoming
        if previous.known and incoming.known:
            return incoming if incoming.similarity > previous.similarity else previous
        return incoming if incoming.similarity > previous.similarity else previous

    def result_for(self, camera_id: str, track_id: TrackId) -> IdentityResult | None:
        return self._recognized.get((camera_id, track_id))

    def discard_track(self, camera_id: str, track_id: TrackId) -> None:
        key = (camera_id, track_id)
        self.fusion.discard(camera_id, track_id)
        self._pending.pop(key, None)
        self._window_started.pop(key, None)
        self._recognized.pop(key, None)
        self._last_recognition_at.pop(key, None)
        self._last_recognition_quality.pop(key, None)


def _seconds_between(later: datetime, earlier: datetime) -> float:
    if later.tzinfo is None and earlier.tzinfo is not None:
        later = later.replace(tzinfo=earlier.tzinfo)
    elif later.tzinfo is not None and earlier.tzinfo is None:
        earlier = earlier.replace(tzinfo=later.tzinfo)
    return (later - earlier).total_seconds()


__all__ = ["FaceRecognitionConfig", "FaceRecognitionService"]
