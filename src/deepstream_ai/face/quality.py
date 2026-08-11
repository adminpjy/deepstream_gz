"""Multi-frame face quality scoring and embedding fusion."""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock

import numpy as np

from deepstream_ai.domain import FaceDetection, TrackId
from deepstream_ai.face.errors import InvalidFaceInput

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceQualityWeights:
    face_score: float = 0.35
    size: float = 0.20
    blur: float = 0.25
    frontal: float = 0.20

    def __post_init__(self) -> None:
        values = (self.face_score, self.size, self.blur, self.frontal)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("face quality weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one face quality weight must be positive")

    @property
    def total(self) -> float:
        return self.face_score + self.size + self.blur + self.frontal

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> FaceQualityWeights:
        if not values:
            return cls()
        defaults = cls()
        return cls(
            face_score=float(values.get("face_score", defaults.face_score)),
            size=float(values.get("size", defaults.size)),
            blur=float(values.get("blur", defaults.blur)),
            frontal=float(values.get("frontal", defaults.frontal)),
        )


@dataclass(frozen=True, slots=True)
class FaceFusionConfig:
    min_candidates: int = 3
    max_candidates: int = 8
    max_track_age_seconds: float = 3.0
    face_area_reference: float = 112.0 * 112.0
    blur_reference: float = 120.0
    min_quality: float = 0.0
    weights: FaceQualityWeights = field(default_factory=FaceQualityWeights)
    frame_color_space: str = "rgba"

    def __post_init__(self) -> None:
        if self.min_candidates < 1:
            raise ValueError("min_candidates must be positive")
        if self.max_candidates < self.min_candidates:
            raise ValueError("max_candidates must be >= min_candidates")
        if self.max_track_age_seconds <= 0:
            raise ValueError("max_track_age_seconds must be positive")
        if self.face_area_reference <= 0 or self.blur_reference <= 0:
            raise ValueError("quality reference values must be positive")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be between 0 and 1")
        color = self.frame_color_space.strip().lower()
        if color not in {"rgb", "rgba", "bgr", "bgra"}:
            raise ValueError("frame_color_space must be rgb, rgba, bgr, or bgra")
        object.__setattr__(self, "frame_color_space", color)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> FaceFusionConfig:
        if not values:
            return cls()
        weights = FaceQualityWeights.from_mapping(
            values.get("quality_weights")
            if isinstance(values.get("quality_weights"), Mapping)
            else None
        )
        return cls(
            min_candidates=int(values.get("min_frames", values.get("min_candidates", 3))),
            max_candidates=int(values.get("max_candidates", 8)),
            max_track_age_seconds=float(
                values.get(
                    "max_track_age_seconds",
                    values.get("decision_timeout_sec", 3.0),
                )
            ),
            face_area_reference=float(values.get("face_area_reference", 112.0 * 112.0)),
            blur_reference=float(values.get("blur_reference", 120.0)),
            min_quality=float(values.get("min_quality", 0.0)),
            weights=weights,
            frame_color_space=str(values.get("frame_color_space", "rgba")),
        )

    @classmethod
    def from_runtime_config(cls, config: object) -> FaceFusionConfig:
        """Adapt the project's typed config without importing its module."""

        return cls(
            min_candidates=int(getattr(config, "min_candidates", 3)),
            max_candidates=int(getattr(config, "max_candidates", 8)),
            max_track_age_seconds=float(getattr(config, "decision_timeout_sec", 3.0)),
            frame_color_space=str(getattr(config, "frame_color_space", "rgba")),
        )


@dataclass(frozen=True, slots=True)
class FaceCandidate:
    detection: FaceDetection
    quality: float
    size_score: float
    blur_score: float
    frontal_score: float
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("quality", "size_score", "blur_score", "frontal_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        if self.embedding is not None:
            object.__setattr__(self, "embedding", _validate_embedding(self.embedding))


def _validate_embedding(embedding: np.ndarray | Sequence[float]) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != 512:
        raise InvalidFaceInput(f"AdaFace must produce 512 values, got {vector.size}")
    if not np.all(np.isfinite(vector)):
        raise InvalidFaceInput("embedding contains NaN or infinity")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise InvalidFaceInput("embedding has zero norm")
    normalized = vector / norm
    normalized.setflags(write=False)
    return normalized


normalize_embedding = _validate_embedding


class FaceQualityScorer:
    """Calculate comparable quality scores without requiring OpenCV."""

    def __init__(self, config: FaceFusionConfig | None = None) -> None:
        self.config = config or FaceFusionConfig()

    def score(
        self,
        detection: FaceDetection,
        *,
        crop: np.ndarray | None = None,
        frame_shape: Sequence[int] | None = None,
        embedding: np.ndarray | Sequence[float] | None = None,
    ) -> FaceCandidate:
        del frame_shape  # Face area uses source coordinates and a configured reference.
        size_score = min(1.0, detection.bbox.area / self.config.face_area_reference)
        blur_score = self._blur_score(crop, detection.metadata)
        frontal_score = self._frontal_score(detection.landmarks, detection.metadata)
        weights = self.config.weights
        quality = (
            weights.face_score * detection.score
            + weights.size * size_score
            + weights.blur * blur_score
            + weights.frontal * frontal_score
        ) / weights.total
        return FaceCandidate(
            detection=detection,
            quality=max(0.0, min(1.0, quality)),
            size_score=size_score,
            blur_score=blur_score,
            frontal_score=frontal_score,
            embedding=None if embedding is None else np.asarray(embedding),
        )

    def _blur_score(self, crop: np.ndarray | None, metadata: Mapping[str, object]) -> float:
        supplied = metadata.get("blur_score")
        if supplied is not None:
            return _unit(float(supplied))
        supplied_variance = metadata.get("blur_variance")
        if supplied_variance is not None:
            return _unit(float(supplied_variance) / self.config.blur_reference)
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            return 0.0
        gray = _grayscale(crop, self.config.frame_color_space)
        if min(gray.shape) < 3:
            return 0.0
        # Discrete Laplacian variance is a useful sharpness signal and keeps
        # this module usable in CPU-only tests where cv2 is intentionally absent.
        center = gray[1:-1, 1:-1]
        laplacian = (
            gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
        )
        return _unit(float(np.var(laplacian)) / self.config.blur_reference)

    @staticmethod
    def _frontal_score(
        landmarks: Sequence[tuple[float, float]], metadata: Mapping[str, object]
    ) -> float:
        supplied = metadata.get("frontal_score")
        if supplied is not None:
            return _unit(float(supplied))
        if len(landmarks) < 3:
            return 0.5
        left_eye = np.asarray(landmarks[0], dtype=np.float64)
        right_eye = np.asarray(landmarks[1], dtype=np.float64)
        nose = np.asarray(landmarks[2], dtype=np.float64)
        eye_span = float(np.linalg.norm(right_eye - left_eye))
        if eye_span <= 1e-6:
            return 0.0
        left_distance = float(np.linalg.norm(nose - left_eye))
        right_distance = float(np.linalg.norm(nose - right_eye))
        yaw_asymmetry = abs(left_distance - right_distance) / eye_span
        eye_tilt = abs(float(right_eye[1] - left_eye[1])) / eye_span
        return _unit(1.0 - 0.8 * yaw_asymmetry - 0.2 * eye_tilt)


def _grayscale(image: np.ndarray, color_space: str = "rgba") -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32, copy=False)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise InvalidFaceInput("face crop must be HxW, HxWx1, HxWx3, or HxWx4")
    if array.shape[2] == 1:
        return array[..., 0].astype(np.float32, copy=False)
    pixels = array[..., :3].astype(np.float32, copy=False)
    if color_space.startswith("bgr"):
        return 0.114 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.299 * pixels[..., 2]
    return 0.299 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.114 * pixels[..., 2]


def _unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


class MultiFrameFaceFusion:
    """Bounded, thread-safe candidate buffers keyed by camera and track."""

    def __init__(
        self,
        config: FaceFusionConfig | None = None,
        scorer: FaceQualityScorer | None = None,
    ) -> None:
        self.config = config or FaceFusionConfig()
        self.scorer = scorer or FaceQualityScorer(self.config)
        self._buffers: dict[tuple[str, TrackId], deque[FaceCandidate]] = {}
        self._last_seen: dict[tuple[str, TrackId], datetime] = {}
        self._lock = RLock()

    def add(
        self,
        detection: FaceDetection,
        *,
        crop: np.ndarray | None = None,
        frame_shape: Sequence[int] | None = None,
        embedding: np.ndarray | Sequence[float] | None = None,
    ) -> FaceCandidate:
        candidate = self.scorer.score(
            detection,
            crop=crop,
            frame_shape=frame_shape,
            embedding=embedding,
        )
        if candidate.quality < self.config.min_quality:
            LOGGER.debug(
                "Rejected low-quality face camera=%s track=%s quality=%.3f",
                detection.camera_id,
                detection.track_id,
                candidate.quality,
            )
            return candidate
        with self._lock:
            buffer = self._buffers.setdefault(
                detection.key, deque(maxlen=self.config.max_candidates)
            )
            buffer.append(candidate)
            self._last_seen[detection.key] = detection.timestamp
        return candidate

    def count(self, camera_id: str, track_id: TrackId) -> int:
        with self._lock:
            return len(self._buffers.get((camera_id, track_id), ()))

    def is_ready(self, camera_id: str, track_id: TrackId) -> bool:
        return self.count(camera_id, track_id) >= self.config.min_candidates

    def candidates(self, camera_id: str, track_id: TrackId) -> tuple[FaceCandidate, ...]:
        with self._lock:
            return tuple(self._buffers.get((camera_id, track_id), ()))

    def best(self, camera_id: str, track_id: TrackId) -> FaceCandidate | None:
        values = self.candidates(camera_id, track_id)
        return max(values, key=lambda candidate: candidate.quality, default=None)

    def fused_embedding(self, camera_id: str, track_id: TrackId) -> np.ndarray:
        values = [
            candidate
            for candidate in self.candidates(camera_id, track_id)
            if candidate.embedding is not None
        ]
        if not values:
            raise InvalidFaceInput("no candidate embeddings are available for this track")
        matrix = np.stack([candidate.embedding for candidate in values], axis=0)
        weights = np.asarray(
            [max(candidate.quality, 1e-6) for candidate in values], dtype=np.float32
        )
        fused = np.average(matrix, axis=0, weights=weights)
        return _validate_embedding(fused)

    def consume(self, camera_id: str, track_id: TrackId) -> tuple[FaceCandidate, ...]:
        key = (camera_id, track_id)
        with self._lock:
            values = tuple(self._buffers.pop(key, ()))
            self._last_seen.pop(key, None)
        return values

    def discard(self, camera_id: str, track_id: TrackId) -> None:
        self.consume(camera_id, track_id)

    def expire(self, now: datetime | None = None) -> tuple[tuple[str, TrackId], ...]:
        now = now or datetime.now(timezone.utc)
        expired: list[tuple[str, TrackId]] = []
        with self._lock:
            for key, last_seen in tuple(self._last_seen.items()):
                reference_now = now
                if last_seen.tzinfo is None and now.tzinfo is not None:
                    reference_now = now.replace(tzinfo=None)
                if last_seen.tzinfo is not None and now.tzinfo is None:
                    reference_now = now.replace(tzinfo=last_seen.tzinfo)
                if (reference_now - last_seen).total_seconds() >= self.config.max_track_age_seconds:
                    expired.append(key)
                    self._buffers.pop(key, None)
                    self._last_seen.pop(key, None)
        return tuple(expired)


FaceCandidateBuffer = MultiFrameFaceFusion


__all__ = [
    "FaceCandidate",
    "FaceCandidateBuffer",
    "FaceFusionConfig",
    "FaceQualityScorer",
    "FaceQualityWeights",
    "MultiFrameFaceFusion",
    "normalize_embedding",
]
