"""High-level multi-frame AdaFace recognition service."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from deepstream_ai.database import FaceVectorRepository
from deepstream_ai.domain import FaceDetection, IdentityResult, TrackId
from deepstream_ai.face.alignment import FivePointFaceAligner
from deepstream_ai.face.embedding import FaceEmbedder
from deepstream_ai.face.errors import InvalidFaceInput
from deepstream_ai.face.quality import FaceFusionConfig, MultiFrameFaceFusion

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceRecognitionConfig:
    similarity_threshold: float = 0.55
    recognize_once_per_track: bool = True
    require_landmarks: bool = True
    fusion: FaceFusionConfig = field(default_factory=FaceFusionConfig)

    def __post_init__(self) -> None:
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> FaceRecognitionConfig:
        if not values:
            return cls()
        if isinstance(values.get("face_recognition"), Mapping):
            values = values["face_recognition"]  # type: ignore[assignment]
        return cls(
            similarity_threshold=float(
                values.get("similarity_threshold", values.get("match_threshold", 0.55))
            ),
            recognize_once_per_track=bool(values.get("recognize_once_per_track", True)),
            require_landmarks=bool(values.get("require_landmarks", True)),
            fusion=FaceFusionConfig.from_mapping(values),
        )

    @classmethod
    def from_runtime_config(cls, config: object) -> FaceRecognitionConfig:
        """Adapt :mod:`deepstream_ai.config`'s slots dataclass by attributes."""

        return cls(
            similarity_threshold=float(
                getattr(config, "match_threshold", getattr(config, "similarity_threshold", 0.55))
            ),
            recognize_once_per_track=bool(getattr(config, "recognize_once_per_track", True)),
            require_landmarks=bool(getattr(config, "require_landmarks", True)),
            fusion=FaceFusionConfig.from_runtime_config(config),
        )


class FaceRecognitionService:
    """Collect, quality-weight, fuse, and identify faces per stable track."""

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
        self.fusion = fusion or MultiFrameFaceFusion(self.config.fusion)
        self.aligner = aligner or FivePointFaceAligner()
        self._recognized: dict[tuple[str, TrackId], IdentityResult] = {}

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
        if len(detection.landmarks) >= 5:
            embedding_crop = self.aligner.align(raw_crop, detection)
        elif self.config.require_landmarks:
            raise InvalidFaceInput(
                "face detector did not provide five landmarks; refusing unaligned AdaFace inference"
            )
        else:
            embedding_crop = raw_crop
        embedding = self.embedder.embed(embedding_crop)
        self.fusion.add(
            detection,
            crop=raw_crop,
            frame_shape=frame_shape,
            embedding=embedding,
        )
        if not self.fusion.is_ready(*key):
            return None
        return self.recognize_track(*key)

    def recognize_track(self, camera_id: str, track_id: TrackId) -> IdentityResult:
        candidates = self.fusion.candidates(camera_id, track_id)
        if len(candidates) < self.config.fusion.min_candidates:
            raise ValueError("track does not yet have enough face candidates")
        embedding = self.fusion.fused_embedding(camera_id, track_id)
        match = self.repository.find_nearest(embedding)
        best = max(candidates, key=lambda candidate: candidate.quality)
        average_quality = float(np.mean([candidate.quality for candidate in candidates]))
        similarity = -1.0 if match is None else match.similarity
        known = match is not None and similarity >= self.config.similarity_threshold
        result = IdentityResult(
            camera_id=camera_id,
            track_id=track_id,
            timestamp=best.detection.timestamp,
            worker_id=match.worker_id if known else None,
            similarity=similarity,
            confidence=max(0.0, min(1.0, average_quality * max(0.0, similarity))),
            sample_count=len(candidates),
        )
        self.fusion.consume(camera_id, track_id)
        if self.config.recognize_once_per_track:
            self._recognized[(camera_id, track_id)] = result
        LOGGER.info(
            "Face recognition camera=%s track=%s known=%s similarity=%.3f samples=%d",
            camera_id,
            track_id,
            result.known,
            result.similarity,
            result.sample_count,
        )
        return result

    def result_for(self, camera_id: str, track_id: TrackId) -> IdentityResult | None:
        return self._recognized.get((camera_id, track_id))

    def discard_track(self, camera_id: str, track_id: TrackId) -> None:
        self.fusion.discard(camera_id, track_id)
        self._recognized.pop((camera_id, track_id), None)


__all__ = ["FaceRecognitionConfig", "FaceRecognitionService"]
