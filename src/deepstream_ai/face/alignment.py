"""Five-point similarity alignment used by AdaFace/ArcFace-style models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from deepstream_ai.domain import FaceDetection
from deepstream_ai.face.errors import FaceRuntimeUnavailable, InvalidFaceInput

_CANONICAL_112 = np.asarray(
    [
        (38.2946, 51.6963),
        (73.5318, 51.5014),
        (56.0252, 71.7366),
        (41.5493, 92.3655),
        (70.7299, 92.2041),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class FivePointFaceAligner:
    output_size: tuple[int, int] = (112, 112)

    def __post_init__(self) -> None:
        if len(self.output_size) != 2 or min(self.output_size) <= 0:
            raise ValueError("output_size must contain positive width and height")

    def align(self, crop: np.ndarray, detection: FaceDetection) -> np.ndarray:
        if len(detection.landmarks) < 5:
            raise InvalidFaceInput("five face landmarks are required for AdaFace alignment")
        image = np.asarray(crop)
        if image.size == 0 or image.ndim != 3:
            raise InvalidFaceInput("face crop must be a non-empty HxWxC image")
        source = np.asarray(detection.landmarks[:5], dtype=np.float64)
        source[:, 0] -= detection.bbox.x1
        source[:, 1] -= detection.bbox.y1
        target = canonical_landmarks(self.output_size)
        matrix = similarity_transform(source, target)
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FaceRuntimeUnavailable(
                "five-point face alignment requires opencv-python-headless"
            ) from exc
        width, height = self.output_size
        try:
            aligned = cv2.warpAffine(
                image,
                matrix.astype(np.float32),
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        except Exception as exc:
            raise InvalidFaceInput("five-point face alignment failed") from exc
        return np.ascontiguousarray(aligned)


def canonical_landmarks(output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    scale = np.asarray((width / 112.0, height / 112.0), dtype=np.float64)
    return _CANONICAL_112 * scale


def similarity_transform(
    source: Sequence[Sequence[float]] | np.ndarray,
    destination: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return a deterministic 2x3 Umeyama similarity transform."""

    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(destination, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or src.shape[0] < 2:
        raise InvalidFaceInput("landmark arrays must have matching Nx2 shapes")
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise InvalidFaceInput("landmarks contain NaN or infinity")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    variance = float(np.sum(src_centered**2) / src.shape[0])
    if variance <= 1e-12:
        raise InvalidFaceInput("face landmarks are degenerate")
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    signs = np.ones(2, dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        signs[-1] = -1.0
    rotation = u @ np.diag(signs) @ vt
    scale = float(np.dot(singular_values, signs) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    return np.column_stack((scale * rotation, translation))


__all__ = ["FivePointFaceAligner", "canonical_landmarks", "similarity_transform"]
