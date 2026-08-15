"""Stable business-domain contracts shared by the video analytics modules.

The classes in this module deliberately contain no DeepStream, GStreamer, CUDA,
or database types.  Pipeline adapters translate their native metadata into
these values, keeping downstream business code independent of the selected
detector and tracker implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, TypeAlias, runtime_checkable

TrackId: TypeAlias = int | str
Point: TypeAlias = tuple[float, float]


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _probability(value: float, name: str) -> float:
    value = _finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned box using half-open ``(x1, y1, x2, y2)`` coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive width and height")

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> BoundingBox:
        if len(values) != 4:
            raise ValueError("a bounding box requires exactly four coordinates")
        return cls(*map(float, values))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clipped(self, width: int, height: int) -> BoundingBox | None:
        """Return this box clipped to an image, or ``None`` if it is outside."""

        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        x2 = min(max(self.x2, 0.0), float(width))
        y2 = min(max(self.y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2, y2)

    def integer_slices(self, width: int, height: int) -> tuple[slice, slice]:
        clipped = self.clipped(width, height)
        if clipped is None:
            raise ValueError("bounding box does not intersect the image")
        left = max(0, math.floor(clipped.x1))
        top = max(0, math.floor(clipped.y1))
        right = min(width, math.ceil(clipped.x2))
        bottom = min(height, math.ceil(clipped.y2))
        return slice(top, bottom), slice(left, right)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


BBox = BoundingBox


@dataclass(frozen=True, slots=True)
class Detection:
    camera_id: str
    timestamp: datetime
    bbox: BoundingBox
    confidence: float
    class_name: str = "person"
    class_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        if not self.class_name:
            raise ValueError("class_name cannot be empty")
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))


PersonDetection = Detection


@dataclass(frozen=True, slots=True)
class Track:
    """Tracker-neutral observation of a person in one frame."""

    camera_id: str
    track_id: TrackId
    timestamp: datetime
    bbox: BoundingBox
    confidence: float = 1.0
    class_name: str = "person"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        if self.track_id == "" or self.track_id is None:
            raise ValueError("track_id cannot be empty")
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))

    @property
    def key(self) -> tuple[str, TrackId]:
        return self.camera_id, self.track_id


TrackObservation = Track


@runtime_checkable
class Tracker(Protocol):
    """Contract implemented by NvDCF, NvSORT, DeepSORT, or test adapters."""

    def update(
        self,
        camera_id: str,
        timestamp: datetime,
        detections: Sequence[Detection],
    ) -> Sequence[Track]: ...


TrackerContract = Tracker


@dataclass(frozen=True, slots=True)
class FaceDetection:
    camera_id: str
    track_id: TrackId
    timestamp: datetime
    bbox: BoundingBox
    score: float
    landmarks: tuple[Point, ...] = ()
    crop: Any | None = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        if self.track_id == "" or self.track_id is None:
            raise ValueError("track_id cannot be empty")
        object.__setattr__(self, "score", _probability(self.score, "face score"))
        points: list[Point] = []
        for point in self.landmarks:
            if len(point) != 2:
                raise ValueError("each landmark must contain x and y")
            points.append((_finite(point[0], "landmark x"), _finite(point[1], "landmark y")))
        object.__setattr__(self, "landmarks", tuple(points))

    @property
    def face_score(self) -> float:
        return self.score

    @property
    def key(self) -> tuple[str, TrackId]:
        return self.camera_id, self.track_id


FaceObservation = FaceDetection


@dataclass(frozen=True, slots=True)
class IdentityResult:
    camera_id: str
    track_id: TrackId
    timestamp: datetime
    worker_id: str | None
    similarity: float
    confidence: float
    sample_count: int = 1

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        if self.track_id == "" or self.track_id is None:
            raise ValueError("track_id cannot be empty")
        similarity = _finite(self.similarity, "similarity")
        if not -1.0 <= similarity <= 1.0:
            raise ValueError("cosine similarity must be between -1 and 1")
        object.__setattr__(self, "similarity", similarity)
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        if self.worker_id is not None and not self.worker_id.strip():
            raise ValueError("worker_id cannot be blank")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")

    @property
    def known(self) -> bool:
        return self.worker_id is not None


Identity = IdentityResult


class BehaviorType(str, Enum):
    SMOKING = "smoking"
    EATING = "eating"
    DRINKING = "drinking"
    PHONE = "phone"
    FIRE = "fire"
    CARRYING = "carrying"

    @classmethod
    def parse(cls, value: BehaviorType | str) -> BehaviorType:
        if isinstance(value, cls):
            return value
        normalized = (
            str(value)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            .replace("/", "_")
        )
        aliases = {
            "smoke": cls.SMOKING,
            "cigarette": cls.SMOKING,
            "eat": cls.EATING,
            "food": cls.EATING,
            "drink": cls.DRINKING,
            "call": cls.PHONE,
            "calling": cls.PHONE,
            "cell_phone": cls.PHONE,
            "cellphone": cls.PHONE,
            "mobile_phone": cls.PHONE,
            "flame": cls.FIRE,
            "carrying_large_item": cls.CARRYING,
            "large_item_carrying": cls.CARRYING,
        }
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


@dataclass(frozen=True, slots=True)
class BehaviorDetection:
    camera_id: str
    track_id: TrackId
    timestamp: datetime
    behavior: BehaviorType
    confidence: float
    bbox: BoundingBox
    model_name: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        if self.track_id == "" or self.track_id is None:
            raise ValueError("track_id cannot be empty")
        object.__setattr__(self, "behavior", BehaviorType.parse(self.behavior))
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))


Behavior = BehaviorDetection


__all__ = [
    "BBox",
    "Behavior",
    "BehaviorDetection",
    "BehaviorType",
    "BoundingBox",
    "Detection",
    "FaceDetection",
    "FaceObservation",
    "Identity",
    "IdentityResult",
    "PersonDetection",
    "Point",
    "Track",
    "TrackId",
    "Tracker",
    "TrackerContract",
    "TrackObservation",
]
