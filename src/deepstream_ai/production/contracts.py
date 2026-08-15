"""Production recognition-session contracts.

This module intentionally contains no DeepStream/GStreamer/CUDA types. The
REST API, GPU supervisor, scenario processors and result publishers share
these immutable contracts so transport/runtime changes cannot leak into the
already tuned person/face recognition core.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

_CAMERA_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def utc_now() -> datetime:
    return datetime.now(UTC)


class SessionState(str, Enum):
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Optional analytics; core person/face analytics cannot be disabled."""

    smoking: bool = False
    eating: bool = False
    drinking: bool = False
    phone: bool = False
    left_object: bool = False
    large_object_moving: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> FeatureSet:
        data = dict(value or {})
        return cls(
            smoking=bool(data.get("smoking", False)),
            eating=bool(data.get("eating", False)),
            drinking=bool(data.get("drinking", False)),
            phone=bool(data.get("phone", data.get("phoneCall", False))),
            left_object=bool(data.get("leftObject", data.get("left_object", False))),
            large_object_moving=bool(
                data.get("largeObjectMoving", data.get("large_object_moving", False))
            ),
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "smoking": self.smoking,
            "eating": self.eating,
            "drinking": self.drinking,
            "phone": self.phone,
            "leftObject": self.left_object,
            "largeObjectMoving": self.large_object_moving,
        }

    def enabled_behavior_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.smoking:
            names.append("smoking")
        if self.eating:
            names.append("eating")
        if self.drinking:
            names.append("drinking")
        if self.phone:
            names.append("phone")
        return tuple(names)


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    person_absent_seconds: float = 30.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ExitPolicy:
        data = dict(value or {})
        seconds = float(
            data.get("personAbsentSeconds", data.get("person_absent_seconds", 30.0))
        )
        if not 1.0 <= seconds <= 3600.0:
            raise ValueError("personAbsentSeconds 必须在 1 到 3600 秒之间")
        return cls(person_absent_seconds=seconds)

    def as_dict(self) -> dict[str, float]:
        return {"personAbsentSeconds": self.person_absent_seconds}


@dataclass(frozen=True, slots=True)
class LeftObjectPolicy:
    pixel_threshold: int = 28
    min_area_ratio: float = 0.0015
    min_component_area_ratio: float = 0.00035
    confirm_frames: int = 3
    max_recent_frames: int = 8

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> LeftObjectPolicy:
        data = dict(value or {})
        result = cls(
            pixel_threshold=int(
                data.get("pixelThreshold", data.get("pixel_threshold", 28))
            ),
            min_area_ratio=float(
                data.get("minAreaRatio", data.get("min_area_ratio", 0.0015))
            ),
            min_component_area_ratio=float(
                data.get(
                    "minComponentAreaRatio",
                    data.get("min_component_area_ratio", 0.00035),
                )
            ),
            confirm_frames=int(
                data.get("confirmFrames", data.get("confirm_frames", 3))
            ),
            max_recent_frames=int(
                data.get("maxRecentFrames", data.get("max_recent_frames", 8))
            ),
        )
        if not 1 <= result.pixel_threshold <= 255:
            raise ValueError("leftObject.pixelThreshold 必须在 1 到 255 之间")
        if not 0 < result.min_area_ratio <= 0.5:
            raise ValueError("leftObject.minAreaRatio 必须在 0 到 0.5 之间")
        if not 0 < result.min_component_area_ratio <= result.min_area_ratio:
            raise ValueError(
                "leftObject.minComponentAreaRatio 必须大于 0 且不大于 minAreaRatio"
            )
        if not 1 <= result.confirm_frames <= 10:
            raise ValueError("leftObject.confirmFrames 必须在 1 到 10 之间")
        if not result.confirm_frames <= result.max_recent_frames <= 30:
            raise ValueError(
                "leftObject.maxRecentFrames 必须不小于 confirmFrames 且不大于 30"
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "pixelThreshold": self.pixel_threshold,
            "minAreaRatio": self.min_area_ratio,
            "minComponentAreaRatio": self.min_component_area_ratio,
            "confirmFrames": self.confirm_frames,
            "maxRecentFrames": self.max_recent_frames,
        }


@dataclass(frozen=True, slots=True)
class SessionRequest:
    camera_id: str
    stream_url: str
    nominal_fps: float = 30.0
    features: FeatureSet = field(default_factory=FeatureSet)
    exit_policy: ExitPolicy = field(default_factory=ExitPolicy)
    left_object: LeftObjectPolicy = field(default_factory=LeftObjectPolicy)
    context: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SessionRequest:
        data = dict(value)
        camera_id = str(data.get("cameraId", data.get("camera_id", ""))).strip()
        if not _CAMERA_ID.fullmatch(camera_id):
            raise ValueError(
                "cameraId 仅可包含字母、数字、点、下划线和连字符，长度 1-64"
            )
        stream_url = str(data.get("streamUrl", data.get("stream_url", ""))).strip()
        parsed = urlsplit(stream_url)
        if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise ValueError("streamUrl 必须是有效的 rtsp:// 或 rtsps:// 地址")
        nominal_fps = float(data.get("nominalFps", data.get("nominal_fps", 30.0)))
        if not 0.1 <= nominal_fps <= 240:
            raise ValueError("nominalFps 必须在 0.1 到 240 之间")
        context = data.get("context") or {}
        if not isinstance(context, Mapping):
            raise ValueError("context 必须是对象")
        return cls(
            camera_id=camera_id,
            stream_url=stream_url,
            nominal_fps=nominal_fps,
            features=FeatureSet.from_mapping(data.get("features")),
            exit_policy=ExitPolicy.from_mapping(data.get("exitPolicy")),
            left_object=LeftObjectPolicy.from_mapping(data.get("leftObject")),
            context=dict(context),
        )

    def as_dict(self, *, redact_url: bool = False) -> dict[str, Any]:
        url = self.stream_url
        if redact_url:
            parsed = urlsplit(url)
            host = parsed.hostname or ""
            if parsed.port:
                host += f":{parsed.port}"
            url = parsed._replace(netloc=host, query="", fragment="").geturl()
        return {
            "cameraId": self.camera_id,
            "streamUrl": url,
            "nominalFps": self.nominal_fps,
            "features": self.features.as_dict(),
            "exitPolicy": self.exit_policy.as_dict(),
            "leftObject": self.left_object.as_dict(),
            "context": dict(self.context),
        }


@dataclass(frozen=True, slots=True)
class RecognitionEvent:
    event_id: str
    session_id: str
    camera_id: str
    event_type: str
    timestamp: datetime
    action: str = "raise"
    track_id: str | None = None
    person_id: str | None = None
    confidence: float | None = None
    snapshot: str | None = None
    video_clip: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        camera_id: str,
        event_type: str,
        timestamp: datetime | None = None,
        action: str = "raise",
        track_id: object | None = None,
        person_id: str | None = None,
        confidence: float | None = None,
        snapshot: str | None = None,
        video_clip: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> RecognitionEvent:
        if confidence is not None and not 0 <= float(confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return cls(
            event_id=uuid4().hex,
            session_id=session_id,
            camera_id=camera_id,
            event_type=str(event_type),
            timestamp=timestamp or utc_now(),
            action=str(action),
            track_id=None if track_id is None else str(track_id),
            person_id=person_id,
            confidence=None if confidence is None else float(confidence),
            snapshot=snapshot,
            video_clip=video_clip,
            extra=dict(extra or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "eventId": self.event_id,
            "sessionId": self.session_id,
            "cameraId": self.camera_id,
            "eventType": self.event_type,
            "action": self.action,
            "trackId": self.track_id,
            "personId": self.person_id,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "snapshot": self.snapshot,
            "videoClip": self.video_clip,
            "extra": dict(self.extra),
        }


__all__ = [
    "ExitPolicy",
    "FeatureSet",
    "LeftObjectPolicy",
    "RecognitionEvent",
    "SessionRequest",
    "SessionState",
    "utc_now",
]
