"""Configuration and metadata routing for independent behavior models."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from deepstream_ai.domain import BehaviorDetection, BehaviorType, BoundingBox, TrackId

LOGGER = logging.getLogger(__name__)


class BehaviorConfigurationError(ValueError):
    """Behavior model configuration is invalid or ambiguous."""


@dataclass(frozen=True, slots=True)
class BehaviorModelConfig:
    behavior: BehaviorType
    enabled: bool
    model_path: Path | None
    gie_unique_id: int
    confidence_threshold: float = 0.5
    class_map: Mapping[int, BehaviorType] = field(default_factory=dict)
    engine_config_path: Path | None = None
    model_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "behavior", BehaviorType.parse(self.behavior))
        if self.gie_unique_id <= 0:
            raise BehaviorConfigurationError("gie_unique_id must be positive")
        threshold = float(self.confidence_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise BehaviorConfigurationError("confidence_threshold must be between 0 and 1")
        object.__setattr__(self, "confidence_threshold", threshold)
        if self.model_path is not None:
            object.__setattr__(self, "model_path", Path(self.model_path))
        if self.engine_config_path is not None:
            object.__setattr__(self, "engine_config_path", Path(self.engine_config_path))
        if self.enabled and self.model_path is None:
            raise BehaviorConfigurationError(
                f"enabled behavior {self.behavior.value!r} requires a model path"
            )
        parsed_map = {
            int(class_id): BehaviorType.parse(behavior)
            for class_id, behavior in self.class_map.items()
        }
        if any(class_id < 0 for class_id in parsed_map):
            raise BehaviorConfigurationError("behavior class ids cannot be negative")
        object.__setattr__(self, "class_map", MappingProxyType(parsed_map))
        if not self.model_name:
            object.__setattr__(self, "model_name", self.behavior.value)

    @property
    def should_load(self) -> bool:
        """Only these specs should be turned into nvinfer elements."""

        return self.enabled


@dataclass(frozen=True, slots=True)
class BehaviorMetadata:
    """Pipeline-neutral subset of an nvinfer object metadata record."""

    camera_id: str
    track_id: TrackId
    timestamp: datetime
    gie_unique_id: int
    class_id: int
    confidence: float
    bbox: BoundingBox
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        if self.track_id == "" or self.track_id is None:
            raise ValueError("track_id cannot be empty")
        if self.gie_unique_id <= 0:
            raise ValueError("gie_unique_id must be positive")
        if self.class_id < 0:
            raise ValueError("class_id cannot be negative")
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> BehaviorMetadata:
        bbox_value = values.get("bbox")
        if isinstance(bbox_value, BoundingBox):
            bbox = bbox_value
        elif isinstance(bbox_value, Sequence) and not isinstance(bbox_value, (str, bytes)):
            bbox = BoundingBox.from_sequence(bbox_value)
        else:
            left = float(values.get("left", 0.0))
            top = float(values.get("top", 0.0))
            width = float(values["width"])
            height = float(values["height"])
            bbox = BoundingBox(left, top, left + width, top + height)
        timestamp = values["timestamp"]
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        return cls(
            camera_id=str(values["camera_id"]),
            track_id=values["track_id"],
            timestamp=timestamp,
            gie_unique_id=int(values.get("gie_unique_id", values.get("component_id"))),
            class_id=int(values["class_id"]),
            confidence=float(values["confidence"]),
            bbox=bbox,
            raw=values,
        )


class BehaviorMetadataRouter:
    """Route inference metadata by nvinfer unique ID and model class map."""

    def __init__(self, models: Iterable[BehaviorModelConfig]) -> None:
        configured = tuple(models)
        enabled = tuple(model for model in configured if model.enabled)
        by_id: dict[int, BehaviorModelConfig] = {}
        for model in enabled:
            if model.gie_unique_id in by_id:
                other = by_id[model.gie_unique_id]
                raise BehaviorConfigurationError(
                    f"duplicate gie_unique_id {model.gie_unique_id} for "
                    f"{other.behavior.value} and {model.behavior.value}"
                )
            by_id[model.gie_unique_id] = model
        self._configured = configured
        self._enabled = enabled
        self._by_id = MappingProxyType(by_id)

    @property
    def configured_models(self) -> tuple[BehaviorModelConfig, ...]:
        return self._configured

    @property
    def enabled_models(self) -> tuple[BehaviorModelConfig, ...]:
        """The pipeline construction plan; disabled models are absent."""

        return self._enabled

    @property
    def models_by_unique_id(self) -> Mapping[int, BehaviorModelConfig]:
        return self._by_id

    def route(self, metadata: BehaviorMetadata | Mapping[str, Any]) -> BehaviorDetection | None:
        if not isinstance(metadata, BehaviorMetadata):
            metadata = BehaviorMetadata.from_mapping(metadata)
        model = self._by_id.get(metadata.gie_unique_id)
        if model is None:
            LOGGER.debug("Ignoring metadata from unconfigured GIE id=%s", metadata.gie_unique_id)
            return None
        if metadata.confidence < model.confidence_threshold:
            return None
        if model.class_map:
            behavior = model.class_map.get(metadata.class_id)
            if behavior is None:
                LOGGER.debug(
                    "Ignoring unmapped class=%s from behavior model=%s",
                    metadata.class_id,
                    model.model_name,
                )
                return None
        else:
            behavior = model.behavior
        return BehaviorDetection(
            camera_id=metadata.camera_id,
            track_id=metadata.track_id,
            timestamp=metadata.timestamp,
            behavior=behavior,
            confidence=metadata.confidence,
            bbox=metadata.bbox,
            model_name=model.model_name,
            metadata=metadata.raw,
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        first_unique_id: int = 20,
    ) -> BehaviorMetadataRouter:
        """Build from either the whole YAML mapping or its ``behavior`` node.

        Model paths under ``models.behaviors`` are fallbacks. The explicit
        ``behavior.<kind>`` switch remains authoritative, ensuring a disabled
        model is never added to the pipeline construction plan.
        """

        behavior_section: Mapping[str, Any]
        if isinstance(values.get("behavior"), Mapping):
            behavior_section = values["behavior"]
        else:
            behavior_section = values

        fallback_models: Mapping[str, Any] = {}
        models_section = values.get("models")
        if isinstance(models_section, Mapping) and isinstance(
            models_section.get("behaviors"), Mapping
        ):
            fallback_models = models_section["behaviors"]

        models: list[BehaviorModelConfig] = []
        for offset, behavior in enumerate(BehaviorType):
            raw = behavior_section.get(behavior.value, {})
            if isinstance(raw, bool):
                raw = {"enabled": raw}
            if raw is None:
                raw = {}
            if not isinstance(raw, Mapping):
                raise BehaviorConfigurationError(f"behavior.{behavior.value} must be a mapping")
            enabled = _as_bool(raw.get("enabled", False))
            fallback = fallback_models.get(behavior.value)
            if isinstance(fallback, Mapping):
                fallback = fallback.get("engine", fallback.get("model", fallback.get("path")))
            model_value = raw.get("model", raw.get("engine", raw.get("path", fallback)))
            model_path = None if model_value in (None, "") else Path(str(model_value))
            class_map_raw = raw.get("labels", raw.get("class_map", {}))
            if class_map_raw is None:
                class_map_raw = {}
            if isinstance(class_map_raw, Sequence) and not isinstance(class_map_raw, (str, bytes)):
                class_map_raw = {index: value for index, value in enumerate(class_map_raw)}
            if not isinstance(class_map_raw, Mapping):
                raise BehaviorConfigurationError(
                    f"behavior.{behavior.value}.labels must be a mapping"
                )
            models.append(
                BehaviorModelConfig(
                    behavior=behavior,
                    enabled=enabled,
                    model_path=model_path,
                    gie_unique_id=int(
                        raw.get("gie_unique_id", raw.get("unique_id", first_unique_id + offset))
                    ),
                    confidence_threshold=float(
                        raw.get("confidence_threshold", raw.get("threshold", 0.5))
                    ),
                    class_map={
                        int(key): BehaviorType.parse(value) for key, value in class_map_raw.items()
                    },
                    engine_config_path=(
                        None
                        if raw.get("config_file") in (None, "")
                        else Path(str(raw["config_file"]))
                    ),
                    model_name=str(raw.get("name", behavior.value)),
                )
            )
        return cls(models)

    @classmethod
    def from_runtime_configs(
        cls,
        models: Iterable[object],
    ) -> BehaviorMetadataRouter:
        """Adapt the project's typed behavior config records."""

        specs: list[BehaviorModelConfig] = []
        for model in models:
            name = str(model.name)
            labels = tuple(getattr(model, "labels", ()))
            class_map = {index: BehaviorType.parse(label) for index, label in enumerate(labels)}
            model_value = str(getattr(model, "model", ""))
            config_file = str(getattr(model, "config_file", ""))
            specs.append(
                BehaviorModelConfig(
                    behavior=BehaviorType.parse(name),
                    enabled=bool(getattr(model, "enabled", False)),
                    model_path=Path(model_value or config_file)
                    if (model_value or config_file)
                    else None,
                    gie_unique_id=int(model.unique_id),
                    confidence_threshold=float(getattr(model, "threshold", 0.5)),
                    class_map=class_map,
                    engine_config_path=Path(config_file) if config_file else None,
                    model_name=name,
                )
            )
        return cls(specs)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0", ""}:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise BehaviorConfigurationError(f"invalid boolean value: {value!r}")


BehaviorRouter = BehaviorMetadataRouter
BehaviorModelSpec = BehaviorModelConfig


__all__ = [
    "BehaviorConfigurationError",
    "BehaviorMetadata",
    "BehaviorMetadataRouter",
    "BehaviorModelConfig",
    "BehaviorModelSpec",
    "BehaviorRouter",
]
