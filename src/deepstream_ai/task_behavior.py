"""Behavior feature selection for stable single-task pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from deepstream_ai.domain import BehaviorType
from deepstream_ai.pipeline.metadata import FramePacket, FramePacketConsumer

TASK_BEHAVIOR_FEATURES = ("smoking", "eating", "drinking")


def normalize_task_behavior_features(value: Mapping[str, Any] | None) -> dict[str, bool]:
    data = dict(value or {})
    unsupported = sorted(
        key
        for key, enabled in data.items()
        if bool(enabled) and key not in TASK_BEHAVIOR_FEATURES
    )
    if unsupported:
        raise ValueError("unsupported task behavior features: " + ", ".join(unsupported))
    return {name: bool(data.get(name, False)) for name in TASK_BEHAVIOR_FEATURES}


def behavior_model_enabled(model_name: str, features: Mapping[str, Any]) -> bool:
    selected = normalize_task_behavior_features(features)
    if model_name == "smoking":
        return selected["smoking"]
    if model_name == "eating":
        # The deployed YOLO11n SGIE is shared by eating and drinking.
        return selected["eating"] or selected["drinking"]
    return False


def allowed_behavior_types(features: Mapping[str, Any]) -> frozenset[BehaviorType]:
    selected = normalize_task_behavior_features(features)
    result: set[BehaviorType] = set()
    if selected["smoking"]:
        result.add(BehaviorType.SMOKING)
    if selected["eating"]:
        result.add(BehaviorType.EATING)
    if selected["drinking"]:
        result.add(BehaviorType.DRINKING)
    return frozenset(result)


class TaskBehaviorFilter(FramePacketConsumer):
    """Filter shared behavior SGIE output to the switches selected for one task."""

    def __init__(self, delegate: FramePacketConsumer, features: Mapping[str, Any]) -> None:
        self.delegate = delegate
        self.allowed = allowed_behavior_types(features)

    def submit(self, packet: FramePacket) -> bool:
        behaviors = tuple(item for item in packet.behaviors if item.behavior in self.allowed)
        if behaviors != packet.behaviors:
            packet = replace(packet, behaviors=behaviors)
        return self.delegate.submit(packet)

    def identity_label(self, camera_id: str, track_id: int | str) -> str | None:
        return self.delegate.identity_label(camera_id, track_id)

    def presentation_track_id(
        self,
        camera_id: str,
        raw_track_id: int | str,
    ) -> int | str | None:
        resolver = getattr(self.delegate, "presentation_track_id", None)
        if callable(resolver):
            return resolver(camera_id, raw_track_id)
        return raw_track_id


__all__ = [
    "TASK_BEHAVIOR_FEATURES",
    "TaskBehaviorFilter",
    "allowed_behavior_types",
    "behavior_model_enabled",
    "normalize_task_behavior_features",
]
