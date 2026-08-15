"""Production feature availability derived from deployed assets."""

from __future__ import annotations

from typing import Any

from deepstream_ai.config import AppConfig
from deepstream_ai.preflight import inspect_nvinfer_config

_OPTIONAL_BEHAVIOR_NAMES = ("smoking", "eating", "drinking")


def behavior_capabilities(config: AppConfig) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    by_name = {model.name: model for model in config.behavior}
    for name in _OPTIONAL_BEHAVIOR_NAMES:
        model = by_name.get(name)
        if model is None or not model.config_file:
            result[name] = {
                "available": False,
                "reason": "not_configured",
                "model": None,
            }
            continue
        report = inspect_nvinfer_config(config, model.config_file)
        missing = list(report.missing)
        if model.model and not config.resolve_path(model.model).is_file():
            missing.append(f"model missing: {config.resolve_path(model.model)}")
        result[name] = {
            "available": not missing,
            "reason": None if not missing else "; ".join(missing),
            "model": model.name,
        }
    return result


def warmable_behavior_names(config: AppConfig) -> tuple[str, ...]:
    capabilities = behavior_capabilities(config)
    return tuple(
        name
        for name in _OPTIONAL_BEHAVIOR_NAMES
        if capabilities[name]["available"]
    )


def production_capabilities(config: AppConfig) -> dict[str, Any]:
    behavior = behavior_capabilities(config)
    return {
        "core": {
            "personDetection": True,
            "personTracking": True,
            "faceDetection": bool(config.pipeline.face.enabled),
            "faceTracking": True,
            "faceRecognition": bool(config.face_recognition.enabled),
            "mandatory": True,
        },
        "optional": {
            "smoking": behavior["smoking"],
            "eating": behavior["eating"],
            "drinking": behavior["drinking"],
            "leftObject": {
                "available": True,
                "reason": None,
                "model": None,
                "mode": "scene_diff",
            },
            "largeObjectMoving": {
                "available": False,
                "reason": "reserved_not_implemented",
                "model": None,
            },
        },
    }


__all__ = [
    "behavior_capabilities",
    "production_capabilities",
    "warmable_behavior_names",
]
