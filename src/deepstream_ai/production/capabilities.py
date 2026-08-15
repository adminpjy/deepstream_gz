"""Production feature availability derived from deployed assets."""

from __future__ import annotations

from typing import Any

from deepstream_ai.config import AppConfig, BehaviorModelConfig
from deepstream_ai.domain import BehaviorType
from deepstream_ai.preflight import inspect_nvinfer_config

_OPTIONAL_BEHAVIOR_FEATURES = ("smoking", "eating", "drinking", "phone")


def _model_features(model: BehaviorModelConfig) -> tuple[str, ...]:
    features: list[str] = []
    candidates = (*model.labels, model.name)
    for candidate in candidates:
        try:
            value = BehaviorType.parse(candidate).value
        except ValueError:
            continue
        if value in _OPTIONAL_BEHAVIOR_FEATURES and value not in features:
            features.append(value)
    return tuple(features)


def _asset_status(config: AppConfig, model: BehaviorModelConfig) -> dict[str, Any]:
    if not model.config_file:
        return {"available": False, "reason": "not_configured", "model": model.name}
    report = inspect_nvinfer_config(config, model.config_file)
    missing = list(report.missing)
    if model.model and not config.resolve_path(model.model).is_file():
        missing.append(f"model missing: {config.resolve_path(model.model)}")
    return {
        "available": not missing,
        "reason": None if not missing else "; ".join(missing),
        "model": model.name,
    }


def behavior_capabilities(config: AppConfig) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        name: {"available": False, "reason": "not_configured", "model": None}
        for name in _OPTIONAL_BEHAVIOR_FEATURES
    }
    for model in config.behavior:
        features = _model_features(model)
        if not features:
            continue
        status = _asset_status(config, model)
        for feature in features:
            # A single multi-class model can provide several independent business
            # features (the local yolo11n.onnx provides eating + drinking).
            result[feature] = dict(status)
    return result


def warmable_behavior_names(config: AppConfig) -> tuple[str, ...]:
    names: list[str] = []
    for model in config.behavior:
        if not _model_features(model):
            continue
        status = _asset_status(config, model)
        if status["available"] and model.name not in names:
            names.append(model.name)
    return tuple(names)


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
            "phone": behavior["phone"],
            "leftObject": {
                "available": True,
                "reason": None,
                "model": None,
                "mode": "scene_diff",
            },
            "fire": {
                "available": False,
                "reason": "converted_as_full_frame_model_not_yet_wired_to_session_gate",
                "model": "fire",
                "mode": "full_frame",
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
