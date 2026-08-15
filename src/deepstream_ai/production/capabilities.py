"""Production feature availability derived from deployed assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepstream_ai.config import AppConfig, BehaviorModelConfig
from deepstream_ai.domain import BehaviorType
from deepstream_ai.preflight import inspect_nvinfer_config

_OPTIONAL_BEHAVIOR_FEATURES = ("smoking", "eating", "drinking")
# Keep the COCO evidence contract aligned with opsvision/eating_drinking.py.
_EAT_DRINK_PROXY_REQUIRED_LABELS = frozenset(
    {
        "bottle",
        "cup",
        "wine_glass",
        "bowl",
        "apple",
        "banana",
        "sandwich",
        "orange",
        "pizza",
        "donut",
        "cake",
        "hot_dog",
    }
)


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


def _labels_from_report(auxiliary_files: tuple[Path, ...]) -> tuple[str, ...] | None:
    candidates = [path for path in auxiliary_files if "label" in path.name.lower()]
    if len(candidates) != 1:
        return None
    path = candidates[0]
    try:
        return tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except OSError:
        return None


def _normalized_labels(labels: tuple[str, ...]) -> frozenset[str]:
    return frozenset(
        value.strip().lower().replace("-", "_").replace(" ", "_")
        for value in labels
        if value.strip()
    )


def _is_eat_drink_proxy(model: BehaviorModelConfig, labels: tuple[str, ...] | None) -> bool:
    if model.name != "eating" or labels is None:
        return False
    return _EAT_DRINK_PROXY_REQUIRED_LABELS.issubset(_normalized_labels(labels))


def _asset_status(config: AppConfig, model: BehaviorModelConfig) -> dict[str, Any]:
    if not model.config_file:
        return {"available": False, "reason": "not_configured", "model": model.name}
    report = inspect_nvinfer_config(config, model.config_file)
    missing = list(report.missing)
    if model.model and not config.resolve_path(model.model).is_file():
        missing.append(f"model missing: {config.resolve_path(model.model)}")
    deployed_labels = _labels_from_report(report.auxiliary_files)
    proxy_mode = _is_eat_drink_proxy(model, deployed_labels)
    if deployed_labels is not None and deployed_labels != model.labels and not proxy_mode:
        missing.append(
            "label order mismatch: "
            f"config={list(model.labels)!r} deployed={list(deployed_labels)!r}"
        )
    return {
        "available": not missing,
        "reason": None if not missing else "; ".join(missing),
        "model": model.name,
        "mode": "person_crop_coco_proxy" if proxy_mode else None,
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
            # A single person-crop model can provide several independent business
            # features (the deployed COCO YOLO11n provides eating + drinking proxy evidence).
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
            "leftObject": {
                "available": True,
                "reason": None,
                "model": None,
                "mode": "scene_diff",
            },
            "fire": {
                "available": False,
                "reason": "classification_model_requires_label_and_preprocessing_contract",
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
