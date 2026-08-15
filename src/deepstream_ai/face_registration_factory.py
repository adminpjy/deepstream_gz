"""Factory for the web face-registration service."""

from __future__ import annotations

from pathlib import Path

from deepstream_ai.config import AppConfig
from deepstream_ai.database import PgVectorFaceRepository
from deepstream_ai.face import AdaFaceONNXAdapter, AdaFacePreprocessor
from deepstream_ai.face_registration import (
    FaceRegistrationError,
    FaceRegistrationService,
    ScrfdOnnxRegistrationDetector,
)


def build_face_registration_service(config: AppConfig) -> FaceRegistrationService:
    if not config.face_recognition.enabled:
        raise FaceRegistrationError("face_recognition 未启用，无法注册人脸")
    if not config.database.enabled or not config.database.dsn:
        raise FaceRegistrationError("database 未启用或 DATABASE_DSN 未配置，无法注册人脸")
    face_config = config.resolve_path(config.pipeline.face.config_file)
    scrfd_onnx = _nvinfer_model_path(config, face_config, "onnx-file")
    adaface_path = config.resolve_path(config.face_recognition.model)
    detector = ScrfdOnnxRegistrationDetector(scrfd_onnx)
    embedder = AdaFaceONNXAdapter(
        model_path=adaface_path,
        providers=["CPUExecutionProvider"],
        input_name=config.face_recognition.input_name,
        output_name=config.face_recognition.output_name or None,
        preprocessor=AdaFacePreprocessor(
            (config.face_recognition.input_width, config.face_recognition.input_height),
            input_color="bgr",
        ),
    )
    repository = PgVectorFaceRepository(config.database.dsn)
    repository.ensure_schema()
    return FaceRegistrationService(detector, embedder, repository)


def _nvinfer_model_path(config: AppConfig, path: Path, key: str) -> Path:
    if not path.is_file():
        raise FaceRegistrationError(f"nvinfer 配置不存在: {path}")
    prefix = f"{key}="
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if not value:
            break
        resolved = config.resolve_path(value)
        if not resolved.is_file():
            raise FaceRegistrationError(f"注册模型不存在: {resolved}")
        return resolved
    raise FaceRegistrationError(f"nvinfer 配置缺少 {key}: {path}")


__all__ = ["build_face_registration_service"]
