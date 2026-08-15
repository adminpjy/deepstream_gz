"""Quality-gated face enrollment for the local web console.

Enrollment deliberately uses the same SCRFD geometry contract and AdaFace model
as runtime recognition. A worker may own multiple templates: one clear frontal
photo is the baseline, while later left/right angle samples can be appended
without deleting the baseline. Recognition already searches the whole pgvector
table, so the nearest template wins naturally.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from deepstream_ai.config import AppConfig
from deepstream_ai.database import PgVectorFaceRepository, StoredFaceVector
from deepstream_ai.domain import BoundingBox, FaceDetection
from deepstream_ai.face import FivePointFaceAligner
from deepstream_ai.face.quality import FaceFusionConfig, FaceQualityScorer
from deepstream_ai.pipeline.scrfd import ScrfdCandidate, decode_scrfd_outputs

LOGGER = logging.getLogger(__name__)
_WORKER_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class FaceRegistrationError(RuntimeError):
    """Enrollment request cannot be evaluated or persisted."""


class RegistrationFaceDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> tuple[FaceDetection, ...]: ...


@dataclass(frozen=True, slots=True)
class FaceRegistrationPolicy:
    max_image_bytes: int = 12 * 1024 * 1024
    min_face_pixels: int = 112
    preferred_face_pixels: int = 160
    min_primary_detector: float = 0.75
    min_supplement_detector: float = 0.68
    min_primary_blur: float = 0.58
    min_supplement_blur: float = 0.52
    min_primary_frontal: float = 0.82
    min_supplement_frontal: float = 0.52
    min_primary_quality: float = 0.72
    min_supplement_quality: float = 0.64
    min_brightness: float = 45.0
    max_brightness: float = 210.0
    duplicate_similarity: float = 0.97
    supplement_identity_floor: float = 0.30
    supplement_competitor_margin: float = 0.05
    max_templates_per_worker: int = 12


@dataclass(frozen=True, slots=True)
class FaceQualityReport:
    accepted: bool
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "metrics": self.metrics,
        }


class ScrfdOnnxRegistrationDetector:
    """CPU-friendly one-shot SCRFD detector for uploaded registration photos."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        network_size: tuple[int, int] = (640, 640),
        threshold: float = 0.35,
        nms_iou: float = 0.40,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FaceRegistrationError("人脸注册需要 onnxruntime") from exc
        path = Path(model_path)
        if not path.is_file():
            raise FaceRegistrationError(f"SCRFD 模型不存在: {path}")
        selected = list(providers) if providers else ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(path), providers=selected)
        except Exception as exc:
            raise FaceRegistrationError(f"无法加载 SCRFD 注册模型: {path}") from exc
        try:
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [item.name for item in self.session.get_outputs()]
        except Exception as exc:
            raise FaceRegistrationError("SCRFD ONNX 输入输出契约无效") from exc
        self.network_width, self.network_height = map(int, network_size)
        self.threshold = float(threshold)
        self.nms_iou = float(nms_iou)

    def detect(self, image_bgr: np.ndarray) -> tuple[FaceDetection, ...]:
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
            raise FaceRegistrationError("上传图片无法解析为彩色图像")
        tensor, scale, pad_x, pad_y = self._preprocess(image[..., :3])
        try:
            outputs = self.session.run(None, {self.input_name: tensor})
        except Exception as exc:
            raise FaceRegistrationError("SCRFD 人脸质量检测失败") from exc
        layers = {
            name: np.asarray(output, dtype=np.float32)
            for name, output in zip(self.output_names, outputs, strict=True)
        }
        candidates = decode_scrfd_outputs(
            layers,
            network_width=self.network_width,
            network_height=self.network_height,
            threshold=self.threshold,
        )
        selected = _nms(candidates, self.nms_iou)
        height, width = image.shape[:2]
        now = datetime.now(UTC)
        detections: list[FaceDetection] = []
        for index, candidate in enumerate(selected):
            mapped = _map_candidate(candidate, scale, pad_x, pad_y, width, height)
            if mapped is None:
                continue
            detections.append(
                FaceDetection(
                    camera_id="face-registration",
                    track_id=index,
                    timestamp=now,
                    bbox=mapped.bbox,
                    score=mapped.score,
                    landmarks=mapped.landmarks,
                )
            )
        return tuple(sorted(detections, key=lambda item: item.score, reverse=True))

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FaceRegistrationError("人脸注册需要 OpenCV") from exc
        height, width = image_bgr.shape[:2]
        scale = min(self.network_width / width, self.network_height / height)
        resized_width = max(1, min(self.network_width, int(round(width * scale))))
        resized_height = max(1, min(self.network_height, int(round(height * scale))))
        resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.network_width - resized_width) // 2
        pad_y = (self.network_height - resized_height) // 2
        canvas = np.full((self.network_height, self.network_width, 3), 127, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        # DeepStream face.example.txt uses RGB, offsets=127.5 and scale=1/128.
        rgb = canvas[..., ::-1].astype(np.float32)
        normalized = (rgb - 127.5) * 0.0078125
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...], dtype=np.float32)
        return tensor, scale, pad_x, pad_y


class FaceRegistrationService:
    """Validate, embed and append registration templates for one worker."""

    def __init__(
        self,
        detector: RegistrationFaceDetector,
        embedder: Any,
        repository: PgVectorFaceRepository,
        *,
        policy: FaceRegistrationPolicy | None = None,
        aligner: FivePointFaceAligner | None = None,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.repository = repository
        self.policy = policy or FaceRegistrationPolicy()
        self.aligner = aligner or FivePointFaceAligner((112, 112))
        self.scorer = FaceQualityScorer(FaceFusionConfig(frame_color_space="bgr"))
        self._lock = threading.RLock()

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "FaceRegistrationService":
        # Kept as a convenience entrypoint; the dedicated factory owns model
        # path resolution so Web and root CLI cannot drift apart.
        from deepstream_ai.face_registration_factory import build_face_registration_service

        return build_face_registration_service(config)

    def worker_summary(self, worker_id: str) -> dict[str, Any]:
        worker = _validate_worker_id(worker_id)
        rows = self.repository.list_worker(worker)
        return {
            "worker_id": worker,
            "template_count": len(rows),
            "templates": [
                {
                    "record_id": row.record_id,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "sample_type": row.sample_type,
                    "pose": row.pose,
                    "quality": row.quality,
                }
                for row in rows
            ],
        }

    def register(
        self,
        worker_id: str,
        image_bytes: bytes,
        *,
        mode: str = "primary",
        filename: str = "",
    ) -> dict[str, Any]:
        worker = _validate_worker_id(worker_id)
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"primary", "supplement"}:
            raise ValueError("mode 仅支持 primary 或 supplement")
        if not image_bytes:
            raise ValueError("人脸照片不能为空")
        if len(image_bytes) > self.policy.max_image_bytes:
            raise ValueError("人脸照片超过 12MB 限制")
        image = _decode_image(image_bytes)
        report, face = self._quality_report(image, normalized_mode)
        response: dict[str, Any] = {
            "worker_id": worker,
            "mode": normalized_mode,
            "filename": Path(filename).name[:200],
            **report.as_dict(),
            "stored": False,
        }
        if not report.accepted or face is None:
            return response

        face_crop = _crop(image, face.bbox)
        if face_crop is None:
            response["accepted"] = False
            response["issues"] = ["人脸框超出图片有效区域，请重新上传。"]
            return response
        aligned = self.aligner.align(face_crop, face)
        embedding = self.embedder.embed(aligned)

        with self._lock:
            existing = self.repository.list_worker(worker)
            response["existing_template_count"] = len(existing)
            if normalized_mode == "supplement" and not existing:
                response["accepted"] = False
                response["issues"] = ["该 workid 还没有主注册照片，请先提交清晰正脸作为首次注册。"]
                return response
            if len(existing) >= self.policy.max_templates_per_worker:
                response["accepted"] = False
                response["issues"] = [
                    f"该人员已保存 {len(existing)} 张模板，达到上限 {self.policy.max_templates_per_worker}；请先清理无效样本。"
                ]
                return response

            same_similarity = _max_similarity(embedding, existing)
            response["same_worker_similarity"] = round(same_similarity, 4) if existing else None
            if existing and same_similarity >= self.policy.duplicate_similarity:
                response["accepted"] = True
                response["stored"] = False
                response["warnings"] = list(response["warnings"]) + [
                    "这张照片与该人员已有模板几乎重复，无需再次保存；建议补充不同角度而不是重复正脸。"
                ]
                response["template_count"] = len(existing)
                return response

            if normalized_mode == "supplement":
                competitor = self.repository.find_nearest_other(embedding, worker)
                response["nearest_other"] = (
                    {
                        "worker_id": competitor.worker_id,
                        "similarity": round(competitor.similarity, 4),
                    }
                    if competitor is not None
                    else None
                )
                competitor_similarity = competitor.similarity if competitor is not None else -1.0
                if same_similarity < self.policy.supplement_identity_floor:
                    response["accepted"] = False
                    response["issues"] = [
                        "补充照片与该 workid 已有模板差异过大，暂不写入，避免把错误人员污染到人脸库。",
                        "建议先补充约 15°~30° 的轻微侧脸，再逐步补充更大角度；或重新确认 workid。",
                    ]
                    return response
                if competitor_similarity >= same_similarity + self.policy.supplement_competitor_margin:
                    response["accepted"] = False
                    response["issues"] = [
                        f"这张照片更接近其他人员 {competitor.worker_id}，为防止串人已拒绝注册，请核对 workid。"
                    ]
                    return response
                if same_similarity < 0.40:
                    response["warnings"] = list(response["warnings"]) + [
                        "该角度与现有模板相似度偏低，已作为人工确认的补充角度保存；建议再补一张更小转角作为过渡样本。"
                    ]

            digest = hashlib.sha256(image_bytes).hexdigest()
            stored = self.repository.add(
                worker,
                embedding,
                sample_type=normalized_mode,
                pose=str(report.metrics.get("pose", "unknown")),
                quality=float(report.metrics.get("quality", 0.0)),
                image_sha256=digest,
            )
            response["stored"] = True
            response["record_id"] = stored.record_id
            response["template_count"] = len(existing) + 1
            response["created_at"] = stored.created_at.isoformat() if stored.created_at else None
            return response

    def _quality_report(
        self,
        image: np.ndarray,
        mode: str,
    ) -> tuple[FaceQualityReport, FaceDetection | None]:
        detections = tuple(item for item in self.detector.detect(image) if item.score >= 0.35)
        issues: list[str] = []
        warnings: list[str] = []
        if not detections:
            issues.append("未检测到人脸。请上传单人、无遮挡、脸部更大的清晰照片。")
            return FaceQualityReport(False, tuple(issues), tuple(warnings), {}), None
        strong_faces = tuple(item for item in detections if item.score >= 0.50)
        if len(strong_faces) > 1:
            issues.append(f"照片中检测到 {len(strong_faces)} 张人脸。注册照片只能包含一个人。")
            return FaceQualityReport(False, tuple(issues), tuple(warnings), {"face_count": len(strong_faces)}), None
        face = strong_faces[0] if strong_faces else detections[0]
        crop = _crop(image, face.bbox)
        if crop is None:
            issues.append("检测到的人脸位于图片边界外，请换一张完整头像。")
            return FaceQualityReport(False, tuple(issues), tuple(warnings), {}), None

        scored = self.scorer.score(face, crop=crop, frame_shape=image.shape)
        brightness = _brightness(crop)
        min_face_side = min(face.bbox.width, face.bbox.height)
        pose = _pose_label(face.landmarks)
        border_ratio = _border_margin_ratio(face.bbox, image.shape[1], image.shape[0])
        metrics: dict[str, Any] = {
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "face_count": len(strong_faces) if strong_faces else 1,
            "face_width": int(round(face.bbox.width)),
            "face_height": int(round(face.bbox.height)),
            "detector": round(face.score, 4),
            "blur": round(scored.blur_score, 4),
            "frontal": round(scored.frontal_score, 4),
            "quality": round(scored.quality, 4),
            "brightness": round(brightness, 1),
            "pose": pose,
        }

        detector_floor = (
            self.policy.min_primary_detector if mode == "primary" else self.policy.min_supplement_detector
        )
        blur_floor = self.policy.min_primary_blur if mode == "primary" else self.policy.min_supplement_blur
        frontal_floor = (
            self.policy.min_primary_frontal if mode == "primary" else self.policy.min_supplement_frontal
        )
        quality_floor = (
            self.policy.min_primary_quality if mode == "primary" else self.policy.min_supplement_quality
        )

        if min_face_side < self.policy.min_face_pixels:
            issues.append(
                f"人脸有效尺寸只有约 {int(round(face.bbox.width))}×{int(round(face.bbox.height))}，太小。"
                f"请让人脸至少达到 {self.policy.min_face_pixels}px，推荐 {self.policy.preferred_face_pixels}px 以上。"
            )
        elif min_face_side < self.policy.preferred_face_pixels:
            warnings.append(
                f"人脸尺寸可用但偏小（约 {int(round(face.bbox.width))}×{int(round(face.bbox.height))}）；"
                f"推荐上传人脸边长 {self.policy.preferred_face_pixels}px 以上的原图。"
            )
        if face.score < detector_floor:
            issues.append(
                f"人脸检测置信度 {face.score:.2f} 偏低。请避免遮挡、低头和过远拍摄，并换更清晰的照片。"
            )
        if scored.blur_score < blur_floor:
            issues.append(
                f"照片清晰度不足（清晰度评分 {scored.blur_score:.2f}）。请使用原图，避免截图放大、运动模糊和强压缩。"
            )
        elif scored.blur_score < 0.75:
            warnings.append("照片清晰度达到最低要求，但仍建议使用更锐利的原始照片以提高识别稳定性。")
        if scored.frontal_score < frontal_floor:
            if mode == "primary":
                issues.append(
                    f"首次注册照片角度过偏（正脸评分 {scored.frontal_score:.2f}）。请正视摄像头、保持双眼和鼻子清晰可见。"
                )
            else:
                issues.append(
                    f"补充侧脸角度过大（姿态评分 {scored.frontal_score:.2f}）。建议先提交约 15°~35° 的左/右侧脸，而不是接近纯侧脸。"
                )
        if not self.policy.min_brightness <= brightness <= self.policy.max_brightness:
            direction = "过暗" if brightness < self.policy.min_brightness else "过亮"
            issues.append(f"人脸区域{direction}（亮度 {brightness:.0f}/255）。请调整光线，避免逆光或过曝。")
        elif brightness < 65 or brightness > 190:
            warnings.append("光照接近可用边界，建议使用均匀正面光线，避免脸部一侧过暗。")
        if border_ratio < 0.015:
            issues.append("人脸紧贴照片边缘，疑似被裁切。请保留完整头部和少量肩部背景。")
        if scored.quality < quality_floor:
            issues.append(
                f"综合质量评分 {scored.quality:.2f} 低于注册标准 {quality_floor:.2f}。请按上方提示重新拍摄。"
            )
        if mode == "supplement" and scored.frontal_score >= 0.92:
            warnings.append("这张补充照片仍接近正脸；若目标是提高侧脸识别率，建议再补充左右各 15°~35° 的清晰照片。")
        if min(image.shape[:2]) < 480 and min_face_side < self.policy.preferred_face_pixels:
            warnings.append("整张图片分辨率偏低，尽量上传相机原图而不是聊天软件缩略图。")

        return (
            FaceQualityReport(not issues, tuple(issues), tuple(warnings), metrics),
            face,
        )


def _decode_image(payload: bytes) -> np.ndarray:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise FaceRegistrationError("人脸注册需要 OpenCV") from exc
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("无法读取图片，仅支持常见 JPEG/PNG/WebP 图片")
    if image.shape[0] > 8192 or image.shape[1] > 8192:
        raise ValueError("图片分辨率过大，最长边不能超过 8192px")
    return np.ascontiguousarray(image)


def _crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    clipped = bbox.clipped(image.shape[1], image.shape[0])
    if clipped is None:
        return None
    rows, cols = clipped.integer_slices(image.shape[1], image.shape[0])
    crop = np.ascontiguousarray(image[rows, cols])
    return crop if crop.size else None


def _nms(candidates: Sequence[ScrfdCandidate], threshold: float) -> tuple[ScrfdCandidate, ...]:
    selected: list[ScrfdCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(_iou(candidate.bbox, kept.bbox) > threshold for kept in selected):
            continue
        selected.append(candidate)
    return tuple(selected)


def _iou(left: BoundingBox, right: BoundingBox) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    union = left.area + right.area - intersection
    return intersection / union if union > 0.0 else 0.0


def _map_candidate(
    candidate: ScrfdCandidate,
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
) -> ScrfdCandidate | None:
    def point(value: tuple[float, float]) -> tuple[float, float]:
        x = min(float(width), max(0.0, (value[0] - pad_x) / scale))
        y = min(float(height), max(0.0, (value[1] - pad_y) / scale))
        return x, y

    x1, y1 = point((candidate.bbox.x1, candidate.bbox.y1))
    x2, y2 = point((candidate.bbox.x2, candidate.bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return ScrfdCandidate(
        BoundingBox(x1, y1, x2, y2),
        candidate.score,
        tuple(point(item) for item in candidate.landmarks),
    )


def _brightness(crop: np.ndarray) -> float:
    pixels = crop[..., :3].astype(np.float32, copy=False)
    # OpenCV input is BGR.
    gray = 0.114 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.299 * pixels[..., 2]
    return float(np.mean(gray)) if gray.size else 0.0


def _pose_label(landmarks: Sequence[tuple[float, float]]) -> str:
    if len(landmarks) < 3:
        return "unknown"
    left_eye = np.asarray(landmarks[0], dtype=np.float64)
    right_eye = np.asarray(landmarks[1], dtype=np.float64)
    nose = np.asarray(landmarks[2], dtype=np.float64)
    eye_mid = (left_eye + right_eye) * 0.5
    eye_span = float(np.linalg.norm(right_eye - left_eye))
    if eye_span <= 1e-6:
        return "unknown"
    offset = float((nose[0] - eye_mid[0]) / eye_span)
    if abs(offset) < 0.10:
        return "front"
    return "right" if offset > 0 else "left"


def _border_margin_ratio(bbox: BoundingBox, width: int, height: int) -> float:
    margins = (
        bbox.x1 / width,
        bbox.y1 / height,
        (width - bbox.x2) / width,
        (height - bbox.y2) / height,
    )
    return max(0.0, min(float(item) for item in margins))


def _max_similarity(embedding: np.ndarray, rows: Sequence[StoredFaceVector]) -> float:
    if not rows:
        return -1.0
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return max(float(np.dot(vector, row.embedding)) for row in rows)


def _validate_worker_id(worker_id: str) -> str:
    value = str(worker_id).strip()
    if not _WORKER_ID.fullmatch(value):
        raise ValueError("workid 仅可包含字母、数字、点、下划线和连字符，长度 1-64")
    return value


__all__ = [
    "FaceQualityReport",
    "FaceRegistrationError",
    "FaceRegistrationPolicy",
    "FaceRegistrationService",
    "ScrfdOnnxRegistrationDetector",
]
