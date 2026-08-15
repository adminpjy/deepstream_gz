"""Persistent per-camera baseline images for left-object detection."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

_CAMERA_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "application/octet-stream": ".jpg",
}


@dataclass(frozen=True, slots=True)
class BaselineRecord:
    camera_id: str
    baseline_id: str
    path: Path
    content_type: str
    size: int
    created_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "cameraId": self.camera_id,
            "baselineId": self.baseline_id,
            "contentType": self.content_type,
            "size": self.size,
            "createdAt": self.created_at,
        }


class BaselineStore:
    def __init__(self, root: str | Path, *, max_bytes: int = 16 * 1024 * 1024) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._lock = threading.RLock()

    @staticmethod
    def validate_camera_id(camera_id: str) -> str:
        value = str(camera_id).strip()
        if not _CAMERA_ID.fullmatch(value):
            raise ValueError(
                "cameraId 仅可包含字母、数字、点、下划线和连字符，长度 1-64"
            )
        return value

    def save(
        self,
        camera_id: str,
        payload: bytes,
        *,
        content_type: str,
    ) -> BaselineRecord:
        camera_id = self.validate_camera_id(camera_id)
        if not payload:
            raise ValueError("基准图片不能为空")
        if len(payload) > self.max_bytes:
            raise ValueError(f"基准图片超过 {self.max_bytes} 字节限制")
        media_type = content_type.split(";", 1)[0].strip().lower()
        suffix = _ALLOWED_CONTENT_TYPES.get(media_type)
        if suffix is None:
            raise ValueError("基准图片仅支持 JPEG、PNG、WebP")
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("OpenCV 不可用，无法验证物品遗留基准图片") from exc
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("基准图片无法解码")
        baseline_id = uuid4().hex
        camera_root = (self.root / camera_id).resolve()
        camera_root.relative_to(self.root)
        camera_root.mkdir(parents=True, exist_ok=True)
        destination = camera_root / f"{baseline_id}{suffix}"
        temp = camera_root / f".{baseline_id}.tmp"
        with self._lock:
            try:
                temp.write_bytes(payload)
                os.replace(temp, destination)
                pointer_tmp = camera_root / ".current.tmp"
                pointer_tmp.write_text(destination.name, encoding="utf-8")
                os.replace(pointer_tmp, camera_root / "current")
            finally:
                temp.unlink(missing_ok=True)
        return BaselineRecord(
            camera_id=camera_id,
            baseline_id=baseline_id,
            path=destination,
            content_type=media_type,
            size=len(payload),
            created_at=datetime.now(UTC).isoformat(),
        )

    def current(self, camera_id: str) -> BaselineRecord | None:
        camera_id = self.validate_camera_id(camera_id)
        camera_root = (self.root / camera_id).resolve()
        try:
            filename = (camera_root / "current").read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            return None
        path = (camera_root / filename).resolve()
        try:
            path.relative_to(camera_root)
        except ValueError:
            return None
        if not path.is_file():
            return None
        stat = path.stat()
        return BaselineRecord(
            camera_id=camera_id,
            baseline_id=path.stem,
            path=path,
            content_type={
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(path.suffix.lower(), "image/jpeg"),
            size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        )


__all__ = ["BaselineRecord", "BaselineStore"]
