"""Manual-start task service with safe history handling.

Opening the web console must never start historical video sources. Historical
runs remain visible as records only. A user can explicitly start a new run from
an old record, or upload/enter a new source and click Start.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from deepstream_ai.task_behavior import normalize_task_behavior_features
from deepstream_ai.task_service import RecognitionTaskService, _atomic_json, _read_json, _utc_now

_ACTIVE_STATES = {"starting", "running", "stopping"}
_TASK_ID_LENGTH = 16


class ManualRecognitionTaskService(RecognitionTaskService):
    """Recognition service that never resumes historical tasks implicitly."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._normalize_orphaned_history()

    def list_tasks(self) -> list[dict[str, Any]]:
        """Return current tasks plus historical records without starting them."""

        with self._lock:
            current = {task_id: task.snapshot() for task_id, task in self._tasks.items()}

        history: dict[str, dict[str, Any]] = {}
        for task_dir in self._history_dirs():
            task_id = task_dir.name
            if task_id in current:
                continue
            snapshot = self._history_snapshot(task_dir)
            if snapshot is not None:
                history[task_id] = snapshot

        return sorted(
            [*current.values(), *history.values()],
            key=lambda item: str(item.get("created_at", "")),
            reverse=True,
        )

    def start_rtsp_with_features(
        self,
        url: str,
        *,
        camera_id: str | None = None,
        nominal_fps: float = 25.0,
        idle_timeout_sec: float | None = None,
        behavior_features: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start the proven one-source RTSP pipeline with selected behavior SGIEs."""

        parsed = urlsplit(url.strip())
        if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise ValueError("RTSP 地址必须以 rtsp:// 或 rtsps:// 开头并包含主机")
        if not 0 < float(nominal_fps) <= 240:
            raise ValueError("nominal_fps 必须在 0 到 240 之间")
        selected = normalize_task_behavior_features(behavior_features)
        resolved_camera_id = self._camera_id(camera_id or f"rtsp-{uuid4().hex[:8]}")
        source = {
            "type": "rtsp",
            "camera_id": resolved_camera_id,
            "url": url.strip(),
            "nominal_fps": float(nominal_fps),
            "enabled": True,
            "behavior_features": selected,
        }
        redacted = urlunsplit(
            (
                parsed.scheme,
                parsed.hostname + (f":{parsed.port}" if parsed.port else ""),
                parsed.path,
                "",
                "",
            )
        )
        return self._start(
            source,
            source_label=redacted,
            upload_id=None,
            idle_timeout_sec=idle_timeout_sec,
        )

    def start_existing(self, task_id: str) -> dict[str, Any]:
        """Explicitly start a new run from a historical task specification."""

        normalized = self._validate_history_task_id(task_id)
        with self._lock:
            current = self._tasks.get(normalized)
            if current is not None and current.is_active():
                raise RuntimeError("该任务当前仍在运行，无需重复启动")

        task_dir = self._task_dir(normalized)
        spec = _read_json(task_dir / "task.json")
        if not spec:
            raise KeyError(normalized)
        source = spec.get("source")
        if not isinstance(source, dict):
            raise ValueError("历史任务缺少有效的视频源配置")
        source_type = str(source.get("type", "")).lower()
        if source_type not in {"file", "rtsp"}:
            raise ValueError("历史任务的视频源类型无效")

        source_copy = dict(source)
        source_copy["enabled"] = True
        if source_type == "file":
            path = Path(str(source_copy.get("path", ""))).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"历史视频文件不存在: {path}")
            source_copy["path"] = str(path)

        source_label = str(spec.get("source_label") or source_copy.get("camera_id") or source_type)
        upload_id = spec.get("upload_id")
        idle_timeout = spec.get("idle_timeout_sec", self.default_idle_timeout_sec)
        return self._start(
            source_copy,
            source_label=source_label,
            upload_id=str(upload_id) if upload_id else None,
            idle_timeout_sec=float(idle_timeout),
        )

    def _normalize_orphaned_history(self) -> None:
        """Mark stale active states as stopped; never resurrect old workers."""

        now = _utc_now()
        for task_dir in self._history_dirs():
            status_path = task_dir / "status.json"
            status = _read_json(status_path)
            if not status:
                continue
            if str(status.get("status", "")).lower() not in _ACTIVE_STATES:
                continue
            _atomic_json(
                status_path,
                {
                    **status,
                    "status": "stopped",
                    "stop_reason": "service_restart_manual_start_required",
                    "ended_at": status.get("ended_at") or now,
                    "updated_at": now,
                },
            )

    def _history_snapshot(self, task_dir: Path) -> dict[str, Any] | None:
        status = _read_json(task_dir / "status.json")
        spec = _read_json(task_dir / "task.json")
        if not status and not spec:
            return None
        task_id = task_dir.name
        source = spec.get("source", {}) if isinstance(spec, dict) else {}
        if not isinstance(source, dict):
            source = {}
        source_type = str(
            (status or {}).get("source_type")
            or source.get("type")
            or "file"
        ).lower()
        created_at = (
            (status or {}).get("created_at")
            or (spec or {}).get("created_at")
            or ""
        )
        source_label = str(
            (status or {}).get("source_label")
            or (spec or {}).get("source_label")
            or source.get("camera_id")
            or source_type
        )
        camera_id = str(
            (status or {}).get("camera_id")
            or source.get("camera_id")
            or ""
        )
        document = {
            **(status or {}),
            "id": task_id,
            "created_at": created_at,
            "source_type": source_type,
            "source_label": source_label,
            "camera_id": camera_id,
            "historical": True,
            "restartable": bool(spec),
            "preview_url": None,
            "preview_image_url": None,
            "exit_code": None,
        }
        return document

    def _history_dirs(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in self.tasks_root.iterdir()
            if path.is_dir() and len(path.name) == _TASK_ID_LENGTH
        )

    def _task_dir(self, task_id: str) -> Path:
        path = (self.tasks_root / task_id).resolve()
        path.relative_to(self.tasks_root)
        if not path.is_dir():
            raise KeyError(task_id)
        return path

    @staticmethod
    def _validate_history_task_id(task_id: str) -> str:
        value = str(task_id).strip().lower()
        if len(value) != _TASK_ID_LENGTH or any(ch not in "0123456789abcdef" for ch in value):
            raise KeyError(task_id)
        return value


__all__ = ["ManualRecognitionTaskService"]
