"""Persistent HTTP-facing supervisor for isolated recognition task processes."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

LOGGER = logging.getLogger(__name__)
_ACTIVE_STATES = {"starting", "running", "stopping"}
_TERMINAL_STATES = {"completed", "stopped", "failed"}
_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,10}$")
_CAMERA_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    codec: str
    fps: float
    width: int
    height: int
    duration_sec: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "fps": round(self.fps, 6),
            "width": self.width,
            "height": self.height,
            "duration_sec": self.duration_sec,
        }


def probe_media(path: Path) -> MediaInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return _probe_media_with_opencv(path)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"无法检查上传视频: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "ffprobe failed").strip()[-500:]
        raise ValueError(f"上传文件不是可读取的视频: {detail}")
    try:
        document = json.loads(result.stdout)
        streams = document.get("streams", [])
        if len(streams) != 1:
            raise ValueError("上传文件必须恰好包含一路视频流")
        stream = streams[0]
        codec = str(stream["codec_name"]).lower()
        fps = float(Fraction(str(stream["avg_frame_rate"])))
        width = int(stream["width"])
        height = int(stream["height"])
        raw_duration = document.get("format", {}).get("duration")
        duration = float(raw_duration) if raw_duration is not None else None
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("无法解析上传视频的编码、分辨率或帧率") from exc
    if codec not in {"h264", "hevc"}:
        raise ValueError(f"仅支持 H.264/H.265 视频，当前编码为 {codec}")
    if not 0 < fps <= 240 or width <= 0 or height <= 0 or width > 8192 or height > 8192:
        raise ValueError("上传视频的帧率或分辨率超出支持范围")
    return MediaInfo(codec, fps, width, height, duration)


def _probe_media_with_opencv(path: Path) -> MediaInfo:
    """Inspect an upload when the slim DeepStream image has no ffprobe binary."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - the runtime image includes OpenCV
        raise ValueError("容器中既没有 ffprobe 也没有 OpenCV，无法检查上传视频") from exc
    capture = cv2.VideoCapture(
        str(path),
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            30_000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            30_000,
        ],
    )
    try:
        if not capture.isOpened():
            raise ValueError("上传文件不是可读取的视频")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
        normalized = fourcc.strip("\x00 ").lower()
        if normalized in {"avc1", "h264", "x264"}:
            codec = "h264"
        elif normalized in {"hev1", "hvc1", "hevc", "h265"}:
            codec = "hevc"
        else:
            raise ValueError(f"仅支持 H.264/H.265 视频，当前编码为 {normalized or 'unknown'}")
        if not 0 < fps <= 240 or width <= 0 or height <= 0 or width > 8192 or height > 8192:
            raise ValueError("上传视频的帧率或分辨率超出支持范围")
        decoded, _frame = capture.read()
        if not decoded:
            raise ValueError("上传视频无法解码首帧")
        duration = frames / fps if frames > 0 else None
        return MediaInfo(codec, fps, width, height, duration)
    finally:
        capture.release()


@dataclass(frozen=True, slots=True)
class UploadRecord:
    upload_id: str
    filename: str
    path: Path
    size: int
    media: MediaInfo
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "size": self.size,
            "created_at": self.created_at,
            **self.media.as_dict(),
        }


class UploadStore:
    def __init__(
        self,
        root: str | Path,
        *,
        max_bytes: int,
        media_probe: Callable[[Path], MediaInfo] = probe_media,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.media_probe = media_probe
        self._records: dict[str, UploadRecord] = {}
        self._lock = threading.RLock()
        self._load_existing()

    def save(self, stream: BinaryIO, length: int, filename: str) -> UploadRecord:
        if length <= 0:
            raise ValueError("上传文件不能为空")
        if length > self.max_bytes:
            raise ValueError(f"上传文件超过 {self.max_bytes} 字节限制")
        original = Path(filename.replace("\\", "/")).name.strip()[:200] or "video.mp4"
        suffix = Path(original).suffix.lower()
        suffix = suffix if _SAFE_SUFFIX.fullmatch(suffix) else ".bin"
        upload_id = uuid4().hex
        directory = (self.root / upload_id).resolve()
        directory.relative_to(self.root)
        directory.mkdir(parents=False, exist_ok=False)
        destination = directory / f"source{suffix}"
        temporary = directory / ".upload.part"
        remaining = length
        try:
            with temporary.open("xb") as output:
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("上传连接提前结束")
                    output.write(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            media = self.media_probe(destination)
            record = UploadRecord(
                upload_id=upload_id,
                filename=original,
                path=destination,
                size=length,
                media=media,
                created_at=_utc_now(),
            )
            _atomic_json(
                directory / "upload.json",
                {**record.as_dict(), "path": destination.name},
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            (directory / "upload.json").unlink(missing_ok=True)
            with suppress(OSError):
                directory.rmdir()
            raise
        with self._lock:
            self._records[upload_id] = record
        return record

    def get(self, upload_id: str) -> UploadRecord:
        with self._lock:
            record = self._records.get(upload_id)
        if record is None:
            raise KeyError(upload_id)
        return record

    def _load_existing(self) -> None:
        for metadata_path in self.root.glob("*/upload.json"):
            document = _read_json(metadata_path)
            if not document:
                continue
            try:
                upload_id = metadata_path.parent.name
                relative = Path(str(document["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                path = (metadata_path.parent / relative).resolve()
                path.relative_to(self.root)
                if not path.is_file():
                    continue
                media = MediaInfo(
                    codec=str(document["codec"]),
                    fps=float(document["fps"]),
                    width=int(document["width"]),
                    height=int(document["height"]),
                    duration_sec=(
                        float(document["duration_sec"])
                        if document.get("duration_sec") is not None
                        else None
                    ),
                )
                self._records[upload_id] = UploadRecord(
                    upload_id=upload_id,
                    filename=str(document["filename"]),
                    path=path,
                    size=int(document["size"]),
                    media=media,
                    created_at=str(document["created_at"]),
                )
            except (KeyError, TypeError, ValueError, OSError):
                continue


class TaskProcess:
    def __init__(
        self,
        task_id: str,
        output_dir: Path,
        metadata: dict[str, Any],
        process: Any,
        log_stream: BinaryIO,
        *,
        stop_grace_sec: float,
    ) -> None:
        self.task_id = task_id
        self.output_dir = output_dir
        self.metadata = dict(metadata)
        self.process = process
        self.log_stream = log_stream
        self.stop_grace_sec = float(stop_grace_sec)
        self.status_path = output_dir / "status.json"
        self.control_path = output_dir / "control.json"
        self._lock = threading.RLock()
        self._forced = threading.Event()
        self._reaper = threading.Thread(
            target=self._reap,
            name=f"reap-{task_id}",
            daemon=True,
        )
        self._reaper.start()

    def snapshot(self) -> dict[str, Any]:
        status = _read_json(self.status_path) or {
            "id": self.task_id,
            "status": "starting",
            "created_at": self.metadata["created_at"],
        }
        control = _read_json(self.control_path) or {}
        if self.process.poll() is None and control.get("stop_reason"):
            # status.json belongs to the worker process. Overlay the parent's
            # stop intent here instead of racing its periodic atomic writer.
            status = {
                **status,
                "status": "stopping",
                "stop_reason": control["stop_reason"],
            }
        return {
            **self.metadata,
            **status,
            "id": self.task_id,
            "preview_url": f"/api/tasks/{self.task_id}/stream.mjpg",
            "preview_image_url": f"/api/tasks/{self.task_id}/preview.jpg",
            "result_url": (
                f"/api/tasks/{self.task_id}/result.mp4"
                if (self.output_dir / "result.mp4").is_file()
                and (self.output_dir / "result.mp4").stat().st_size > 0
                and status.get("status") in {"completed", "stopped"}
                else None
            ),
            "events_url": (
                f"/api/tasks/{self.task_id}/events.jsonl"
                if (self.output_dir / "events.jsonl").is_file()
                else None
            ),
            "exit_code": self.process.poll(),
        }

    def is_active(self) -> bool:
        return self.process.poll() is None and self.snapshot().get("status") in _ACTIVE_STATES

    def stop(self, reason: str = "requested") -> bool:
        with self._lock:
            if self.process.poll() is not None:
                return False
            if (_read_json(self.control_path) or {}).get("stop_reason"):
                return False
            _atomic_json(self.control_path, {"stop_reason": reason, "requested_at": _utc_now()})
            try:
                self.process.terminate()
            except ProcessLookupError:
                return False
            threading.Thread(
                target=self._force_kill_after_grace,
                name=f"kill-{self.task_id}",
                daemon=True,
            ).start()
            return True

    def wait(self, timeout: float | None = None) -> int | None:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def join_reaper(self, timeout: float | None = None) -> None:
        self._reaper.join(timeout)

    def _force_kill_after_grace(self) -> None:
        deadline = time.monotonic() + self.stop_grace_sec
        while self.process.poll() is None and time.monotonic() < deadline:
            threading.Event().wait(0.1)
        if self.process.poll() is None:
            LOGGER.error("任务优雅停止超时，强制终止 task_id=%s", self.task_id)
            self._forced.set()
            with suppress(ProcessLookupError):
                self.process.kill()

    def _reap(self) -> None:
        exit_code = self.process.wait()
        with suppress(OSError):
            self.log_stream.close()
        status = _read_json(self.status_path)
        if not self._forced.is_set() and status and status.get("status") == "failed":
            return
        if exit_code == 0 and status and status.get("status") in _TERMINAL_STATES:
            return
        control = _read_json(self.control_path) or {}
        reason = control.get("stop_reason")
        if reason and not self._forced.is_set() and exit_code in {0, -signal.SIGTERM}:
            terminal = "stopped"
            error = None
        elif exit_code != 0:
            terminal = "failed"
            reason = "forced_stop" if self._forced.is_set() else "worker_exit"
            error = f"task worker exited with code {exit_code}"
        else:
            terminal = "completed"
            reason = "eos"
            error = None
        base = status or {"id": self.task_id, "created_at": self.metadata["created_at"]}
        _atomic_json(
            self.status_path,
            {
                **base,
                "status": terminal,
                "stop_reason": reason,
                "error": error,
                "ended_at": _utc_now(),
                "updated_at": _utc_now(),
            },
        )


class RecognitionTaskService:
    """Start, stop and restart isolated recognition task processes."""

    def __init__(
        self,
        base_config: str | Path,
        *,
        uploads_root: str | Path,
        tasks_root: str | Path,
        default_idle_timeout_sec: float = 10.0,
        max_upload_bytes: int = 2 * 1024 * 1024 * 1024,
        max_active_tasks: int = 2,
        preview_fps: float = 5.0,
        preview_width: int = 960,
        stop_grace_sec: float = 25.0,
        media_probe: Callable[[Path], MediaInfo] = probe_media,
        process_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        if not 1 <= max_active_tasks <= 32:
            raise ValueError("max_active_tasks must be between 1 and 32")
        if not 1 <= default_idle_timeout_sec <= 3600:
            raise ValueError("default_idle_timeout_sec must be between 1 and 3600")
        if not math.isfinite(preview_fps) or not 0.1 <= preview_fps <= 30:
            raise ValueError("preview_fps must be between 0.1 and 30")
        if not 160 <= preview_width <= 3840:
            raise ValueError("preview_width must be between 160 and 3840")
        if not math.isfinite(stop_grace_sec) or stop_grace_sec <= 0:
            raise ValueError("stop_grace_sec must be positive")
        self.base_config = Path(base_config).resolve()
        if not self.base_config.is_file():
            raise FileNotFoundError(self.base_config)
        self.tasks_root = Path(tasks_root).resolve()
        self.tasks_root.mkdir(parents=True, exist_ok=True)
        self.uploads = UploadStore(
            uploads_root,
            max_bytes=max_upload_bytes,
            media_probe=media_probe,
        )
        self.default_idle_timeout_sec = float(default_idle_timeout_sec)
        self.max_active_tasks = int(max_active_tasks)
        self.preview_fps = float(preview_fps)
        self.preview_width = int(preview_width)
        self.stop_grace_sec = float(stop_grace_sec)
        self.process_factory = process_factory
        self.service_id = uuid4().hex
        self.started_at = _utc_now()
        self._tasks: dict[str, TaskProcess] = {}
        self._lock = threading.RLock()
        self._restarting = False
        self._closing = False

    def upload(self, stream: BinaryIO, length: int, filename: str) -> dict[str, Any]:
        with self._lock:
            if self._closing:
                raise RuntimeError("识别服务正在停止")
            if self._restarting:
                raise RuntimeError("识别服务正在重启")
        return self.uploads.save(stream, length, filename).as_dict()

    def start_file(
        self,
        upload_id: str,
        *,
        camera_id: str | None = None,
        idle_timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        upload = self.uploads.get(upload_id)
        resolved_camera_id = self._camera_id(camera_id or f"file-{upload_id[:8]}")
        source = {
            "type": "file",
            "camera_id": resolved_camera_id,
            "path": str(upload.path),
            "nominal_fps": upload.media.fps,
            "enabled": True,
        }
        return self._start(
            source,
            source_label=upload.filename,
            upload_id=upload_id,
            idle_timeout_sec=idle_timeout_sec,
        )

    def start_rtsp(
        self,
        url: str,
        *,
        camera_id: str | None = None,
        nominal_fps: float = 25.0,
        idle_timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        parsed = urlsplit(url.strip())
        if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise ValueError("RTSP 地址必须以 rtsp:// 或 rtsps:// 开头并包含主机")
        if not 0 < float(nominal_fps) <= 240:
            raise ValueError("nominal_fps 必须在 0 到 240 之间")
        resolved_camera_id = self._camera_id(camera_id or f"rtsp-{uuid4().hex[:8]}")
        source = {
            "type": "rtsp",
            "camera_id": resolved_camera_id,
            "url": url.strip(),
            "nominal_fps": float(nominal_fps),
            "enabled": True,
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

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            tasks = tuple(self._tasks.values())
        return sorted(
            (task.snapshot() for task in tasks),
            key=lambda item: str(item.get("created_at", "")),
            reverse=True,
        )

    def get_task(self, task_id: str) -> TaskProcess:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    def stop(self, task_id: str, reason: str = "requested") -> dict[str, Any]:
        task = self.get_task(task_id)
        task.stop(reason)
        return task.snapshot()

    def restart(self) -> dict[str, Any]:
        with self._lock:
            if self._closing:
                raise RuntimeError("识别服务正在停止")
            if self._restarting:
                raise RuntimeError("识别服务正在重启")
            self._restarting = True
            tasks = tuple(self._tasks.values())
            old_service_id = self.service_id
            self.service_id = uuid4().hex
            self.started_at = _utc_now()
        try:
            for task in tasks:
                if task.is_active():
                    task.stop("service_restart")
            deadline = time.monotonic() + self.stop_grace_sec + 2.0
            for task in tasks:
                remaining = max(0.0, deadline - time.monotonic())
                if task.process.poll() is None:
                    task.wait(remaining)
                task.join_reaper(max(0.0, deadline - time.monotonic()))
            survivors = [task.task_id for task in tasks if task.process.poll() is None]
            if survivors:
                raise RuntimeError("重启超时，任务仍未退出: " + ",".join(survivors))
        finally:
            with self._lock:
                self._restarting = False
        result = self.status()
        result["previous_service_id"] = old_service_id
        return result

    def close(self) -> None:
        with self._lock:
            self._closing = True
            tasks = tuple(self._tasks.values())
        for task in tasks:
            if task.is_active():
                task.stop("service_shutdown")
        deadline = time.monotonic() + self.stop_grace_sec + 2.0
        for task in tasks:
            if task.process.poll() is None:
                task.wait(max(0.0, deadline - time.monotonic()))
            task.join_reaper(max(0.0, deadline - time.monotonic()))

    def status(self) -> dict[str, Any]:
        with self._lock:
            task_processes = tuple(self._tasks.values())
            restarting = self._restarting
            closing = self._closing
        tasks = [task.snapshot() for task in task_processes]
        return {
            "service_id": self.service_id,
            "status": "stopping" if closing else "restarting" if restarting else "ready",
            "started_at": self.started_at,
            "default_idle_timeout_sec": self.default_idle_timeout_sec,
            "max_active_tasks": self.max_active_tasks,
            "active_tasks": sum(task.is_active() for task in task_processes),
            "total_tasks": len(tasks),
        }

    def _start(
        self,
        source: dict[str, Any],
        *,
        source_label: str,
        upload_id: str | None,
        idle_timeout_sec: float | None,
    ) -> dict[str, Any]:
        idle_timeout = (
            self.default_idle_timeout_sec if idle_timeout_sec is None else float(idle_timeout_sec)
        )
        if not 1 <= idle_timeout <= 3600:
            raise ValueError("idle_timeout_sec 必须在 1 到 3600 之间")
        with self._lock:
            if self._closing:
                raise RuntimeError("识别服务正在停止")
            if self._restarting:
                raise RuntimeError("识别服务正在重启")
            active = sum(task.is_active() for task in self._tasks.values())
            if active >= self.max_active_tasks:
                raise RuntimeError(f"活动任务已达到上限 {self.max_active_tasks}")
            task_id = uuid4().hex[:16]
            output_dir = (self.tasks_root / task_id).resolve()
            output_dir.relative_to(self.tasks_root)
            output_dir.mkdir(parents=False, exist_ok=False)
            created_at = _utc_now()
            spec = {
                "version": 1,
                "task_id": task_id,
                "created_at": created_at,
                "base_config": str(self.base_config),
                "source": source,
                "source_label": source_label,
                "upload_id": upload_id,
                "output_dir": str(output_dir),
                "idle_timeout_sec": idle_timeout,
                "preview_fps": self.preview_fps,
                "preview_width": self.preview_width,
            }
            spec_path = output_dir / "task.json"
            _atomic_json(spec_path, spec)
            metadata = {
                "id": task_id,
                "created_at": created_at,
                "source_type": source["type"],
                "source_label": source_label,
                "camera_id": source["camera_id"],
                "idle_timeout_sec": idle_timeout,
                "upload_id": upload_id,
            }
            _atomic_json(
                output_dir / "status.json",
                {**metadata, "status": "starting", "updated_at": created_at},
            )
            log_stream = (output_dir / "pipeline.log").open("ab", buffering=0)
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            command = [
                sys.executable,
                "-m",
                "deepstream_ai.task_worker",
                "--spec",
                str(spec_path),
            ]
            try:
                process = self.process_factory(
                    command,
                    cwd=str(self.base_config.parent.parent),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    start_new_session=os.name != "nt",
                )
            except Exception:
                log_stream.close()
                _atomic_json(
                    output_dir / "status.json",
                    {
                        **metadata,
                        "status": "failed",
                        "updated_at": _utc_now(),
                        "ended_at": _utc_now(),
                        "error": "task worker could not be started",
                    },
                )
                raise
            task = TaskProcess(
                task_id,
                output_dir,
                metadata,
                process,
                log_stream,
                stop_grace_sec=self.stop_grace_sec,
            )
            self._tasks[task_id] = task
        LOGGER.info(
            "识别任务已创建 task_id=%s source_type=%s camera_id=%s",
            task_id,
            source["type"],
            source["camera_id"],
        )
        return task.snapshot()

    @staticmethod
    def _camera_id(value: str) -> str:
        normalized = str(value).strip()
        if not _CAMERA_ID.fullmatch(normalized):
            raise ValueError("camera_id 仅可包含字母、数字、点、下划线和连字符，长度 1-64")
        return normalized


__all__ = [
    "MediaInfo",
    "RecognitionTaskService",
    "TaskProcess",
    "UploadRecord",
    "UploadStore",
    "probe_media",
]
