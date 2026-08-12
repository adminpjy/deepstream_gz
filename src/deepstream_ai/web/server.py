"""Small same-origin HTTP API for recognition tasks and live MJPEG previews."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import signal
import threading
import time
from contextlib import suppress
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from deepstream_ai.config import AppConfig
from deepstream_ai.doctor import run_doctor
from deepstream_ai.manual_task_service import ManualRecognitionTaskService
from deepstream_ai.preflight import validate_assets
from deepstream_ai.task_service import RecognitionTaskService, TaskProcess

LOGGER = logging.getLogger(__name__)
_TASK_PATH = re.compile(r"^/api/tasks/([a-f0-9]{16})(?:/(.*))?$")
_TERMINAL_STATES = {"completed", "stopped", "failed"}


class _CountingReader:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        payload = self.stream.read(size)
        self.bytes_read += len(payload)
        return payload


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)


class RecognitionHTTPServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        service: RecognitionTaskService,
        static_root: Path,
    ) -> None:
        self.service = service
        self.static_root = static_root.resolve()
        super().__init__(address, RecognitionRequestHandler)


class RecognitionRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: RecognitionHTTPServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30.0)
        self._request_body_consumed = False

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._do_get()
        except Exception as exc:
            self._handle_error(exc)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._require_same_origin()
            self._do_post()
        except Exception as exc:
            self._handle_error(exc)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.debug("HTTP %s - %s", self.client_address[0], format % args)

    def _do_get(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/health/live":
            self._send_json({"status": "live"})
            return
        if path == "/health/ready":
            status = self.server.service.status()
            self._send_json(
                status,
                status=HTTPStatus.OK
                if status.get("status") == "ready"
                else HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        if path == "/api/service":
            self._send_json(self.server.service.status())
            return
        if path == "/api/tasks":
            self._send_json({"tasks": self.server.service.list_tasks()})
            return
        match = _TASK_PATH.fullmatch(path)
        if match:
            task = self._task(match.group(1))
            artifact = match.group(2)
            if artifact is None:
                self._send_json(task.snapshot())
            elif artifact == "stream.mjpg":
                self._stream_mjpeg(task)
            elif artifact == "preview.jpg":
                self._serve_file(task.output_dir / "preview.jpg", "image/jpeg", no_store=True)
            elif artifact == "result.mp4":
                if task.snapshot().get("result_url") is None:
                    raise ApiError(HTTPStatus.CONFLICT, "结果视频尚未完成封装")
                self._serve_file(task.output_dir / "result.mp4", "video/mp4")
            elif artifact == "events.jsonl":
                self._serve_file(
                    task.output_dir / "events.jsonl",
                    "application/x-ndjson; charset=utf-8",
                )
            else:
                raise ApiError(HTTPStatus.NOT_FOUND, "接口不存在")
            return
        if path in {"/", "/index.html"}:
            self._serve_static("index.html")
            return
        if path in {"/app.js", "/styles.css"}:
            self._serve_static(path.removeprefix("/"))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "接口不存在")

    def _do_post(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/api/uploads":
            self._require_content_type("upload")
            query = parse_qs(parsed.query)
            filename = str(query.get("filename", [""])[0]).strip()
            if not filename:
                raise ApiError(HTTPStatus.BAD_REQUEST, "filename 不能为空")
            length = self._content_length()
            reader = _CountingReader(self.rfile)
            try:
                record = self.server.service.upload(reader, length, filename)
            finally:
                self._request_body_consumed = reader.bytes_read >= length
            self._send_json(record, status=HTTPStatus.CREATED)
            return
        if path == "/api/tasks":
            self._require_content_type("json")
            body = self._read_json()
            source_type = str(body.get("source_type", "")).lower()
            camera_id = str(body.get("camera_id", "")).strip() or None
            idle_timeout = body.get("idle_timeout_sec")
            if source_type == "file":
                upload_id = str(body.get("upload_id", "")).strip()
                if not upload_id:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "upload_id 不能为空")
                task = self.server.service.start_file(
                    upload_id,
                    camera_id=camera_id,
                    idle_timeout_sec=idle_timeout,
                )
            elif source_type == "rtsp":
                task = self.server.service.start_rtsp(
                    str(body.get("url", "")),
                    camera_id=camera_id,
                    nominal_fps=float(body.get("nominal_fps", 25.0)),
                    idle_timeout_sec=idle_timeout,
                )
            else:
                raise ApiError(HTTPStatus.BAD_REQUEST, "source_type 仅支持 file 或 rtsp")
            self._send_json(task, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/service/restart":
            self._require_optional_json_content_type()
            self._read_json(optional=True)
            self._send_json(self.server.service.restart())
            return
        match = _TASK_PATH.fullmatch(path)
        if match and match.group(2) == "start":
            self._require_optional_json_content_type()
            self._read_json(optional=True)
            starter = getattr(self.server.service, "start_existing", None)
            if not callable(starter):
                raise ApiError(HTTPStatus.CONFLICT, "当前服务不支持从历史任务手动启动")
            task = starter(match.group(1))
            self._send_json(task, status=HTTPStatus.ACCEPTED)
            return
        if match and match.group(2) == "stop":
            self._require_optional_json_content_type()
            self._read_json(optional=True)
            task = self.server.service.stop(match.group(1))
            self._send_json(task, status=HTTPStatus.ACCEPTED)
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "接口不存在")

    def _task(self, task_id: str) -> TaskProcess:
        try:
            return self.server.service.get_task(task_id)
        except KeyError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, "识别任务不存在") from exc

    def _content_length(self) -> int:
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise ApiError(HTTPStatus.LENGTH_REQUIRED, "请求必须提供 Content-Length")
        try:
            length = int(raw)
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Content-Length 无效") from exc
        if length < 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Content-Length 无效")
        return length

    def _require_same_origin(self) -> None:
        origin = self.headers.get("Origin")
        if not origin:
            return
        host = self.headers.get("Host", "")
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.netloc.lower() != host.lower()
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ApiError(HTTPStatus.FORBIDDEN, "拒绝非同源请求")

    def _require_content_type(self, kind: str) -> None:
        media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if kind == "json" and media_type != "application/json":
            raise ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "请求必须使用 application/json")
        if kind == "upload" and not (
            media_type == "application/octet-stream" or media_type.startswith("video/")
        ):
            raise ApiError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "上传必须使用 application/octet-stream 或 video/*",
            )

    def _require_optional_json_content_type(self) -> None:
        raw_length = self.headers.get("Content-Length")
        if raw_length is not None and raw_length != "0":
            self._require_content_type("json")

    def _read_json(self, *, optional: bool = False) -> dict[str, Any]:
        if optional and self.headers.get("Content-Length") is None:
            return {}
        length = self._content_length()
        if length == 0 and optional:
            self._request_body_consumed = True
            return {}
        if length > 64 * 1024:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "JSON 请求过大")
        payload = self.rfile.read(length)
        self._request_body_consumed = len(payload) >= length
        if len(payload) != length:
            raise ApiError(HTTPStatus.BAD_REQUEST, "请求体提前结束")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "JSON 请求无效") from exc
        if not isinstance(value, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "JSON 根节点必须是对象")
        return value

    def _send_json(self, document: Any, *, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_static(self, name: str) -> None:
        path = (self.server.static_root / name).resolve()
        try:
            path.relative_to(self.server.static_root)
        except ValueError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, "静态资源不存在") from exc
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._serve_file(path, content_type, static=True)

    def _serve_file(
        self,
        path: Path,
        content_type: str,
        *,
        static: bool = False,
        no_store: bool = False,
    ) -> None:
        try:
            stream = path.open("rb")
            size = os.fstat(stream.fileno()).st_size
        except (FileNotFoundError, IsADirectoryError) as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, "文件尚不可用") from exc
        start, end = 0, max(0, size - 1)
        partial = False
        raw_range = self.headers.get("Range") if not static else None
        if raw_range:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", raw_range.strip())
            if not match or (not match.group(1) and not match.group(2)):
                stream.close()
                self._send_range_error(size)
                return
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            else:
                suffix = int(match.group(2))
                start = max(0, size - suffix)
            end = min(end, max(0, size - 1))
            if size == 0 or start < 0 or start > end or start >= size:
                stream.close()
                self._send_range_error(size)
                return
            partial = True
        length = max(0, end - start + 1) if size else 0
        try:
            self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header(
                "Cache-Control",
                "no-cache, must-revalidate"
                if static
                else "no-store"
                if no_store
                else "private, no-cache",
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if static:
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' data:; style-src 'self'; "
                    "script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
                )
            self.send_header("Content-Length", str(length))
            self.end_headers()
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return
        finally:
            stream.close()

    def _send_range_error(self, size: int) -> None:
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _stream_mjpeg(self, task: TaskProcess) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        preview = task.output_dir / "preview.jpg"
        last_mtime = -1
        terminal_since: float | None = None
        try:
            while True:
                snapshot = task.snapshot()
                try:
                    stat = preview.stat()
                    if stat.st_mtime_ns != last_mtime:
                        payload = preview.read_bytes()
                        last_mtime = stat.st_mtime_ns
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(payload)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except FileNotFoundError:
                    pass
                if snapshot.get("status") in _TERMINAL_STATES:
                    terminal_since = terminal_since or time.monotonic()
                    if time.monotonic() - terminal_since >= 1.0:
                        return
                threading.Event().wait(0.1)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def _handle_error(self, exc: Exception) -> None:
        if isinstance(exc, ApiError):
            status = exc.status
            message = str(exc)
        elif isinstance(exc, KeyError):
            status = HTTPStatus.NOT_FOUND
            message = "资源不存在"
        elif isinstance(exc, FileNotFoundError):
            status = HTTPStatus.CONFLICT
            message = str(exc)
        elif isinstance(exc, ValueError):
            status = HTTPStatus.UNPROCESSABLE_ENTITY
            message = str(exc)
        elif isinstance(exc, RuntimeError):
            status = HTTPStatus.CONFLICT
            message = str(exc)
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            message = "服务内部错误"
            LOGGER.exception("HTTP 请求处理失败 path=%s", self.path)
        # Some errors are raised before a rejected upload/JSON request body is
        # consumed. Closing this HTTP/1.1 connection prevents those bytes from
        # being parsed as a second request.
        self._discard_small_request_body()
        self.close_connection = True
        with suppress(BrokenPipeError, ConnectionResetError, OSError):
            self._send_json({"error": message}, status=status)

    def _discard_small_request_body(self) -> None:
        if self._request_body_consumed:
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return
        if not 0 < length <= 64 * 1024:
            return
        previous_timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(0.5)
            payload = self.rfile.read(length)
            self._request_body_consumed = len(payload) >= length
        except (OSError, TimeoutError):
            return
        finally:
            self.connection.settimeout(previous_timeout)


def run_web_service(
    config: AppConfig,
    *,
    host: str,
    port: int,
    uploads_root: str | Path,
    tasks_root: str | Path,
    idle_timeout_sec: float,
    max_upload_mb: int,
    max_tasks: int,
    preview_fps: float,
    preview_width: int,
) -> None:
    # The supervisor is only declared ready after the immutable model contract,
    # DeepStream runtime, GPU and required GStreamer factories are available.
    # Dynamic input media is intentionally excluded from this service check.
    validate_assets(replace(config, sources=()))
    failed_checks = [check for check in run_doctor(config) if not check.ok]
    if failed_checks:
        detail = "; ".join(f"{check.name}: {check.detail}" for check in failed_checks)
        raise RuntimeError(f"识别服务运行环境检查失败: {detail}")
    service = ManualRecognitionTaskService(
        config.config_path,
        uploads_root=uploads_root,
        tasks_root=tasks_root,
        default_idle_timeout_sec=idle_timeout_sec,
        max_upload_bytes=max_upload_mb * 1024 * 1024,
        max_active_tasks=max_tasks,
        preview_fps=preview_fps,
        preview_width=preview_width,
        stop_grace_sec=config.runtime.shutdown_timeout_sec + 65,
    )
    static_root = Path(__file__).resolve().parent / "static"
    server = RecognitionHTTPServer((host, port), service, static_root)
    health_path = Path(config.runtime.health_file)
    temporary_health = health_path.with_suffix(health_path.suffix + ".tmp")
    health_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_health.write_text(f"{os.getpid()} {host}:{server.server_port}", encoding="ascii")
    os.replace(temporary_health, health_path)
    stop_once = threading.Event()

    def request_shutdown(_signum: int, _frame: Any) -> None:
        if stop_once.is_set():
            return
        stop_once.set()
        threading.Thread(target=server.shutdown, name="http-shutdown", daemon=True).start()

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_shutdown)
    LOGGER.info("识别服务已启动: http://%s:%d", host, server.server_port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.shutdown()
        service.close()
        server.server_close()
        health_path.unlink(missing_ok=True)
        temporary_health.unlink(missing_ok=True)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        LOGGER.info("识别服务已停止")


__all__ = ["RecognitionHTTPServer", "RecognitionRequestHandler", "run_web_service"]
