"""Production REST control plane layered beside the existing test task API."""

from __future__ import annotations

import logging
import os
import re
import signal
import threading
import time
from contextlib import suppress
from dataclasses import replace
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from deepstream_ai.config import AppConfig
from deepstream_ai.doctor import run_doctor
from deepstream_ai.face_registration import FaceRegistrationService
from deepstream_ai.face_registration_factory import build_face_registration_service
from deepstream_ai.manual_task_service import ManualRecognitionTaskService
from deepstream_ai.preflight import validate_assets
from deepstream_ai.production.contracts import SessionRequest
from deepstream_ai.production.manager import (
    ProductionRecognitionService,
    ProductionServiceError,
)
from deepstream_ai.production.unavailable import UnavailableProductionRecognitionService
from deepstream_ai.web.server import ApiError, RecognitionRequestHandler
from deepstream_ai.web.server import RecognitionHTTPServer as LegacyRecognitionHTTPServer

LOGGER = logging.getLogger(__name__)
_SESSION_PATH = re.compile(r"^/api/v1/recognition/sessions/([a-f0-9]{16})(?:/(.*))?$")
_CAMERA_BASELINE_PATH = re.compile(r"^/api/v1/recognition/cameras/([^/]+)/baseline$")
_CAMERA_STOP_PATH = re.compile(r"^/api/v1/recognition/cameras/([^/]+)/stop$")
_PRODUCTION_TERMINAL_STATES = {"completed", "stopped", "failed"}


class ProductionHTTPServer(LegacyRecognitionHTTPServer):
    """Expose legacy tests and production sessions from the same process."""

    def __init__(
        self,
        address: tuple[str, int],
        legacy_service: ManualRecognitionTaskService,
        production_service: (
            ProductionRecognitionService | UnavailableProductionRecognitionService
        ),
        static_root: Path,
        face_registration: FaceRegistrationService,
    ) -> None:
        # LegacyRecognitionHTTPServer.__init__ hard-codes its handler class, so
        # initialize ThreadingHTTPServer directly while retaining the exact
        # attributes expected by the inherited request handler.
        self.service = legacy_service
        self.production_service = production_service
        self.face_registration = face_registration
        self.static_root = static_root.resolve()
        ThreadingHTTPServer.__init__(self, address, ProductionRequestHandler)


class ProductionRequestHandler(RecognitionRequestHandler):
    server: ProductionHTTPServer

    def _do_get(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/health/ready":
            legacy = self.server.service.status()
            production = self.server.production_service.status()
            legacy_ready = legacy.get("status") == "ready"
            production_ready = production.get("status") == "ready"
            if legacy_ready and production_ready:
                overall = "ready"
            elif legacy_ready:
                # Production is additive. Keep the proven `/api/tasks` path
                # healthy when a host cannot start the new multi-GPU workers.
                overall = "degraded"
            else:
                overall = "unavailable"
            self._send_json(
                {
                    "status": overall,
                    "legacy": legacy,
                    "production": production,
                },
                status=HTTPStatus.OK if legacy_ready else HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        if path == "/api/v1/recognition/service":
            self._send_json(self.server.production_service.status())
            return
        if path == "/api/v1/recognition/capabilities":
            self._send_json(self.server.production_service.capabilities)
            return
        if path == "/api/v1/recognition/gpus":
            self._send_json({"gpus": self.server.production_service.gpu_status()})
            return
        if path == "/api/v1/recognition/sessions":
            self._send_json({"sessions": self.server.production_service.list_sessions()})
            return
        session_match = _SESSION_PATH.fullmatch(path)
        if session_match:
            session_id, suffix = session_match.groups()
            if suffix is None:
                self._send_json(self.server.production_service.session(session_id))
                return
            if suffix == "stream.mjpg":
                self._stream_production_mjpeg(session_id)
                return
            if suffix == "preview.jpg":
                self._serve_file(
                    self.server.production_service.preview_path(session_id),
                    "image/jpeg",
                    no_store=True,
                )
                return
        baseline_match = _CAMERA_BASELINE_PATH.fullmatch(path)
        if baseline_match:
            camera_id = unquote(baseline_match.group(1))
            value = self.server.production_service.baseline(camera_id)
            if value is None:
                raise ApiError(HTTPStatus.NOT_FOUND, "该摄像头尚未配置正常场景基准图")
            self._send_json(value)
            return
        if path == "/production.js":
            self._serve_static("production.js")
            return
        super()._do_get()

    def _stream_production_mjpeg(self, session_id: str) -> None:
        """Stream the per-session preview JPEG without opening another RTSP source."""

        # Resolve once before sending HTTP headers so an unknown session can
        # still return a normal JSON/404 response rather than a half-open MJPEG stream.
        self.server.production_service.session(session_id)
        preview = self.server.production_service.preview_path(session_id)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        last_mtime = -1
        terminal_since: float | None = None
        next_state_check = 0.0
        terminal = False
        try:
            while True:
                try:
                    stat = preview.stat()
                    if stat.st_mtime_ns != last_mtime:
                        payload = preview.read_bytes()
                        if payload:
                            last_mtime = stat.st_mtime_ns
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                            )
                            self.wfile.write(payload)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                except FileNotFoundError:
                    pass

                now = time.monotonic()
                if now >= next_state_check:
                    next_state_check = now + 1.0
                    try:
                        snapshot = self.server.production_service.session(session_id)
                        terminal = str(snapshot.get("state", "")) in _PRODUCTION_TERMINAL_STATES
                    except KeyError:
                        terminal = True
                if terminal:
                    terminal_since = terminal_since or now
                    if now - terminal_since >= 1.0:
                        return
                else:
                    terminal_since = None
                threading.Event().wait(0.1)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def _validate_optional_model_availability(self, request: SessionRequest) -> None:
        optional = self.server.production_service.capabilities.get("optional", {})
        requested = {
            "smoking": request.features.smoking,
            "eating": request.features.eating,
            "drinking": request.features.drinking,
            "phone": request.features.phone,
        }
        for name, enabled in requested.items():
            info = optional.get(name, {})
            if enabled and not bool(info.get("available")):
                raise ProductionServiceError(
                    "FEATURE_UNAVAILABLE",
                    f"识别能力 {name} 的模型资产尚未就绪",
                    detail={"feature": name, "reason": info.get("reason")},
                )

    def _do_post(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/api/v1/recognition/sessions/start":
            self._require_content_type("json")
            body = self._read_json()
            request = SessionRequest.from_mapping(body)
            self._validate_optional_model_availability(request)
            result = self.server.production_service.start_session(request)
            self._send_json(result, status=HTTPStatus.CREATED)
            return
        session_match = _SESSION_PATH.fullmatch(path)
        if session_match and session_match.group(2) == "stop":
            self._require_optional_json_content_type()
            body = self._read_json(optional=True)
            result = self.server.production_service.stop_session(
                session_match.group(1),
                reason=str(body.get("reason", "requested")),
            )
            self._send_json(result, status=HTTPStatus.ACCEPTED)
            return
        baseline_match = _CAMERA_BASELINE_PATH.fullmatch(path)
        if baseline_match:
            self._require_content_type("image")
            length = self._content_length()
            maximum = self.server.production_service.baselines.max_bytes
            if length <= 0:
                raise ApiError(HTTPStatus.BAD_REQUEST, "基准图片不能为空")
            if length > maximum:
                raise ApiError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    f"基准图片超过 {maximum} 字节限制",
                )
            payload = self.rfile.read(length)
            self._request_body_consumed = len(payload) >= length
            if len(payload) != length:
                raise ApiError(HTTPStatus.BAD_REQUEST, "基准图片上传提前结束")
            result = self.server.production_service.upload_baseline(
                unquote(baseline_match.group(1)),
                payload,
                self.headers.get("Content-Type", "application/octet-stream"),
            )
            self._send_json(result, status=HTTPStatus.CREATED)
            return
        camera_stop_match = _CAMERA_STOP_PATH.fullmatch(path)
        if camera_stop_match:
            self._require_optional_json_content_type()
            self._read_json(optional=True)
            result = self.server.production_service.stop_camera(
                unquote(camera_stop_match.group(1))
            )
            self._send_json(result, status=HTTPStatus.ACCEPTED)
            return
        super()._do_post()

    def _handle_error(self, exc: Exception) -> None:
        if isinstance(exc, ProductionServiceError):
            status_by_code = {
                "SESSION_ALREADY_ACTIVE": HTTPStatus.CONFLICT,
                "FEATURE_NOT_IMPLEMENTED": HTTPStatus.UNPROCESSABLE_ENTITY,
                "FEATURE_UNAVAILABLE": HTTPStatus.UNPROCESSABLE_ENTITY,
                "BASELINE_REQUIRED": HTTPStatus.UNPROCESSABLE_ENTITY,
                "INVALID_EXIT_POLICY": HTTPStatus.UNPROCESSABLE_ENTITY,
                "NO_GPU_CAPACITY": HTTPStatus.SERVICE_UNAVAILABLE,
                "GPU_WORKER_START_FAILED": HTTPStatus.SERVICE_UNAVAILABLE,
                "GPU_WORKER_STOP_FAILED": HTTPStatus.CONFLICT,
                "GPU_WORKER_UNAVAILABLE": HTTPStatus.SERVICE_UNAVAILABLE,
                "SERVICE_STOPPING": HTTPStatus.SERVICE_UNAVAILABLE,
            }
            self._discard_small_request_body()
            self.close_connection = True
            with suppress(BrokenPipeError, ConnectionResetError, OSError):
                self._send_json(
                    {
                        "error": str(exc),
                        "code": exc.code,
                        "detail": exc.detail,
                    },
                    status=status_by_code.get(exc.code, HTTPStatus.CONFLICT),
                )
            return
        super()._handle_error(exc)


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
    # Preserve the existing service preflight exactly. Optional behavior assets
    # are validated separately by every GPU worker only when they are warmable.
    validate_assets(replace(config, sources=()))
    failed_checks = [check for check in run_doctor(config) if not check.ok]
    if failed_checks:
        detail = "; ".join(f"{check.name}: {check.detail}" for check in failed_checks)
        raise RuntimeError(f"识别服务运行环境检查失败: {detail}")

    legacy_service = ManualRecognitionTaskService(
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
    production_root = Path(
        os.environ.get("PRODUCTION_OUTPUT_ROOT", "/workspace/output/production")
    )
    production_service: ProductionRecognitionService | UnavailableProductionRecognitionService
    try:
        production_service = ProductionRecognitionService(
            config,
            output_root=production_root,
            sessions_per_gpu=int(os.environ.get("SERVICE_SESSIONS_PER_GPU", "16")),
            preview_fps=preview_fps,
            preview_width=preview_width,
        )
    except Exception as exc:
        LOGGER.exception(
            "[PRODUCTION_DEGRADED] production workers unavailable; preserving legacy /api/tasks"
        )
        production_service = UnavailableProductionRecognitionService(
            config,
            output_root=production_root,
            error=exc,
        )

    face_registration = build_face_registration_service(config)
    static_root = Path(__file__).resolve().parent / "static"
    server = ProductionHTTPServer(
        (host, port),
        legacy_service,
        production_service,
        static_root,
        face_registration,
    )
    health_path = Path(config.runtime.health_file)
    temporary_health = health_path.with_suffix(health_path.suffix + ".tmp")
    health_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_health.write_text(
        f"{os.getpid()} {host}:{server.server_port}",
        encoding="ascii",
    )
    os.replace(temporary_health, health_path)
    stop_once = threading.Event()

    def request_shutdown(_signum: int, _frame: Any) -> None:
        if stop_once.is_set():
            return
        stop_once.set()
        threading.Thread(
            target=server.shutdown,
            name="http-shutdown",
            daemon=True,
        ).start()

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_shutdown)
    LOGGER.info("识别服务已启动: http://%s:%d", host, server.server_port)
    LOGGER.info(
        "生产识别 REST: http://%s:%d/api/v1/recognition/service",
        host,
        server.server_port,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.shutdown()
        production_service.close()
        legacy_service.close()
        server.server_close()
        health_path.unlink(missing_ok=True)
        temporary_health.unlink(missing_ok=True)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        LOGGER.info("识别服务已停止")


__all__ = ["ProductionHTTPServer", "ProductionRequestHandler", "run_web_service"]
