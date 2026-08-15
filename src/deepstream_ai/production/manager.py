"""Production REST-facing multi-GPU session supervisor."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from deepstream_ai.config import AppConfig
from deepstream_ai.production.baseline import BaselineStore
from deepstream_ai.production.capabilities import production_capabilities
from deepstream_ai.production.contracts import SessionRequest, SessionState, utc_now

LOGGER = logging.getLogger(__name__)
_ACTIVE_STATES = {"starting", "active", "stopping"}
_TERMINAL_STATES = {"completed", "stopped", "failed"}


class ProductionServiceError(RuntimeError):
    def __init__(self, code: str, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = dict(detail or {})


@dataclass(slots=True)
class SupervisorSession:
    session_id: str
    request: SessionRequest
    gpu_id: int
    output_dir: Path
    state: str
    created_at: str
    updated_at: str
    stop_reason: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "cameraId": self.request.camera_id,
            "gpuId": self.gpu_id,
            "state": self.state,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "stopReason": self.stop_reason,
            "error": self.error,
            "features": self.request.features.as_dict(),
            "exitPolicy": self.request.exit_policy.as_dict(),
            "request": self.request.as_dict(redact_url=True),
            "previewUrl": f"/api/v1/recognition/sessions/{self.session_id}/preview.jpg",
        }


def discover_gpu_ids() -> tuple[int, ...]:
    configured = os.environ.get("SERVICE_GPU_IDS", "").strip()
    if configured:
        try:
            values = tuple(dict.fromkeys(int(item.strip()) for item in configured.split(",") if item.strip()))
        except ValueError as exc:
            raise ValueError("SERVICE_GPU_IDS 必须是逗号分隔的物理 GPU 编号") from exc
        if not values or min(values) < 0:
            raise ValueError("SERVICE_GPU_IDS 必须至少包含一个非负 GPU 编号")
        return values

    visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip()
    if visible and visible.lower() not in {"all", "void", "none"}:
        tokens = [item.strip() for item in visible.split(",") if item.strip()]
        if tokens and all(item.isdigit() for item in tokens):
            return tuple(dict.fromkeys(int(item) for item in tokens))

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        values = tuple(
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip().isdigit()
        )
        if values:
            return tuple(dict.fromkeys(values))
    except (OSError, subprocess.SubprocessError):
        pass
    # The existing single-GPU deployment always uses logical gpu-id=0. Keep a
    # conservative fallback so development environments retain compatibility.
    return (0,)


class GpuWorkerClient:
    def __init__(
        self,
        *,
        gpu_id: int,
        config_path: str | Path,
        socket_path: str | Path,
        output_root: str | Path,
        capacity: int,
        preview_fps: float,
        preview_width: int,
    ) -> None:
        self.gpu_id = int(gpu_id)
        self.config_path = Path(config_path).resolve()
        self.socket_path = Path(socket_path).resolve()
        self.output_root = Path(output_root).resolve()
        self.capacity = int(capacity)
        self.preview_fps = float(preview_fps)
        self.preview_width = int(preview_width)
        self.process: subprocess.Popen[bytes] | None = None
        self._log_stream: Any | None = None
        self.last_error: str | None = None

    def spawn(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        log_path = self.output_root / f"gpu-{self.gpu_id}" / "worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_stream = log_path.open("ab", buffering=0)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["DEEPSTREAM_PHYSICAL_GPU_ID"] = str(self.gpu_id)
        command = [
            sys.executable,
            "-m",
            "deepstream_ai.production.worker",
            "--config",
            str(self.config_path),
            "--gpu-id",
            str(self.gpu_id),
            "--socket",
            str(self.socket_path),
            "--output-root",
            str(self.output_root),
            "--capacity",
            str(self.capacity),
            "--preview-fps",
            str(self.preview_fps),
            "--preview-width",
            str(self.preview_width),
        ]
        self.process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        LOGGER.info(
            "[GPU_SUPERVISOR_SPAWN] gpu=%d pid=%d socket=%s log=%s",
            self.gpu_id,
            self.process.pid,
            self.socket_path,
            log_path,
        )

    def wait_ready(self, deadline: float) -> dict[str, Any]:
        while time.monotonic() < deadline:
            process = self.process
            if process is None:
                raise RuntimeError("GPU worker has not been spawned")
            code = process.poll()
            if code is not None:
                self.last_error = f"worker exited during startup with code {code}"
                raise RuntimeError(self.last_error)
            if self.socket_path.exists():
                try:
                    result = self.request({"action": "ping"}, timeout=2.0)
                    worker = result.get("worker") or {}
                    if worker.get("status") == "ready":
                        return worker
                except Exception:
                    pass
            time.sleep(0.25)
        self.last_error = "worker startup timed out"
        raise TimeoutError(f"GPU {self.gpu_id}: {self.last_error}")

    def request(self, document: dict[str, Any], *, timeout: float = 20.0) -> dict[str, Any]:
        payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        try:
            client.connect(str(self.socket_path))
            client.sendall(payload)
            response = b""
            while b"\n" not in response:
                chunk = client.recv(65536)
                if not chunk:
                    break
                response += chunk
                if len(response) > 512 * 1024:
                    raise RuntimeError("GPU worker response too large")
        finally:
            client.close()
        if not response:
            raise RuntimeError(f"GPU {self.gpu_id} worker returned no response")
        value = json.loads(response.split(b"\n", 1)[0].decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("invalid GPU worker response")
        if not value.get("ok"):
            raise RuntimeError(str(value.get("error") or "GPU worker command failed"))
        return value

    def status(self) -> dict[str, Any]:
        process = self.process
        if process is None:
            return {
                "status": "not_started",
                "physicalGpuId": self.gpu_id,
                "capacity": self.capacity,
                "activeSessions": 0,
                "availableSlots": 0,
                "error": self.last_error,
            }
        code = process.poll()
        if code is not None:
            return {
                "status": "failed",
                "physicalGpuId": self.gpu_id,
                "capacity": self.capacity,
                "activeSessions": 0,
                "availableSlots": 0,
                "exitCode": code,
                "error": self.last_error or f"worker exited with code {code}",
            }
        try:
            return dict(self.request({"action": "ping"}, timeout=2.0).get("worker") or {})
        except Exception as exc:
            return {
                "status": "unavailable",
                "physicalGpuId": self.gpu_id,
                "capacity": self.capacity,
                "activeSessions": 0,
                "availableSlots": 0,
                "error": str(exc),
            }

    def send_shutdown(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        with suppress(Exception):
            self.request({"action": "shutdown"}, timeout=2.0)

    def wait_stopped(self, timeout: float = 30.0) -> None:
        process = self.process
        if process is None:
            return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        self.socket_path.unlink(missing_ok=True)
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None


class ProductionRecognitionService:
    def __init__(
        self,
        config: AppConfig,
        *,
        output_root: str | Path,
        sessions_per_gpu: int = 16,
        preview_fps: float = 5.0,
        preview_width: int = 960,
    ) -> None:
        self.config = config
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.sessions_root = self.output_root / "sessions"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.baselines = BaselineStore(self.output_root / "baselines")
        self.capabilities = production_capabilities(config)
        self.sessions_per_gpu = int(sessions_per_gpu)
        if self.sessions_per_gpu < 1:
            raise ValueError("sessions_per_gpu must be positive")
        self.preview_fps = float(preview_fps)
        self.preview_width = int(preview_width)
        self.gpu_ids = discover_gpu_ids()
        self.workers = {
            gpu_id: GpuWorkerClient(
                gpu_id=gpu_id,
                config_path=config.config_path,
                socket_path=self.output_root / "control" / f"gpu-{gpu_id}.sock",
                output_root=self.output_root,
                capacity=self.sessions_per_gpu,
                preview_fps=self.preview_fps,
                preview_width=self.preview_width,
            )
            for gpu_id in self.gpu_ids
        }
        self._sessions: dict[str, SupervisorSession] = {}
        self._camera_to_session: dict[str, str] = {}
        self._history: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._load_history()
        self._start_workers()

    def _load_history(self) -> None:
        now = utc_now().isoformat()
        for directory in self.sessions_root.iterdir():
            if not directory.is_dir() or len(directory.name) != 16:
                continue
            path = directory / "status.json"
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            if str(value.get("state", "")) in _ACTIVE_STATES:
                value["state"] = "stopped"
                value["stopReason"] = "service_restart_manual_start_required"
                value["updatedAt"] = now
                with suppress(OSError):
                    path.write_text(
                        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8",
                    )
            value["historical"] = True
            self._history[directory.name] = value

    def _start_workers(self) -> None:
        # Spawn first, wait second: model deserialization and CUDA initialization
        # happen concurrently across all physical GPUs.
        for worker in self.workers.values():
            worker.spawn()
        deadline = time.monotonic() + float(self.config.runtime.startup_timeout_sec) + 60.0
        failures: dict[int, str] = {}
        ready = 0
        for gpu_id, worker in self.workers.items():
            try:
                worker.wait_ready(deadline)
                ready += 1
            except Exception as exc:
                failures[gpu_id] = str(exc)
                worker.last_error = str(exc)
                LOGGER.exception("GPU worker 启动失败 gpu=%d", gpu_id)
        if ready == 0:
            raise RuntimeError(f"没有可用的 GPU worker: {failures}")
        LOGGER.info(
            "[GPU_SUPERVISOR_READY] ready=%d total=%d failures=%s",
            ready,
            len(self.workers),
            failures,
        )

    def _validate_request(self, request: SessionRequest) -> Path | None:
        optional = self.capabilities["optional"]
        requested = {
            "smoking": request.features.smoking,
            "eating": request.features.eating,
            "drinking": request.features.drinking,
        }
        if request.features.large_object_moving:
            raise ProductionServiceError(
                "FEATURE_NOT_IMPLEMENTED",
                "大件物品搬运已预留接口，当前版本尚未实现",
                detail={"feature": "largeObjectMoving"},
            )
        for name, enabled in requested.items():
            if enabled and not bool(optional[name]["available"]):
                raise ProductionServiceError(
                    "FEATURE_UNAVAILABLE",
                    f"识别能力 {name} 的模型资产尚未就绪",
                    detail={"feature": name, "reason": optional[name].get("reason")},
                )
        baseline: Path | None = None
        if request.features.left_object:
            record = self.baselines.current(request.camera_id)
            if record is None:
                raise ProductionServiceError(
                    "BASELINE_REQUIRED",
                    "启用物品遗留识别前必须为该摄像头上传人员进入前的正常场景图片",
                    detail={"cameraId": request.camera_id},
                )
            baseline = record.path
            minimum_timeout = max(2.0, request.left_object.confirm_frames * 0.5)
            if request.exit_policy.person_absent_seconds < minimum_timeout:
                raise ProductionServiceError(
                    "INVALID_EXIT_POLICY",
                    f"启用物品遗留时 personAbsentSeconds 至少需要 {minimum_timeout:g} 秒",
                )
        return baseline

    def _worker_candidates(self) -> list[tuple[GpuWorkerClient, dict[str, Any]]]:
        values: list[tuple[GpuWorkerClient, dict[str, Any]]] = []
        for worker in self.workers.values():
            status = worker.status()
            if status.get("status") != "ready":
                continue
            if int(status.get("availableSlots", 0)) <= 0:
                continue
            values.append((worker, status))
        return sorted(
            values,
            key=lambda item: (
                int(item[1].get("activeSessions", 0)),
                int(item[0].gpu_id),
            ),
        )

    def start_session(self, request: SessionRequest) -> dict[str, Any]:
        baseline = self._validate_request(request)
        with self._lock:
            existing_id = self._camera_to_session.get(request.camera_id)
            if existing_id is not None:
                existing = self._sessions[existing_id]
                raise ProductionServiceError(
                    "SESSION_ALREADY_ACTIVE",
                    "该摄像头已经存在运行中的识别 Session",
                    detail={
                        "sessionId": existing.session_id,
                        "gpuId": existing.gpu_id,
                    },
                )
        candidates = self._worker_candidates()
        if not candidates:
            raise ProductionServiceError(
                "NO_GPU_CAPACITY",
                "当前没有 READY 且有可用容量的 GPU Worker",
                detail={"gpus": self.gpu_status()},
            )
        worker, _status = candidates[0]
        session_id = uuid4().hex[:16]
        output_dir = self.sessions_root / session_id
        output_dir.mkdir(parents=True, exist_ok=False)
        now = utc_now().isoformat()
        session = SupervisorSession(
            session_id=session_id,
            request=request,
            gpu_id=worker.gpu_id,
            output_dir=output_dir,
            state=SessionState.STARTING.value,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._sessions[session_id] = session
            self._camera_to_session[request.camera_id] = session_id
        self._persist(session)
        try:
            response = worker.request(
                {
                    "action": "start",
                    "sessionId": session_id,
                    "request": request.as_dict(),
                    "outputDir": str(output_dir),
                    "baselinePath": None if baseline is None else str(baseline),
                },
                timeout=30.0,
            )
            snapshot = dict(response.get("session") or {})
            with self._lock:
                session.state = str(snapshot.get("state", SessionState.ACTIVE.value))
                session.updated_at = utc_now().isoformat()
            self._persist(session, overlay=snapshot)
            return self.session(session_id)
        except Exception as exc:
            with self._lock:
                session.state = SessionState.FAILED.value
                session.error = str(exc)
                session.updated_at = utc_now().isoformat()
                self._camera_to_session.pop(request.camera_id, None)
            self._persist(session)
            raise ProductionServiceError(
                "GPU_WORKER_START_FAILED",
                f"GPU {worker.gpu_id} 启动识别 Session 失败: {exc}",
                detail={"gpuId": worker.gpu_id, "sessionId": session_id},
            ) from exc

    def _persist(self, session: SupervisorSession, *, overlay: dict[str, Any] | None = None) -> None:
        document = session.as_dict()
        if overlay:
            document.update(overlay)
        path = session.output_dir / "status.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(document, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temp, path)

    def _refresh_from_worker(self, session: SupervisorSession) -> dict[str, Any]:
        if session.state not in _ACTIVE_STATES:
            return session.as_dict()
        worker = self.workers.get(session.gpu_id)
        if worker is None:
            return session.as_dict()
        try:
            response = worker.request({"action": "sessions"}, timeout=3.0)
            item = next(
                (
                    value
                    for value in response.get("sessions", [])
                    if value.get("sessionId") == session.session_id
                ),
                None,
            )
            if item is None:
                return session.as_dict()
            state = str(item.get("state", session.state))
            with self._lock:
                session.state = state
                session.updated_at = str(item.get("updatedAt", utc_now().isoformat()))
                session.stop_reason = item.get("stopReason")
                session.error = item.get("error")
                if state in _TERMINAL_STATES:
                    self._camera_to_session.pop(session.request.camera_id, None)
            self._persist(session, overlay=item)
            return {**session.as_dict(), **item}
        except Exception:
            LOGGER.exception("刷新 Session 状态失败 session=%s", session.session_id)
            return session.as_dict()

    def session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            current = self._sessions.get(session_id)
            historical = self._history.get(session_id)
        if current is not None:
            return self._refresh_from_worker(current)
        if historical is not None:
            return dict(historical)
        raise KeyError(session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            current = tuple(self._sessions.values())
            history = dict(self._history)
        values = [self._refresh_from_worker(item) for item in current]
        current_ids = {item.session_id for item in current}
        values.extend(value for key, value in history.items() if key not in current_ids)
        return sorted(values, key=lambda item: str(item.get("createdAt", "")), reverse=True)

    def stop_session(self, session_id: str, *, reason: str = "requested") -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            if session_id in self._history:
                return dict(self._history[session_id])
            raise KeyError(session_id)
        if session.state in _TERMINAL_STATES:
            return self.session(session_id)
        worker = self.workers.get(session.gpu_id)
        if worker is None:
            raise ProductionServiceError("GPU_WORKER_UNAVAILABLE", "Session 对应的 GPU Worker 不存在")
        try:
            response = worker.request(
                {"action": "stop", "sessionId": session_id, "reason": reason},
                timeout=30.0,
            )
            item = dict(response.get("session") or {})
            with self._lock:
                session.state = str(item.get("state", SessionState.STOPPED.value))
                session.stop_reason = reason
                session.updated_at = utc_now().isoformat()
                self._camera_to_session.pop(session.request.camera_id, None)
            self._persist(session, overlay=item)
            return {**session.as_dict(), **item}
        except Exception as exc:
            raise ProductionServiceError(
                "GPU_WORKER_STOP_FAILED",
                f"停止识别 Session 失败: {exc}",
                detail={"sessionId": session_id, "gpuId": session.gpu_id},
            ) from exc

    def stop_camera(self, camera_id: str) -> dict[str, Any]:
        with self._lock:
            session_id = self._camera_to_session.get(camera_id)
        if session_id is None:
            raise KeyError(camera_id)
        return self.stop_session(session_id)

    def preview_path(self, session_id: str) -> Path:
        with self._lock:
            current = self._sessions.get(session_id)
        if current is not None:
            return current.output_dir / "preview.jpg"
        directory = (self.sessions_root / session_id).resolve()
        directory.relative_to(self.sessions_root)
        if not directory.is_dir():
            raise KeyError(session_id)
        return directory / "preview.jpg"

    def upload_baseline(self, camera_id: str, payload: bytes, content_type: str) -> dict[str, Any]:
        return self.baselines.save(camera_id, payload, content_type=content_type).as_dict()

    def baseline(self, camera_id: str) -> dict[str, Any] | None:
        record = self.baselines.current(camera_id)
        return None if record is None else record.as_dict()

    def gpu_status(self) -> list[dict[str, Any]]:
        return [self.workers[gpu_id].status() for gpu_id in self.gpu_ids]

    def status(self) -> dict[str, Any]:
        gpus = self.gpu_status()
        ready = [item for item in gpus if item.get("status") == "ready"]
        with self._lock:
            active = sum(1 for item in self._sessions.values() if item.state in _ACTIVE_STATES)
        return {
            "status": "ready" if ready else "unavailable",
            "gpuCount": len(gpus),
            "readyGpuCount": len(ready),
            "activeSessions": active,
            "allocationPolicy": "least_active_sessions",
            "coreAnalytics": "always_enabled",
            "gpus": gpus,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Send shutdown to every worker first so GPUs begin releasing together.
        for worker in self.workers.values():
            worker.send_shutdown()
        for worker in self.workers.values():
            worker.wait_stopped(self.config.runtime.shutdown_timeout_sec + 15.0)


__all__ = [
    "GpuWorkerClient",
    "ProductionRecognitionService",
    "ProductionServiceError",
    "discover_gpu_ids",
]
