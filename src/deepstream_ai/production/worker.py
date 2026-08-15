"""Long-lived production GPU worker.

One process owns exactly one physical GPU via CUDA_VISIBLE_DEVICES.  The tuned
DeepStream core sees logical gpu-id=0 exactly as before. Models/Pipeline remain
resident while RTSP sources are attached and detached through a private Unix
socket control plane.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from deepstream_ai.config import SourceConfig, load_config
from deepstream_ai.logging_config import configure_logging
from deepstream_ai.pipeline.runtime import load_runtime, runtime_versions
from deepstream_ai.preflight import validate_assets
from deepstream_ai.production.capabilities import warmable_behavior_names
from deepstream_ai.production.consumer import MultiSessionConsumer
from deepstream_ai.production.contracts import SessionRequest, SessionState, utc_now
from deepstream_ai.production.feature_gate import FeatureRegistry
from deepstream_ai.production.pipeline import (
    DynamicPipelineRunner,
    DynamicSourceController,
    WarmDynamicPipelineBuilder,
    build_warm_config,
)
from deepstream_ai.production.publishers import AlarmResultAdapter, build_result_publisher
from deepstream_ai.provisional_analytics import ProvisionalAwareAnalyticsDispatcher

LOGGER = logging.getLogger(__name__)
_MAX_MESSAGE_BYTES = 256 * 1024


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temp, path)


@dataclass(slots=True)
class WorkerSession:
    session_id: str
    request: SessionRequest
    output_dir: Path
    slot: int
    state: SessionState
    created_at: str
    updated_at: str
    stop_reason: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "cameraId": self.request.camera_id,
            "state": self.state.value,
            "slot": self.slot,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "stopReason": self.stop_reason,
            "error": self.error,
            "request": self.request.as_dict(redact_url=True),
        }


class ProductionGpuWorker:
    def __init__(
        self,
        *,
        config_path: str | Path,
        physical_gpu_id: int,
        socket_path: str | Path,
        output_root: str | Path,
        capacity: int,
        preview_fps: float,
        preview_width: int,
    ) -> None:
        self.physical_gpu_id = int(physical_gpu_id)
        self.socket_path = Path(socket_path).resolve()
        self.output_root = Path(output_root).resolve()
        self.worker_root = self.output_root / f"gpu-{self.physical_gpu_id}"
        self.worker_root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.worker_root / "worker-status.json"
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("worker capacity must be positive")
        self.preview_fps = float(preview_fps)
        self.preview_width = int(preview_width)
        self.base_config = load_config(config_path)
        configure_logging(
            self.base_config.runtime.log_level,
            self.base_config.runtime.json_logs,
        )
        warm_behaviors = warmable_behavior_names(self.base_config)
        self.config = build_warm_config(
            self.base_config,
            capacity=self.capacity,
            worker_root=self.worker_root,
            enabled_behavior_names=warm_behaviors,
        )
        self.warm_behaviors = warm_behaviors
        self.runtime: Any | None = None
        self.publisher: Any | None = None
        self.dispatcher: ProvisionalAwareAnalyticsDispatcher | None = None
        self.consumer: MultiSessionConsumer | None = None
        self.feature_registry = FeatureRegistry()
        self.builder: WarmDynamicPipelineBuilder | None = None
        self.graph: Any | None = None
        self.runner: DynamicPipelineRunner | None = None
        self.controller: DynamicSourceController | None = None
        self.pipeline_thread: threading.Thread | None = None
        self._sessions: dict[str, WorkerSession] = {}
        self._camera_to_session: dict[str, str] = {}
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._fatal_error: BaseException | None = None

    def start(self) -> None:
        LOGGER.info(
            "[GPU_WORKER_START] physical_gpu=%d logical_gpu=0 capacity=%d warm_behaviors=%s",
            self.physical_gpu_id,
            self.capacity,
            list(self.warm_behaviors),
        )
        reports = validate_assets(self.config)
        LOGGER.info(
            "[GPU_WORKER_PREFLIGHT] physical_gpu=%d engines=%s",
            self.physical_gpu_id,
            [str(item.engine_path) for item in reports if item.engine_path is not None],
        )
        self.runtime = load_runtime()
        LOGGER.info(
            "[GPU_WORKER_RUNTIME] physical_gpu=%d versions=%s",
            self.physical_gpu_id,
            runtime_versions(self.runtime),
        )
        self.publisher = build_result_publisher(self.output_root)
        consumer_holder: dict[str, MultiSessionConsumer] = {}
        alarm_adapter = AlarmResultAdapter(
            self.publisher,
            lambda camera_id: (
                consumer_holder["consumer"].session_id_for_camera(camera_id)
                if "consumer" in consumer_holder
                else None
            ),
            evidence_root=self.output_root / "alarm-evidence",
        )
        self.dispatcher = ProvisionalAwareAnalyticsDispatcher(
            self.config,
            queue_size=self.config.runtime.analytics_queue_size,
            alarm_publisher=alarm_adapter,
        )
        self.dispatcher.start()
        self.consumer = MultiSessionConsumer(
            self.dispatcher,
            self.publisher,
            on_idle=self._on_idle,
            preview_fps=self.preview_fps,
            preview_width=self.preview_width,
        )
        consumer_holder["consumer"] = self.consumer
        self.builder = WarmDynamicPipelineBuilder(
            self.runtime,
            self.config,
            self.consumer,
            self.feature_registry,
        )
        self.graph = self.builder.build()

        runner_holder: dict[str, DynamicPipelineRunner] = {}

        def on_started() -> None:
            runner = runner_holder["runner"]
            runner._install_physical_gpu_monitor()
            self._ready.set()
            self._write_status("ready")
            LOGGER.info(
                "[GPU_WORKER_READY] physical_gpu=%d capacity=%d warm_behaviors=%s",
                self.physical_gpu_id,
                self.capacity,
                list(self.warm_behaviors),
            )

        self.runner = DynamicPipelineRunner(
            self.runtime,
            self.config,
            self.graph,
            physical_gpu_id=self.physical_gpu_id,
            on_started=on_started,
        )
        runner_holder["runner"] = self.runner
        self.controller = DynamicSourceController(
            self.runtime,
            self.config,
            self.graph,
            self.feature_registry,
            capacity=self.capacity,
            shadow_registry_getter=lambda: self.runner.shadow_registry if self.runner else None,
        )
        self.runner.source_controller = self.controller
        self.pipeline_thread = threading.Thread(
            target=self._run_pipeline,
            name=f"gpu-pipeline-{self.physical_gpu_id}",
            daemon=True,
        )
        self.pipeline_thread.start()
        timeout = float(self.config.runtime.startup_timeout_sec) + 30.0
        deadline = time.monotonic() + timeout
        while not self._ready.wait(0.25):
            if self._fatal_error is not None:
                raise RuntimeError("GPU worker pipeline failed during startup") from self._fatal_error
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"GPU {self.physical_gpu_id} worker did not become ready in {timeout:.0f}s"
                )

    def _run_pipeline(self) -> None:
        assert self.runner is not None
        try:
            self.runner.run()
        except BaseException as exc:
            self._fatal_error = exc
            LOGGER.exception("GPU worker Pipeline 异常退出 gpu=%d", self.physical_gpu_id)
            self._write_status("failed", error=str(exc))
        finally:
            self._ready.clear()

    def _write_status(self, state: str, *, error: str | None = None) -> None:
        with self._lock:
            sessions = [item.as_dict() for item in self._sessions.values()]
        _atomic_json(
            self.status_path,
            {
                "status": state,
                "physicalGpuId": self.physical_gpu_id,
                "logicalGpuId": 0,
                "capacity": self.capacity,
                "activeSessions": sum(
                    1
                    for item in sessions
                    if item["state"] in {"starting", "active", "stopping"}
                ),
                "warmBehaviors": list(self.warm_behaviors),
                "updatedAt": utc_now().isoformat(),
                "error": error,
                "sessions": sessions,
            },
        )

    def _glib_call(self, callback: Callable[[], Any], *, timeout: float = 15.0) -> Any:
        if not self._ready.is_set() or self.runtime is None:
            raise RuntimeError("GPU worker is not ready")
        done = threading.Event()
        result: dict[str, Any] = {}

        def invoke() -> bool:
            try:
                result["value"] = callback()
            except BaseException as exc:
                result["error"] = exc
            finally:
                done.set()
            return False

        self.runtime.GLib.idle_add(invoke)
        if not done.wait(timeout):
            raise TimeoutError("GPU worker GLib operation timed out")
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def start_session(
        self,
        session_id: str,
        request: SessionRequest,
        *,
        output_dir: str | Path,
        baseline_path: str | Path | None,
    ) -> dict[str, Any]:
        if not self._ready.is_set():
            raise RuntimeError("GPU worker is not ready")
        if request.features.large_object_moving:
            raise NotImplementedError("largeObjectMoving is reserved but not implemented")
        with self._lock:
            if session_id in self._sessions:
                raise RuntimeError(f"session already exists: {session_id}")
            if request.camera_id in self._camera_to_session:
                raise RuntimeError(f"camera already active: {request.camera_id}")
            active = sum(
                1
                for item in self._sessions.values()
                if item.state in {SessionState.STARTING, SessionState.ACTIVE, SessionState.STOPPING}
            )
            if active >= self.capacity:
                raise RuntimeError("GPU worker session capacity reached")
        assert self.consumer is not None
        assert self.controller is not None
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now().isoformat()
        session = WorkerSession(
            session_id=session_id,
            request=request,
            output_dir=output_dir,
            slot=-1,
            state=SessionState.STARTING,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._sessions[session_id] = session
            self._camera_to_session[request.camera_id] = session_id
        try:
            self.consumer.add_session(
                session_id,
                request,
                output_dir,
                baseline_path=baseline_path,
            )
            source = SourceConfig(
                camera_id=request.camera_id,
                type="rtsp",
                url=request.stream_url,
                enabled=True,
                nominal_fps=request.nominal_fps,
                latency_ms=int(request.context.get("rtspLatencyMs", 200)),
                reconnect_interval_sec=int(request.context.get("reconnectIntervalSec", 10)),
            )
            slot = int(self._glib_call(lambda: self.controller.add(source, request.features)))
            with self._lock:
                session.slot = slot
                session.state = SessionState.ACTIVE
                session.updated_at = utc_now().isoformat()
            self._write_session_status(session)
            self._write_status("ready")
            return self.session_snapshot(session_id)
        except Exception as exc:
            LOGGER.exception(
                "启动生产 Session 失败 session=%s camera=%s gpu=%d",
                session_id,
                request.camera_id,
                self.physical_gpu_id,
            )
            with suppress(Exception):
                self.consumer.remove_session(session_id, run_absent_hooks=False)
            with self._lock:
                self._camera_to_session.pop(request.camera_id, None)
                session.state = SessionState.FAILED
                session.error = str(exc)
                session.updated_at = utc_now().isoformat()
            self._write_session_status(session)
            self._write_status("ready")
            raise

    def _write_session_status(self, session: WorkerSession) -> None:
        _atomic_json(session.output_dir / "status.json", self.session_snapshot(session.session_id))

    def session_snapshot(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            document = session.as_dict()
        if self.consumer is not None:
            consumer = self.consumer.snapshot(session_id)
            if consumer is not None:
                document.update(consumer)
        document.update(
            {
                "gpuId": self.physical_gpu_id,
                "previewUrl": f"/api/v1/recognition/sessions/{session_id}/preview.jpg",
            }
        )
        return document

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            ids = tuple(self._sessions)
        return [self.session_snapshot(session_id) for session_id in ids]

    def _on_idle(self, session_id: str, camera_id: str) -> None:
        LOGGER.info(
            "[PERSON_ABSENT_TIMEOUT] session=%s camera=%s gpu=%d",
            session_id,
            camera_id,
            self.physical_gpu_id,
        )
        try:
            self.stop_session(
                session_id,
                reason="person_absent_timeout",
                run_absent_hooks=True,
            )
        except Exception:
            LOGGER.exception("人员离场自动停止失败 session=%s", session_id)

    def stop_session(
        self,
        session_id: str,
        *,
        reason: str,
        run_absent_hooks: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            if session.state in {SessionState.STOPPED, SessionState.COMPLETED, SessionState.FAILED}:
                return self.session_snapshot(session_id)
            session.state = SessionState.STOPPING
            session.stop_reason = reason
            session.updated_at = utc_now().isoformat()
        self._write_session_status(session)
        assert self.consumer is not None
        assert self.controller is not None
        try:
            # LeftObject post-check runs before detaching because it consumes the
            # recent no-person frames already captured by this session. It is CPU
            # OpenCV work and never blocks the GStreamer streaming thread.
            self.consumer.remove_session(
                session_id,
                run_absent_hooks=run_absent_hooks,
            )
            self._glib_call(lambda: self.controller.remove(session.request.camera_id))
            with self._lock:
                session.state = SessionState.COMPLETED if run_absent_hooks else SessionState.STOPPED
                session.updated_at = utc_now().isoformat()
                self._camera_to_session.pop(session.request.camera_id, None)
            self._write_session_status(session)
            self._write_status("ready")
            return self.session_snapshot(session_id)
        except Exception as exc:
            with self._lock:
                session.state = SessionState.FAILED
                session.error = str(exc)
                session.updated_at = utc_now().isoformat()
                self._camera_to_session.pop(session.request.camera_id, None)
            self._write_session_status(session)
            self._write_status("ready")
            raise

    def status(self) -> dict[str, Any]:
        with self._lock:
            active = sum(
                1
                for item in self._sessions.values()
                if item.state in {SessionState.STARTING, SessionState.ACTIVE, SessionState.STOPPING}
            )
        return {
            "status": "ready" if self._ready.is_set() else "failed" if self._fatal_error else "starting",
            "physicalGpuId": self.physical_gpu_id,
            "logicalGpuId": 0,
            "capacity": self.capacity,
            "activeSessions": active,
            "availableSlots": max(0, self.capacity - active),
            "warmBehaviors": list(self.warm_behaviors),
            "pid": os.getpid(),
            "fatalError": None if self._fatal_error is None else str(self._fatal_error),
        }

    def handle_command(self, command: dict[str, Any]) -> dict[str, Any]:
        action = str(command.get("action", "")).strip().lower()
        if action == "ping":
            return {"ok": True, "worker": self.status()}
        if action == "start":
            session_id = str(command.get("sessionId", "")).strip()
            if not session_id:
                raise ValueError("sessionId is required")
            request = SessionRequest.from_mapping(command.get("request") or {})
            return {
                "ok": True,
                "session": self.start_session(
                    session_id,
                    request,
                    output_dir=str(command.get("outputDir", "")),
                    baseline_path=command.get("baselinePath"),
                ),
            }
        if action == "stop":
            session_id = str(command.get("sessionId", "")).strip()
            return {
                "ok": True,
                "session": self.stop_session(
                    session_id,
                    reason=str(command.get("reason", "requested")),
                    run_absent_hooks=False,
                ),
            }
        if action == "sessions":
            return {"ok": True, "sessions": self.list_sessions()}
        if action == "shutdown":
            self._stop.set()
            return {"ok": True}
        raise ValueError(f"unknown worker command: {action}")

    def serve(self) -> None:
        self.start()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        server.listen(32)
        server.settimeout(0.5)
        try:
            while not self._stop.is_set():
                if self._fatal_error is not None:
                    raise RuntimeError("GPU worker pipeline terminated") from self._fatal_error
                try:
                    connection, _address = server.accept()
                except socket.timeout:
                    continue
                threading.Thread(
                    target=self._serve_connection,
                    args=(connection,),
                    name=f"gpu-{self.physical_gpu_id}-control",
                    daemon=True,
                ).start()
        finally:
            server.close()
            self.socket_path.unlink(missing_ok=True)
            self.close()

    def _serve_connection(self, connection: socket.socket) -> None:
        with connection:
            connection.settimeout(15.0)
            try:
                payload = b""
                while b"\n" not in payload:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    payload += chunk
                    if len(payload) > _MAX_MESSAGE_BYTES:
                        raise ValueError("worker control message too large")
                command = json.loads(payload.split(b"\n", 1)[0].decode("utf-8"))
                if not isinstance(command, dict):
                    raise ValueError("worker control message must be an object")
                response = self.handle_command(command)
            except Exception as exc:
                LOGGER.exception("GPU worker control command failed")
                response = {
                    "ok": False,
                    "error": str(exc),
                    "errorType": type(exc).__name__,
                }
            with suppress(OSError):
                connection.sendall(
                    json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    + b"\n"
                )

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            active_ids = [
                item.session_id
                for item in self._sessions.values()
                if item.state in {SessionState.STARTING, SessionState.ACTIVE, SessionState.STOPPING}
            ]
        for session_id in active_ids:
            with suppress(Exception):
                self.stop_session(session_id, reason="worker_shutdown", run_absent_hooks=False)
        if self.runner is not None and self.runtime is not None and self._ready.is_set():
            with suppress(Exception):
                self.runtime.GLib.idle_add(lambda: (self.runner.stop(send_eos=False), False)[1])
        if self.pipeline_thread is not None:
            self.pipeline_thread.join(self.config.runtime.shutdown_timeout_sec + 5.0)
        if self.consumer is not None:
            with suppress(Exception):
                self.consumer.close()
        if self.dispatcher is not None:
            with suppress(Exception):
                self.dispatcher.close()
        if self.publisher is not None:
            with suppress(Exception):
                self.publisher.close()
        self._ready.clear()
        self._write_status("stopped")
        LOGGER.info("[GPU_WORKER_STOPPED] physical_gpu=%d", self.physical_gpu_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepStream production GPU worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--preview-fps", type=float, default=5.0)
    parser.add_argument("--preview-width", type=int, default=960)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # The supervisor sets this before exec. Keep an explicit guard because a
    # wrong device mapping would invalidate the logical-gpu=0 preservation rule.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != str(args.gpu_id):
        raise RuntimeError(
            f"worker requires CUDA_VISIBLE_DEVICES={args.gpu_id}, got {visible!r}"
        )
    worker = ProductionGpuWorker(
        config_path=args.config,
        physical_gpu_id=args.gpu_id,
        socket_path=args.socket,
        output_root=args.output_root,
        capacity=args.capacity,
        preview_fps=args.preview_fps,
        preview_width=args.preview_width,
    )

    def stop(_signum: int, _frame: Any) -> None:
        worker._stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    worker.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
