"""Per-camera session fan-out on top of the tuned analytics dispatcher."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from deepstream_ai.activity import ActivityAwareConsumer, PersonActivityTracker
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.preview import PreviewWriter
from deepstream_ai.production.contracts import SessionRequest
from deepstream_ai.production.publishers import ResultPublisher
from deepstream_ai.production.scenarios import ScenarioManager

LOGGER = logging.getLogger(__name__)


class VisibleSessionSink:
    """Receive only business-visible tracks after continuity/provisional guards."""

    def __init__(self, preview: PreviewWriter, scenarios: ScenarioManager) -> None:
        self.preview = preview
        self.scenarios = scenarios
        self._person_count = 0
        self._last_timestamp: str | None = None
        self._last_person_seen_at: str | None = None
        self._lock = threading.RLock()

    def submit(self, packet: FramePacket) -> None:
        count = len(packet.tracks)
        timestamp = packet.timestamp.isoformat()
        with self._lock:
            self._person_count = count
            self._last_timestamp = timestamp
            if count:
                self._last_person_seen_at = timestamp
        self.preview.submit(packet)
        self.scenarios.process(packet)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "personCount": self._person_count,
                "lastFrameAt": self._last_timestamp,
                "lastPersonSeenAt": self._last_person_seen_at,
            }


@dataclass(slots=True)
class SessionConsumer:
    session_id: str
    request: SessionRequest
    output_dir: Path
    activity: PersonActivityTracker
    preview: PreviewWriter
    scenarios: ScenarioManager
    visible: VisibleSessionSink
    wrapper: ActivityAwareConsumer
    stop_event: threading.Event
    idle_thread: threading.Thread

    def snapshot(self) -> dict[str, Any]:
        activity = self.activity.snapshot()
        return {
            "sessionId": self.session_id,
            "cameraId": self.request.camera_id,
            "features": self.request.features.as_dict(),
            "exitPolicy": self.request.exit_policy.as_dict(),
            "frames": activity.frames,
            "personFrames": activity.person_frames,
            "personDetections": activity.person_detections,
            "faceDetections": activity.face_detections,
            "idleSeconds": activity.idle_seconds,
            "idleTriggered": activity.idle_triggered,
            "previewPath": str(self.output_dir / "preview.jpg"),
            **self.visible.snapshot(),
        }


class MultiSessionConsumer:
    """Route dynamic-source packets into isolated per-session business state."""

    def __init__(
        self,
        core_dispatcher: Any,
        publisher: ResultPublisher,
        *,
        on_idle: Callable[[str, str], None],
        preview_fps: float = 5.0,
        preview_width: int = 960,
    ) -> None:
        self.core_dispatcher = core_dispatcher
        self.publisher = publisher
        self.on_idle = on_idle
        self.preview_fps = float(preview_fps)
        self.preview_width = int(preview_width)
        self._sessions: dict[str, SessionConsumer] = {}
        self._by_camera: dict[str, str] = {}
        self._lock = threading.RLock()

    def add_session(
        self,
        session_id: str,
        request: SessionRequest,
        output_dir: str | Path,
        *,
        baseline_path: str | Path | None,
    ) -> SessionConsumer:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if session_id in self._sessions:
                raise RuntimeError(f"session already exists: {session_id}")
            if request.camera_id in self._by_camera:
                raise RuntimeError(f"camera already has a session: {request.camera_id}")

        activity = PersonActivityTracker(request.exit_policy.person_absent_seconds)
        preview = PreviewWriter(
            output_dir / "preview.jpg",
            identity_label=self.core_dispatcher.identity_label,
            max_fps=self.preview_fps,
            max_width=self.preview_width,
        )
        scenarios = ScenarioManager(
            session_id=session_id,
            camera_id=request.camera_id,
            features=request.features,
            publisher=self.publisher,
            output_dir=output_dir,
            left_object_policy=request.left_object,
            baseline_path=baseline_path,
        )
        visible = VisibleSessionSink(preview, scenarios)
        wrapper = ActivityAwareConsumer(
            self.core_dispatcher,
            activity,
            preview=visible,
        )
        stop_event = threading.Event()
        idle_thread = threading.Thread(
            target=self._idle_waiter,
            args=(session_id, request.camera_id, activity, stop_event),
            name=f"idle-{session_id}",
            daemon=True,
        )
        session = SessionConsumer(
            session_id=session_id,
            request=request,
            output_dir=output_dir,
            activity=activity,
            preview=preview,
            scenarios=scenarios,
            visible=visible,
            wrapper=wrapper,
            stop_event=stop_event,
            idle_thread=idle_thread,
        )
        preview.start()
        with self._lock:
            self._sessions[session_id] = session
            self._by_camera[request.camera_id] = session_id
        idle_thread.start()
        return session

    def _idle_waiter(
        self,
        session_id: str,
        camera_id: str,
        activity: PersonActivityTracker,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            if activity.idle_event.wait(0.25):
                if stop_event.is_set():
                    return
                try:
                    self.on_idle(session_id, camera_id)
                except Exception:
                    LOGGER.exception(
                        "处理人员离场超时失败 session=%s camera=%s",
                        session_id,
                        camera_id,
                    )
                return

    def remove_session(
        self,
        session_id: str,
        *,
        run_absent_hooks: bool,
    ) -> SessionConsumer | None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return None
            self._by_camera.pop(session.request.camera_id, None)
        session.stop_event.set()
        if run_absent_hooks:
            session.scenarios.on_person_absent()
        session.preview.close()
        session.scenarios.close()
        if session.idle_thread is not threading.current_thread():
            session.idle_thread.join(1.0)
        return session

    def submit(self, packet: FramePacket) -> bool:
        with self._lock:
            session_id = self._by_camera.get(packet.camera_id)
            session = self._sessions.get(session_id) if session_id is not None else None
        if session is None:
            return True
        return bool(session.wrapper.submit(packet))

    def identity_label(self, camera_id: str, track_id: int | str) -> str | None:
        with self._lock:
            session_id = self._by_camera.get(camera_id)
            session = self._sessions.get(session_id) if session_id is not None else None
        if session is None:
            return self.core_dispatcher.identity_label(camera_id, track_id)
        return session.wrapper.identity_label(camera_id, track_id)

    def presentation_track_id(
        self,
        camera_id: str,
        raw_track_id: int | str,
    ) -> int | str | None:
        with self._lock:
            session_id = self._by_camera.get(camera_id)
            session = self._sessions.get(session_id) if session_id is not None else None
        if session is None:
            return None
        return session.wrapper.presentation_track_id(camera_id, raw_track_id)

    def queue_metrics(self) -> dict[str, float | int]:
        return self.core_dispatcher.queue_metrics()

    def snapshot(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
        return None if session is None else session.snapshot()

    def session_id_for_camera(self, camera_id: str) -> str | None:
        with self._lock:
            return self._by_camera.get(camera_id)

    def has_camera(self, camera_id: str) -> bool:
        return self.session_id_for_camera(camera_id) is not None

    def close(self) -> None:
        with self._lock:
            session_ids = tuple(self._sessions)
        for session_id in session_ids:
            try:
                self.remove_session(session_id, run_absent_hooks=False)
            except Exception:
                LOGGER.exception("关闭 Session Consumer 失败 session=%s", session_id)


__all__ = ["MultiSessionConsumer", "SessionConsumer", "VisibleSessionSink"]
