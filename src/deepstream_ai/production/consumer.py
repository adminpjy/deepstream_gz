"""Per-camera session fan-out on top of the tuned analytics dispatcher."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepstream_ai.activity import ActivityAwareConsumer, PersonActivityTracker
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.preview import PreviewWriter
from deepstream_ai.production.contracts import SessionRequest
from deepstream_ai.production.publishers import ResultPublisher
from deepstream_ai.production.scenarios import ScenarioManager
from deepstream_ai.production.session_reconcile import (
    ReconcileAnalysisDelegate,
    SessionFinalReconciler,
)

LOGGER = logging.getLogger(__name__)


def _mock_person_id(identity_label: Callable[[str, int | str], str | None], camera_id: str, track_id: int | str) -> str | None:
    """Extract only a verified worker id from the existing presentation label.

    ``AnalyticsDispatcher.identity_label`` intentionally formats UI text as
    ``id=<worker> sim=<score>`` or ``unknown sim=<score>``.  The mock production
    event layer must never treat the latter non-empty display string as a known
    employee.  Pure IDs remain accepted for compatible/test dispatchers.
    """

    value = identity_label(camera_id, track_id)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("unknown"):
        return None
    if lowered.startswith("id="):
        worker_id = text[3:].split(" sim=", 1)[0].strip()
        return worker_id or None
    # Compatibility boundary for delegates that already return a raw worker id.
    return text if " " not in text else None


class VisibleSessionSink:
    """Receive only business-visible tracks after continuity/provisional guards."""

    def __init__(
        self,
        preview: PreviewWriter,
        scenarios: ScenarioManager,
        reconciler: SessionFinalReconciler,
    ) -> None:
        self.preview = preview
        self.scenarios = scenarios
        self.reconciler = reconciler
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
        # Final-reconcile evidence collection is intentionally best-effort and
        # downstream of every existing live consumer.  A mock-push/evidence
        # failure must never interrupt the tuned recognition path.
        try:
            self.reconciler.observe(packet)
        except Exception:
            LOGGER.exception(
                "Session 补偿证据收集失败 camera=%s frame=%s",
                packet.camera_id,
                packet.frame_number,
            )

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
    reconciler: SessionFinalReconciler
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
            **self.reconciler.snapshot(),
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
        # output_dir = <production>/sessions/<session-id>; keep simulation output
        # separate from current session/status/result files.
        mock_root = output_dir.parent.parent / "mock-push"
        reconciler = SessionFinalReconciler(
            session_id=session_id,
            camera_id=request.camera_id,
            mock_root=mock_root,
            identity_label=lambda camera_id, track_id: _mock_person_id(
                self.core_dispatcher.identity_label,
                camera_id,
                track_id,
            ),
        )
        scenarios = ScenarioManager(
            session_id=session_id,
            camera_id=request.camera_id,
            features=request.features,
            publisher=reconciler.scenario_publisher(self.publisher),
            output_dir=output_dir,
            left_object_policy=request.left_object,
            baseline_path=baseline_path,
        )
        visible = VisibleSessionSink(preview, scenarios, reconciler)
        # The existing weak-track guard intentionally sends provisional tracks
        # to analytics but hides them from business preview/events.  This thin
        # pass-through only retains those already-produced observations for a
        # strict end-of-session recovery check; it never makes them live-visible.
        analysis_delegate = ReconcileAnalysisDelegate(self.core_dispatcher, reconciler)
        wrapper = ActivityAwareConsumer(
            analysis_delegate,
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
            reconciler=reconciler,
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
            # Preserve the existing order: LEFT_OBJECT first consumes the recent
            # no-person frames; Final Reconcile then sees that scenario result
            # through its passive publisher tap and chooses/finalizes evidence.
            session.scenarios.on_person_absent()
            try:
                session.reconciler.finalize()
            except Exception:
                LOGGER.exception(
                    "Session 结束补偿失败 session=%s camera=%s；继续正常释放摄像头",
                    session.session_id,
                    session.request.camera_id,
                )
        session.preview.close()
        session.scenarios.close()
        session.reconciler.close()
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


__all__ = [
    "MultiSessionConsumer",
    "SessionConsumer",
    "VisibleSessionSink",
    "_mock_person_id",
]
