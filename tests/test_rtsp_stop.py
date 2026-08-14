from __future__ import annotations

from types import SimpleNamespace

from deepstream_ai.pipeline.runner import PipelineRunner


class _Loop:
    def __init__(self) -> None:
        self.quit_calls = 0

    def quit(self) -> None:
        self.quit_calls += 1

    def is_running(self) -> bool:
        return True


class _GLib:
    def __init__(self) -> None:
        self.loop = _Loop()
        self.timeouts: list[tuple[int, object]] = []

    def MainLoop(self) -> _Loop:
        return self.loop

    def timeout_add_seconds(self, delay: int, callback: object) -> int:
        self.timeouts.append((delay, callback))
        return 1


class _Event:
    @staticmethod
    def new_eos() -> object:
        return object()


class _Pipeline:
    def __init__(self) -> None:
        self.events: list[object] = []

    def send_event(self, event: object) -> bool:
        self.events.append(event)
        return True


def test_live_rtsp_user_stop_skips_eos_and_quits_immediately() -> None:
    glib = _GLib()
    runtime = SimpleNamespace(GLib=glib, Gst=SimpleNamespace(Event=_Event))
    config = SimpleNamespace(
        runtime=SimpleNamespace(shutdown_timeout_sec=20),
        enabled_sources=(SimpleNamespace(type="rtsp", camera_id="camera-a"),),
    )
    pipeline = _Pipeline()
    runner = PipelineRunner(runtime, config, SimpleNamespace(pipeline=pipeline))

    runner.stop()

    assert glib.loop.quit_calls == 1
    assert pipeline.events == []
    assert glib.timeouts == []
