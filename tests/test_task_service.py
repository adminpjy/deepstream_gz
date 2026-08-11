from __future__ import annotations

import io
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepstream_ai.task_service import MediaInfo, RecognitionTaskService, probe_media


class FakeProcess:
    def __init__(self, *, exit_on_terminate: bool = True) -> None:
        self.exit_on_terminate = exit_on_terminate
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self._done = threading.Event()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if not self._done.wait(timeout):
            raise subprocess.TimeoutExpired("fake-task-worker", timeout)
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.exit_on_terminate:
            self.finish(0)

    def kill(self) -> None:
        self.kill_calls += 1
        self.finish(-9)

    def finish(self, returncode: int) -> None:
        if self.returncode is None:
            self.returncode = returncode
            self._done.set()


class FakeProcessFactory:
    def __init__(self, *, exit_on_terminate: bool = True) -> None:
        self.exit_on_terminate = exit_on_terminate
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, command: list[str], **kwargs: object) -> FakeProcess:
        process = FakeProcess(exit_on_terminate=self.exit_on_terminate)
        self.calls.append((list(command), dict(kwargs)))
        self.processes.append(process)
        return process


def media_probe(_path: Path) -> MediaInfo:
    return MediaInfo("h264", 25.0, 1920, 1080, 30.0)


def make_service(
    tmp_path: Path,
    factory: FakeProcessFactory,
    *,
    max_active_tasks: int = 2,
) -> RecognitionTaskService:
    config = tmp_path / "configs" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("source: {type: file, path: placeholder.mp4}\n", encoding="utf-8")
    return RecognitionTaskService(
        config,
        uploads_root=tmp_path / "uploads",
        tasks_root=tmp_path / "tasks",
        max_upload_bytes=1024,
        max_active_tasks=max_active_tasks,
        stop_grace_sec=0.1,
        media_probe=media_probe,
        process_factory=factory,
    )


def upload(service: RecognitionTaskService, filename: str = "video.mp4") -> dict[str, object]:
    payload = b"fake-video"
    return service.upload(io.BytesIO(payload), len(payload), filename)


def wait_for_status(
    service: RecognitionTaskService,
    task_id: str,
    expected: str,
    *,
    timeout: float = 1.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = service.get_task(task_id).snapshot()
        if snapshot.get("status") == expected:
            return snapshot
        threading.Event().wait(0.005)
    raise AssertionError(
        f"task {task_id} did not reach {expected}: {service.get_task(task_id).snapshot()}"
    )


def set_worker_status(service: RecognitionTaskService, task_id: str, status: str) -> None:
    task = service.get_task(task_id)
    current = json.loads(task.status_path.read_text(encoding="utf-8"))
    task.status_path.write_text(
        json.dumps({**current, "status": status}),
        encoding="utf-8",
    )


def test_upload_filename_is_confined_and_early_eof_is_cleaned_up(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeProcessFactory())

    saved = upload(service, "..\\..\\outside.mp4")
    record = service.uploads.get(str(saved["upload_id"]))
    assert record.filename == "outside.mp4"
    assert record.path.name == "source.mp4"
    assert record.path.parent.parent == (tmp_path / "uploads").resolve()
    assert not (tmp_path / "outside.mp4").exists()

    before = {path.name for path in (tmp_path / "uploads").iterdir()}
    with pytest.raises(ValueError, match="提前结束"):
        service.upload(io.BytesIO(b"short"), 10, "truncated.mp4")
    after = {path.name for path in (tmp_path / "uploads").iterdir()}
    assert after == before

    with pytest.raises(ValueError, match="超过 1024"):
        service.upload(io.BytesIO(b"x"), 1025, "large.mp4")


def test_probe_media_uses_ffprobe_contract_without_running_real_ffprobe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_name": "h264",
                            "avg_frame_rate": "30000/1001",
                            "width": 1920,
                            "height": 1080,
                        }
                    ],
                    "format": {"duration": "12.5"},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("deepstream_ai.task_service.subprocess.run", fake_run)
    source = tmp_path / "source.mp4"
    info = probe_media(source)

    assert info.codec == "h264"
    assert info.fps == pytest.approx(29.97003)
    assert info.width == 1920
    assert info.height == 1080
    assert info.duration_sec == 12.5
    assert calls[0][0][0] == "ffprobe"
    assert calls[0][0][-1] == str(source)
    assert calls[0][1]["timeout"] == 30


def test_probe_media_falls_back_when_slim_image_has_no_ffprobe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = MediaInfo("h264", 10.0, 1920, 1080, 22.3)

    def missing_ffprobe(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr("deepstream_ai.task_service.subprocess.run", missing_ffprobe)
    monkeypatch.setattr(
        "deepstream_ai.task_service._probe_media_with_opencv",
        lambda _path: expected,
    )

    assert probe_media(tmp_path / "source.mp4") == expected


def test_start_returns_unique_id_and_redacts_rtsp_credentials(tmp_path: Path) -> None:
    factory = FakeProcessFactory()
    service = make_service(tmp_path, factory)
    first_upload = upload(service)

    first = service.start_file(str(first_upload["upload_id"]), camera_id="file-camera")
    second = service.start_rtsp(
        "rtsp://user:secret@example.test:8554/live?token=secret",
        camera_id="rtsp-camera",
    )

    assert len(str(first["id"])) == 16
    assert len(str(second["id"])) == 16
    assert first["id"] != second["id"]
    assert second["source_label"] == "rtsp://example.test:8554/live"
    assert "secret" not in json.dumps(second)
    assert len(factory.calls) == 2
    for command, options in factory.calls:
        assert command[1:3] == ["-m", "deepstream_ai.task_worker"]
        assert command[3] == "--spec"
        assert options["shell"] is False
        assert options["stdin"] is subprocess.DEVNULL

    service.close()


def test_max_concurrency_releases_capacity_after_process_exit(tmp_path: Path) -> None:
    factory = FakeProcessFactory()
    service = make_service(tmp_path, factory, max_active_tasks=1)
    first_upload = upload(service, "first.mp4")
    second_upload = upload(service, "second.mp4")
    first = service.start_file(str(first_upload["upload_id"]))

    with pytest.raises(RuntimeError, match="达到上限 1"):
        service.start_file(str(second_upload["upload_id"]))

    factory.processes[0].finish(0)
    completed = wait_for_status(service, str(first["id"]), "completed")
    assert completed["stop_reason"] == "eos"

    second = service.start_file(str(second_upload["upload_id"]))
    assert second["status"] == "starting"
    service.close()


def test_stop_by_id_is_graceful_and_unknown_id_is_rejected(tmp_path: Path) -> None:
    factory = FakeProcessFactory(exit_on_terminate=False)
    service = make_service(tmp_path, factory)
    item = service.start_rtsp("rtsp://example.test/live", camera_id="gate-a")
    task_id = str(item["id"])
    set_worker_status(service, task_id, "running")

    response = service.stop(task_id)
    assert response["status"] == "stopping"
    assert factory.processes[0].terminate_calls == 1
    assert service.stop(task_id)["stop_reason"] == "requested"
    assert factory.processes[0].terminate_calls == 1
    control = json.loads(service.get_task(task_id).control_path.read_text(encoding="utf-8"))
    assert control["stop_reason"] == "requested"

    factory.processes[0].finish(0)
    stopped = wait_for_status(service, task_id, "stopped")
    assert stopped["stop_reason"] == "requested"
    with pytest.raises(KeyError):
        service.stop("missing-task")


def test_restart_rotates_service_id_and_stops_all_active_tasks(tmp_path: Path) -> None:
    factory = FakeProcessFactory()
    service = make_service(tmp_path, factory, max_active_tasks=2)
    first = service.start_rtsp("rtsp://example.test/one")
    second = service.start_rtsp("rtsp://example.test/two")
    old_service_id = service.service_id

    result = service.restart()

    assert result["previous_service_id"] == old_service_id
    assert result["service_id"] != old_service_id
    assert result["status"] == "ready"
    assert result["active_tasks"] == 0
    assert result["total_tasks"] == 2
    assert [process.terminate_calls for process in factory.processes] == [1, 1]
    assert wait_for_status(service, str(first["id"]), "stopped")["stop_reason"] == (
        "service_restart"
    )
    assert wait_for_status(service, str(second["id"]), "stopped")["stop_reason"] == (
        "service_restart"
    )


def test_process_exit_transitions_to_completed_or_failed(tmp_path: Path) -> None:
    factory = FakeProcessFactory()
    service = make_service(tmp_path, factory, max_active_tasks=2)
    completed = service.start_rtsp("rtsp://example.test/completed")
    failed = service.start_rtsp("rtsp://example.test/failed")
    set_worker_status(service, str(completed["id"]), "running")
    set_worker_status(service, str(failed["id"]), "running")

    factory.processes[0].finish(0)
    factory.processes[1].finish(7)

    completed_status = wait_for_status(service, str(completed["id"]), "completed")
    failed_status = wait_for_status(service, str(failed["id"]), "failed")
    assert completed_status["stop_reason"] == "eos"
    assert completed_status["error"] is None
    assert failed_status["stop_reason"] == "worker_exit"
    assert failed_status["error"] == "task worker exited with code 7"


def test_worker_failure_detail_is_preserved_and_starting_cancel_is_stopped(
    tmp_path: Path,
) -> None:
    factory = FakeProcessFactory(exit_on_terminate=False)
    service = make_service(tmp_path, factory, max_active_tasks=2)
    failed = service.start_rtsp("rtsp://example.test/failed-detail")
    cancelled = service.start_rtsp("rtsp://example.test/cancelled")

    failed_task = service.get_task(str(failed["id"]))
    failed_document = json.loads(failed_task.status_path.read_text(encoding="utf-8"))
    failed_task.status_path.write_text(
        json.dumps(
            {
                **failed_document,
                "status": "failed",
                "stop_reason": None,
                "error": "PipelineError: decoder failed",
            }
        ),
        encoding="utf-8",
    )
    factory.processes[0].finish(1)

    service.stop(str(cancelled["id"]))
    factory.processes[1].finish(-15)

    assert wait_for_status(service, str(failed["id"]), "failed")["error"] == (
        "PipelineError: decoder failed"
    )
    cancelled_status = wait_for_status(service, str(cancelled["id"]), "stopped")
    assert cancelled_status["stop_reason"] == "requested"


def test_service_rejects_unsafe_camera_and_preview_settings(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeProcessFactory())
    item = upload(service)

    with pytest.raises(ValueError, match="camera_id"):
        service.start_file(str(item["upload_id"]), camera_id="../camera")
    with pytest.raises(ValueError, match="preview_fps"):
        RecognitionTaskService(
            tmp_path / "configs/config.yaml",
            uploads_root=tmp_path / "more-uploads",
            tasks_root=tmp_path / "more-tasks",
            preview_fps=float("nan"),
        )
