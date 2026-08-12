from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

from deepstream_ai.manual_task_service import ManualRecognitionTaskService


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self._done = threading.Event()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if not self._done.wait(timeout):
            raise subprocess.TimeoutExpired("fake-task-worker", timeout)
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = 0
            self._done.set()

    def kill(self) -> None:
        if self.returncode is None:
            self.returncode = -9
            self._done.set()


class FakeFactory:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, command: list[str], **_kwargs: object) -> FakeProcess:
        process = FakeProcess()
        self.calls.append(list(command))
        self.processes.append(process)
        return process


def _service(tmp_path: Path, factory: FakeFactory, *, max_tasks: int = 4) -> ManualRecognitionTaskService:
    config = tmp_path / "configs" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("source: {type: file, path: placeholder.mp4}\n", encoding="utf-8")
    return ManualRecognitionTaskService(
        config,
        uploads_root=tmp_path / "uploads",
        tasks_root=tmp_path / "tasks",
        max_active_tasks=max_tasks,
        stop_grace_sec=0.1,
        process_factory=factory,
    )


def _historical_rtsp(
    root: Path,
    task_id: str,
    *,
    status: str = "starting",
    camera_id: str = "camera-01",
    url: str = "rtsp://example.test/live",
) -> None:
    directory = root / task_id
    directory.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": 1,
        "task_id": task_id,
        "created_at": "2026-08-12T08:00:00+00:00",
        "base_config": str(root.parent / "configs" / "config.yaml"),
        "source": {
            "type": "rtsp",
            "camera_id": camera_id,
            "url": url,
            "nominal_fps": 25.0,
            "enabled": True,
        },
        "source_label": "rtsp://example.test/live",
        "upload_id": None,
        "output_dir": str(directory),
        "idle_timeout_sec": 30.0,
        "preview_fps": 5.0,
        "preview_width": 960,
    }
    (directory / "task.json").write_text(json.dumps(spec), encoding="utf-8")
    (directory / "status.json").write_text(
        json.dumps(
            {
                "id": task_id,
                "status": status,
                "created_at": spec["created_at"],
                "updated_at": spec["created_at"],
                "source_type": "rtsp",
                "camera_id": camera_id,
            }
        ),
        encoding="utf-8",
    )


def test_opening_service_never_starts_historical_tasks(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    _historical_rtsp(tasks, "aaaaaaaaaaaaaaaa", status="starting")
    _historical_rtsp(tasks, "bbbbbbbbbbbbbbbb", status="running", camera_id="camera-02")
    factory = FakeFactory()

    service = _service(tmp_path, factory)

    assert factory.calls == []
    listed = service.list_tasks()
    assert len(listed) == 2
    assert {item["status"] for item in listed} == {"stopped"}
    assert {item["stop_reason"] for item in listed} == {
        "service_restart_manual_start_required"
    }
    assert all(item["historical"] is True for item in listed)
    assert service.status()["active_tasks"] == 0


def test_historical_task_only_starts_after_explicit_start(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    old_id = "aaaaaaaaaaaaaaaa"
    _historical_rtsp(tasks, old_id, status="stopped")
    factory = FakeFactory()
    service = _service(tmp_path, factory)

    assert factory.calls == []
    created = service.start_existing(old_id)

    assert created["id"] != old_id
    assert created["camera_id"] == "camera-01"
    assert created["status"] == "starting"
    assert len(factory.calls) == 1
    service.close()


def test_manual_history_start_supports_multiple_concurrent_sources(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    _historical_rtsp(tasks, "aaaaaaaaaaaaaaaa", status="stopped", camera_id="camera-01")
    _historical_rtsp(
        tasks,
        "bbbbbbbbbbbbbbbb",
        status="stopped",
        camera_id="camera-02",
        url="rtsp://example.test/two",
    )
    factory = FakeFactory()
    service = _service(tmp_path, factory, max_tasks=4)

    first = service.start_existing("aaaaaaaaaaaaaaaa")
    second = service.start_existing("bbbbbbbbbbbbbbbb")

    assert first["id"] != second["id"]
    assert service.status()["active_tasks"] == 2
    assert len(factory.calls) == 2
    service.close()
