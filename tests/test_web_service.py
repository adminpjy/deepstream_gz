from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from deepstream_ai.web.server import RecognitionHTTPServer

_TASK_ID = "0123456789abcdef"


class _Task:
    def __init__(self, output_dir: Path, *, status: str = "running") -> None:
        self.output_dir = output_dir
        self._status = status

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": _TASK_ID,
            "status": self._status,
            "camera_id": "camera-a",
            "source_type": "file",
            "result_url": (
                f"/api/tasks/{_TASK_ID}/result.mp4"
                if self._status == "completed" and (self.output_dir / "result.mp4").is_file()
                else None
            ),
        }


class _Service:
    def __init__(self, tmp_path: Path) -> None:
        self.task = _Task(tmp_path / "task")
        self.calls: list[tuple[Any, ...]] = []

    def status(self) -> dict[str, Any]:
        return {
            "service_id": "service-a",
            "status": "ready",
            "active_tasks": 1,
            "max_active_tasks": 2,
        }

    def list_tasks(self) -> list[dict[str, Any]]:
        return [self.task.snapshot()]

    def upload(self, stream: Any, length: int, filename: str) -> dict[str, Any]:
        payload = stream.read(length)
        self.calls.append(("upload", filename, payload))
        return {
            "upload_id": "upload-a",
            "filename": filename,
            "size": len(payload),
            "codec": "h264",
            "fps": 10.0,
        }

    def start_file(
        self,
        upload_id: str,
        *,
        camera_id: str | None,
        idle_timeout_sec: Any,
    ) -> dict[str, Any]:
        if upload_id == "missing":
            raise KeyError(upload_id)
        self.calls.append(("start_file", upload_id, camera_id, idle_timeout_sec))
        return {"id": _TASK_ID, "status": "starting", "source_type": "file"}

    def start_rtsp(
        self,
        url: str,
        *,
        camera_id: str | None,
        nominal_fps: float,
        idle_timeout_sec: Any,
    ) -> dict[str, Any]:
        self.calls.append(("start_rtsp", url, camera_id, nominal_fps, idle_timeout_sec))
        return {"id": _TASK_ID, "status": "starting", "source_type": "rtsp"}

    def get_task(self, task_id: str) -> _Task:
        if task_id != _TASK_ID:
            raise KeyError(task_id)
        return self.task

    def stop(self, task_id: str) -> dict[str, Any]:
        self.calls.append(("stop", task_id))
        return {"id": task_id, "status": "stopping"}

    def restart(self) -> dict[str, Any]:
        self.calls.append(("restart",))
        return {
            "service_id": "service-b",
            "previous_service_id": "service-a",
            "status": "ready",
        }


@pytest.fixture
def http_server(tmp_path: Path) -> Iterator[tuple[RecognitionHTTPServer, _Service]]:
    service = _Service(tmp_path)
    static_root = Path(__file__).parents[1] / "src/deepstream_ai/web/static"
    server = RecognitionHTTPServer(("127.0.0.1", 0), service, static_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, service
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(
    server: RecognitionHTTPServer,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    host, port = server.server_address
    connection = http.client.HTTPConnection(str(host), int(port), timeout=3)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = response.read()
        return (
            response.status,
            {key.lower(): value for key, value in response.getheaders()},
            payload,
        )
    finally:
        connection.close()


def _post_json(
    server: RecognitionHTTPServer,
    path: str,
    document: dict[str, Any],
) -> tuple[int, dict[str, str], bytes]:
    payload = json.dumps(document).encode("utf-8")
    return _request(
        server,
        "POST",
        path,
        body=payload,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        },
    )


def _json(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"))


def test_static_console_is_served_with_same_origin_security_headers(
    http_server: tuple[RecognitionHTTPServer, _Service],
) -> None:
    server, _service = http_server

    status, headers, payload = _request(server, "GET", "/")

    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in headers["content-security-policy"]
    assert "智能视频分析控制台" in payload.decode("utf-8")


def test_service_and_task_collection_endpoints(
    http_server: tuple[RecognitionHTTPServer, _Service],
) -> None:
    server, _service = http_server

    service_status, service_headers, service_payload = _request(server, "GET", "/api/service")
    tasks_status, _tasks_headers, tasks_payload = _request(server, "GET", "/api/tasks")

    assert service_status == 200
    assert service_headers["cache-control"] == "no-store"
    assert _json(service_payload)["service_id"] == "service-a"
    assert tasks_status == 200
    assert _json(tasks_payload) == {"tasks": [_service.task.snapshot()]}


def test_raw_upload_and_file_and_rtsp_task_start(
    http_server: tuple[RecognitionHTTPServer, _Service],
) -> None:
    server, service = http_server
    video = b"raw-video-bytes"

    upload_status, _headers, upload_payload = _request(
        server,
        "POST",
        "/api/uploads?filename=shift%20one.mp4",
        body=video,
        headers={
            "Content-Length": str(len(video)),
            "Content-Type": "application/octet-stream",
        },
    )
    file_status, _headers, file_payload = _post_json(
        server,
        "/api/tasks",
        {
            "source_type": "file",
            "upload_id": "upload-a",
            "camera_id": "gate-file",
            "idle_timeout_sec": 12,
        },
    )
    rtsp_status, _headers, rtsp_payload = _post_json(
        server,
        "/api/tasks",
        {
            "source_type": "rtsp",
            "url": "rtsp://camera.example/live",
            "camera_id": "gate-rtsp",
            "nominal_fps": 20,
            "idle_timeout_sec": 15,
        },
    )

    assert upload_status == 201
    assert _json(upload_payload)["upload_id"] == "upload-a"
    assert file_status == 202
    assert _json(file_payload)["source_type"] == "file"
    assert rtsp_status == 202
    assert _json(rtsp_payload)["source_type"] == "rtsp"
    assert service.calls == [
        ("upload", "shift one.mp4", video),
        ("start_file", "upload-a", "gate-file", 12),
        ("start_rtsp", "rtsp://camera.example/live", "gate-rtsp", 20.0, 15),
    ]


def test_stop_task_and_restart_service(
    http_server: tuple[RecognitionHTTPServer, _Service],
) -> None:
    server, service = http_server

    stop_status, _headers, stop_payload = _post_json(server, f"/api/tasks/{_TASK_ID}/stop", {})
    restart_status, _headers, restart_payload = _post_json(server, "/api/service/restart", {})

    assert stop_status == 202
    assert _json(stop_payload)["status"] == "stopping"
    assert restart_status == 200
    assert _json(restart_payload)["previous_service_id"] == "service-a"
    assert service.calls == [("stop", _TASK_ID), ("restart",)]


def test_http_errors_use_stable_json_statuses(
    http_server: tuple[RecognitionHTTPServer, _Service],
) -> None:
    server, _service = http_server

    invalid_source = _post_json(server, "/api/tasks", {"source_type": "camera"})
    missing_upload = _post_json(
        server,
        "/api/tasks",
        {"source_type": "file", "upload_id": "missing"},
    )
    missing_task = _request(server, "GET", "/api/tasks/ffffffffffffffff")
    active_result = _request(server, "GET", f"/api/tasks/{_TASK_ID}/result.mp4")

    assert invalid_source[0] == 400
    assert "source_type" in _json(invalid_source[2])["error"]
    assert missing_upload[0] == 404
    assert _json(missing_upload[2]) == {"error": "资源不存在"}
    assert missing_task[0] == 404
    assert _json(missing_task[2]) == {"error": "识别任务不存在"}
    assert active_result[0] == 409
    assert "尚未完成封装" in _json(active_result[2])["error"]


def test_post_rejects_cross_origin_and_unsafe_content_type(
    http_server: tuple[RecognitionHTTPServer, _Service],
) -> None:
    server, _service = http_server

    cross_origin = _request(
        server,
        "POST",
        "/api/tasks",
        body=b"{}",
        headers={
            "Content-Type": "application/json",
            "Content-Length": "2",
            "Origin": "https://attacker.example",
        },
    )
    unsafe_upload = _request(
        server,
        "POST",
        "/api/uploads?filename=test.mp4",
        body=b"video",
        headers={"Content-Type": "text/plain", "Content-Length": "5"},
    )

    assert cross_origin[0] == 403
    assert unsafe_upload[0] == 415


def test_completed_result_supports_bounded_http_ranges(
    http_server: tuple[RecognitionHTTPServer, _Service],
) -> None:
    server, service = http_server
    service.task.output_dir.mkdir(parents=True)
    (service.task.output_dir / "result.mp4").write_bytes(b"0123456789")
    service.task._status = "completed"

    status, headers, payload = _request(
        server,
        "GET",
        f"/api/tasks/{_TASK_ID}/result.mp4",
        headers={"Range": "bytes=2-5"},
    )

    assert status == 206
    assert headers["content-range"] == "bytes 2-5/10"
    assert payload == b"2345"
