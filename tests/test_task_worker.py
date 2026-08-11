from pathlib import Path

from deepstream_ai.config import load_config
from deepstream_ai.task_worker import build_task_config


def _spec(tmp_path: Path, source: dict[str, object]) -> dict[str, object]:
    return {
        "task_id": "0123456789abcdef",
        "output_dir": str(tmp_path / "task"),
        "source": source,
    }


def test_file_task_is_namespaced_recorded_and_clock_paced(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DATABASE_DSN", "postgresql://test:test@localhost/test")
    base = load_config(Path("configs/config.yaml"))
    config = build_task_config(
        base,
        _spec(
            tmp_path,
            {
                "type": "file",
                "camera_id": "file-a",
                "path": str(Path("videos/test.mp4").resolve()),
                "nominal_fps": 10,
            },
        ),
    )

    assert config.output.enabled is True
    assert config.output.sync is True
    assert Path(config.output.path).parent == (tmp_path / "task").resolve()
    assert config.runtime.health_file.endswith("pipeline.ready")


def test_rtsp_task_has_preview_events_without_unbounded_mp4(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DATABASE_DSN", "postgresql://test:test@localhost/test")
    base = load_config(Path("configs/config.yaml"))
    config = build_task_config(
        base,
        _spec(
            tmp_path,
            {
                "type": "rtsp",
                "camera_id": "rtsp-a",
                "url": "rtsp://camera.example/live",
                "nominal_fps": 25,
            },
        ),
    )

    assert config.output.enabled is False
    assert config.output.sync is False
    assert config.output.events_enabled is True
    assert config.output.snapshot.root.endswith("snapshot")
