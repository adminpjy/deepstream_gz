from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from deepstream_ai.cli import main


def test_non_strict_validate_reports_missing_assets_and_nonzero(tmp_path: Path, capsys) -> None:
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs/config.yaml"
    config.write_text(
        """source: {type: file, path: videos/missing.mp4}
person: {enabled: true, config_file: configs/person.txt}
tracker: {config_file: configs/tracker.yml}
runtime: {strict_assets: true}
""",
        encoding="utf-8",
    )

    code = main(["validate", "--config", str(config), "--no-strict-assets"])

    output = capsys.readouterr().out
    assert code == 4
    assert '"ok": false' in output
    assert "missing.mp4" in output


def test_serve_forwards_http_task_service_options(monkeypatch, tmp_path: Path) -> None:
    config = SimpleNamespace(runtime=SimpleNamespace(log_level="INFO", json_logs=False))
    captured: dict[str, object] = {}

    monkeypatch.setattr("deepstream_ai.cli.load_config", lambda _path: config)
    monkeypatch.setattr("deepstream_ai.cli.configure_logging", lambda *_args: None)

    def run_web_service(received_config: object, **options: object) -> None:
        captured["config"] = received_config
        captured.update(options)

    monkeypatch.setattr("deepstream_ai.web.run_web_service", run_web_service)

    code = main(
        [
            "serve",
            "--config",
            str(tmp_path / "config.yaml"),
            "--host",
            "127.0.0.1",
            "--port",
            "9080",
            "--uploads-root",
            str(tmp_path / "uploads"),
            "--tasks-root",
            str(tmp_path / "tasks"),
            "--idle-timeout-sec",
            "17.5",
            "--max-upload-mb",
            "512",
            "--max-tasks",
            "3",
            "--preview-fps",
            "4",
            "--preview-width",
            "800",
        ]
    )

    assert code == 0
    assert captured == {
        "config": config,
        "host": "127.0.0.1",
        "port": 9080,
        "uploads_root": str(tmp_path / "uploads"),
        "tasks_root": str(tmp_path / "tasks"),
        "idle_timeout_sec": 17.5,
        "max_upload_mb": 512,
        "max_tasks": 3,
        "preview_fps": 4.0,
        "preview_width": 800,
    }
