"""One isolated DeepStream task process managed by the recognition service."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepstream_ai.activity import ActivityAwareConsumer, PersonActivityTracker
from deepstream_ai.analytics import AnalyticsDispatcher
from deepstream_ai.config import AppConfig, SourceConfig, load_config
from deepstream_ai.logging_config import configure_logging
from deepstream_ai.pipeline.builder import DeepStreamPipelineBuilder
from deepstream_ai.pipeline.runner import PipelineRunner
from deepstream_ai.pipeline.runtime import load_runtime, runtime_versions
from deepstream_ai.preflight import validate_assets
from deepstream_ai.preview import PreviewWriter

LOGGER = logging.getLogger(__name__)
_TASK_ID = re.compile(r"^[a-f0-9]{12,32}$")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class TaskStatusWriter:
    def __init__(self, path: Path, initial: dict[str, Any]) -> None:
        self.path = path
        self._document = dict(initial)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.update()

    def update(self, **values: Any) -> dict[str, Any]:
        with self._lock:
            self._document.update(values)
            self._document["updated_at"] = _utc_now()
            payload = json.dumps(
                self._document,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            try:
                temporary.write_bytes(payload)
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            return dict(self._document)


def build_task_config(base: AppConfig, spec: dict[str, Any]) -> AppConfig:
    task_id = str(spec.get("task_id", ""))
    if not _TASK_ID.fullmatch(task_id):
        raise ValueError("invalid task_id")
    source_raw = spec.get("source")
    if not isinstance(source_raw, dict):
        raise ValueError("task source must be an object")
    source = SourceConfig.from_mapping(source_raw, 0)
    output_dir = Path(str(spec["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = replace(base.output.snapshot, root=str(output_dir / "snapshot"))
    output = replace(
        base.output,
        # File jobs produce a finalized MP4. An unbounded RTSP recording would
        # eventually fill the host volume, so live jobs publish preview/events
        # and snapshots without a monolithic result file.
        enabled=source.type == "file",
        path=str(output_dir / "result.mp4"),
        sync=source.type == "file",
        events_enabled=True,
        events_path=str(output_dir / "events.jsonl"),
        snapshot=snapshot,
    )
    runtime = replace(
        base.runtime,
        health_file=str(output_dir / "pipeline.ready"),
    )
    return replace(base, sources=(source,), output=output, runtime=runtime)


def _log_contract(config: AppConfig) -> None:
    detector_type = config.pipeline.person.detector_type
    tracker_backend = config.pipeline.tracker.backend
    LOGGER.info("[DETECTOR] %s", "PeopleNet" if detector_type == "peoplenet" else detector_type)
    LOGGER.info("[TRACKER] %s", "NvDCF" if tracker_backend == "nvdcf" else tracker_backend)
    LOGGER.info("[TRACKER_CONFIG] %s", config.pipeline.tracker.config_file)
    LOGGER.info(
        "[PEOPLENET_CLASSES] %s",
        ", ".join(f"{name}={class_id}" for name, class_id in config.pipeline.person.people_classes),
    )


def _read_control_reason(path: Path) -> str | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    reason = str(document.get("stop_reason", "")).strip()
    return reason or None


def run_task(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    task_id = str(spec.get("task_id", ""))
    output_dir = Path(str(spec["output_dir"])).resolve()
    status = TaskStatusWriter(
        output_dir / "status.json",
        {
            "id": task_id,
            "status": "starting",
            "created_at": str(spec.get("created_at") or _utc_now()),
            "started_at": None,
            "ended_at": None,
            "stop_reason": None,
            "error": None,
            "source_type": spec.get("source", {}).get("type"),
            "camera_id": spec.get("source", {}).get("camera_id"),
        },
    )
    base = load_config(Path(str(spec["base_config"])))
    configure_logging(base.runtime.log_level, base.runtime.json_logs)
    config = build_task_config(base, spec)
    idle_timeout_sec = float(spec.get("idle_timeout_sec", 10.0))
    preview_fps = float(spec.get("preview_fps", 5.0))
    preview_width = int(spec.get("preview_width", 960))
    control_path = output_dir / "control.json"
    activity = PersonActivityTracker(idle_timeout_sec)
    dispatcher: AnalyticsDispatcher | None = None
    preview: PreviewWriter | None = None
    reporter_stop = threading.Event()
    reporter: threading.Thread | None = None
    error: BaseException | None = None

    try:
        reports = validate_assets(config)
        for report in reports:
            LOGGER.info(
                "模型预检通过 config=%s engine=%s source_models=%s",
                report.config_path,
                report.engine_path,
                [str(path) for path in report.source_models],
            )
        _log_contract(config)
        runtime = load_runtime()
        LOGGER.info("DeepStream Python runtime 已加载: %s", runtime_versions(runtime))
        dispatcher = AnalyticsDispatcher(
            config,
            queue_size=config.runtime.analytics_queue_size,
        )
        dispatcher.start()
        preview = PreviewWriter(
            output_dir / "preview.jpg",
            identity_label=dispatcher.identity_label,
            max_fps=preview_fps,
            max_width=preview_width,
        )
        preview.start()
        consumer = ActivityAwareConsumer(dispatcher, activity, preview)
        graph = DeepStreamPipelineBuilder(runtime, config, consumer).build()

        def on_started() -> None:
            status.update(status="running", started_at=_utc_now())

        runner = PipelineRunner(runtime, config, graph, on_started=on_started)

        def report_status() -> None:
            while not reporter_stop.wait(0.5):
                snapshot = activity.snapshot()
                values: dict[str, Any] = {
                    "frames": snapshot.frames,
                    "person_frames": snapshot.person_frames,
                    "person_detections": snapshot.person_detections,
                    "face_detections": snapshot.face_detections,
                    "idle_seconds": round(snapshot.idle_seconds, 3),
                    "idle_timeout_sec": idle_timeout_sec,
                }
                if preview is not None:
                    values.update(preview.stats())
                status.update(**values)

        reporter = threading.Thread(target=report_status, name=f"status-{task_id}", daemon=True)
        reporter.start()

        def stop_when_idle() -> None:
            activity.idle_event.wait()
            if reporter_stop.is_set():
                return
            LOGGER.info(
                "连续 %.3f 秒视频时间未检测到人员，自动停止 task_id=%s",
                activity.snapshot().idle_seconds,
                task_id,
            )
            status.update(status="stopping", stop_reason="idle_timeout")

            def request_stop() -> bool:
                runner.stop(send_eos=True)
                return False

            runtime.GLib.idle_add(request_stop)

        idle_monitor = threading.Thread(
            target=stop_when_idle,
            name=f"idle-{task_id}",
            daemon=True,
        )
        idle_monitor.start()
        runner.run()
    except BaseException as exc:
        error = exc
        LOGGER.exception("识别任务失败 task_id=%s", task_id)
    finally:
        reporter_stop.set()
        if reporter is not None:
            reporter.join(2.0)
        close_error: BaseException | None = None
        if dispatcher is not None:
            try:
                dispatcher.close()
            except BaseException as exc:
                close_error = exc
                LOGGER.exception("关闭分析组件失败 task_id=%s", task_id)
        if preview is not None:
            preview.close()
        if error is None and close_error is not None:
            error = close_error

    activity_snapshot = activity.snapshot()
    preview_stats = preview.stats() if preview is not None else {}
    common = {
        "ended_at": _utc_now(),
        "frames": activity_snapshot.frames,
        "person_frames": activity_snapshot.person_frames,
        "person_detections": activity_snapshot.person_detections,
        "face_detections": activity_snapshot.face_detections,
        "idle_seconds": round(activity_snapshot.idle_seconds, 3),
        "idle_timeout_sec": idle_timeout_sec,
        **preview_stats,
    }
    if error is not None:
        status.update(status="failed", error=f"{type(error).__name__}: {error}", **common)
        return 1
    control_reason = _read_control_reason(control_path)
    if control_reason:
        reason = control_reason
        terminal_status = "stopped"
    elif activity.idle_event.is_set():
        reason = "idle_timeout"
        terminal_status = "completed"
    else:
        reason = "eos"
        terminal_status = "completed"
    status.update(status=terminal_status, stop_reason=reason, **common)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated DeepStream recognition task")
    parser.add_argument("--spec", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_task(args.spec.resolve())
    except Exception:
        logging.basicConfig(level=logging.INFO)
        LOGGER.exception("任务进程启动失败")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_task_config", "main", "run_task"]
