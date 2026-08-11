"""Command-line interface for run, validation, and runtime diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from deepstream_ai import __version__
from deepstream_ai.app import DeepStreamApplication
from deepstream_ai.config import load_config
from deepstream_ai.doctor import checks_as_dict, run_doctor
from deepstream_ai.errors import DeepStreamAIError
from deepstream_ai.logging_config import configure_logging
from deepstream_ai.preflight import inspect_assets

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepstream-ai",
        description="NVIDIA DeepStream industrial video analytics platform",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=("run", "serve", "validate", "doctor"),
        help="run one pipeline, serve the task API/UI, validate assets, or inspect runtime",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="YAML configuration path",
    )
    parser.add_argument(
        "--no-strict-assets",
        action="store_true",
        help="only list missing assets during validate; never use for production run",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--host", default="127.0.0.1", help="serve command listen host")
    parser.add_argument("--port", type=int, default=8080, help="serve command listen port")
    parser.add_argument(
        "--idle-timeout-sec",
        type=float,
        default=10.0,
        help="default person-free video time before a task exits",
    )
    parser.add_argument(
        "--uploads-root",
        default="/workspace/uploads",
        help="serve command upload directory",
    )
    parser.add_argument(
        "--tasks-root",
        default="/workspace/output/tasks",
        help="serve command per-task output directory",
    )
    parser.add_argument("--max-upload-mb", type=int, default=2048)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--preview-fps", type=float, default=5.0)
    parser.add_argument("--preview-width", type=int, default=960)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_strict_assets and args.command != "validate":
        build_parser().error("--no-strict-assets 仅可与 validate 一起使用")
    try:
        config = load_config(Path(args.config))
        configure_logging(config.runtime.log_level, config.runtime.json_logs)
        if args.command == "validate":
            reports, failures = inspect_assets(config)
            summary = {
                "ok": not failures,
                "config": str(config.config_path),
                "sources": [source.camera_id for source in config.enabled_sources],
                # Kept for automation compatibility; this now includes semantic
                # config/parser mismatches as well as absent files.
                "missing": list(failures),
                "nvinfer": [
                    {
                        "config": str(report.config_path),
                        "engine": str(report.engine_path) if report.engine_path else None,
                        "source_models": [str(path) for path in report.source_models],
                        "missing": list(report.missing),
                    }
                    for report in reports
                ],
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if summary["ok"] else 4
        if args.command == "doctor":
            result = checks_as_dict(run_doctor(config))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ok"] else 3
        if args.command == "serve":
            from deepstream_ai.web import run_web_service

            run_web_service(
                config,
                host=args.host,
                port=args.port,
                uploads_root=args.uploads_root,
                tasks_root=args.tasks_root,
                idle_timeout_sec=args.idle_timeout_sec,
                max_upload_mb=args.max_upload_mb,
                max_tasks=args.max_tasks,
                preview_fps=args.preview_fps,
                preview_width=args.preview_width,
            )
            return 0
        DeepStreamApplication(config).run()
        return 0
    except KeyboardInterrupt:
        LOGGER.info("用户中断")
        return 130
    except DeepStreamAIError as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception:
        LOGGER.exception("未处理的致命错误")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
