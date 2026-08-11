#!/usr/bin/env python3
"""Host-side file-source codec/FPS checks used by both preflight frontends."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from deepstream_ai.config import load_config


def _frame_rate(stream: dict[str, Any]) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = str(stream.get(key, "0/0"))
        try:
            value = Fraction(raw)
        except (ValueError, ZeroDivisionError):
            continue
        if value > 0:
            return float(value)
    raise ValueError("ffprobe 未返回有效视频帧率")


def _probe(ffprobe: str, path: Path) -> tuple[str, float]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit={result.returncode}"
        raise ValueError(f"ffprobe 失败: {detail}")
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError("ffprobe 未发现可解析的视频流") from exc
    return str(stream.get("codec_name", "")).lower(), _frame_rate(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    file_sources = [source for source in config.enabled_sources if source.type == "file"]
    failures: list[str] = []
    resolved: list[tuple[Any, Path]] = []
    for source in file_sources:
        path = config.resolve_path(source.location)
        if not path.is_file():
            failures.append(f"file source {source.camera_id} 不存在: {path}")
        else:
            resolved.append((source, path))

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        if resolved:
            print("[preflight] ffprobe 不可用；已跳过文件编码/FPS 检查。")
    else:
        for source, path in resolved:
            try:
                codec, actual_fps = _probe(ffprobe, path)
            except ValueError as exc:
                failures.append(f"file source {source.camera_id}: {exc}")
                continue
            if codec not in {"h264", "hevc"}:
                failures.append(
                    f"file source {source.camera_id} codec={codec or 'unknown'}；"
                    "仅允许 NVDEC 目标 H.264/H.265(HEVC)"
                )
            nominal = float(source.nominal_fps)
            tolerance = max(0.5, nominal * 0.05)
            if abs(actual_fps - nominal) > tolerance:
                failures.append(
                    f"file source {source.camera_id} nominal_fps={nominal:.3f} "
                    f"与 ffprobe fps={actual_fps:.3f} 明显不一致 "
                    f"(tolerance={tolerance:.3f})"
                )
            print(
                f"[preflight] media camera={source.camera_id} "
                f"codec={codec} fps={actual_fps:.3f} nominal={nominal:.3f} path={path}"
            )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
