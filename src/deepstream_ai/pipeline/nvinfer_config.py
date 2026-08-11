"""Materialize immutable nvinfer templates with runtime-safe overrides."""

from __future__ import annotations

import configparser
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from deepstream_ai.errors import PipelineError

_PATH_KEYS = {
    "onnx-file",
    "model-engine-file",
    "labelfile-path",
    "int8-calib-file",
    "custom-lib-path",
    "tlt-encoded-model",
    "model-file",
    "proto-file",
    "uff-file",
    "mean-file",
}


def materialize_nvinfer_config(
    source: Path,
    destination: Path,
    overrides: Mapping[str, str | int | float],
) -> Path:
    """Copy an nvinfer config and atomically apply authoritative YAML values.

    SGIE cadence is a config-file-only option in DeepStream 9.0, so a GObject
    property override cannot implement it. Relative asset paths are absolutized
    before moving the runtime copy to the writable output directory.
    """

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str  # type: ignore[method-assign]
    try:
        with source.open("r", encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, configparser.Error) as exc:
        raise PipelineError(f"无法读取 nvinfer 配置 {source}: {exc}") from exc
    section_name = next((name for name in parser.sections() if name.lower() == "property"), None)
    if section_name is None:
        raise PipelineError(f"nvinfer 配置缺少 [property] 节: {source}")
    section = parser[section_name]
    original_keys = {key.lower(): key for key in section}
    for lowered, original in tuple(original_keys.items()):
        if lowered not in _PATH_KEYS:
            continue
        raw = section[original].strip().strip('"').strip("'")
        if not raw or raw.replace("\\", "/").startswith("/workspace/"):
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            section[original] = str((source.parent / candidate).resolve())
    for key, value in overrides.items():
        existing = original_keys.get(key.lower(), key)
        section[existing] = str(value)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            parser.write(temporary, space_around_delimiters=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except Exception as exc:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise PipelineError(f"无法写入运行时 nvinfer 配置 {destination}: {exc}") from exc
    return destination


__all__ = ["materialize_nvinfer_config"]
