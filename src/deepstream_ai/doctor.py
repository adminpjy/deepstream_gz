"""Runtime diagnostics used by local development and deployment preflight."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from deepstream_ai.config import AppConfig
from deepstream_ai.pipeline.runtime import load_runtime, runtime_versions


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def _command(name: str, args: list[str], timeout: int = 15) -> Check:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check(name, False, str(exc))
    detail = (result.stdout or result.stderr).strip().replace("\n", " | ")
    return Check(name, result.returncode == 0, detail[:1000])


def run_doctor(config: AppConfig) -> tuple[Check, ...]:
    checks: list[Check] = [
        _command(
            "nvidia-gpu",
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
        ),
        _command("deepstream-version", ["deepstream-app", "--version-all"]),
    ]
    try:
        runtime = load_runtime()
    except Exception as exc:
        checks.append(Check("python-runtime", False, str(exc)))
        return tuple(checks)
    versions = runtime_versions(runtime)
    checks.append(Check("python-runtime", True, str(versions)))
    factories = {
        "nvstreammux",
        "nvurisrcbin",
        "nvinfer",
        "nvtracker",
        "nvvideoconvert",
        "nvmultistreamtiler",
        "nvdsosd",
        "qtmux",
        "filesink",
    }
    if config.output.enabled:
        factories.update(
            {
                f"nvv4l2{config.output.codec}enc",
                f"{config.output.codec}parse",
            }
        )
    for factory in sorted(factories):
        available = runtime.Gst.ElementFactory.find(factory) is not None
        checks.append(
            Check(
                f"gstreamer:{factory}",
                available,
                "available" if available else "plugin factory not found",
            )
        )
    return tuple(checks)


def checks_as_dict(checks: tuple[Check, ...]) -> dict[str, Any]:
    return {
        "ok": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
    }
