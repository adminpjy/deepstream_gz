"""Lazy access to DeepStream runtime modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepstream_ai.errors import RuntimeUnavailableError


@dataclass(frozen=True, slots=True)
class DeepStreamRuntime:
    Gst: Any
    GLib: Any
    pyds: Any


def load_runtime() -> DeepStreamRuntime:
    """Load GI and pyds only inside the NVIDIA DeepStream container."""

    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst
    except (ImportError, ValueError) as exc:
        raise RuntimeUnavailableError(
            "无法导入 GStreamer Python bindings。请在项目 DeepStream Docker 镜像内运行。"
        ) from exc
    try:
        import pyds
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "无法导入 pyds。请使用 docker/Dockerfile 构建的镜像，或安装匹配当前 DeepStream 的 Python bindings。"
        ) from exc
    Gst.init(None)
    return DeepStreamRuntime(Gst=Gst, GLib=GLib, pyds=pyds)


def runtime_versions(runtime: DeepStreamRuntime) -> dict[str, str]:
    Gst, pyds = runtime.Gst, runtime.pyds
    gst_version = ".".join(str(part) for part in Gst.version())
    pyds_version = str(getattr(pyds, "__version__", "unknown"))
    return {"gstreamer": gst_version, "pyds": pyds_version}
