"""Safe GStreamer element construction helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from deepstream_ai.errors import PipelineError


def make_element(Gst: Any, factory: str, name: str) -> Any:
    element = Gst.ElementFactory.make(factory, name)
    if element is None:
        raise PipelineError(
            f"无法创建 GStreamer 元素 {factory} ({name})；请确认 DeepStream 插件已安装并可被 gst-inspect-1.0 发现"
        )
    return element


def add_many(pipeline: Any, elements: Iterable[Any]) -> None:
    for element in elements:
        pipeline.add(element)


def link_many(elements: Iterable[Any]) -> None:
    chain = list(elements)
    for upstream, downstream in zip(chain, chain[1:], strict=False):
        if not upstream.link(downstream):
            raise PipelineError(f"GStreamer 元素连接失败: {upstream.name} -> {downstream.name}")


def set_if_supported(element: Any, name: str, value: Any) -> bool:
    if element.find_property(name) is None:
        return False
    element.set_property(name, value)
    return True


def existing_path(path: Path, label: str) -> str:
    if not path.exists():
        raise PipelineError(f"{label} 不存在: {path}")
    return str(path)
