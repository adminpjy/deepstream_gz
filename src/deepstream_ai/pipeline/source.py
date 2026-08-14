"""File and RTSP source bins backed by NVIDIA hardware decode."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any
from urllib.parse import urlparse

from deepstream_ai.config import AppConfig, SourceConfig
from deepstream_ai.errors import PipelineError

from .elements import make_element, set_if_supported

LOGGER = logging.getLogger(__name__)


def source_uri(config: AppConfig, source: SourceConfig) -> str:
    if source.type == "rtsp":
        parsed = urlparse(source.location)
        if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
            raise PipelineError(f"RTSP 源 {source.camera_id} 的 URL 无效: {source.location}")
        return source.location
    path = config.resolve_path(source.location)
    return path.as_uri()


class SourceBin:
    """Own an ``nvurisrcbin`` and expose one stable ghost source pad."""

    def __init__(self, runtime: Any, app_config: AppConfig, source: SourceConfig, index: int):
        self.runtime = runtime
        self.config = source
        self.index = index
        self._linked = False
        self._attach_system_timestamp = app_config.pipeline.streammux.attach_system_timestamp
        self._ntp_synced_sources: set[int] = set()
        Gst = runtime.Gst
        self.bin = Gst.Bin.new(f"source-bin-{index:02d}")
        if self.bin is None:
            raise PipelineError(f"无法创建视频源 Bin: {source.camera_id}")
        self.decoder = make_element(Gst, "nvurisrcbin", f"uri-source-{index:02d}")
        self.decoder.set_property("uri", source_uri(app_config, source))
        set_if_supported(self.decoder, "gpu-id", app_config.pipeline.streammux.gpu_id)
        # dGPU/WSL and production L20 both use CUDA device memory. DeepStream's
        # dGPU troubleshooting guidance recommends cudadec-memtype=0, and the
        # metadata probe already performs an explicit device-to-host copy from
        # the NvBufSurface CUDA pointer, so unified memory is neither required
        # nor desirable here.
        set_if_supported(self.decoder, "cudadec-memtype", 0)
        set_if_supported(self.decoder, "num-extra-surfaces", 4)
        set_if_supported(self.decoder, "drop-frame-interval", 0)
        if source.type == "rtsp":
            # Live analytics must stay close to wall-clock time. Keep only a
            # small RTP jitter reserve and allow rtspsrc/nvurisrcbin to discard
            # packets that have already fallen outside that latency budget.
            # NvDCF/continuity absorb short misses; accumulating old frames would
            # instead make the operator watch increasingly stale video.
            rtsp_latency_ms = max(200, int(source.latency_ms))
            set_if_supported(self.decoder, "latency", rtsp_latency_ms)
            # Docker Desktop/WSL cannot reliably receive the UDP RTP ports
            # negotiated by RTSP. Start directly with interleaved RTP-over-TCP
            # instead of waiting for rtspsrc's UDP timeout and fallback.
            set_if_supported(self.decoder, "select-rtp-protocol", 4)
            # DS 9 nvurisrcbin calls this property rtsp-reconnect-interval;
            # rtsp-reconnect-interval-sec is a deepstream-app config key, not
            # a GObject property on this element.
            set_if_supported(self.decoder, "rtsp-reconnect-interval", source.reconnect_interval_sec)
            set_if_supported(self.decoder, "rtsp-reconnect-attempts", -1)
            set_if_supported(self.decoder, "drop-on-latency", True)
            LOGGER.info(
                "RTSP 实时模式: camera_id=%s transport=tcp latency_ms=%d "
                "drop_on_latency=true reconnect_interval_sec=%d",
                source.camera_id,
                rtsp_latency_ms,
                source.reconnect_interval_sec,
            )
        self.decoder.connect("pad-added", self._on_pad_added)
        self.decoder.connect("child-added", self._on_child_added)
        self.bin.add(self.decoder)
        ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        if ghost_pad is None or not self.bin.add_pad(ghost_pad):
            raise PipelineError(f"无法为视频源创建 ghost pad: {source.camera_id}")

    def _on_pad_added(self, _element: Any, pad: Any) -> None:
        if self._linked:
            return
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            return
        structure = caps.get_structure(0)
        media_name = structure.get_name() if structure else ""
        if not media_name.startswith("video/"):
            return
        features = caps.get_features(0)
        if features is not None and not features.contains("memory:NVMM"):
            LOGGER.error(
                "视频源 %s 未协商 NVIDIA NVMM 内存；该编码可能未使用 NVIDIA 解码器",
                self.config.camera_id,
            )
            return
        ghost_pad = self.bin.get_static_pad("src")
        if ghost_pad is None or not ghost_pad.set_target(pad):
            LOGGER.error("视频源 %s 的动态 pad 连接失败", self.config.camera_id)
            return
        self._linked = True
        LOGGER.info("视频源已连接: camera_id=%s uri=%s", self.config.camera_id, pad.name)

    def _on_child_added(self, _parent: Any, child: Any, name: str) -> None:
        lowered = name.lower()
        if "decodebin" in lowered:
            with suppress(TypeError):
                child.connect("child-added", self._on_child_added)
        if "nvv4l2decoder" in lowered:
            # Set the concrete decoder as well as nvurisrcbin so dynamically
            # created children cannot fall back to a different memory mode.
            set_if_supported(child, "cudadec-memtype", 0)
            set_if_supported(child, "enable-max-performance", True)
            set_if_supported(child, "drop-frame-interval", 0)
            set_if_supported(child, "num-extra-surfaces", 4)
            LOGGER.info(
                "[NVDEC] camera_id=%s decoder=%s cudadec_memtype=device",
                self.config.camera_id,
                name,
            )
        factory = child.get_factory() if hasattr(child, "get_factory") else None
        factory_name = factory.get_name().lower() if factory is not None else ""
        if (
            self.config.type == "rtsp"
            and not self._attach_system_timestamp
            and factory_name == "rtspsrc"
        ):
            source_pointer = hash(child)
            if source_pointer in self._ntp_synced_sources:
                return
            configure = getattr(self.runtime.pyds, "configure_source_for_ntp_sync", None)
            if not callable(configure):
                LOGGER.error(
                    "PyDS 缺少 configure_source_for_ntp_sync；RTSP 源 %s 无法启用 RTCP NTP 同步",
                    self.config.camera_id,
                )
                return
            configure(source_pointer)
            self._ntp_synced_sources.add(source_pointer)
            LOGGER.info("RTSP 源已启用 RTCP NTP 同步: camera_id=%s", self.config.camera_id)


def link_source_to_mux(runtime: Any, source_bin: SourceBin, streammux: Any, index: int) -> None:
    Gst = runtime.Gst
    src_pad = source_bin.bin.get_static_pad("src")
    request_name = f"sink_{index}"
    sink_pad = None
    if hasattr(streammux, "request_pad_simple"):
        sink_pad = streammux.request_pad_simple(request_name)
    if sink_pad is None:
        template = streammux.get_pad_template("sink_%u")
        sink_pad = streammux.request_pad(template, request_name, None)
    if src_pad is None or sink_pad is None:
        raise PipelineError(f"无法请求 nvstreammux pad: {request_name}")
    result = src_pad.link(sink_pad)
    if result != Gst.PadLinkReturn.OK:
        raise PipelineError(f"视频源 {source_bin.config.camera_id} 连接 nvstreammux 失败: {result}")