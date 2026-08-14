"""Pipeline lifecycle, bus diagnostics, and graceful shutdown."""

from __future__ import annotations

import logging
import os
import re
import signal
from contextlib import suppress
from pathlib import Path
from threading import Event
from typing import Any

from deepstream_ai.config import AppConfig
from deepstream_ai.errors import PipelineError
from deepstream_ai.stream_epoch import bump_stream_generation

from .adaptive_realtime import RealtimeAdaptiveInferenceController
from .builder import PipelineGraph
from .shadow_tracking import ShadowTrackRegistry

LOGGER = logging.getLogger(__name__)
_RTSP_SOURCE_NAME = re.compile(r"uri-source-(\d+)$")


class PipelineRunner:
    def __init__(
        self,
        runtime: Any,
        config: AppConfig,
        graph: PipelineGraph,
        *,
        on_started: Any | None = None,
    ):
        self.runtime = runtime
        self.config = config
        self.graph = graph
        self.loop = runtime.GLib.MainLoop()
        self._failed: PipelineError | None = None
        self._stopping = Event()
        self._previous_handlers: dict[int, Any] = {}
        self._on_started = on_started
        self._signal_stop_count = 0
        self._adaptive: RealtimeAdaptiveInferenceController | None = None
        self._shadow: ShadowTrackRegistry | None = None

    def run(self) -> None:
        Gst = self.runtime.Gst
        bus = self.graph.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)
        self._install_signals()
        health_path = Path(self.config.runtime.health_file)
        health_path.unlink(missing_ok=True)
        try:
            # Read NvDCF shadow-list metadata directly at tracker output. This
            # probe stores only bbox/ID data for preview and never enters the
            # business FramePacket path used by SCRFD/AdaFace/evidence.
            tracker = self.graph.pipeline.get_by_name("person-tracker")
            if tracker is not None:
                tracker_pad = tracker.get_static_pad("src")
                if tracker_pad is not None:
                    self._shadow = ShadowTrackRegistry(
                        self.runtime,
                        self.config,
                        self.graph.metadata_probe.consumer,
                    )
                    tracker_pad.add_probe(Gst.PadProbeType.BUFFER, self._shadow.callback, None)
                    LOGGER.info(
                        "[TRACK_SHADOW] preview bridge enabled max_age=%.2fs",
                        self._shadow.config.display_max_age_sec,
                    )

            state_result = self.graph.pipeline.set_state(Gst.State.PLAYING)
            if state_result == Gst.StateChangeReturn.FAILURE:
                raise PipelineError("Pipeline 无法进入 PLAYING 状态")
            wait_result, current_state, pending_state = self.graph.pipeline.get_state(
                self.config.runtime.startup_timeout_sec * Gst.SECOND
            )
            if wait_result == Gst.StateChangeReturn.FAILURE or current_state != Gst.State.PLAYING:
                raise PipelineError(
                    "Pipeline 未在启动超时内进入 PLAYING 状态: "
                    f"result={wait_result.value_nick} current={current_state.value_nick} "
                    f"pending={pending_state.value_nick}"
                )
            health_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_health = health_path.with_suffix(health_path.suffix + ".tmp")
            temporary_health.write_text(str(os.getpid()), encoding="ascii")
            temporary_health.replace(health_path)
            LOGGER.info("DeepStream Pipeline 已启动，sources=%d", len(self.config.enabled_sources))
            # Start only after PLAYING so negotiated source caps contain the real
            # camera/file frame rate. Runtime property changes happen on GLib's
            # main-loop thread, not in the streaming probe thread.
            self._adaptive = RealtimeAdaptiveInferenceController(self.config, self.graph)
            self._adaptive.start(self.runtime.GLib)
            if self._on_started is not None:
                self._on_started()
            self.loop.run()
            if self._failed is not None:
                raise self._failed
        finally:
            if self._adaptive is not None:
                self._adaptive.stop()
                self._adaptive = None
            health_path.unlink(missing_ok=True)
            health_path.with_suffix(health_path.suffix + ".tmp").unlink(missing_ok=True)
            self.graph.pipeline.set_state(Gst.State.NULL)
            try:
                self.graph.metadata_probe.log_performance()
            except Exception:
                LOGGER.exception("输出 Probe 性能统计失败")
            if self._shadow is not None:
                stats = self._shadow.stats()
                LOGGER.info(
                    "[TRACK_SHADOW_SUMMARY] frames=%d objects=%d hidden_by_age=%d errors=%d",
                    stats.frames,
                    stats.objects,
                    stats.hidden_by_age,
                    stats.errors,
                )
                self._shadow.close()
                self._shadow = None
            if self.graph.pretracker_guard is not None:
                stats = self.graph.pretracker_guard.stats()
                LOGGER.info(
                    "[PRETRACKER_GUARD_SUMMARY] frames=%d verified_people=%d "
                    "suppressed=%d errors=%d",
                    stats.frames,
                    stats.verified_people,
                    stats.suppressed,
                    stats.errors,
                )
            bus.remove_signal_watch()
            self._restore_signals()
            LOGGER.info("DeepStream Pipeline 已停止")

    def stop(self, send_eos: bool = True, *, force: bool = False) -> None:
        if self._stopping.is_set():
            if force:
                LOGGER.warning("再次收到停止信号，立即强制停止主循环")
                self.loop.quit()
            else:
                LOGGER.debug("忽略重复的程序化停止请求")
            return
        self._stopping.set()
        LOGGER.info("收到停止请求")

        live_rtsp = any(
            getattr(source, "type", None) == "rtsp"
            for source in getattr(self.config, "enabled_sources", ())
        )
        if send_eos and live_rtsp:
            # Sending EOS through nvurisrcbin can leave the live source waiting
            # long enough for its reconnect watchdog to create a fresh decoder
            # and reset NvDCF IDs. A user/idle stop on RTSP should instead leave
            # PLAYING immediately; pipeline NULL in finally releases resources.
            send_eos = False
            LOGGER.info("[RTSP_STOP] live source: skip EOS and stop main loop immediately")

        if send_eos:
            accepted = bool(self.graph.pipeline.send_event(self.runtime.Gst.Event.new_eos()))
            if not accepted:
                LOGGER.warning("Pipeline 拒绝 EOS 事件，立即停止主循环")
                self.loop.quit()
                return
            self.runtime.GLib.timeout_add_seconds(
                self.config.runtime.shutdown_timeout_sec,
                self._force_stop_after_timeout,
            )
        else:
            self.loop.quit()

    def _force_stop_after_timeout(self) -> bool:
        if self.loop.is_running():
            LOGGER.warning(
                "等待 EOS 超过 %s 秒，强制停止主循环",
                self.config.runtime.shutdown_timeout_sec,
            )
            self.loop.quit()
        return False

    def _on_message(self, _bus: Any, message: Any) -> None:
        Gst = self.runtime.Gst
        if message.type == Gst.MessageType.EOS:
            LOGGER.info("所有文件源处理完成 (EOS)")
            self.loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            source = message.src.get_name() if message.src is not None else "unknown"
            self._failed = PipelineError(f"GStreamer 错误 source={source}: {error}; debug={debug}")
            LOGGER.error("%s", self._failed)
            self.loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            source = message.src.get_name() if message.src is not None else "unknown"
            LOGGER.warning("GStreamer 警告 source=%s: %s; debug=%s", source, warning, debug)
            if not self._stopping.is_set() and "Trying reconnection" in str(warning):
                self._mark_rtsp_reconnect(source)
        elif message.type == Gst.MessageType.STATE_CHANGED and message.src == self.graph.pipeline:
            old, new, pending = message.parse_state_changed()
            LOGGER.debug(
                "Pipeline 状态: %s -> %s (pending=%s)",
                old.value_nick,
                new.value_nick,
                pending.value_nick,
            )

    def _mark_rtsp_reconnect(self, source_name: str) -> None:
        match = _RTSP_SOURCE_NAME.fullmatch(source_name)
        if match is None:
            return
        index = int(match.group(1))
        sources = getattr(self.config, "enabled_sources", ())
        if not 0 <= index < len(sources):
            return
        source = sources[index]
        if getattr(source, "type", None) != "rtsp":
            return
        bump_stream_generation(
            source.camera_id,
            reason=f"rtsp_reconnect:{source_name}",
        )

    def _install_signals(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._on_stop_signal)
            except (ValueError, OSError):
                LOGGER.debug("当前线程无法注册信号 %s", signum)

    def _on_stop_signal(self, _signum: int, _frame: Any) -> None:
        self._signal_stop_count += 1
        self.stop(force=self._signal_stop_count > 1)

    def _restore_signals(self) -> None:
        for signum, handler in self._previous_handlers.items():
            with suppress(ValueError, OSError):
                signal.signal(signum, handler)
