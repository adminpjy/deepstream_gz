"""Pipeline lifecycle, bus diagnostics, and graceful shutdown."""

from __future__ import annotations

import logging
import os
import signal
from contextlib import suppress
from pathlib import Path
from threading import Event
from typing import Any

from deepstream_ai.config import AppConfig
from deepstream_ai.errors import PipelineError

from .builder import PipelineGraph

LOGGER = logging.getLogger(__name__)


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

    def run(self) -> None:
        Gst = self.runtime.Gst
        bus = self.graph.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)
        self._install_signals()
        health_path = Path(self.config.runtime.health_file)
        health_path.unlink(missing_ok=True)
        try:
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
            if self._on_started is not None:
                self._on_started()
            self.loop.run()
            if self._failed is not None:
                raise self._failed
        finally:
            health_path.unlink(missing_ok=True)
            health_path.with_suffix(health_path.suffix + ".tmp").unlink(missing_ok=True)
            self.graph.pipeline.set_state(Gst.State.NULL)
            try:
                self.graph.metadata_probe.log_performance()
            except Exception:
                LOGGER.exception("输出 Probe 性能统计失败")
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
        elif message.type == Gst.MessageType.STATE_CHANGED and message.src == self.graph.pipeline:
            old, new, pending = message.parse_state_changed()
            LOGGER.debug(
                "Pipeline 状态: %s -> %s (pending=%s)",
                old.value_nick,
                new.value_nick,
                pending.value_nick,
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
