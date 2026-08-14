"""Realtime-safe adaptive inference policy for the pinned DeepStream runtime.

DeepStream exposes the primary GIE ``interval`` as a writable Gst property, but
``secondary-reinfer-interval`` is a config-file key rather than a runtime Gst
property in the deployed GstNvInfer. Keep SCRFD/behavior cadence fixed at
pipeline construction and adapt only PeopleNet while the pipeline is PLAYING.

GPU utilization by itself is not treated as overload. TensorRT is expected to
use the GPU aggressively; person cadence is never reduced below the configured
tracking-safe ``inference.person_fps`` floor. Queue pressure may still move the
capacity profile so service admission can react, but it must not trade away
person tracking continuity.
"""

from __future__ import annotations

import logging

from .adaptive import AdaptiveInferenceController, GpuLoad, InferenceProfile, _skip_interval

LOGGER = logging.getLogger(__name__)


class RealtimeAdaptiveInferenceController(AdaptiveInferenceController):
    """Preserve the configured PeopleNet tracking floor under realtime load."""

    def __init__(self, config, graph) -> None:
        super().__init__(config, graph)
        self._static_sgie_logged: set[str] = set()

    def _is_critical(
        self,
        gpu: GpuLoad,
        queue_ratio: float | None,
        drops_delta: int,
    ) -> bool:
        if drops_delta > 0:
            return True
        if queue_ratio is not None and queue_ratio >= self.config.queue_critical_ratio:
            return True
        if (
            queue_ratio is not None
            and queue_ratio >= self.config.queue_high_ratio
            and gpu.gpu_util is not None
            and gpu.gpu_util >= self.config.gpu_critical
        ):
            return True
        return False

    def _sustained_high(self, now: float) -> bool:
        values = [
            item
            for item in self._history
            if now - item.timestamp <= self.config.decision_window_sec
        ]
        if not values or now - values[0].timestamp < self.config.decision_window_sec * 0.8:
            return False
        if any(item.queue_drops_delta > 0 for item in values):
            return True
        queue_values = [item.queue_ratio for item in values if item.queue_ratio is not None]
        if not queue_values:
            return False
        average_queue = sum(queue_values) / len(queue_values)
        if average_queue >= self.config.queue_high_ratio:
            return True
        gpu_values = [item.gpu.gpu_util for item in values if item.gpu.gpu_util is not None]
        return bool(
            average_queue >= self.config.queue_low_ratio
            and gpu_values
            and sum(gpu_values) / len(gpu_values) >= self.config.gpu_high
        )

    def _apply_profile(
        self,
        profile: InferenceProfile,
        source_fps: float,
        *,
        reason: str,
    ) -> None:
        elements = getattr(self.graph, "inference_elements", {}) or {}
        for key, element in elements.items():
            if key == "person":
                configured_floor = float(self.app_config.inference.person_fps)
                target = max(float(profile.person_fps), configured_floor)
                skip = _skip_interval(source_fps, target)
                try:
                    element.set_property("interval", int(skip))
                except Exception:
                    LOGGER.exception(
                        "[ADAPTIVE_RATE_ERROR] component=%s property=interval value=%s",
                        getattr(element, "name", "person-detector"),
                        skip,
                    )
                    continue
                actual = source_fps / (skip + 1)
                LOGGER.info(
                    "[ADAPTIVE_RATE] component=person target=%.2f floor=%.2f actual≈%.2f "
                    "interval=%d reason=%s",
                    target,
                    configured_floor,
                    actual,
                    skip,
                    reason,
                )
                continue

            if key == "face":
                target = float(self.app_config.inference.face_fps)
            elif key.startswith("behavior:"):
                target = float(self.app_config.inference.behavior_fps)
            else:
                continue
            if key in self._static_sgie_logged:
                continue
            skip = _skip_interval(source_fps, target)
            reinfer = 0 if skip == 0 else skip + 1
            actual = source_fps if reinfer == 0 else source_fps / reinfer
            LOGGER.info(
                "[ADAPTIVE_RATE_STATIC] component=%s target=%.2f actual≈%.2f "
                "reinfer=%d reason=config_startup runtime_update=unsupported",
                key,
                target,
                actual,
                reinfer,
            )
            self._static_sgie_logged.add(key)

        self._last_source_fps = source_fps


__all__ = ["RealtimeAdaptiveInferenceController"]
