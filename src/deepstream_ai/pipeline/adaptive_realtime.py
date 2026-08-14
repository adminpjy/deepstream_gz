"""Realtime-safe adaptive inference policy for the pinned DeepStream runtime.

DeepStream exposes the primary GIE ``interval`` as a writable Gst property, but
``secondary-reinfer-interval`` is a config-file key rather than a runtime Gst
property in the deployed GstNvInfer. Keep SCRFD/behavior cadence fixed at
pipeline construction and adapt only PeopleNet while the pipeline is PLAYING.
"""

from __future__ import annotations

import logging

from .adaptive import AdaptiveInferenceController, InferenceProfile, _skip_interval

LOGGER = logging.getLogger(__name__)


class RealtimeAdaptiveInferenceController(AdaptiveInferenceController):
    """Preserve person-tracking cadence and never fake unsupported SGIE updates."""

    def __init__(self, config, graph) -> None:
        super().__init__(config, graph)
        self._static_sgie_logged: set[str] = set()

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
                target = profile.person_fps
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
                    "[ADAPTIVE_RATE] component=person target=%.2f actual≈%.2f "
                    "interval=%d reason=%s",
                    target,
                    actual,
                    skip,
                    reason,
                )
                continue

            # SGIE cadence was materialized into its nvinfer config before
            # PLAYING. Do not call the non-existent Gst property and do not log
            # a dynamic success that never happened.
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