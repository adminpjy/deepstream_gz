"""Adaptive inference-rate control for live DeepStream pipelines.

The controller deliberately keeps decoding and NvDCF tracking continuous.  It
only changes detector/classifier cadence at runtime, based on negotiated source
FPS plus shared GPU/decoder pressure and business-queue backpressure.

Bitrate is intentionally not a scheduling input: two streams with the same
resolution/FPS can have very different compressed bitrates while costing nearly
the same TensorRT work.  Source FPS/resolution are read from negotiated caps;
bitrate can be added later as an observability-only metric.
"""

from __future__ import annotations

import logging
import math
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceProfile:
    name: str
    person_fps: float
    face_fps: float
    behavior_fps: float


@dataclass(frozen=True, slots=True)
class AdaptiveInferenceConfig:
    enabled: bool = True
    sample_interval_sec: float = 2.0
    decision_window_sec: float = 10.0
    upgrade_stable_sec: float = 20.0
    change_cooldown_sec: float = 10.0
    gpu_low: float = 55.0
    gpu_high: float = 85.0
    gpu_critical: float = 92.0
    nvdec_low: float = 60.0
    nvdec_high: float = 85.0
    nvdec_critical: float = 95.0
    queue_low_ratio: float = 0.25
    queue_high_ratio: float = 0.60
    queue_critical_ratio: float = 0.85
    profiles: tuple[InferenceProfile, ...] = (
        InferenceProfile("boost", 8.0, 8.0, 2.0),
        InferenceProfile("normal", 5.0, 5.0, 1.0),
        InferenceProfile("balanced", 5.0, 4.0, 0.5),
        InferenceProfile("protect", 4.0, 3.0, 0.25),
        InferenceProfile("emergency", 3.0, 3.0, 0.2),
    )
    initial_profile: str = "normal"
    reject_new_tasks_at_floor: bool = True

    @classmethod
    def from_file(cls, path: str | Path) -> "AdaptiveInferenceConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        section = raw.get("adaptive_inference") or {}
        if not isinstance(section, dict):
            return cls(enabled=False)
        monitor = section.get("monitor") or {}
        load = section.get("load") or {}
        overload = section.get("overload") or {}
        raw_profiles = section.get("profiles") or {}
        defaults = {profile.name: profile for profile in cls().profiles}
        profiles: list[InferenceProfile] = []
        for name in ("boost", "normal", "balanced", "protect", "emergency"):
            base = defaults[name]
            item = raw_profiles.get(name) or {}
            profiles.append(
                InferenceProfile(
                    name=name,
                    person_fps=float(item.get("person_fps", base.person_fps)),
                    face_fps=float(item.get("face_fps", base.face_fps)),
                    behavior_fps=float(item.get("behavior_fps", base.behavior_fps)),
                )
            )
        result = cls(
            enabled=bool(section.get("enabled", True)),
            sample_interval_sec=float(monitor.get("sample_interval_sec", 2.0)),
            decision_window_sec=float(monitor.get("decision_window_sec", 10.0)),
            upgrade_stable_sec=float(monitor.get("upgrade_stable_sec", 20.0)),
            change_cooldown_sec=float(monitor.get("change_cooldown_sec", 10.0)),
            gpu_low=float(load.get("gpu_low", 55.0)),
            gpu_high=float(load.get("gpu_high", 85.0)),
            gpu_critical=float(load.get("gpu_critical", 92.0)),
            nvdec_low=float(load.get("nvdec_low", 60.0)),
            nvdec_high=float(load.get("nvdec_high", 85.0)),
            nvdec_critical=float(load.get("nvdec_critical", 95.0)),
            queue_low_ratio=float(load.get("queue_low_ratio", 0.25)),
            queue_high_ratio=float(load.get("queue_high_ratio", 0.60)),
            queue_critical_ratio=float(load.get("queue_critical_ratio", 0.85)),
            profiles=tuple(profiles),
            initial_profile=str(section.get("initial_profile", "normal")).lower(),
            reject_new_tasks_at_floor=bool(
                overload.get("reject_new_tasks_at_floor", True)
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.sample_interval_sec <= 0 or self.decision_window_sec <= 0:
            raise ValueError("adaptive monitor intervals must be positive")
        if self.upgrade_stable_sec <= 0 or self.change_cooldown_sec < 0:
            raise ValueError("adaptive hysteresis intervals are invalid")
        for low, high, critical, label in (
            (self.gpu_low, self.gpu_high, self.gpu_critical, "gpu"),
            (self.nvdec_low, self.nvdec_high, self.nvdec_critical, "nvdec"),
        ):
            if not 0 <= low < high < critical <= 100:
                raise ValueError(f"adaptive {label} thresholds must satisfy low < high < critical")
        if not 0 <= self.queue_low_ratio < self.queue_high_ratio < self.queue_critical_ratio <= 1:
            raise ValueError("adaptive queue thresholds must satisfy low < high < critical")
        if self.initial_profile not in {profile.name for profile in self.profiles}:
            raise ValueError("adaptive initial_profile is not configured")
        for profile in self.profiles:
            if min(profile.person_fps, profile.face_fps, profile.behavior_fps) <= 0:
                raise ValueError(f"adaptive profile {profile.name} FPS values must be positive")


@dataclass(frozen=True, slots=True)
class GpuLoad:
    gpu_util: float | None
    memory_util: float | None
    memory_used_mb: float | None
    memory_total_mb: float | None
    decoder_util: float | None


@dataclass(frozen=True, slots=True)
class SourceProfile:
    camera_id: str
    fps: float
    width: int | None
    height: int | None
    negotiated: bool


@dataclass(frozen=True, slots=True)
class LoadSample:
    timestamp: float
    gpu: GpuLoad
    queue_ratio: float | None
    queue_drops_delta: int


class NvidiaSmiMonitor:
    """Read shared GPU pressure without adding a Python NVML dependency."""

    def __init__(self, gpu_id: int = 0) -> None:
        self.gpu_id = int(gpu_id)
        self._warned = False

    def sample(self) -> GpuLoad:
        base = self._query(
            "utilization.gpu,utilization.memory,memory.used,memory.total"
        )
        if base is None or len(base) < 4:
            if not self._warned:
                LOGGER.warning(
                    "[ADAPTIVE_GPU_UNAVAILABLE] nvidia-smi metrics unavailable; "
                    "queue backpressure remains active but automatic boost is disabled"
                )
                self._warned = True
            return GpuLoad(None, None, None, None, None)
        decoder_values = self._query("utilization.decoder")
        return GpuLoad(
            gpu_util=base[0],
            memory_util=base[1],
            memory_used_mb=base[2],
            memory_total_mb=base[3],
            decoder_util=decoder_values[0] if decoder_values else None,
        )

    def _query(self, fields: str) -> list[float] | None:
        command = [
            "nvidia-smi",
            f"--id={self.gpu_id}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        line = result.stdout.strip().splitlines()
        if not line:
            return None
        try:
            return [float(item.strip()) for item in line[0].split(",")]
        except ValueError:
            return None


class AdaptiveInferenceController:
    """Dynamically apply a bounded inference profile with hysteresis."""

    def __init__(self, config: Any, graph: Any) -> None:
        self.app_config = config
        self.graph = graph
        self.config = AdaptiveInferenceConfig.from_file(config.config_path)
        self._gpu = NvidiaSmiMonitor(config.pipeline.streammux.gpu_id)
        self._history: deque[LoadSample] = deque()
        self._profile_index = next(
            index
            for index, profile in enumerate(self.config.profiles)
            if profile.name == self.config.initial_profile
        )
        self._last_change = 0.0
        self._last_source_fps: float | None = None
        self._last_queue_drops = 0
        self._timer_id: int | None = None
        self._glib: Any | None = None
        self._overload_logged = False
        self._source_logged: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def current_profile(self) -> InferenceProfile:
        return self.config.profiles[self._profile_index]

    def start(self, glib: Any) -> None:
        if not self.enabled:
            LOGGER.info("[ADAPTIVE] disabled")
            return
        self._glib = glib
        self.tick(force_apply=True)
        interval_ms = max(250, int(round(self.config.sample_interval_sec * 1000)))
        self._timer_id = int(glib.timeout_add(interval_ms, self._on_timer))
        LOGGER.info(
            "[ADAPTIVE] enabled profile=%s sample=%.1fs window=%.1fs",
            self.current_profile.name,
            self.config.sample_interval_sec,
            self.config.decision_window_sec,
        )

    def stop(self) -> None:
        if self._timer_id is not None and self._glib is not None:
            try:
                self._glib.source_remove(self._timer_id)
            except Exception:
                LOGGER.debug("adaptive timer already removed", exc_info=True)
        self._timer_id = None

    def _on_timer(self) -> bool:
        try:
            self.tick()
        except Exception:
            LOGGER.exception("[ADAPTIVE_ERROR] inference-rate controller tick failed")
        return True

    def tick(self, *, force_apply: bool = False) -> None:
        now = time.monotonic()
        source_profiles = self._source_profiles()
        source_fps = max(
            (profile.fps for profile in source_profiles),
            default=max(source.nominal_fps for source in self.app_config.enabled_sources),
        )
        for profile in source_profiles:
            if profile.camera_id not in self._source_logged and profile.negotiated:
                LOGGER.info(
                    "[SOURCE_PROFILE] camera=%s fps=%.3f resolution=%sx%s source=negotiated_caps",
                    profile.camera_id,
                    profile.fps,
                    profile.width or "?",
                    profile.height or "?",
                )
                self._source_logged.add(profile.camera_id)

        gpu = self._gpu.sample()
        queue_ratio = _queue_ratio(self.graph.metadata_probe.consumer)
        perf = self.graph.metadata_probe.performance()
        drops_delta = max(0, perf.queue_drops - self._last_queue_drops)
        self._last_queue_drops = perf.queue_drops
        self._history.append(LoadSample(now, gpu, queue_ratio, drops_delta))
        horizon = max(self.config.decision_window_sec, self.config.upgrade_stable_sec) + 5
        while self._history and now - self._history[0].timestamp > horizon:
            self._history.popleft()

        source_changed = (
            self._last_source_fps is None
            or abs(source_fps - self._last_source_fps) / max(source_fps, 1.0) >= 0.05
        )
        if force_apply or source_changed:
            self._apply_profile(self.current_profile, source_fps, reason="source_profile")
            self._last_source_fps = source_fps

        if self._is_critical(gpu, queue_ratio, drops_delta):
            if self._profile_index < len(self.config.profiles) - 1:
                self._change_profile(self._profile_index + 1, source_fps, "critical_load", now)
            else:
                if not self._overload_logged:
                    LOGGER.error(
                        "[SYSTEM_CAPACITY_EXCEEDED] profile=emergency gpu=%s nvdec=%s "
                        "queue=%s drops_delta=%d reject_new_tasks=%s",
                        _fmt(gpu.gpu_util),
                        _fmt(gpu.decoder_util),
                        _fmt_ratio(queue_ratio),
                        drops_delta,
                        self.config.reject_new_tasks_at_floor,
                    )
                    self._overload_logged = True
            self._log_sample(gpu, queue_ratio, source_fps, drops_delta)
            return

        if now - self._last_change >= self.config.change_cooldown_sec:
            if self._sustained_high(now):
                if self._profile_index < len(self.config.profiles) - 1:
                    self._change_profile(self._profile_index + 1, source_fps, "sustained_high", now)
            elif self._sustained_low(now):
                if self._profile_index > 0:
                    self._change_profile(self._profile_index - 1, source_fps, "stable_headroom", now)
        self._overload_logged = False
        self._log_sample(gpu, queue_ratio, source_fps, drops_delta)

    def _is_critical(
        self,
        gpu: GpuLoad,
        queue_ratio: float | None,
        drops_delta: int,
    ) -> bool:
        return bool(
            drops_delta > 0
            or (gpu.gpu_util is not None and gpu.gpu_util >= self.config.gpu_critical)
            or (
                gpu.decoder_util is not None
                and gpu.decoder_util >= self.config.nvdec_critical
            )
            or (
                queue_ratio is not None
                and queue_ratio >= self.config.queue_critical_ratio
            )
        )

    def _sustained_high(self, now: float) -> bool:
        values = [
            item
            for item in self._history
            if now - item.timestamp <= self.config.decision_window_sec
        ]
        if not values or now - values[0].timestamp < self.config.decision_window_sec * 0.8:
            return False
        gpu_values = [item.gpu.gpu_util for item in values if item.gpu.gpu_util is not None]
        decoder_values = [
            item.gpu.decoder_util for item in values if item.gpu.decoder_util is not None
        ]
        queue_values = [item.queue_ratio for item in values if item.queue_ratio is not None]
        return bool(
            any(item.queue_drops_delta > 0 for item in values)
            or (gpu_values and sum(gpu_values) / len(gpu_values) >= self.config.gpu_high)
            or (
                decoder_values
                and sum(decoder_values) / len(decoder_values) >= self.config.nvdec_high
            )
            or (
                queue_values
                and sum(queue_values) / len(queue_values) >= self.config.queue_high_ratio
            )
        )

    def _sustained_low(self, now: float) -> bool:
        # Never boost solely because queue pressure is low; we need real GPU
        # telemetry to prove that compute headroom exists.
        values = [
            item
            for item in self._history
            if now - item.timestamp <= self.config.upgrade_stable_sec
        ]
        if not values or now - values[0].timestamp < self.config.upgrade_stable_sec * 0.8:
            return False
        if any(item.gpu.gpu_util is None for item in values):
            return False
        for item in values:
            if item.queue_drops_delta > 0:
                return False
            if item.gpu.gpu_util is not None and item.gpu.gpu_util >= self.config.gpu_low:
                return False
            if (
                item.gpu.decoder_util is not None
                and item.gpu.decoder_util >= self.config.nvdec_low
            ):
                return False
            if (
                item.queue_ratio is not None
                and item.queue_ratio >= self.config.queue_low_ratio
            ):
                return False
        return True

    def _change_profile(
        self,
        target_index: int,
        source_fps: float,
        reason: str,
        now: float,
    ) -> None:
        old = self.current_profile
        self._profile_index = max(0, min(target_index, len(self.config.profiles) - 1))
        new = self.current_profile
        if new.name == old.name:
            return
        self._last_change = now
        self._apply_profile(new, source_fps, reason=reason)
        LOGGER.warning(
            "[ADAPTIVE_PROFILE] %s -> %s reason=%s source_fps=%.3f",
            old.name,
            new.name,
            reason,
            source_fps,
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
                target = profile.person_fps
                skip = _skip_interval(source_fps, target)
                _set_property(element, "interval", skip)
                actual = source_fps / (skip + 1)
                LOGGER.info(
                    "[ADAPTIVE_RATE] component=person target=%.2f actual≈%.2f interval=%d reason=%s",
                    target,
                    actual,
                    skip,
                    reason,
                )
            elif key == "face":
                target = profile.face_fps
                skip = _skip_interval(source_fps, target)
                reinfer = 0 if skip == 0 else skip + 1
                _set_property(element, "secondary-reinfer-interval", reinfer)
                actual = source_fps if reinfer == 0 else source_fps / reinfer
                LOGGER.info(
                    "[ADAPTIVE_RATE] component=face target=%.2f actual≈%.2f reinfer=%d reason=%s",
                    target,
                    actual,
                    reinfer,
                    reason,
                )
            elif key.startswith("behavior:"):
                target = profile.behavior_fps
                skip = _skip_interval(source_fps, target)
                reinfer = 0 if skip == 0 else skip + 1
                _set_property(element, "secondary-reinfer-interval", reinfer)
        self._last_source_fps = source_fps

    def _source_profiles(self) -> tuple[SourceProfile, ...]:
        result: list[SourceProfile] = []
        for source_bin in self.graph.source_bins:
            fallback = float(source_bin.config.nominal_fps)
            pad = source_bin.bin.get_static_pad("src")
            caps = pad.get_current_caps() if pad is not None else None
            if caps is None and pad is not None:
                try:
                    caps = pad.query_caps(None)
                except Exception:
                    caps = None
            width: int | None = None
            height: int | None = None
            fps: float | None = None
            if caps is not None and caps.get_size() > 0:
                structure = caps.get_structure(0)
                if structure is not None:
                    try:
                        width = int(structure.get_value("width"))
                        height = int(structure.get_value("height"))
                    except Exception:
                        width = height = None
                    try:
                        fps = _fraction_to_float(structure.get_value("framerate"))
                    except Exception:
                        fps = None
            negotiated = fps is not None and fps > 0
            result.append(
                SourceProfile(
                    camera_id=source_bin.config.camera_id,
                    fps=fps if negotiated else fallback,
                    width=width,
                    height=height,
                    negotiated=negotiated,
                )
            )
        return tuple(result)

    def _log_sample(
        self,
        gpu: GpuLoad,
        queue_ratio: float | None,
        source_fps: float,
        drops_delta: int,
    ) -> None:
        LOGGER.info(
            "[ADAPTIVE_LOAD] profile=%s source_fps=%.2f gpu=%s mem=%s nvdec=%s "
            "queue=%s drops_delta=%d",
            self.current_profile.name,
            source_fps,
            _fmt(gpu.gpu_util),
            _fmt(gpu.memory_util),
            _fmt(gpu.decoder_util),
            _fmt_ratio(queue_ratio),
            drops_delta,
        )


def _skip_interval(source_fps: float, target_fps: float) -> int:
    if source_fps <= 0 or target_fps <= 0:
        return 0
    return max(0, round(source_fps / min(source_fps, target_fps)) - 1)


def _set_property(element: Any, name: str, value: int) -> None:
    try:
        element.set_property(name, int(value))
    except Exception:
        LOGGER.exception(
            "[ADAPTIVE_RATE_ERROR] component=%s property=%s value=%s",
            getattr(element, "name", "unknown"),
            name,
            value,
        )


def _fraction_to_float(value: Any) -> float | None:
    if value is None:
        return None
    numerator = getattr(value, "numerator", getattr(value, "num", None))
    denominator = getattr(value, "denominator", getattr(value, "denom", None))
    if numerator is not None and denominator:
        return float(numerator) / float(denominator)
    text = str(value)
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            denominator_value = float(right)
            return float(left) / denominator_value if denominator_value else None
        except ValueError:
            return None
    try:
        numeric = float(text)
        return numeric if math.isfinite(numeric) and numeric > 0 else None
    except ValueError:
        return None


def _queue_ratio(consumer: Any) -> float | None:
    current = consumer
    visited: set[int] = set()
    for _ in range(6):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        provider = getattr(current, "queue_metrics", None)
        if callable(provider):
            try:
                metrics = provider()
                ratio = metrics.get("ratio") if isinstance(metrics, dict) else None
                if ratio is not None:
                    return max(0.0, min(1.0, float(ratio)))
            except Exception:
                LOGGER.debug("queue_metrics provider failed", exc_info=True)
        queue = getattr(current, "_queue", None)
        if queue is not None:
            maxsize = int(getattr(queue, "maxsize", 0) or 0)
            if maxsize > 0:
                try:
                    return max(0.0, min(1.0, queue.qsize() / maxsize))
                except Exception:
                    pass
        current = getattr(current, "delegate", None)
    return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


__all__ = [
    "AdaptiveInferenceConfig",
    "AdaptiveInferenceController",
    "GpuLoad",
    "InferenceProfile",
    "NvidiaSmiMonitor",
    "SourceProfile",
]
