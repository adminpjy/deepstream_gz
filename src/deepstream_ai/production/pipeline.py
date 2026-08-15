"""Warm dynamic DeepStream pipeline for production RTSP sessions.

The tuned person/face chain is reused from :mod:`deepstream_ai.pipeline.builder`.
Only source lifecycle and optional behavior admission are added here. Each GPU
worker is isolated with CUDA_VISIBLE_DEVICES, therefore the validated DeepStream
configuration continues to use logical gpu-id=0 unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

from deepstream_ai.config import AppConfig, BehaviorModelConfig, InferComponentConfig, SourceConfig
from deepstream_ai.errors import PipelineError
from deepstream_ai.pipeline.adaptive import NvidiaSmiMonitor
from deepstream_ai.pipeline.builder import DeepStreamPipelineBuilder, PipelineGraph
from deepstream_ai.pipeline.elements import add_many, link_many, make_element, set_if_supported
from deepstream_ai.pipeline.nvinfer_config import materialize_nvinfer_config
from deepstream_ai.pipeline.peoplenet_pretracker_guard import (
    PeopleNetPretrackerGuard,
    PeopleNetPretrackerGuardConfig,
)
from deepstream_ai.pipeline.runner import PipelineRunner
from deepstream_ai.pipeline.source import SourceBin
from deepstream_ai.preflight import inspect_nvinfer_config
from deepstream_ai.production.contracts import FeatureSet
from deepstream_ai.production.feature_gate import BehaviorInferenceGate, FeatureRegistry
from deepstream_ai.stream_epoch import bump_stream_generation

LOGGER = logging.getLogger(__name__)
_NVBUF_MEM_CUDA_DEVICE = 2


def build_warm_config(
    base: AppConfig,
    *,
    capacity: int,
    worker_root: str | Path,
    enabled_behavior_names: tuple[str, ...],
) -> AppConfig:
    """Create a worker config without altering the validated core model contract."""

    if capacity < 1:
        raise ValueError("capacity must be positive")
    worker_root = Path(worker_root).resolve()
    worker_root.mkdir(parents=True, exist_ok=True)
    # Dummy RTSP sources exist only so existing batch-size/rate helpers retain
    # their tested semantics. WarmDynamicPipelineBuilder intentionally does not
    # instantiate these sources; real RTSP bins are added at runtime.
    sources = tuple(
        SourceConfig(
            camera_id=f"warm-slot-{index:03d}",
            type="rtsp",
            url=f"rtsp://127.0.0.1:9/warm-slot-{index:03d}",
            enabled=True,
            nominal_fps=30.0,
            latency_ms=200,
            reconnect_interval_sec=10,
        )
        for index in range(capacity)
    )
    enabled_names = frozenset(enabled_behavior_names)
    behaviors: list[BehaviorModelConfig] = []
    for model in base.behavior:
        behaviors.append(replace(model, enabled=model.name in enabled_names))
    output = replace(
        base.output,
        enabled=False,
        path=str(worker_root / "discard.mp4"),
        events_enabled=False,
        events_path=str(worker_root / "core-events.jsonl"),
        snapshot=replace(base.output.snapshot, root=str(worker_root / "snapshot")),
    )
    runtime = replace(
        base.runtime,
        health_file=str(worker_root / "pipeline.ready"),
    )
    result = replace(
        base,
        sources=sources,
        behavior=tuple(behaviors),
        output=output,
        runtime=runtime,
    )
    component_ids = [result.pipeline.person.unique_id]
    if result.pipeline.face.enabled:
        component_ids.append(result.pipeline.face.unique_id)
    component_ids.extend(model.unique_id for model in result.behavior if model.enabled)
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("warm worker nvinfer unique_id must be unique")
    return result


class WarmDynamicPipelineBuilder(DeepStreamPipelineBuilder):
    """Build the existing inference chain once, with no live source initially."""

    def __init__(
        self,
        runtime: Any,
        config: AppConfig,
        consumer: Any,
        feature_registry: FeatureRegistry,
    ) -> None:
        super().__init__(runtime, config, consumer)
        self.feature_registry = feature_registry
        self.behavior_gates: dict[str, BehaviorInferenceGate] = {}

    def _primary_engine_fingerprint(self, component: InferComponentConfig) -> str:
        source_config = self.config.resolve_path(component.config_file)
        digest = hashlib.sha256(source_config.read_bytes())
        report = inspect_nvinfer_config(self.config, component.config_file)
        for model_path in report.source_models:
            try:
                stat = model_path.stat()
                digest.update(str(model_path).encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
            except OSError:
                digest.update(str(model_path).encode("utf-8"))
        return digest.hexdigest()[:12]

    def _infer_element(
        self,
        name: str,
        config: InferComponentConfig,
        target_fps: float,
        *,
        primary: bool,
    ) -> Any:
        """Reuse the existing nvinfer setup, isolating only the multi-stream PGIE engine.

        The deployed PeopleNet engine is a tuned batch-1 asset used by the
        existing task pipeline. A multi-stream worker needs batch=capacity. We
        must never let Gst-nvinfer rebuild and overwrite the legacy b1 file, so
        the production PGIE gets a persistent worker-local engine path. Face and
        behavior SGIE contracts remain exactly as deployed.
        """

        element = super()._infer_element(name, config, target_fps, primary=primary)
        if not primary:
            return element
        source_path = self.config.resolve_path(config.config_file)
        capacity = len(self.config.enabled_sources)
        worker_root = self.config.resolve_path(self.config.output.path).parent
        fingerprint = self._primary_engine_fingerprint(config)
        engine_path = (
            worker_root
            / ".engines"
            / f"{name}-b{capacity}-{fingerprint}.engine"
        )
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path = worker_root / ".runtime" / "nvinfer" / f"{name}.txt"
        skip_interval = self.config.interval_for(target_fps)
        materialize_nvinfer_config(
            source_path,
            runtime_path,
            {
                "gie-unique-id": config.unique_id,
                "gpu-id": self.config.pipeline.streammux.gpu_id,
                "interval": skip_interval,
                "batch-size": capacity,
                "model-engine-file": str(engine_path),
            },
        )
        element.set_property("config-file-path", str(runtime_path))
        LOGGER.info(
            "[PRODUCTION_ENGINE] component=%s batch=%d isolated_engine=%s legacy_engine_untouched=true",
            name,
            capacity,
            engine_path,
        )
        return element

    def build(self) -> PipelineGraph:
        Gst = self.runtime.Gst
        pipeline = Gst.Pipeline.new("deepstream-ai-production-worker")
        if pipeline is None:
            raise PipelineError("无法创建生产 DeepStream Pipeline")
        streammux = make_element(Gst, "nvstreammux", "stream-muxer")
        capacity = len(self.config.enabled_sources)
        self._configure_streammux(streammux, capacity)
        add_many(pipeline, [streammux])

        pgie = self._infer_element(
            "person-detector",
            self.config.pipeline.person,
            self.config.inference.person_fps,
            primary=True,
        )
        inference_elements: dict[str, Any] = {"person": pgie}
        tracker = self._tracker_element()
        secondary_chain: list[Any] = []
        if self.config.pipeline.face.enabled:
            face_element = self._infer_element(
                "face-detector",
                self.config.pipeline.face,
                self.config.inference.face_fps,
                primary=False,
            )
            secondary_chain.append(face_element)
            inference_elements["face"] = face_element

        for model in self.config.behavior:
            if not model.enabled:
                continue
            component = InferComponentConfig(
                enabled=True,
                config_file=model.config_file,
                unique_id=model.unique_id,
                label=model.name,
            )
            element = self._infer_element(
                f"behavior-{model.name}",
                component,
                self.config.inference.behavior_fps,
                primary=False,
            )
            gate = BehaviorInferenceGate(
                self.runtime,
                self.feature_registry,
                feature_name=model.name,
                person_unique_id=self.config.pipeline.person.unique_id,
                gate_unique_id=model.unique_id,
            )
            gate.install(element)
            self.behavior_gates[model.name] = gate
            secondary_chain.append(element)
            inference_elements[f"behavior:{model.name}"] = element

        snapshot_convert = make_element(Gst, "nvvideoconvert", "snapshot-rgba-convert")
        set_if_supported(snapshot_convert, "nvbuf-memory-type", _NVBUF_MEM_CUDA_DEVICE)
        set_if_supported(snapshot_convert, "gpu-id", self.config.pipeline.streammux.gpu_id)
        snapshot_caps = make_element(Gst, "capsfilter", "snapshot-rgba-caps")
        snapshot_caps.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
        )
        tiler = make_element(Gst, "nvmultistreamtiler", "stream-tiler")
        self._configure_tiler(tiler, capacity)
        osd = make_element(Gst, "nvdsosd", "onscreen-display")
        set_if_supported(osd, "gpu-id", self.config.pipeline.streammux.gpu_id)
        set_if_supported(osd, "display-text", True)
        set_if_supported(osd, "display-bbox", True)
        sink = make_element(Gst, "fakesink", "discard-sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)

        # Keep the tuned core ordering identical to DeepStreamPipelineBuilder:
        # PGIE -> NvDCF -> RGBA -> face SGIE -> optional behavior SGIEs -> probe.
        elements = [
            pgie,
            tracker,
            snapshot_convert,
            snapshot_caps,
            *secondary_chain,
            tiler,
            osd,
            sink,
        ]
        add_many(pipeline, elements)
        link_many([streammux, *elements])

        guard_config = PeopleNetPretrackerGuardConfig.from_file(self.config.config_path)
        pretracker_guard: PeopleNetPretrackerGuard | None = None
        if guard_config.enabled:
            person = self.config.pipeline.person
            if person.detector_type != "peoplenet":
                raise PipelineError("person_pretracker_guard 仅可用于已验证 class_id 的 PeopleNet")
            pretracker_guard = PeopleNetPretrackerGuard(
                self.runtime,
                guard_config,
                pgie_unique_id=person.unique_id,
                person_class_ids=person.person_class_ids,
                frame_width=self.config.pipeline.streammux.width,
                frame_height=self.config.pipeline.streammux.height,
            )
            guard_pad = pgie.get_static_pad("src")
            if guard_pad is None:
                raise PipelineError("无法获取 PeopleNet pre-tracker guard pad")
            guard_pad.add_probe(Gst.PadProbeType.BUFFER, pretracker_guard.callback, None)

        # Import here to keep the exact existing MetadataProbe implementation.
        from deepstream_ai.pipeline.metadata import MetadataProbe

        probe = MetadataProbe(self.runtime, self.config, self.consumer)
        probe.camera_by_pad.clear()
        probe_element = secondary_chain[-1] if secondary_chain else snapshot_caps
        probe_pad = probe_element.get_static_pad("src")
        if probe_pad is None:
            raise PipelineError("无法获取生产 metadata probe pad")
        probe_pad.add_probe(Gst.PadProbeType.BUFFER, probe.callback, None)
        return PipelineGraph(
            pipeline=pipeline,
            metadata_probe=probe,
            source_bins=(),
            inference_elements=inference_elements,
            pretracker_guard=pretracker_guard,
        )


class DynamicSourceController:
    """Add/remove RTSP SourceBins on the worker GLib thread."""

    def __init__(
        self,
        runtime: Any,
        config: AppConfig,
        graph: PipelineGraph,
        feature_registry: FeatureRegistry,
        *,
        capacity: int,
        shadow_registry_getter: Callable[[], Any | None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.graph = graph
        self.feature_registry = feature_registry
        self.capacity = int(capacity)
        self.shadow_registry_getter = shadow_registry_getter
        self._sources: dict[int, SourceBin] = {}
        self._sink_pads: dict[int, Any] = {}
        self._camera_to_slot: dict[str, int] = {}
        self._lock = threading.RLock()

    def _streammux(self) -> Any:
        value = self.graph.pipeline.get_by_name("stream-muxer")
        if value is None:
            raise RuntimeError("production streammux not found")
        return value

    def active_count(self) -> int:
        with self._lock:
            return len(self._sources)

    def slot_for_camera(self, camera_id: str) -> int | None:
        with self._lock:
            return self._camera_to_slot.get(camera_id)

    def camera_for_source_name(self, source_name: str) -> str | None:
        prefix = "uri-source-"
        if not source_name.startswith(prefix):
            return None
        try:
            slot = int(source_name[len(prefix) :])
        except ValueError:
            return None
        with self._lock:
            source = self._sources.get(slot)
            return source.config.camera_id if source is not None else None

    def _free_slot(self) -> int:
        with self._lock:
            for slot in range(self.capacity):
                if slot not in self._sources:
                    return slot
        raise RuntimeError("GPU worker session capacity reached")

    def add(self, source: SourceConfig, features: FeatureSet) -> int:
        if source.type != "rtsp":
            raise ValueError("production dynamic worker only accepts RTSP")
        with self._lock:
            if source.camera_id in self._camera_to_slot:
                raise RuntimeError(f"camera already attached: {source.camera_id}")
        slot = self._free_slot()
        Gst = self.runtime.Gst
        pipeline = self.graph.pipeline
        streammux = self._streammux()
        source_bin = SourceBin(self.runtime, self.config, source, slot)
        if not pipeline.add(source_bin.bin):
            raise PipelineError(f"无法把视频源加入生产 Pipeline: {source.camera_id}")
        sink_pad = None
        try:
            request_name = f"sink_{slot}"
            if hasattr(streammux, "request_pad_simple"):
                sink_pad = streammux.request_pad_simple(request_name)
            if sink_pad is None:
                template = streammux.get_pad_template("sink_%u")
                sink_pad = streammux.request_pad(template, request_name, None)
            src_pad = source_bin.bin.get_static_pad("src")
            if src_pad is None or sink_pad is None:
                raise PipelineError(f"无法请求动态 nvstreammux pad: {request_name}")
            result = src_pad.link(sink_pad)
            if result != Gst.PadLinkReturn.OK:
                raise PipelineError(
                    f"视频源 {source.camera_id} 动态连接 nvstreammux 失败: {result}"
                )
            self.feature_registry.register(slot, source.camera_id, features)
            self.graph.metadata_probe.camera_by_pad[slot] = source.camera_id
            shadow = self.shadow_registry_getter() if self.shadow_registry_getter else None
            if shadow is not None:
                shadow.camera_by_pad[slot] = source.camera_id
            with self._lock:
                self._sources[slot] = source_bin
                self._sink_pads[slot] = sink_pad
                self._camera_to_slot[source.camera_id] = slot
                self.graph.source_bins = tuple(
                    self._sources[index] for index in sorted(self._sources)
                )
            bump_stream_generation(source.camera_id, reason="production_session_attach")
            if not source_bin.bin.sync_state_with_parent():
                raise PipelineError(f"视频源 {source.camera_id} 无法同步到 PLAYING")
            LOGGER.info(
                "[SESSION_ATTACH] camera=%s slot=%d active=%d features=%s",
                source.camera_id,
                slot,
                self.active_count(),
                features.as_dict(),
            )
            return slot
        except Exception:
            self.feature_registry.unregister(slot)
            self.graph.metadata_probe.camera_by_pad.pop(slot, None)
            if sink_pad is not None:
                with suppress(Exception):
                    streammux.release_request_pad(sink_pad)
            with suppress(Exception):
                source_bin.bin.set_state(Gst.State.NULL)
            with suppress(Exception):
                pipeline.remove(source_bin.bin)
            raise

    def remove(self, camera_id: str) -> bool:
        Gst = self.runtime.Gst
        with self._lock:
            slot = self._camera_to_slot.get(camera_id)
            if slot is None:
                return False
            source_bin = self._sources[slot]
            sink_pad = self._sink_pads[slot]
        streammux = self._streammux()
        pipeline = self.graph.pipeline
        src_pad = source_bin.bin.get_static_pad("src")
        source_bin.bin.set_state(Gst.State.NULL)
        if src_pad is not None and sink_pad is not None:
            with suppress(Exception):
                src_pad.unlink(sink_pad)
        if sink_pad is not None:
            with suppress(Exception):
                streammux.release_request_pad(sink_pad)
        if not pipeline.remove(source_bin.bin):
            LOGGER.warning("动态视频源已置 NULL 但从 Pipeline 移除失败 camera=%s", camera_id)
        self.feature_registry.unregister(slot)
        self.graph.metadata_probe.camera_by_pad.pop(slot, None)
        shadow = self.shadow_registry_getter() if self.shadow_registry_getter else None
        if shadow is not None:
            shadow.camera_by_pad.pop(slot, None)
        with self._lock:
            self._sources.pop(slot, None)
            self._sink_pads.pop(slot, None)
            self._camera_to_slot.pop(camera_id, None)
            self.graph.source_bins = tuple(
                self._sources[index] for index in sorted(self._sources)
            )
        bump_stream_generation(camera_id, reason="production_session_detach")
        LOGGER.info(
            "[SESSION_DETACH] camera=%s slot=%d active=%d",
            camera_id,
            slot,
            self.active_count(),
        )
        return True

    def close(self) -> None:
        with self._lock:
            cameras = tuple(self._camera_to_slot)
        for camera_id in cameras:
            try:
                self.remove(camera_id)
            except Exception:
                LOGGER.exception("关闭动态视频源失败 camera=%s", camera_id)


class DynamicPipelineRunner(PipelineRunner):
    """Existing runner with physical-GPU telemetry and dynamic reconnect mapping."""

    def __init__(self, *args: Any, physical_gpu_id: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.physical_gpu_id = int(physical_gpu_id)
        self.source_controller: DynamicSourceController | None = None

    @property
    def shadow_registry(self) -> Any | None:
        return self._shadow

    def _mark_rtsp_reconnect(self, source_name: str) -> None:
        if self.source_controller is not None:
            camera_id = self.source_controller.camera_for_source_name(source_name)
            if camera_id is not None:
                bump_stream_generation(camera_id, reason=f"rtsp_reconnect:{source_name}")
                return
        super()._mark_rtsp_reconnect(source_name)

    def _install_physical_gpu_monitor(self) -> None:
        if self._adaptive is not None:
            self._adaptive._gpu = NvidiaSmiMonitor(self.physical_gpu_id)


__all__ = [
    "DynamicPipelineRunner",
    "DynamicSourceController",
    "WarmDynamicPipelineBuilder",
    "build_warm_config",
]
