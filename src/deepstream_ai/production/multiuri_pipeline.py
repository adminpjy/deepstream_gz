"""Official DeepStream dynamic-source implementation for production workers.

The production worker keeps the tuned inference graph resident, but delegates
runtime RTSP add/remove to NVIDIA's ``nvmultiurisrcbin`` instead of manually
adding ``nvurisrcbin`` elements and requesting/releasing ``nvstreammux`` pads.
This keeps source lifecycle inside the DeepStream component designed for that
purpose while leaving PeopleNet/NvDCF/SCRFD/business consumers unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from deepstream_ai.config import AppConfig, InferComponentConfig, SourceConfig
from deepstream_ai.errors import PipelineError
from deepstream_ai.pipeline.builder import DeepStreamPipelineBuilder, PipelineGraph
from deepstream_ai.pipeline.elements import add_many, link_many, make_element, set_if_supported
from deepstream_ai.pipeline.metadata import MetadataProbe
from deepstream_ai.pipeline.nvinfer_config import materialize_nvinfer_config
from deepstream_ai.pipeline.peoplenet_pretracker_guard import (
    PeopleNetPretrackerGuard,
    PeopleNetPretrackerGuardConfig,
)
from deepstream_ai.preflight import inspect_nvinfer_config
from deepstream_ai.production.contracts import FeatureSet
from deepstream_ai.production.feature_gate import BehaviorInferenceGate, FeatureRegistry
from deepstream_ai.stream_epoch import bump_stream_generation

LOGGER = logging.getLogger(__name__)
_NVBUF_MEM_CUDA_DEVICE = 2
_DEFAULT_REST_PORT_BASE = 9100


def production_multiuri_port() -> int:
    """Return a process-unique loopback REST port for one physical GPU worker."""

    try:
        base = int(os.environ.get("PRODUCTION_MULTIURI_PORT_BASE", _DEFAULT_REST_PORT_BASE))
        physical_gpu = int(os.environ.get("DEEPSTREAM_PHYSICAL_GPU_ID", "0"))
    except ValueError as exc:
        raise ValueError("PRODUCTION_MULTIURI_PORT_BASE/DEEPSTREAM_PHYSICAL_GPU_ID must be integers") from exc
    port = base + physical_gpu
    if not 1 <= port <= 65535:
        raise ValueError(f"production nvmultiurisrcbin REST port is invalid: {port}")
    return port


class MultiUriPipelineBuilder(DeepStreamPipelineBuilder):
    """Build one resident inference graph fed by NVIDIA nvmultiurisrcbin."""

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
        element = super()._infer_element(name, config, target_fps, primary=primary)
        if not primary:
            return element

        source_path = self.config.resolve_path(config.config_file)
        capacity = len(self.config.enabled_sources)
        worker_root = self.config.resolve_path(self.config.output.path).parent
        fingerprint = self._primary_engine_fingerprint(config)
        engine_path = worker_root / ".engines" / f"{name}-b{capacity}-{fingerprint}.engine"
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

    def _multiuri_source(self, capacity: int) -> Any:
        Gst = self.runtime.Gst
        source = make_element(Gst, "nvmultiurisrcbin", "production-multiuri-source")
        cfg = self.config.pipeline.streammux
        port = production_multiuri_port()

        # These are direct nvmultiurisrcbin properties and must exist in the
        # deployed DeepStream runtime. Fail early rather than silently falling
        # back to the old manual source lifecycle.
        source.set_property("mode", 0)
        source.set_property("ip-address", "127.0.0.1")
        source.set_property("port", str(port))
        source.set_property("max-batch-size", capacity)

        # Properties below are forwarded by nvmultiurisrcbin to each
        # nvurisrcbin / its internal nvstreammux. Keep the same RTSP defaults as
        # the validated SourceBin path: CUDA device decode and RTP-over-TCP.
        for name, value in {
            "gpu-id": cfg.gpu_id,
            "cudadec-memtype": 0,
            "num-extra-surfaces": 4,
            "drop-frame-interval": 0,
            "select-rtp-protocol": 4,
            "rtsp-reconnect-interval": 10,
            "rtsp-reconnect-attempts": -1,
            "latency": 200,
            "disable-audio": True,
            "async-handling": True,
            "drop-pipeline-eos": True,
            "live-source": True,
            "batched-push-timeout": cfg.batch_timeout_us,
            "width": cfg.width,
            "height": cfg.height,
            "attach-sys-ts": cfg.attach_system_timestamp,
            "sync-inputs": cfg.sync_inputs,
        }.items():
            set_if_supported(source, name, value)

        LOGGER.info(
            "[PRODUCTION_MULTIURI] port=%d capacity=%d live_source=true "
            "drop_pipeline_eos=true transport=tcp latency_ms=200",
            port,
            capacity,
        )
        return source

    def build(self) -> PipelineGraph:
        Gst = self.runtime.Gst
        pipeline = Gst.Pipeline.new("deepstream-ai-production-worker")
        if pipeline is None:
            raise PipelineError("无法创建生产 DeepStream Pipeline")

        capacity = len(self.config.enabled_sources)
        multi_source = self._multiuri_source(capacity)
        add_many(pipeline, [multi_source])

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

        # Downstream ordering remains identical to the tuned core. Only the
        # source/mux owner changes from manual SourceBins+nvstreammux to the
        # NVIDIA component that contains those two pieces internally.
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
        link_many([multi_source, *elements])

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

        probe = MetadataProbe(self.runtime, self.config, self.consumer)
        # Warm slots are capacity placeholders, not real cameras. The official
        # controller fills the real source_id -> camera mapping after REST add.
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


@dataclass(slots=True)
class _SourceBinding:
    source: SourceConfig
    source_id: int


class MultiUriSourceController:
    """Use nvmultiurisrcbin's loopback REST API for dynamic RTSP lifecycle."""

    def __init__(
        self,
        runtime: Any,
        config: AppConfig,
        graph: PipelineGraph,
        feature_registry: FeatureRegistry,
        *,
        capacity: int,
        shadow_registry_getter: Any | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.graph = graph
        self.feature_registry = feature_registry
        self.capacity = int(capacity)
        self.shadow_registry_getter = shadow_registry_getter
        self.port = production_multiuri_port()
        self.base_url = f"http://127.0.0.1:{self.port}/api/v1"
        self._by_camera: dict[str, _SourceBinding] = {}
        self._by_source_id: dict[int, str] = {}
        self._lock = threading.RLock()

    def active_count(self) -> int:
        with self._lock:
            return len(self._by_camera)

    def slot_for_camera(self, camera_id: str) -> int | None:
        with self._lock:
            item = self._by_camera.get(camera_id)
            return item.source_id if item is not None else None

    def camera_for_source_name(self, source_name: str) -> str | None:
        digits = "".join(ch for ch in source_name.rsplit("-", 1)[-1] if ch.isdigit())
        if not digits:
            return None
        source_id = int(digits)
        with self._lock:
            return self._by_source_id.get(source_id)

    def _free_local_source_id(self) -> int:
        with self._lock:
            for value in range(self.capacity):
                if value not in self._by_source_id:
                    return value
        raise RuntimeError("GPU worker session capacity reached")

    @staticmethod
    def _payload(source: SourceConfig, change: str) -> dict[str, Any]:
        return {
            "key": "sensor",
            "value": {
                "camera_id": source.camera_id,
                "camera_name": source.camera_id,
                "camera_url": source.location,
                "change": change,
                "metadata": {"framerate": float(source.nominal_fps)},
            },
        }

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        body = None
        headers: dict[str, str] = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback only
                raw = response.read()
                status = int(getattr(response, "status", 200))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[-1000:]
            raise PipelineError(
                f"nvmultiurisrcbin REST {method} {path} failed: HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise PipelineError(
                f"nvmultiurisrcbin REST {method} {path} unavailable on 127.0.0.1:{self.port}: {exc}"
            ) from exc
        if not 200 <= status < 300:
            raise PipelineError(f"nvmultiurisrcbin REST {method} {path} returned HTTP {status}")
        if not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"raw": raw.decode("utf-8", "replace")[-2000:]}
        return value if isinstance(value, dict) else {"value": value}

    @staticmethod
    def _response_failed(document: dict[str, Any]) -> str | None:
        status = str(document.get("status", "")).upper()
        reason = str(document.get("reason", "")).upper()
        if " 4" in status or " 5" in status or "FAIL" in reason or "ERROR" in reason:
            return f"status={document.get('status')} reason={document.get('reason')}"
        return None

    def _register_mapping(
        self,
        source_id: int,
        source: SourceConfig,
        features: FeatureSet,
    ) -> None:
        self.feature_registry.register(source_id, source.camera_id, features)
        self.graph.metadata_probe.camera_by_pad[source_id] = source.camera_id
        shadow = self.shadow_registry_getter() if self.shadow_registry_getter else None
        if shadow is not None:
            shadow.camera_by_pad[source_id] = source.camera_id

    def _unregister_mapping(self, source_id: int) -> None:
        self.feature_registry.unregister(source_id)
        self.graph.metadata_probe.camera_by_pad.pop(source_id, None)
        shadow = self.shadow_registry_getter() if self.shadow_registry_getter else None
        if shadow is not None:
            shadow.camera_by_pad.pop(source_id, None)

    def _source_id_from_metrics(self, camera_id: str, *, timeout: float = 2.0) -> int | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                document = self._request_json("GET", "/metrics", timeout=1.0)
            except PipelineError:
                return None
            metrics = document.get("metrics-info") or {}
            stats = metrics.get("stream-stats") if isinstance(metrics, dict) else None
            if isinstance(stats, list):
                for item in stats:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("sensor_id", "")) != camera_id:
                        continue
                    try:
                        return int(item["source_id"])
                    except (KeyError, TypeError, ValueError):
                        return None
            time.sleep(0.1)
        return None

    def add(self, source: SourceConfig, features: FeatureSet) -> int:
        if source.type != "rtsp":
            raise ValueError("production multiuri worker only accepts RTSP")
        with self._lock:
            if source.camera_id in self._by_camera:
                raise RuntimeError(f"camera already attached: {source.camera_id}")
            if len(self._by_camera) >= self.capacity:
                raise RuntimeError("GPU worker session capacity reached")

        provisional_id = self._free_local_source_id()
        self._register_mapping(provisional_id, source, features)
        document: dict[str, Any] = {}
        try:
            document = self._request_json(
                "POST",
                "/stream/add",
                self._payload(source, "camera_add"),
            )
            failed = self._response_failed(document)
            if failed:
                raise PipelineError(f"nvmultiurisrcbin stream add failed: {failed}")

            # DeepStream 9 exposes source_id through its metrics endpoint. Use
            # it when available; otherwise the component's documented reusable
            # pad IDs and our lowest-free allocation give a conservative fallback.
            actual_id = self._source_id_from_metrics(source.camera_id) or provisional_id
            if actual_id != provisional_id:
                self._unregister_mapping(provisional_id)
                self._register_mapping(actual_id, source, features)
            with self._lock:
                self._by_camera[source.camera_id] = _SourceBinding(source, actual_id)
                self._by_source_id[actual_id] = source.camera_id
            bump_stream_generation(source.camera_id, reason="production_session_attach")
            LOGGER.info(
                "[SESSION_ATTACH_MULTIURI] camera=%s source_id=%d active=%d features=%s",
                source.camera_id,
                actual_id,
                self.active_count(),
                features.as_dict(),
            )
            return actual_id
        except Exception:
            self._unregister_mapping(provisional_id)
            # If REST accepted the add but local bookkeeping failed, best-effort
            # remove prevents an unowned stream from remaining resident.
            if document:
                try:
                    self._request_json(
                        "POST",
                        "/stream/remove",
                        self._payload(source, "camera_remove"),
                        timeout=2.0,
                    )
                except Exception:
                    LOGGER.exception(
                        "[SESSION_ATTACH_MULTIURI_ROLLBACK] camera=%s",
                        source.camera_id,
                    )
            raise

    def remove(self, camera_id: str) -> bool:
        with self._lock:
            binding = self._by_camera.get(camera_id)
        if binding is None:
            return False
        document = self._request_json(
            "POST",
            "/stream/remove",
            self._payload(binding.source, "camera_remove"),
        )
        failed = self._response_failed(document)
        if failed:
            raise PipelineError(f"nvmultiurisrcbin stream remove failed: {failed}")
        with self._lock:
            self._by_camera.pop(camera_id, None)
            self._by_source_id.pop(binding.source_id, None)
        self._unregister_mapping(binding.source_id)
        bump_stream_generation(camera_id, reason="production_session_detach")
        LOGGER.info(
            "[SESSION_DETACH_MULTIURI] camera=%s source_id=%d active=%d",
            camera_id,
            binding.source_id,
            self.active_count(),
        )
        return True

    def close(self) -> None:
        with self._lock:
            cameras = tuple(self._by_camera)
        for camera_id in cameras:
            try:
                self.remove(camera_id)
            except Exception:
                LOGGER.exception("关闭 nvmultiurisrcbin 动态视频源失败 camera=%s", camera_id)


__all__ = [
    "MultiUriPipelineBuilder",
    "MultiUriSourceController",
    "production_multiuri_port",
]
