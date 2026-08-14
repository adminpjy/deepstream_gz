"""Build the hardware-accelerated DeepStream graph."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepstream_ai.config import AppConfig, InferComponentConfig
from deepstream_ai.errors import PipelineError

from .elements import add_many, link_many, make_element, set_if_supported
from .metadata import FramePacketConsumer, MetadataProbe
from .nvinfer_config import materialize_nvinfer_config
from .peoplenet_pretracker_guard import (
    PeopleNetPretrackerGuard,
    PeopleNetPretrackerGuardConfig,
)
from .source import SourceBin, link_source_to_mux

LOGGER = logging.getLogger(__name__)
_NVBUF_MEM_CUDA_UNIFIED = 3
_NVBUF_MEM_CUDA_DEVICE = 2
_NVBUF_MEM_CUDA_PINNED = 1


@dataclass(slots=True)
class PipelineGraph:
    pipeline: Any
    metadata_probe: MetadataProbe
    source_bins: tuple[SourceBin, ...]
    inference_elements: dict[str, Any]
    pretracker_guard: PeopleNetPretrackerGuard | None = None


class DeepStreamPipelineBuilder:
    def __init__(self, runtime: Any, config: AppConfig, consumer: FramePacketConsumer):
        self.runtime = runtime
        self.config = config
        self.consumer = consumer

    def build(self) -> PipelineGraph:
        Gst = self.runtime.Gst
        pipeline = Gst.Pipeline.new("deepstream-ai-platform")
        if pipeline is None:
            raise PipelineError("无法创建 GStreamer Pipeline")
        sources = tuple(
            SourceBin(self.runtime, self.config, source, index)
            for index, source in enumerate(self.config.enabled_sources)
        )
        streammux = make_element(Gst, "nvstreammux", "stream-muxer")
        self._configure_streammux(streammux, len(sources))
        add_many(pipeline, [source.bin for source in sources] + [streammux])
        for index, source in enumerate(sources):
            link_source_to_mux(self.runtime, source, streammux, index)

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
                LOGGER.info("行为模型已关闭，不创建 nvinfer: %s", model.name)
                continue
            component = InferComponentConfig(
                enabled=True,
                config_file=model.config_file,
                unique_id=model.unique_id,
                label=model.name,
            )
            behavior_element = self._infer_element(
                f"behavior-{model.name}",
                component,
                self.config.inference.behavior_fps,
                primary=False,
            )
            secondary_chain.append(behavior_element)
            inference_elements[f"behavior:{model.name}"] = behavior_element

        snapshot_convert = make_element(Gst, "nvvideoconvert", "snapshot-rgba-convert")
        # WSL must not expose CUDA unified NvBufSurface memory to CPU paths.
        # MetadataProbe performs an explicit DtoH copy, so device memory is the
        # safe common contract on both native Linux and Docker Desktop/WSL.
        set_if_supported(snapshot_convert, "nvbuf-memory-type", _NVBUF_MEM_CUDA_DEVICE)
        set_if_supported(snapshot_convert, "gpu-id", self.config.pipeline.streammux.gpu_id)
        snapshot_caps = make_element(Gst, "capsfilter", "snapshot-rgba-caps")
        snapshot_caps.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
        )
        tiler = make_element(Gst, "nvmultistreamtiler", "stream-tiler")
        self._configure_tiler(tiler, len(sources))
        osd = make_element(Gst, "nvdsosd", "onscreen-display")
        set_if_supported(osd, "gpu-id", self.config.pipeline.streammux.gpu_id)
        set_if_supported(osd, "display-text", True)
        set_if_supported(osd, "display-bbox", True)
        # Convert to the business layer's RGBA format before secondary GIEs.
        # Gst-nvinfer attaches detector children and tensor output meta to the
        # parent object. A downstream nvvideoconvert buffer/meta copy can lose
        # that parent relationship, so the metadata probe must be immediately
        # after the last SGIE with no transforming element in between.
        elements = [
            pgie,
            tracker,
            snapshot_convert,
            snapshot_caps,
            *secondary_chain,
            tiler,
            osd,
        ]

        if self.config.output.enabled:
            elements.extend(self._output_elements())
        else:
            sink = make_element(Gst, "fakesink", "discard-sink")
            sink.set_property("sync", False)
            elements.append(sink)
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

        probe = MetadataProbe(self.runtime, self.config, self.consumer)
        probe_element = secondary_chain[-1] if secondary_chain else snapshot_caps
        probe_pad = probe_element.get_static_pad("src")
        if probe_pad is None:
            raise PipelineError("无法获取截图 metadata probe pad")
        probe_pad.add_probe(Gst.PadProbeType.BUFFER, probe.callback, None)
        return PipelineGraph(
            pipeline=pipeline,
            metadata_probe=probe,
            source_bins=sources,
            inference_elements=inference_elements,
            pretracker_guard=pretracker_guard,
        )

    def _configure_streammux(self, streammux: Any, count: int) -> None:
        cfg = self.config.pipeline.streammux
        for name, value in {
            "batch-size": count,
            "width": cfg.width,
            "height": cfg.height,
            "batched-push-timeout": cfg.batch_timeout_us,
            "live-source": any(source.type == "rtsp" for source in self.config.enabled_sources),
            "gpu-id": cfg.gpu_id,
            "attach-sys-ts": cfg.attach_system_timestamp,
            "sync-inputs": cfg.sync_inputs,
            "nvbuf-memory-type": _NVBUF_MEM_CUDA_UNIFIED,
        }.items():
            set_if_supported(streammux, name, value)

    def _infer_element(
        self,
        name: str,
        config: InferComponentConfig,
        target_fps: float,
        *,
        primary: bool,
    ) -> Any:
        element = make_element(self.runtime.Gst, "nvinfer", name)
        path = self.config.resolve_path(config.config_file)
        skip_interval = self.config.interval_for(target_fps)
        if primary:
            overrides = {
                "gie-unique-id": config.unique_id,
                "gpu-id": self.config.pipeline.streammux.gpu_id,
                "interval": skip_interval,
                "batch-size": len(self.config.enabled_sources),
            }
            rate_property = "interval"
            rate_value = skip_interval
        else:
            # Gst-nvinfer's `interval` is a primary-GIE property. Secondary GIE
            # cadence is object based: 0 means every frame, otherwise the number
            # of frames after which the tracked object is reinferred.
            reinfer_interval = 0 if skip_interval == 0 else skip_interval + 1
            overrides = {
                "gie-unique-id": config.unique_id,
                "gpu-id": self.config.pipeline.streammux.gpu_id,
                "secondary-reinfer-interval": reinfer_interval,
            }
            rate_property = "secondary-reinfer-interval"
            rate_value = reinfer_interval
        # Service tasks override output.path per task. Keep the materialized
        # nvinfer config beside that task's artifacts so concurrent pipelines
        # never overwrite each other's runtime contract.
        runtime_path = (
            self.config.resolve_path(self.config.output.path).parent
            / ".runtime"
            / "nvinfer"
            / f"{name}.txt"
        )
        materialize_nvinfer_config(path, runtime_path, overrides)
        element.set_property("config-file-path", str(runtime_path))
        LOGGER.info(
            "配置推理组件 name=%s unique_id=%s target_fps=%.2f %s=%s runtime_config=%s",
            name,
            config.unique_id,
            target_fps,
            rate_property,
            rate_value,
            runtime_path,
        )
        return element

    def _tracker_element(self) -> Any:
        tracker = make_element(self.runtime.Gst, "nvtracker", "person-tracker")
        cfg = self.config.pipeline.tracker
        person_class_ids = ";".join(
            str(class_id) for class_id in self.config.pipeline.person.person_class_ids
        )
        config_path = self.config.resolve_path(cfg.config_file)
        library_path = Path(cfg.library_file)
        for name, value in {
            "tracker-width": cfg.width,
            "tracker-height": cfg.height,
            "gpu-id": cfg.gpu_id,
            "ll-config-file": str(config_path),
            "display-tracking-id": cfg.display_tracking_id,
            # PeopleNet also emits bag and face. Restrict NvDCF to the class IDs
            # validated from the deployed labels so auxiliary objects neither
            # consume tracker IDs nor compete with person association.
            "operate-on-class-ids": person_class_ids,
            # ReID vectors are copied into business continuity state before
            # their GstBuffer expires. Size this for dense/multi-stream scenes
            # instead of relying on nvtracker's small default pool of 32.
            "user-meta-pool-size": 256,
        }.items():
            set_if_supported(tracker, name, value)
        if cfg.library_file:
            set_if_supported(tracker, "ll-lib-file", str(library_path))
        return tracker

    def _configure_tiler(self, tiler: Any, count: int) -> None:
        rows = max(1, int(math.sqrt(count)))
        columns = math.ceil(count / rows)
        for name, value in {
            "rows": rows,
            "columns": columns,
            "width": self.config.pipeline.tiler_width,
            "height": self.config.pipeline.tiler_height,
            "gpu-id": self.config.pipeline.streammux.gpu_id,
        }.items():
            set_if_supported(tiler, name, value)

    def _output_elements(self) -> list[Any]:
        Gst = self.runtime.Gst
        output = self.config.output
        output_path = self.config.resolve_path(output.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prefix: list[Any] = []
        if output.sync:
            # Pacing at filesink is unreliable after qtmux on some drivers.
            # Synchronize raw buffers so the service preview follows file time.
            pacer = make_element(Gst, "identity", "file-realtime-pacer")
            pacer.set_property("sync", True)
            prefix.append(pacer)
        convert = make_element(Gst, "nvvideoconvert", "encoder-convert")
        set_if_supported(convert, "gpu-id", self.config.pipeline.streammux.gpu_id)
        caps_filter = make_element(Gst, "capsfilter", "encoder-caps")
        codec = "h264" if output.codec == "h264" else "h265"
        if output.encoder == "x264":
            # Provide a portable CPU fallback for hosts where the NVIDIA video
            # encoder is unavailable. x264 consumes system memory and expresses
            # bitrate in kbit/s rather than bit/s.
            set_if_supported(convert, "nvbuf-memory-type", _NVBUF_MEM_CUDA_PINNED)
            caps_filter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420"))
            encoder = make_element(Gst, "x264enc", "h264-encoder")
            set_if_supported(encoder, "bitrate", max(1, output.bitrate // 1000))
            set_if_supported(encoder, "speed-preset", 3)  # veryfast
            set_if_supported(encoder, "tune", 4)  # zerolatency
            set_if_supported(encoder, "key-int-max", 30)
            set_if_supported(encoder, "byte-stream", False)
        else:
            caps_filter.set_property(
                "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")
            )
            encoder = make_element(Gst, f"nvv4l2{codec}enc", f"{codec}-encoder")
            set_if_supported(encoder, "gpu-id", self.config.pipeline.streammux.gpu_id)
            set_if_supported(encoder, "bitrate", output.bitrate)
            set_if_supported(encoder, "insert-sps-pps", True)
            set_if_supported(encoder, "iframeinterval", 30)
            # iframeinterval alone can create non-IDR I-frames. Periodic IDRs
            # are required for correct MP4 seeking and independent thumbnails.
            set_if_supported(encoder, "idrinterval", 30)
        parser = make_element(Gst, f"{codec}parse", f"{codec}-parser")
        mux = make_element(Gst, "qtmux", "mp4-muxer")
        set_if_supported(mux, "faststart", True)
        sink = make_element(Gst, "filesink", "result-file")
        sink.set_property("location", str(output_path))
        sink.set_property("sync", output.sync)
        sink.set_property("async", False)
        LOGGER.info("结果视频输出: %s encoder=%s", output_path, output.encoder)
        return [*prefix, convert, caps_filter, encoder, parser, mux, sink]
