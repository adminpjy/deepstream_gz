"""Typed YAML configuration loading, normalization, and validation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .errors import AssetValidationError, ConfigurationError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")
_REFERENCE_TRACKER_LIBRARY = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigurationError(f"环境变量 {name} 未设置，且配置未提供默认值")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"配置节 {section} 必须是对象")
    return dict(value)


def _positive(value: float | int, label: str, *, allow_zero: bool = False) -> None:
    if value < 0 if allow_zero else value <= 0:
        qualifier = "非负数" if allow_zero else "正数"
        raise ConfigurationError(f"{label} 必须是{qualifier}，当前值为 {value}")


@dataclass(frozen=True, slots=True)
class SourceConfig:
    camera_id: str
    type: str = "file"
    path: str | None = None
    url: str | None = None
    enabled: bool = True
    nominal_fps: float = 25.0
    latency_ms: int = 200
    reconnect_interval_sec: int = 10

    @property
    def location(self) -> str:
        value = self.path if self.type == "file" else self.url
        return value or ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], index: int) -> SourceConfig:
        item = dict(data)
        source_type = str(item.get("type", "file")).lower()
        if source_type not in {"file", "rtsp"}:
            raise ConfigurationError(f"sources[{index}].type 仅支持 file 或 rtsp")
        camera_id = str(item.get("camera_id") or item.get("id") or f"camera-{index + 1}")
        result = cls(
            camera_id=camera_id,
            type=source_type,
            path=item.get("path"),
            url=item.get("url"),
            enabled=bool(item.get("enabled", True)),
            nominal_fps=float(item.get("nominal_fps", 25.0)),
            latency_ms=int(item.get("latency_ms", item.get("latency", 200))),
            reconnect_interval_sec=int(item.get("reconnect_interval_sec", 10)),
        )
        if not result.location:
            key = "path" if source_type == "file" else "url"
            raise ConfigurationError(f"sources[{index}].{key} 不能为空")
        _positive(result.nominal_fps, f"sources[{index}].nominal_fps")
        _positive(result.latency_ms, f"sources[{index}].latency_ms", allow_zero=True)
        return result


@dataclass(frozen=True, slots=True)
class InferenceRateConfig:
    person_fps: float = 5.0
    face_fps: float = 2.0
    behavior_fps: float = 1.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> InferenceRateConfig:
        result = cls(
            person_fps=float(data.get("person_fps", 5.0)),
            face_fps=float(data.get("face_fps", 2.0)),
            behavior_fps=float(data.get("behavior_fps", 1.0)),
        )
        for field_name in ("person_fps", "face_fps", "behavior_fps"):
            _positive(getattr(result, field_name), f"inference.{field_name}")
        return result


@dataclass(frozen=True, slots=True)
class InferComponentConfig:
    enabled: bool = True
    config_file: str = ""
    unique_id: int = 1
    person_class_ids: tuple[int, ...] = (0,)
    detector_type: str = "generic"
    people_classes: tuple[tuple[str, int], ...] = ()
    label: str = ""
    landmark_source: str = "none"
    landmark_coordinates: str = "absolute"
    landmark_scale: float = 1.0
    landmark_threshold: float = 0.65

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        default_enabled: bool,
        default_unique_id: int,
        default_label: str,
    ) -> InferComponentConfig:
        raw_classes = data.get("people_classes", {})
        if not isinstance(raw_classes, Mapping):
            raise ConfigurationError("person.people_classes 必须是 class_name: class_id 映射")
        people_classes = tuple(
            (str(name).strip().lower(), int(class_id)) for name, class_id in raw_classes.items()
        )
        class_map = dict(people_classes)
        raw_ids = data.get(
            "person_class_ids",
            data.get("class_ids", [class_map["person"]] if "person" in class_map else [0]),
        )
        if isinstance(raw_ids, int):
            raw_ids = [raw_ids]
        result = cls(
            enabled=bool(data.get("enabled", default_enabled)),
            config_file=str(data.get("config_file", data.get("config", ""))),
            unique_id=int(data.get("unique_id", default_unique_id)),
            person_class_ids=tuple(int(item) for item in raw_ids),
            detector_type=str(data.get("type", data.get("detector_type", "generic"))).lower(),
            people_classes=people_classes,
            label=str(data.get("label", default_label)),
            landmark_source=str(data.get("landmark_source", "none")).lower(),
            landmark_coordinates=str(data.get("landmark_coordinates", "absolute")).lower(),
            landmark_scale=float(data.get("landmark_scale", 1.0)),
            landmark_threshold=float(data.get("landmark_threshold", 0.65)),
        )
        if result.landmark_source not in {"none", "tensor", "mask"}:
            raise ConfigurationError("face.landmark_source 仅支持 none、tensor 或兼容模式 mask")
        if not result.detector_type:
            raise ConfigurationError("person.type 不能为空")
        if any(not name or class_id < 0 for name, class_id in result.people_classes):
            raise ConfigurationError("person.people_classes 名称不能为空且 class_id 必须非负")
        if len({class_id for _, class_id in result.people_classes}) != len(result.people_classes):
            raise ConfigurationError("person.people_classes 的 class_id 必须唯一")
        if result.detector_type == "peoplenet":
            configured_classes = dict(result.people_classes)
            if "person" not in configured_classes:
                raise ConfigurationError("PeopleNet 配置必须从实际 labels 提供 person class_id")
            if result.person_class_ids != (configured_classes["person"],):
                raise ConfigurationError("person.person_class_ids 必须等于 people_classes.person")
        if result.landmark_coordinates not in {"absolute", "bbox", "normalized"}:
            raise ConfigurationError(
                "face.landmark_coordinates 仅支持 absolute、bbox 或 normalized"
            )
        _positive(result.landmark_scale, "face.landmark_scale")
        if not 0.0 <= result.landmark_threshold <= 1.0:
            raise ConfigurationError("face.landmark_threshold 必须在 0 到 1 之间")
        return result


@dataclass(frozen=True, slots=True)
class BehaviorModelConfig:
    name: str
    enabled: bool = False
    config_file: str = ""
    model: str = ""
    unique_id: int = 10
    labels: tuple[str, ...] = ()
    threshold: float = 0.5

    @classmethod
    def from_mapping(
        cls, name: str, data: Mapping[str, Any], unique_id: int
    ) -> BehaviorModelConfig:
        labels = data.get("labels", [name])
        if isinstance(labels, str):
            labels = [labels]
        result = cls(
            name=name,
            enabled=bool(data.get("enabled", False)),
            config_file=str(data.get("config_file", data.get("config", ""))),
            model=str(data.get("model", "")),
            unique_id=int(data.get("unique_id", unique_id)),
            labels=tuple(str(item) for item in labels),
            threshold=float(data.get("threshold", 0.5)),
        )
        if not 0 <= result.threshold <= 1:
            raise ConfigurationError(f"behavior.{name}.threshold 必须在 0 到 1 之间")
        return result


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    backend: str = "nvdcf"
    config_file: str = ""
    library_file: str = _REFERENCE_TRACKER_LIBRARY
    width: int = 960
    height: int = 544
    gpu_id: int = 0
    display_tracking_id: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrackerConfig:
        result = cls(
            backend=str(data.get("backend", "nvdcf")).lower(),
            config_file=str(data.get("config_file", data.get("ll_config_file", ""))),
            library_file=str(
                data.get(
                    "library_file",
                    data.get("ll_lib_file", _REFERENCE_TRACKER_LIBRARY),
                )
            ),
            width=int(data.get("width", 960)),
            height=int(data.get("height", 544)),
            gpu_id=int(data.get("gpu_id", 0)),
            display_tracking_id=bool(data.get("display_tracking_id", True)),
        )
        if result.backend not in {"nvdcf", "nvsort", "deepsort"}:
            raise ConfigurationError("tracker.backend 仅支持 nvdcf、nvsort 或 deepsort")
        if (
            Path(result.library_file.replace("\\", "/")).name
            != Path(_REFERENCE_TRACKER_LIBRARY).name
        ):
            raise ConfigurationError(
                "tracker.backend 仅表示 NVIDIA NvMultiObjectTracker 内置算法；"
                "library_file 必须指向 libnvds_nvmultiobjecttracker.so"
            )
        if result.gpu_id != 0:
            raise ConfigurationError(
                "当前 PyDS/TensorRT 运行时仅支持 gpu_id=0；禁止部分组件落在不同 GPU"
            )
        _positive(result.width, "tracker.width")
        _positive(result.height, "tracker.height")
        return result


@dataclass(frozen=True, slots=True)
class StreamMuxConfig:
    width: int = 1920
    height: int = 1080
    batch_timeout_us: int = 40000
    gpu_id: int = 0
    attach_system_timestamp: bool = True
    sync_inputs: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> StreamMuxConfig:
        result = cls(
            width=int(data.get("width", 1920)),
            height=int(data.get("height", 1080)),
            batch_timeout_us=int(
                data.get("batch_timeout_us", data.get("batched_push_timeout", 40000))
            ),
            gpu_id=int(data.get("gpu_id", 0)),
            attach_system_timestamp=bool(data.get("attach_system_timestamp", True)),
            sync_inputs=bool(data.get("sync_inputs", False)),
        )
        _positive(result.width, "pipeline.streammux.width")
        _positive(result.height, "pipeline.streammux.height")
        _positive(result.batch_timeout_us, "pipeline.streammux.batch_timeout_us")
        if result.gpu_id != 0:
            raise ConfigurationError(
                "当前 PyDS/TensorRT 运行时仅支持 pipeline.streammux.gpu_id=0；"
                "禁止部分组件落在不同 GPU"
            )
        return result


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    streammux: StreamMuxConfig = field(default_factory=StreamMuxConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    person: InferComponentConfig = field(default_factory=InferComponentConfig)
    face: InferComponentConfig = field(
        default_factory=lambda: InferComponentConfig(
            enabled=False,
            config_file="",
            unique_id=2,
            person_class_ids=(0,),
            label="face",
        )
    )
    tiler_width: int = 1920
    tiler_height: int = 1080


@dataclass(frozen=True, slots=True)
class FaceRecognitionConfig:
    enabled: bool = False
    backend: str = "tensorrt"
    model: str = ""
    input_name: str = "input"
    output_name: str = "output"
    embedding_size: int = 512
    match_threshold: float = 0.4
    min_candidates: int = 3
    max_candidates: int = 8
    decision_timeout_sec: float = 2.0
    input_width: int = 112
    input_height: int = 112
    require_landmarks: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> FaceRecognitionConfig:
        result = cls(
            enabled=bool(data.get("enabled", False)),
            backend=str(data.get("backend", "tensorrt")).lower(),
            model=str(data.get("model", data.get("engine", ""))),
            input_name=str(data.get("input_name", "input")),
            output_name=str(data.get("output_name", "output")),
            embedding_size=int(data.get("embedding_size", 512)),
            match_threshold=float(data.get("match_threshold", 0.4)),
            min_candidates=int(data.get("min_candidates", 3)),
            max_candidates=int(data.get("max_candidates", 8)),
            decision_timeout_sec=float(data.get("decision_timeout_sec", 2.0)),
            input_width=int(data.get("input_width", 112)),
            input_height=int(data.get("input_height", 112)),
            require_landmarks=bool(data.get("require_landmarks", True)),
        )
        if result.backend not in {"tensorrt", "onnxruntime"}:
            raise ConfigurationError("face_recognition.backend 仅支持 tensorrt 或 onnxruntime")
        if not 0 <= result.match_threshold <= 1:
            raise ConfigurationError("face_recognition.match_threshold 必须在 0 到 1 之间")
        if not 1 <= result.min_candidates <= result.max_candidates:
            raise ConfigurationError("face_recognition 候选帧数量必须满足 1 <= min <= max")
        if result.embedding_size != 512:
            raise ConfigurationError("AdaFace embedding_size 必须为 512")
        return result


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    enabled: bool = False
    dsn: str = ""
    dsn_env: str = "DATABASE_DSN"
    min_similarity: float = 0.4
    pool_min_size: int = 1
    pool_max_size: int = 4
    connect_timeout_sec: int = 5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DatabaseConfig:
        dsn_env = str(data.get("dsn_env", "DATABASE_DSN"))
        dsn = str(data.get("dsn", "")) or os.environ.get(dsn_env, "")
        result = cls(
            enabled=bool(data.get("enabled", False)),
            dsn=dsn,
            dsn_env=dsn_env,
            min_similarity=float(data.get("min_similarity", data.get("match_threshold", 0.4))),
            pool_min_size=int(data.get("pool_min_size", 1)),
            pool_max_size=int(data.get("pool_max_size", 4)),
            connect_timeout_sec=int(data.get("connect_timeout_sec", 5)),
        )
        if not 0 <= result.min_similarity <= 1:
            raise ConfigurationError("database.min_similarity 必须在 0 到 1 之间")
        return result


@dataclass(frozen=True, slots=True)
class SnapshotConfig:
    enabled: bool = True
    root: str = "/workspace/output/snapshot"
    image_format: str = "jpg"
    jpeg_quality: int = 92
    person_decision_delay_sec: float = 2.5
    behavior_cooldown_sec: float = 10.0
    max_pending_tracks: int = 10000
    padding_x_ratio: float = 0.20
    padding_top_ratio: float = 0.20
    upper_body_height_ratio: float = 0.75
    min_crop_width: int = 16
    min_crop_height: int = 32
    min_visible_ratio: float = 0.50

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SnapshotConfig:
        raw_person_crop = data.get("person_crop", {})
        person_crop = raw_person_crop if isinstance(raw_person_crop, Mapping) else {}
        result = cls(
            enabled=bool(data.get("enabled", True)),
            root=str(data.get("root", data.get("path", "/workspace/output/snapshot"))),
            image_format=str(data.get("image_format", "jpg")).lower(),
            jpeg_quality=int(data.get("jpeg_quality", 92)),
            person_decision_delay_sec=float(data.get("person_decision_delay_sec", 2.5)),
            behavior_cooldown_sec=float(data.get("behavior_cooldown_sec", 10.0)),
            max_pending_tracks=int(data.get("max_pending_tracks", 10000)),
            padding_x_ratio=float(person_crop.get("padding_x_ratio", 0.20)),
            padding_top_ratio=float(person_crop.get("padding_top_ratio", 0.20)),
            upper_body_height_ratio=float(person_crop.get("upper_body_height_ratio", 0.75)),
            min_crop_width=int(person_crop.get("min_crop_width", 16)),
            min_crop_height=int(person_crop.get("min_crop_height", 32)),
            min_visible_ratio=float(person_crop.get("min_visible_ratio", 0.50)),
        )
        if result.image_format not in {"jpg", "jpeg", "png"}:
            raise ConfigurationError("snapshot.image_format 仅支持 jpg/jpeg/png")
        if not 1 <= result.jpeg_quality <= 100:
            raise ConfigurationError("snapshot.jpeg_quality 必须在 1 到 100 之间")
        if result.max_pending_tracks < 1:
            raise ConfigurationError("snapshot.max_pending_tracks 必须为正整数")
        if not 0.0 <= result.padding_x_ratio <= 1.0:
            raise ConfigurationError("person_crop.padding_x_ratio 必须在 0 到 1 之间")
        if not 0.0 <= result.padding_top_ratio <= 1.0:
            raise ConfigurationError("person_crop.padding_top_ratio 必须在 0 到 1 之间")
        if not 0.1 <= result.upper_body_height_ratio <= 1.0:
            raise ConfigurationError("person_crop.upper_body_height_ratio 必须在 0.1 到 1 之间")
        if result.min_crop_width < 1 or result.min_crop_height < 1:
            raise ConfigurationError("person_crop 最小宽高必须为正整数")
        if not 0.0 <= result.min_visible_ratio <= 1.0:
            raise ConfigurationError("person_crop.min_visible_ratio 必须在 0 到 1 之间")
        return result


@dataclass(frozen=True, slots=True)
class OutputConfig:
    enabled: bool = True
    path: str = "/workspace/output/result.mp4"
    codec: str = "h264"
    encoder: str = "nvidia"
    bitrate: int = 8_000_000
    sync: bool = False
    events_enabled: bool = True
    events_path: str = "/workspace/output/events.jsonl"
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], snapshot_data: Mapping[str, Any]
    ) -> OutputConfig:
        result = cls(
            enabled=bool(data.get("enabled", True)),
            path=str(data.get("path", data.get("result_video", "/workspace/output/result.mp4"))),
            codec=str(data.get("codec", "h264")).lower(),
            encoder=str(data.get("encoder", "nvidia")).lower(),
            bitrate=int(data.get("bitrate", 8_000_000)),
            sync=bool(data.get("sync", False)),
            events_enabled=bool(data.get("events_enabled", True)),
            events_path=str(data.get("events_path", "/workspace/output/events.jsonl")),
            snapshot=SnapshotConfig.from_mapping(snapshot_data),
        )
        if result.codec not in {"h264", "h265"}:
            raise ConfigurationError("output.codec 仅支持 h264 或 h265")
        if result.encoder not in {"nvidia", "x264"}:
            raise ConfigurationError("output.encoder 仅支持 nvidia 或 x264")
        if result.encoder == "x264" and result.codec != "h264":
            raise ConfigurationError("output.encoder=x264 仅支持 output.codec=h264")
        _positive(result.bitrate, "output.bitrate")
        return result


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    strict_assets: bool = True
    headless: bool = True
    health_file: str = "/tmp/deepstream-ai.ready"
    log_level: str = "INFO"
    json_logs: bool = False
    startup_timeout_sec: int = 600
    shutdown_timeout_sec: int = 20
    analytics_queue_size: int = 8

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RuntimeConfig:
        result = cls(
            strict_assets=bool(data.get("strict_assets", True)),
            headless=bool(data.get("headless", True)),
            health_file=str(data.get("health_file", "/tmp/deepstream-ai.ready")),
            log_level=str(data.get("log_level", "INFO")),
            json_logs=bool(data.get("json_logs", False)),
            startup_timeout_sec=int(data.get("startup_timeout_sec", 600)),
            shutdown_timeout_sec=int(data.get("shutdown_timeout_sec", 20)),
            analytics_queue_size=int(data.get("analytics_queue_size", 8)),
        )
        _positive(result.startup_timeout_sec, "runtime.startup_timeout_sec")
        _positive(result.shutdown_timeout_sec, "runtime.shutdown_timeout_sec")
        _positive(result.analytics_queue_size, "runtime.analytics_queue_size")
        return result


@dataclass(frozen=True, slots=True)
class AppConfig:
    config_path: Path
    sources: tuple[SourceConfig, ...]
    inference: InferenceRateConfig
    pipeline: PipelineConfig
    face_recognition: FaceRecognitionConfig
    behavior: tuple[BehaviorModelConfig, ...]
    database: DatabaseConfig
    output: OutputConfig
    runtime: RuntimeConfig

    @property
    def enabled_sources(self) -> tuple[SourceConfig, ...]:
        return tuple(source for source in self.sources if source.enabled)

    def interval_for(self, target_fps: float) -> int:
        """DeepStream interval: number of batches skipped between inferences."""

        fastest_source = max(source.nominal_fps for source in self.enabled_sources)
        return max(0, round(fastest_source / target_fps) - 1)

    def resolve_path(self, value: str) -> Path:
        normalized = value.replace("\\", "/")
        posix_path = PurePosixPath(normalized)
        # All Compose bind mounts share /workspace. Mapping this prefix back to
        # the repository also makes `validate` useful on the Windows host.
        if posix_path == PurePosixPath("/workspace") or str(posix_path).startswith("/workspace/"):
            relative = posix_path.relative_to("/workspace")
            return (self.config_path.parent.parent / Path(*relative.parts)).resolve()
        path = Path(value)
        if path.is_absolute():
            return path
        if posix_path.is_absolute():
            return Path(str(posix_path))
        return (self.config_path.parent.parent / path).resolve()

    def required_assets(self) -> list[tuple[str, Path]]:
        required: list[tuple[str, Path]] = []
        for source in self.enabled_sources:
            if source.type == "file":
                required.append((f"视频源 {source.camera_id}", self.resolve_path(source.location)))
        if self.pipeline.person.enabled:
            required.append(
                ("人员检测 nvinfer 配置", self.resolve_path(self.pipeline.person.config_file))
            )
        if self.pipeline.face.enabled:
            required.append(
                ("人脸检测 nvinfer 配置", self.resolve_path(self.pipeline.face.config_file))
            )
        if self.face_recognition.enabled:
            required.append(("AdaFace 模型", self.resolve_path(self.face_recognition.model)))
        required.append(("Tracker 配置", self.resolve_path(self.pipeline.tracker.config_file)))
        if self.pipeline.tracker.library_file and (
            not self.pipeline.tracker.library_file.startswith("/opt/nvidia/")
            or Path("/opt/nvidia/deepstream").exists()
        ):
            required.append(("Tracker 库", Path(self.pipeline.tracker.library_file)))
        for model in self.behavior:
            if model.enabled:
                required.append(
                    (f"行为 {model.name} nvinfer 配置", self.resolve_path(model.config_file))
                )
                if model.model:
                    required.append((f"行为 {model.name} 模型", self.resolve_path(model.model)))
        return required

    def validate_assets(self) -> list[str]:
        missing = [f"{name}: {path}" for name, path in self.required_assets() if not path.exists()]
        if missing and self.runtime.strict_assets:
            details = "\n  - ".join(missing)
            raise AssetValidationError(
                "启用的组件缺少必要资产：\n  - "
                + details
                + "\n请运行 scripts/download-models.sh / scripts/convert-model.sh，或修正 configs/config.yaml。"
            )
        return missing


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"配置文件不存在: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"YAML 解析失败 {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError("配置文件根节点必须是对象")
    data = _expand_environment(dict(raw))

    sources_raw = data.get("sources")
    if sources_raw:
        if not isinstance(sources_raw, list):
            raise ConfigurationError("sources 必须是数组")
    else:
        source_raw = data.get("source")
        sources_raw = [source_raw] if source_raw else []
    if not sources_raw:
        raise ConfigurationError("至少需要配置一个 source 或 sources 项")
    sources = tuple(
        SourceConfig.from_mapping(_mapping(item, "source"), i) for i, item in enumerate(sources_raw)
    )
    if not any(source.enabled for source in sources):
        raise ConfigurationError("至少需要启用一个视频源")
    camera_ids = [source.camera_id for source in sources]
    if len(camera_ids) != len(set(camera_ids)):
        raise ConfigurationError("camera_id 必须唯一")

    pipeline_data = _mapping(data.get("pipeline"), "pipeline")
    person_data = _mapping(data.get("person", pipeline_data.get("person")), "person")
    face_data = _mapping(data.get("face", pipeline_data.get("face")), "face")
    tracker_data = _mapping(data.get("tracker", pipeline_data.get("tracker")), "tracker")
    streammux_data = _mapping(pipeline_data.get("streammux"), "pipeline.streammux")
    pipeline = PipelineConfig(
        streammux=StreamMuxConfig.from_mapping(streammux_data),
        tracker=TrackerConfig.from_mapping(tracker_data),
        person=InferComponentConfig.from_mapping(
            person_data, default_enabled=True, default_unique_id=1, default_label="person"
        ),
        face=InferComponentConfig.from_mapping(
            face_data, default_enabled=False, default_unique_id=2, default_label="face"
        ),
        tiler_width=int(pipeline_data.get("tiler_width", 1920)),
        tiler_height=int(pipeline_data.get("tiler_height", 1080)),
    )
    if not pipeline.person.enabled:
        raise ConfigurationError("人员检测是主干组件，person.enabled 必须为 true")
    component_ids = [pipeline.person.unique_id]
    if pipeline.face.enabled:
        component_ids.append(pipeline.face.unique_id)

    behavior_data = _mapping(data.get("behavior"), "behavior")
    behavior_models = tuple(
        BehaviorModelConfig.from_mapping(name, _mapping(item, f"behavior.{name}"), 10 + index)
        for index, (name, item) in enumerate(behavior_data.items())
    )
    component_ids.extend(model.unique_id for model in behavior_models if model.enabled)
    if len(component_ids) != len(set(component_ids)):
        raise ConfigurationError("所有已启用 nvinfer 组件的 unique_id 必须唯一")

    inference = InferenceRateConfig.from_mapping(_mapping(data.get("inference"), "inference"))
    face_recognition = FaceRecognitionConfig.from_mapping(
        _mapping(data.get("face_recognition", data.get("adaface")), "face_recognition")
    )
    if face_recognition.enabled and not pipeline.face.enabled:
        raise ConfigurationError("启用 AdaFace 前必须启用 face.enabled")
    if (
        face_recognition.enabled
        and face_recognition.require_landmarks
        and pipeline.face.landmark_source == "none"
    ):
        raise ConfigurationError("AdaFace require_landmarks=true 时必须配置 face.landmark_source")
    database = DatabaseConfig.from_mapping(_mapping(data.get("database"), "database"))
    if face_recognition.enabled and not database.enabled:
        raise ConfigurationError("启用 AdaFace 身份比对时必须启用 database.enabled")
    if face_recognition.enabled and not database.dsn:
        raise ConfigurationError(
            f"启用 AdaFace 时必须设置 database.dsn 或环境变量 {database.dsn_env}"
        )
    output_data = _mapping(data.get("output"), "output")
    snapshot_data = dict(_mapping(data.get("snapshot", output_data.get("snapshot")), "snapshot"))
    person_crop_data = _mapping(data.get("person_crop"), "person_crop")
    if person_crop_data:
        snapshot_data["person_crop"] = person_crop_data
    output = OutputConfig.from_mapping(output_data, snapshot_data)
    runtime = RuntimeConfig.from_mapping(_mapping(data.get("runtime"), "runtime"))
    return AppConfig(
        config_path=config_path,
        sources=sources,
        inference=inference,
        pipeline=pipeline,
        face_recognition=face_recognition,
        behavior=behavior_models,
        database=database,
        output=output,
        runtime=runtime,
    )
