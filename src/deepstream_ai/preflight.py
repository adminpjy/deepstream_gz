"""Model/config preflight checks performed before allocating GPU resources."""

from __future__ import annotations

import configparser
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from deepstream_ai.config import AppConfig
from deepstream_ai.errors import AssetValidationError, ConfigurationError

_MODEL_KEYS = (
    "onnx-file",
    "tlt-encoded-model",
    "model-file",
    "uff-file",
    "proto-file",
)
_DIRECT_FILE_KEYS = (
    "labelfile-path",
    "custom-lib-path",
    "int8-calib-file",
    "mean-file",
)


@dataclass(frozen=True, slots=True)
class NvinferAssetReport:
    config_path: Path
    engine_path: Path | None
    source_models: tuple[Path, ...]
    auxiliary_files: tuple[Path, ...]
    missing: tuple[str, ...]


def _read_nvinfer(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str  # type: ignore[method-assign]
    try:
        with path.open("r", encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, configparser.Error) as exc:
        raise ConfigurationError(f"无法读取 nvinfer 配置 {path}: {exc}") from exc
    return parser


def _section(path: Path, requested: str) -> dict[str, str]:
    parser = _read_nvinfer(path)
    section_name = next((name for name in parser.sections() if name.lower() == requested), None)
    if section_name is None:
        return {}
    return {key.lower(): value.strip() for key, value in parser[section_name].items()}


def _property_section(path: Path) -> dict[str, str]:
    parser = _read_nvinfer(path)
    section_name = next((name for name in parser.sections() if name.lower() == "property"), None)
    if section_name is None:
        raise ConfigurationError(f"nvinfer 配置缺少 [property] 节: {path}")
    return {key.lower(): value.strip() for key, value in parser[section_name].items()}


def _resolve_reference(config: AppConfig, nvinfer_path: Path, value: str) -> Path:
    value = value.strip().strip('"').strip("'")
    if value.replace("\\", "/").startswith("/workspace/"):
        return config.resolve_path(value)
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (nvinfer_path.parent / candidate).resolve()


def inspect_nvinfer_config(config: AppConfig, value: str) -> NvinferAssetReport:
    path = config.resolve_path(value)
    if not path.is_file():
        return NvinferAssetReport(path, None, (), (), (f"nvinfer 配置不存在: {path}",))
    properties = _property_section(path)
    engine = (
        _resolve_reference(config, path, properties["model-engine-file"])
        if properties.get("model-engine-file")
        else None
    )
    sources = tuple(
        _resolve_reference(config, path, properties[key])
        for key in _MODEL_KEYS
        if properties.get(key)
    )
    auxiliary_items = tuple(
        (key, _resolve_reference(config, path, properties[key]), properties[key])
        for key in _DIRECT_FILE_KEYS
        if properties.get(key)
        and (key != "int8-calib-file" or properties.get("network-mode", "0") == "1")
    )
    auxiliaries = tuple(item for _, item, _ in auxiliary_items)
    missing: list[str] = []
    try:
        configured_gpu = int(properties.get("gpu-id", "0"))
    except ValueError:
        missing.append(f"{path}: gpu-id 必须是整数")
    else:
        if configured_gpu != 0:
            missing.append(f"{path}: 当前运行时仅支持 gpu-id=0，检测到 gpu-id={configured_gpu}")
    # An existing serialized engine is sufficient. If it is absent, nvinfer
    # needs at least one source-model artifact from which to build it.
    if engine is None and not sources:
        missing.append(f"{path}: 未配置 model-engine-file 或源模型文件")
    elif engine is None or not engine.is_file():
        existing_sources = [source for source in sources if source.is_file()]
        if not existing_sources:
            expected = ", ".join(str(item) for item in sources) or str(engine)
            missing.append(f"{path}: TensorRT engine 不存在且无可用源模型（{expected}）")
    for key, auxiliary, raw_reference in auxiliary_items:
        # Calibration data is unnecessary when a prebuilt engine is present.
        if key == "int8-calib-file" and engine is not None and engine.is_file():
            continue
        # Container-provided DeepStream libraries cannot be checked from the
        # Windows host; `doctor` validates their plugin factories in-container.
        if (
            raw_reference.strip()
            .strip('"')
            .strip("'")
            .replace("\\", "/")
            .startswith("/opt/nvidia/")
            and not Path("/opt/nvidia/deepstream").exists()
        ):
            continue
        if not auxiliary.is_file():
            missing.append(f"{path}: 引用文件不存在: {auxiliary}")
    return NvinferAssetReport(path, engine, sources, auxiliaries, tuple(missing))


def _tracker_section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _tracker_int(section: Mapping[str, Any], name: str, default: int = 0) -> int:
    try:
        return int(section.get(name, default))
    except (TypeError, ValueError):
        return default


def validate_tracker_backend(config: AppConfig) -> str | None:
    """Ensure the declarative backend agrees with NvMultiObjectTracker YAML.

    DeepStream uses one low-level library for NvDCF, NvSORT and NvDeepSORT;
    the YAML modules select the actual algorithm. Treating ``backend`` as a
    runtime switch without checking that file would therefore be misleading.
    """

    path = config.resolve_path(config.pipeline.tracker.config_file)
    if not path.is_file():
        return None  # The ordinary required-assets check reports this case.
    try:
        text = path.read_text(encoding="utf-8")
        # NVIDIA sample files commonly start with the OpenCV-style
        # ``%YAML:1.0`` directive, which PyYAML does not accept as YAML 1.2.
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("%YAML:"):
            text = "\n".join(lines[1:])
        raw = yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError) as exc:
        return f"Tracker 配置无法解析 {path}: {exc}"
    if not isinstance(raw, Mapping):
        return f"Tracker 配置根节点必须是对象: {path}"

    visual_type = _tracker_int(_tracker_section(raw, "VisualTracker"), "visualTrackerType")
    reid_type = _tracker_int(_tracker_section(raw, "ReID"), "reidType")
    state_type = _tracker_int(_tracker_section(raw, "StateEstimator"), "stateEstimatorType")
    backend = config.pipeline.tracker.backend
    if backend == "nvdcf" and visual_type not in {1, 2}:
        return (
            f"tracker.backend=nvdcf 与 {path} 不一致：VisualTracker.visualTrackerType 必须为 1 或 2"
        )
    if backend == "nvsort" and (visual_type != 0 or reid_type in {1, 3} or state_type == 0):
        return (
            f"tracker.backend=nvsort 与 {path} 不一致：需启用 StateEstimator，"
            "且不能启用 NvDCF VisualTracker 或 NvDeepSORT ReID"
        )
    if backend == "deepsort" and reid_type not in {1, 3}:
        return f"tracker.backend=deepsort 与 {path} 不一致：ReID.reidType 必须为 1 或 3"
    return None


def validate_person_detector(config: AppConfig, report: NvinferAssetReport) -> tuple[str, ...]:
    """Cross-check a declared PeopleNet detector against its effective files."""

    person = config.pipeline.person
    if person.detector_type != "peoplenet" or not report.config_path.is_file():
        return ()
    properties = _property_section(report.config_path)
    failures: list[str] = []
    if not properties.get("onnx-file"):
        failures.append("PeopleNet 必须配置可审计的 onnx-file")
    else:
        onnx_path = _resolve_reference(config, report.config_path, properties["onnx-file"])
        if not onnx_path.is_file():
            failures.append(f"PeopleNet ONNX 不存在: {onnx_path}")
    label_reference = properties.get("labelfile-path")
    if not label_reference:
        failures.append("PeopleNet 必须配置 labelfile-path")
        actual_classes: tuple[tuple[str, int], ...] = ()
    else:
        labels_path = _resolve_reference(config, report.config_path, label_reference)
        try:
            labels = tuple(
                line.strip().lower()
                for line in labels_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError as exc:
            failures.append(f"PeopleNet labels 无法读取 {labels_path}: {exc}")
            labels = ()
        if len(labels) != len(set(labels)):
            failures.append(f"PeopleNet labels 包含重复名称: {labels_path}")
        actual_classes = tuple((name, class_id) for class_id, name in enumerate(labels))
    if actual_classes != person.people_classes:
        failures.append(
            "people_classes 与实际 labels 顺序不一致: "
            f"configured={dict(person.people_classes)} actual={dict(actual_classes)}"
        )
    try:
        detected_classes = int(properties.get("num-detected-classes", "0"))
    except ValueError:
        detected_classes = 0
    if detected_classes != len(actual_classes):
        failures.append(
            "num-detected-classes 与实际 labels 数量不一致: "
            f"{detected_classes} != {len(actual_classes)}"
        )
    if properties.get("infer-dims") != "3;544;960":
        failures.append("当前 PeopleNet v2.6.3 必须配置 infer-dims=3;544;960")
    if properties.get("cluster-mode") != "3":
        failures.append("PeopleNet + NvDCF 必须使用 cluster-mode=3 Hybrid 聚类")
    if properties.get("maintain-aspect-ratio") != "1":
        failures.append("PeopleNet + NvDCF 必须配置 maintain-aspect-ratio=1")
    output_names = tuple(
        item.strip() for item in properties.get("output-blob-names", "").split(";") if item.strip()
    )
    expected_output_names = ("output_cov/Sigmoid:0", "output_bbox/BiasAdd:0")
    if output_names != expected_output_names:
        failures.append(
            "PeopleNet output-blob-names 与实际 ONNX 输出顺序不一致: "
            f"configured={output_names} expected={expected_output_names}"
        )
    if properties.get("custom-lib-path") or properties.get("parse-bbox-func-name"):
        failures.append("PeopleNet DetectNet_v2 不得继续加载 YOLO custom parser")
    return tuple(failures)


def iter_enabled_nvinfer_configs(config: AppConfig) -> Iterable[tuple[str, str]]:
    yield "person", config.pipeline.person.config_file
    if config.pipeline.face.enabled:
        yield "face", config.pipeline.face.config_file
    for model in config.behavior:
        if model.enabled:
            yield f"behavior.{model.name}", model.config_file


def inspect_assets(
    config: AppConfig,
) -> tuple[tuple[NvinferAssetReport, ...], tuple[str, ...]]:
    """Return all asset and semantic failures without changing strictness."""

    failures = [f"{name}: {path}" for name, path in config.required_assets() if not path.exists()]
    tracker_error = validate_tracker_backend(config)
    if tracker_error:
        failures.append(tracker_error)
    reports: list[NvinferAssetReport] = []
    for label, value in iter_enabled_nvinfer_configs(config):
        if not value:
            failures.append(f"{label}: config_file 不能为空")
            continue
        report = inspect_nvinfer_config(config, value)
        reports.append(report)
        failures.extend(f"{label}: {message}" for message in report.missing)
        if label == "person":
            failures.extend(
                f"person: {message}" for message in validate_person_detector(config, report)
            )
        if (
            label == "face"
            and config.pipeline.face.landmark_source == "tensor"
            and report.config_path.is_file()
        ):
            properties = _property_section(report.config_path)
            if properties.get("output-tensor-meta") != "1":
                failures.append("face: landmark_source=tensor 要求 output-tensor-meta=1")
            if properties.get("disable-output-host-copy", "0") == "1":
                failures.append(
                    "face: SCRFD tensor landmark 要求 host output copy，不能设 disable-output-host-copy=1"
                )
            if properties.get("num-detected-classes") != "1":
                failures.append("face: SCRFD nvinfer 必须配置 num-detected-classes=1")
            if not properties.get("custom-lib-path") or not properties.get("parse-bbox-func-name"):
                failures.append(
                    "face: SCRFD tensor 路径要求 custom-lib-path 与 parse-bbox-func-name"
                )
            attributes = _section(report.config_path, "class-attrs-all")
            try:
                parser_threshold = float(attributes.get("pre-cluster-threshold", "nan"))
            except ValueError:
                parser_threshold = float("nan")
            if parser_threshold != config.pipeline.face.landmark_threshold:
                failures.append(
                    "face: landmark_threshold 必须与 nvinfer pre-cluster-threshold 完全一致 "
                    f"({config.pipeline.face.landmark_threshold} != {parser_threshold})"
                )
        if label.startswith("behavior."):
            name = label.split(".", 1)[1]
            model = next(item for item in config.behavior if item.name == name)
            declared = config.resolve_path(model.model) if model.model else None
            if (
                declared is not None
                and report.engine_path is not None
                and declared.resolve() != report.engine_path.resolve()
            ):
                failures.append(
                    f"{label}: YAML model={declared} 与 nvinfer model-engine-file={report.engine_path} 不一致"
                )
    return tuple(reports), tuple(failures)


def validate_assets(config: AppConfig) -> tuple[NvinferAssetReport, ...]:
    """Validate enabled assets, raising before GPU allocation in strict mode."""

    reports, failures = inspect_assets(config)
    if failures and config.runtime.strict_assets:
        raise AssetValidationError("启动预检发现缺失或无效资产：\n  - " + "\n  - ".join(failures))
    return reports


__all__ = [
    "NvinferAssetReport",
    "inspect_assets",
    "inspect_nvinfer_config",
    "iter_enabled_nvinfer_configs",
    "validate_assets",
    "validate_person_detector",
]
