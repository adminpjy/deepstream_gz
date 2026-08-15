#!/usr/bin/env python3
"""Convert the user's local YOLO behavior models into DeepStream/TensorRT assets.

Expected local inputs under ``models/``:

- yolo11n.onnx : standard COCO detector reused for eating/drinking proxy evidence
- smoking.pt    : smoking detector
- fire.onnx     : fire/flame detector (full-frame config generated for later use)

The model weights remain local and ignored by Git. This tool is intended to run
inside the repository's ``model-converter`` Docker target so TensorRT engines are
built with the same DeepStream/TensorRT stack used in production.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PARSER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvdsinfer_custom_yolo_dynamic.so"
PARSER_FUNC = "NvDsInferParseCustomYoloDynamic"
EAT_DRINK_PARSER_FUNC = "NvDsInferParseCustomYoloEatDrinkCoco"
EAT_DRINK_BUSINESS_LABELS = ("eating", "drinking")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    source_name: str
    onnx_name: str
    engine_name: str
    labels_name: str
    config_name: str
    unique_id: int
    fallback_labels: tuple[str, ...]
    scope: str  # person | frame


SPECS = (
    ModelSpec(
        name="eat_drink",
        source_name="yolo11n.onnx",
        onnx_name="yolo11n.onnx",
        engine_name="yolo11n.engine",
        labels_name="yolo11n.labels.txt",
        config_name="eat-drink.txt",
        unique_id=11,
        # This fallback is used only when model metadata is absent. Production
        # expects the actual local model to expose standard COCO names.
        fallback_labels=EAT_DRINK_BUSINESS_LABELS,
        scope="person",
    ),
    ModelSpec(
        name="smoking",
        source_name="smoking.pt",
        onnx_name="smoking.onnx",
        engine_name="smoking.engine",
        labels_name="smoking.labels.txt",
        config_name="smoking.txt",
        unique_id=12,
        fallback_labels=("smoking",),
        scope="person",
    ),
    ModelSpec(
        name="fire",
        source_name="fire.onnx",
        onnx_name="fire.onnx",
        engine_name="fire.engine",
        labels_name="fire.labels.txt",
        config_name="fire.txt",
        unique_id=14,
        fallback_labels=("fire",),
        scope="frame",
    ),
)


@dataclass(frozen=True, slots=True)
class OnnxContract:
    input_name: str
    input_shape: tuple[int | None, int, int, int]
    labels: tuple[str, ...]
    dynamic_batch: bool
    output_shape: tuple[int | None, ...]

    @property
    def batch_size(self) -> int:
        return 16 if self.dynamic_batch else int(self.input_shape[0] or 1)


_CANONICAL_LABELS = {
    "eat": "eating",
    "eating": "eating",
    "food": "eating",
    "drink": "drinking",
    "drinking": "drinking",
    "smoke": "smoking",
    "smoking": "smoking",
    "cigarette": "smoking",
    "fire": "fire",
    "flame": "fire",
}


def _normalize_label(value: str) -> str:
    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
    return _CANONICAL_LABELS.get(normalized, normalized)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _find_trtexec() -> str:
    candidates = [
        shutil.which("trtexec"),
        "/usr/src/tensorrt/bin/trtexec",
        "/opt/nvidia/deepstream/deepstream/bin/trtexec",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("trtexec not found; run with the repository model-converter image")


def _onnx_metadata_names(model: Any) -> tuple[str, ...]:
    properties = {item.key: item.value for item in model.metadata_props}
    raw = properties.get("names", "").strip()
    if not raw:
        return ()
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return ()
    if isinstance(value, dict):
        try:
            ordered = [value[key] for key in sorted(value, key=lambda item: int(item))]
        except (TypeError, ValueError):
            ordered = list(value.values())
    elif isinstance(value, (list, tuple)):
        ordered = list(value)
    else:
        return ()
    return tuple(_normalize_label(str(item)) for item in ordered)


def _dim_value(dim: Any) -> int | None:
    value = int(getattr(dim, "dim_value", 0) or 0)
    return value if value > 0 else None


def _inspect_onnx(path: Path, spec: ModelSpec) -> OnnxContract:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - converter image always provides it
        raise RuntimeError("onnx package is required in the model-converter image") from exc

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise ValueError(f"{spec.name}: raw YOLO detect ONNX must have exactly one input")
    if len(model.graph.output) != 1:
        raise ValueError(
            f"{spec.name}: expected one raw YOLO output; embedded NMS/end-to-end models are unsupported"
        )

    model_input = model.graph.input[0]
    dims = model_input.type.tensor_type.shape.dim
    if len(dims) != 4:
        raise ValueError(f"{spec.name}: ONNX input must be NCHW rank 4")
    batch = _dim_value(dims[0])
    channels = _dim_value(dims[1])
    height = _dim_value(dims[2])
    width = _dim_value(dims[3])
    if channels != 3 or height is None or width is None:
        raise ValueError(
            f"{spec.name}: expected input [N,3,H,W] with fixed H/W, got "
            f"[{batch or 'dynamic'},{channels},{height},{width}]"
        )

    output_dims = model.graph.output[0].type.tensor_type.shape.dim
    if len(output_dims) != 3:
        raise ValueError(
            f"{spec.name}: raw YOLO output must be rank 3 [N,4+C,rows] or [N,rows,4+C]"
        )
    output_shape = tuple(_dim_value(dim) for dim in output_dims)
    metadata_labels = _onnx_metadata_names(model)
    labels = metadata_labels or spec.fallback_labels
    expected_channels = 4 + len(labels)
    static_tail = {value for value in output_shape[1:] if value is not None}
    if expected_channels not in static_tail:
        fallback_channels = 4 + len(spec.fallback_labels)
        if metadata_labels or fallback_channels not in static_tail:
            raise ValueError(
                f"{spec.name}: output shape {output_shape} does not expose 4+C={expected_channels}; "
                "the supplied model is not compatible with the repository raw YOLO11 parser"
            )
        labels = spec.fallback_labels

    return OnnxContract(
        input_name=model_input.name,
        input_shape=(batch, channels, height, width),
        labels=tuple(labels),
        dynamic_batch=batch is None,
        output_shape=output_shape,
    )


def _pt_names(model: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    raw = getattr(model, "names", None)
    if raw is None:
        raw = getattr(getattr(model, "model", None), "names", None)
    if isinstance(raw, dict):
        try:
            values = [raw[key] for key in sorted(raw, key=lambda item: int(item))]
        except (TypeError, ValueError):
            values = list(raw.values())
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = list(fallback)
    return tuple(_normalize_label(str(value)) for value in values)


def _export_pt(source: Path, target: Path, spec: ModelSpec, imgsz: int) -> tuple[str, ...]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - converter image always provides it
        raise RuntimeError(
            "Ultralytics is required for .pt export; use docker compose --profile tools run --rm model-converter"
        ) from exc

    model = YOLO(str(source), task="detect")
    labels = _pt_names(model, spec.fallback_labels)
    if not labels:
        raise ValueError(f"{spec.name}: checkpoint contains no class names")
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=8,
        dynamic=True,
        simplify=False,
        opset=17,
        nms=False,
        device="cpu",
    )
    exported_path = Path(str(exported)).resolve()
    if not exported_path.is_file():
        raise RuntimeError(f"{spec.name}: Ultralytics did not create ONNX: {exported_path}")
    if exported_path != target.resolve():
        shutil.copy2(exported_path, target)
    return labels


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    contract: OnnxContract,
    *,
    device: int,
    workspace_mib: int,
    precision: str,
) -> None:
    trtexec = _find_trtexec()
    _, channels, height, width = contract.input_shape
    args = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mib}",
        f"--device={device}",
        "--skipInference",
    ]
    if contract.dynamic_batch:
        args.extend(
            [
                f"--minShapes={contract.input_name}:1x{channels}x{height}x{width}",
                f"--optShapes={contract.input_name}:8x{channels}x{height}x{width}",
                f"--maxShapes={contract.input_name}:16x{channels}x{height}x{width}",
            ]
        )
    if precision == "fp16":
        args.append("--fp16")
    elif precision != "fp32":
        raise ValueError(f"unsupported precision: {precision}")
    subprocess.run(args, check=True)
    if not engine_path.is_file() or engine_path.stat().st_size <= 0:
        raise RuntimeError(f"TensorRT engine was not created: {engine_path}")


def _config_reference(config_path: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), config_path.parent.resolve()).replace(os.sep, "/")


def _business_labels(spec: ModelSpec, contract: OnnxContract) -> tuple[str, ...]:
    if spec.name == "eat_drink":
        return EAT_DRINK_BUSINESS_LABELS
    return contract.labels


def _parser_func(spec: ModelSpec) -> str:
    return EAT_DRINK_PARSER_FUNC if spec.name == "eat_drink" else PARSER_FUNC


def _write_nvinfer_config(
    spec: ModelSpec,
    contract: OnnxContract,
    *,
    onnx_path: Path,
    engine_path: Path,
    labels_path: Path,
    config_path: Path,
    precision: str,
) -> None:
    _, _channels, height, width = contract.input_shape
    network_mode = 2 if precision == "fp16" else 0
    business_labels = _business_labels(spec, contract)
    properties = [
        ("gpu-id", "0"),
        ("net-scale-factor", "0.00392156862745098"),
        ("model-color-format", "0"),
        ("onnx-file", _config_reference(config_path, onnx_path)),
        ("model-engine-file", _config_reference(config_path, engine_path)),
        ("labelfile-path", _config_reference(config_path, labels_path)),
        ("infer-dims", f"3;{height};{width}"),
        ("batch-size", str(contract.batch_size)),
        ("network-mode", str(network_mode)),
        ("num-detected-classes", str(len(business_labels))),
        ("gie-unique-id", str(spec.unique_id)),
        ("network-type", "0"),
        ("process-mode", "2" if spec.scope == "person" else "1"),
    ]
    if spec.scope == "person":
        properties.extend(
            [
                ("operate-on-gie-id", "1"),
                ("operate-on-class-ids", "0"),
                ("secondary-reinfer-interval", "1"),
            ]
        )
    else:
        properties.append(("interval", "0"))
    properties.extend(
        [
            ("cluster-mode", "2"),
            ("maintain-aspect-ratio", "1"),
            ("symmetric-padding", "1"),
            ("scaling-filter", "1"),
            ("parse-bbox-func-name", _parser_func(spec)),
            ("disable-output-host-copy", "0"),
            ("custom-lib-path", PARSER_LIB),
        ]
    )
    lines = ["# Generated by scripts/prepare-local-behavior-models.py", "[property]"]
    lines.extend(f"{key}={value}" for key, value in properties)
    lines.extend(
        [
            "",
            "[class-attrs-all]",
            "nms-iou-threshold=0.65",
            "pre-cluster-threshold=0.35",
            "topk=100",
            "",
        ]
    )
    _atomic_text(config_path, "\n".join(lines))


def _convert_one(
    spec: ModelSpec,
    *,
    model_root: Path,
    config_root: Path,
    device: int,
    precision: str,
    workspace_mib: int,
    imgsz: int,
    force: bool,
) -> dict[str, Any]:
    source = model_root / spec.source_name
    onnx_path = model_root / spec.onnx_name
    engine_path = model_root / spec.engine_name
    labels_path = model_root / spec.labels_name
    config_path = config_root / spec.config_name
    if not source.is_file():
        raise FileNotFoundError(f"missing local model: {source}")
    if not force:
        existing = [path for path in (engine_path, labels_path, config_path) if path.exists()]
        if spec.source_name.endswith(".pt") and onnx_path.exists():
            existing.append(onnx_path)
        if existing:
            raise FileExistsError(
                "outputs already exist; rerun converter with --force after review: "
                + ", ".join(str(path) for path in existing)
            )

    exported_labels: tuple[str, ...] = ()
    if source.suffix.lower() == ".pt":
        exported_labels = _export_pt(source, onnx_path, spec, imgsz)
    elif source.resolve() != onnx_path.resolve():
        shutil.copy2(source, onnx_path)

    contract = _inspect_onnx(onnx_path, spec)
    if exported_labels and exported_labels != contract.labels:
        raise ValueError(
            f"{spec.name}: checkpoint labels {exported_labels} != exported ONNX labels {contract.labels}"
        )
    business_labels = _business_labels(spec, contract)
    _atomic_text(labels_path, "\n".join(business_labels) + "\n")
    engine_path.unlink(missing_ok=True)
    _build_engine(
        onnx_path,
        engine_path,
        contract,
        device=device,
        workspace_mib=workspace_mib,
        precision=precision,
    )
    _write_nvinfer_config(
        spec,
        contract,
        onnx_path=onnx_path,
        engine_path=engine_path,
        labels_path=labels_path,
        config_path=config_path,
        precision=precision,
    )
    result = {
        "name": spec.name,
        "scope": spec.scope,
        "source": str(source),
        "sourceSha256": _sha256(source),
        "onnx": str(onnx_path),
        "onnxSha256": _sha256(onnx_path),
        "engine": str(engine_path),
        "engineSha256": _sha256(engine_path),
        "labelsFile": str(labels_path),
        "labels": list(business_labels),
        "nvinferConfig": str(config_path),
        "dynamicBatch": contract.dynamic_batch,
        "nvinferBatchSize": contract.batch_size,
        "inputName": contract.input_name,
        "inputShape": [value if value is not None else -1 for value in contract.input_shape],
        "outputShape": [value if value is not None else -1 for value in contract.output_shape],
        "parser": _parser_func(spec),
        "precision": precision,
        "productionSessionIntegrated": spec.scope == "person" and spec.name != "fire",
    }
    if spec.name == "eat_drink":
        result["sourceLabels"] = list(contract.labels)
        result["mode"] = "person_crop_coco_proxy"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert local YOLO behavior models for DeepStream")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--config-root", default="configs/nvinfer")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(spec.name for spec in SPECS),
        help="Convert only selected model(s); may be repeated",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_root = Path(args.model_root).expanduser().resolve()
    config_root = Path(args.config_root).expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    if args.device < 0:
        raise SystemExit("--device must be non-negative")
    if args.workspace_mib <= 0 or args.imgsz <= 0:
        raise SystemExit("--workspace-mib and --imgsz must be positive")
    if not Path(PARSER_LIB).is_file():
        raise SystemExit(
            f"DeepStream YOLO parser not found at {PARSER_LIB}; use the repository model-converter image"
        )

    selected = [spec for spec in SPECS if not args.only or spec.name in set(args.only)]
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for spec in selected:
        print(f"\n=== Converting {spec.name}: {spec.source_name} ===", flush=True)
        try:
            result = _convert_one(
                spec,
                model_root=model_root,
                config_root=config_root,
                device=args.device,
                precision=args.precision,
                workspace_mib=args.workspace_mib,
                imgsz=args.imgsz,
                force=args.force,
            )
        except Exception as exc:
            failures.append(f"{spec.name}: {type(exc).__name__}: {exc}")
            print(f"FAILED {failures[-1]}", file=sys.stderr, flush=True)
            continue
        results.append(result)
        print(
            f"READY {spec.name}: engine={result['engine']} labels={result['labels']} "
            f"batch={result['nvinferBatchSize']}",
            flush=True,
        )

    manifest = {
        "schemaVersion": 1,
        "generator": "scripts/prepare-local-behavior-models.py",
        "models": results,
        "failures": failures,
    }
    manifest_path = model_root / "deepstream-local-models.manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"\nManifest: {manifest_path}")
    if failures:
        print("\nConversion completed with failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2
    print("\nAll requested local models are DeepStream-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
