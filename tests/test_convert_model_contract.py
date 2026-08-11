from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert-model.sh"


def _bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = "/".join(resolved.parts[1:])
    return f"/mnt/{drive}/{tail}"


def _valid_contract(checkpoint: bytes = b"test-only-placeholder") -> dict[str, object]:
    return {
        "schema_version": 2,
        "name": "smoking",
        "architecture": "yolo11",
        "task": "detect",
        "checkpoint": {"sha256": hashlib.sha256(checkpoint).hexdigest()},
        "exporter": {"name": "ultralytics", "version": "8.3.0"},
        "input": {
            "name": "images",
            "shape": [-1, 3, 640, 640],
            "layout": "NCHW",
            "color_format": "RGB",
            "scale_factor": 1 / 255,
            "dynamic_batch": True,
            "profile": {
                "min": [1, 3, 640, 640],
                "opt": [8, 3, 640, 640],
                "max": [16, 3, 640, 640],
            },
        },
        "onnx": {"opset": 17, "raw_output": True},
        "labels": ["smoking"],
        "deepstream": {
            "network_type": 0,
            "cluster_mode": 2,
            "process_mode": 2,
            "operate_on_gie_id": 1,
            "operate_on_class_ids": [0],
            "batch_size": 16,
            "num_detected_classes": 1,
            "gie_unique_id": 11,
            "maintain_aspect_ratio": True,
            "symmetric_padding": True,
            "pre_cluster_threshold": 0.5,
            "nms_iou_threshold": 0.65,
            "topk": 100,
            "parser_contract": "ultralytics-yolo8-9-11-detect-raw-v1",
            "custom_lib_path": (
                "/opt/nvidia/deepstream/deepstream/lib/libnvdsinfer_custom_yolo_dynamic.so"
            ),
            "parse_bbox_func_name": "NvDsInferParseCustomYoloDynamic",
        },
        "calibration": {
            "required_for_int8": True,
            "cache_file": "/workspace/models/smoking.calib",
            "dataset_id": "approved-v1",
            "preprocessing_id": "rgb-letterbox-v1",
        },
    }


def _validate(tmp_path: Path, contract: dict[str, object]) -> subprocess.CompletedProcess[str]:
    checkpoint = tmp_path / "smoking.pt"
    checkpoint.write_bytes(b"test-only-placeholder")
    metadata = tmp_path / "smoking.metadata.json"
    metadata.write_text(json.dumps(contract), encoding="utf-8")
    arguments = [
        _bash_path(SCRIPT),
        "--input",
        _bash_path(checkpoint),
        "--metadata",
        _bash_path(metadata),
        "--onnx",
        _bash_path(tmp_path / "smoking.onnx"),
        "--engine",
        _bash_path(tmp_path / "smoking.engine"),
        "--labels-file",
        _bash_path(tmp_path / "smoking.labels.txt"),
        "--nvinfer-config",
        _bash_path(tmp_path / "smoking.txt"),
        "--validate-only",
    ]
    command = (
        ["wsl.exe", "-d", "Ubuntu-22.04", "--", "bash", *arguments]
        if os.name == "nt"
        else ["bash", *arguments]
    )
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _fake_conversion_environment(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    binaries = tmp_path / "fake-bin"
    binaries.mkdir()
    yolo_log = tmp_path / "yolo.args"
    trt_log = tmp_path / "trtexec.args"
    _write_executable(
        binaries / "yolo",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "${YOLO_ARGS_LOG}"
model=''
for argument in "$@"; do
  case "${argument}" in model=*) model="${argument#model=}" ;; esac
done
[[ -n "${model}" ]]
printf 'fake-onnx' > "${model%.pt}.onnx"
""",
    )
    _write_executable(
        binaries / "trtexec",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "${TRT_ARGS_LOG}"
engine=''
for argument in "$@"; do
  case "${argument}" in --saveEngine=*) engine="${argument#--saveEngine=}" ;; esac
done
[[ -n "${engine}" ]]
printf 'fake-engine' > "${engine}"
""",
    )

    python_modules = tmp_path / "fake-python"
    onnx_package = python_modules / "onnx"
    onnx_package.mkdir(parents=True)
    (onnx_package / "__init__.py").write_text(
        """from types import SimpleNamespace as NS

def _dim(value=0, parameter=''):
    return NS(dim_value=value, dim_param=parameter)

def _value(name, dimensions):
    return NS(name=name, type=NS(tensor_type=NS(shape=NS(dim=dimensions))))

def load(_path):
    graph = NS(
        input=[_value('images', [_dim(parameter='batch'), _dim(3), _dim(parameter='height'), _dim(parameter='width')])],
        output=[_value('output0', [_dim(parameter='batch'), _dim(5), _dim(8400)])],
    )
    properties = [
        NS(key='task', value='detect'),
        NS(key='version', value='8.3.0'),
        NS(key='names', value="{0: 'smoking'}"),
        NS(key='description', value='Ultralytics YOLO11n behavior model'),
    ]
    return NS(graph=graph, metadata_props=properties)

checker = NS(check_model=lambda _model: None)
""",
        encoding="utf-8",
        newline="\n",
    )
    distribution = python_modules / "ultralytics-8.3.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ultralytics\nVersion: 8.3.0\n",
        encoding="utf-8",
        newline="\n",
    )
    return binaries, python_modules, yolo_log, trt_log


def test_dynamic_batch_contract_is_accepted(tmp_path: Path) -> None:
    result = _validate(tmp_path, _valid_contract())

    assert result.returncode == 0, result.stderr
    assert "dynamic batch 1/8/16, nvinfer batch 16" in result.stdout


def test_conversion_stages_profile_and_generates_matching_sgie(tmp_path: Path) -> None:
    checkpoint_payload = b"trusted-test-checkpoint"
    checkpoint = tmp_path / "smoking.pt"
    checkpoint.write_bytes(checkpoint_payload)
    metadata = tmp_path / "smoking.metadata.json"
    metadata.write_text(json.dumps(_valid_contract(checkpoint_payload)), encoding="utf-8")
    onnx_path = tmp_path / "smoking.onnx"
    engine_path = tmp_path / "smoking.engine"
    labels_path = tmp_path / "smoking.labels.txt"
    config_path = tmp_path / "smoking.txt"
    binaries, python_modules, yolo_log, trt_log = _fake_conversion_environment(tmp_path)

    arguments = [
        _bash_path(SCRIPT),
        "--input",
        _bash_path(checkpoint),
        "--metadata",
        _bash_path(metadata),
        "--onnx",
        _bash_path(onnx_path),
        "--engine",
        _bash_path(engine_path),
        "--labels-file",
        _bash_path(labels_path),
        "--nvinfer-config",
        _bash_path(config_path),
        "--precision",
        "fp16",
    ]
    if os.name == "nt":
        command = [
            "wsl.exe",
            "-d",
            "Ubuntu-22.04",
            "--",
            "env",
            f"PATH={_bash_path(binaries)}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            f"PYTHONPATH={_bash_path(python_modules)}",
            f"YOLO_ARGS_LOG={_bash_path(yolo_log)}",
            f"TRT_ARGS_LOG={_bash_path(trt_log)}",
            "bash",
            *arguments,
        ]
    else:
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{binaries}:{environment['PATH']}",
                "PYTHONPATH": str(python_modules),
                "YOLO_ARGS_LOG": str(yolo_log),
                "TRT_ARGS_LOG": str(trt_log),
            }
        )
        command = ["bash", *arguments]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=None if os.name == "nt" else environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert onnx_path.read_bytes() == b"fake-onnx"
    assert engine_path.read_bytes() == b"fake-engine"
    assert labels_path.read_text(encoding="utf-8") == "smoking\n"
    config = config_path.read_text(encoding="utf-8")
    assert "onnx-file=smoking.onnx" in config
    assert "model-engine-file=smoking.engine" in config
    assert "batch-size=16" in config
    assert "num-detected-classes=1" in config
    assert "parse-bbox-func-name=NvDsInferParseCustomYoloDynamic" in config
    assert "disable-output-host-copy=0" in config
    trt_arguments = trt_log.read_text(encoding="utf-8")
    assert "--minShapes=images:1x3x640x640" in trt_arguments
    assert "--optShapes=images:8x3x640x640" in trt_arguments
    assert "--maxShapes=images:16x3x640x640" in trt_arguments
    yolo_arguments = yolo_log.read_text(encoding="utf-8")
    assert "dynamic=True" in yolo_arguments
    assert "nms=False" in yolo_arguments
    assert "batch=8" in yolo_arguments
    committed_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    assert committed_metadata["artifacts"]["build"]["explicit_batch"] is True
    assert committed_metadata["artifacts"]["build"]["nvinfer_batch_size"] == 16
    assert committed_metadata["artifacts"]["build"]["profile"]["max"][0] == 16
    for key in ("checkpoint", "onnx", "engine", "labels", "nvinfer_config"):
        assert len(committed_metadata["artifacts"][key]["sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (("architecture",), "architecture must be exactly one of"),
        (("checkpoint", "sha256"), "Checkpoint SHA256 mismatch"),
        (("input", "shape"), "input.shape must be dynamic-batch"),
        (("input", "profile", "max"), "profile.max batch must be at least 16"),
        (
            ("deepstream", "num_detected_classes"),
            "num_detected_classes must equal len(labels)",
        ),
        (("deepstream", "batch_size"), "batch_size must be at least 16"),
        (("deepstream", "parse_bbox_func_name"), "parse_bbox_func_name must be"),
    ],
)
def test_incompatible_contract_is_rejected(
    tmp_path: Path,
    mutation: tuple[str, ...],
    expected_error: str,
) -> None:
    contract = copy.deepcopy(_valid_contract())
    target: object = contract
    for key in mutation[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    invalid_values: dict[tuple[str, ...], object] = {
        ("architecture",): "yolov7",
        ("checkpoint", "sha256"): "0" * 64,
        ("input", "shape"): [1, 3, 640, 640],
        ("input", "profile", "max"): [8, 3, 640, 640],
        ("deepstream", "num_detected_classes"): 2,
        ("deepstream", "batch_size"): 8,
        ("deepstream", "parse_bbox_func_name"): "WrongParser",
    }
    target[mutation[-1]] = invalid_values[mutation]

    result = _validate(tmp_path, contract)

    assert result.returncode == 2
    assert expected_error in result.stderr
