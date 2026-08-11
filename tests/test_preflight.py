from __future__ import annotations

from pathlib import Path

import pytest

from deepstream_ai.config import load_config
from deepstream_ai.errors import AssetValidationError
from deepstream_ai.preflight import inspect_assets, inspect_nvinfer_config, validate_assets


def _project(tmp_path: Path, *, source_model: bool) -> Path:
    for directory in ("configs", "configs/nvinfer", "models", "videos"):
        (tmp_path / directory).mkdir(exist_ok=True, parents=True)
    (tmp_path / "videos/test.mp4").touch()
    (tmp_path / "configs/tracker.yml").write_text(
        "VisualTracker:\n  visualTrackerType: 1\n", encoding="utf-8"
    )
    (tmp_path / "models/labels.txt").write_text("person\n", encoding="utf-8")
    if source_model:
        (tmp_path / "models/person.onnx").touch()
    (tmp_path / "configs/nvinfer/person.txt").write_text(
        """[property]
model-engine-file=../../models/person.engine
onnx-file=../../models/person.onnx
labelfile-path=../../models/labels.txt
""",
        encoding="utf-8",
    )
    config = tmp_path / "configs/config.yaml"
    config.write_text(
        """source: {type: file, path: videos/test.mp4}
person: {enabled: true, config_file: configs/nvinfer/person.txt}
tracker: {config_file: configs/tracker.yml}
runtime: {strict_assets: true}
""",
        encoding="utf-8",
    )
    return config


def test_source_model_can_build_missing_engine(tmp_path: Path) -> None:
    config = load_config(_project(tmp_path, source_model=True))

    report = inspect_nvinfer_config(config, config.pipeline.person.config_file)

    assert report.missing == ()
    assert report.source_models[0].name == "person.onnx"
    assert len(validate_assets(config)) == 1


def test_container_custom_library_is_not_required_on_host(tmp_path: Path) -> None:
    config_path = _project(tmp_path, source_model=True)
    nvinfer = tmp_path / "configs/nvinfer/person.txt"
    nvinfer.write_text(
        nvinfer.read_text(encoding="utf-8")
        + "custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/custom.so\n",
        encoding="utf-8",
    )
    config = load_config(config_path)

    report = inspect_nvinfer_config(config, config.pipeline.person.config_file)

    assert report.missing == ()


def test_missing_engine_and_source_model_fails(tmp_path: Path) -> None:
    config = load_config(_project(tmp_path, source_model=False))

    with pytest.raises(AssetValidationError, match="无可用源模型"):
        validate_assets(config)


def test_enabled_behavior_model_must_match_nvinfer_engine(tmp_path: Path) -> None:
    config_path = _project(tmp_path, source_model=True)
    (tmp_path / "models/actual.engine").touch()
    (tmp_path / "models/declared.engine").touch()
    (tmp_path / "configs/nvinfer/smoking.txt").write_text(
        "[property]\nmodel-engine-file=../../models/actual.engine\n",
        encoding="utf-8",
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + """
behavior:
  smoking:
    enabled: true
    config_file: configs/nvinfer/smoking.txt
    model: models/declared.engine
""",
        encoding="utf-8",
    )
    config = load_config(config_path)

    with pytest.raises(AssetValidationError, match="不一致"):
        validate_assets(config)


def test_tracker_backend_must_match_low_level_yaml(tmp_path: Path) -> None:
    config_path = _project(tmp_path, source_model=True)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "tracker: {config_file: configs/tracker.yml}",
            "tracker: {backend: nvsort, config_file: configs/tracker.yml}",
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    with pytest.raises(AssetValidationError, match="backend=nvsort"):
        validate_assets(config)


def test_non_strict_inspection_keeps_semantic_failures(tmp_path: Path) -> None:
    config_path = _project(tmp_path, source_model=True)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        .replace(
            "tracker: {config_file: configs/tracker.yml}",
            "tracker: {backend: nvsort, config_file: configs/tracker.yml}",
        )
        .replace("strict_assets: true", "strict_assets: false"),
        encoding="utf-8",
    )
    config = load_config(config_path)

    _reports, failures = inspect_assets(config)

    assert any("backend=nvsort" in failure for failure in failures)


def test_nvinfer_on_nonzero_gpu_is_rejected(tmp_path: Path) -> None:
    config_path = _project(tmp_path, source_model=True)
    nvinfer = tmp_path / "configs/nvinfer/person.txt"
    nvinfer.write_text(
        nvinfer.read_text(encoding="utf-8").replace("[property]", "[property]\ngpu-id=1"),
        encoding="utf-8",
    )
    config = load_config(config_path)

    with pytest.raises(AssetValidationError, match="仅支持 gpu-id=0"):
        validate_assets(config)


def test_peoplenet_labels_are_the_authoritative_class_mapping(tmp_path: Path) -> None:
    config_path = _project(tmp_path, source_model=True)
    (tmp_path / "models/labels.txt").write_text("person\nbag\nface\n", encoding="utf-8")
    nvinfer = tmp_path / "configs/nvinfer/person.txt"
    nvinfer.write_text(
        """[property]
model-engine-file=../../models/peoplenet.engine
onnx-file=../../models/person.onnx
labelfile-path=../../models/labels.txt
infer-dims=3;544;960
network-type=0
num-detected-classes=3
cluster-mode=3
maintain-aspect-ratio=1
output-blob-names=output_cov/Sigmoid:0;output_bbox/BiasAdd:0
""",
        encoding="utf-8",
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "person: {enabled: true, config_file: configs/nvinfer/person.txt}",
            """person:
  enabled: true
  type: peoplenet
  config_file: configs/nvinfer/person.txt
  people_classes: {person: 0, bag: 1, face: 2}""",
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    validate_assets(config)

    (tmp_path / "models/labels.txt").write_text("bag\nperson\nface\n", encoding="utf-8")
    with pytest.raises(AssetValidationError, match="actual"):
        validate_assets(config)


def test_peoplenet_rejects_reversed_output_binding_order(tmp_path: Path) -> None:
    config_path = _project(tmp_path, source_model=True)
    (tmp_path / "models/labels.txt").write_text("person\nbag\nface\n", encoding="utf-8")
    nvinfer = tmp_path / "configs/nvinfer/person.txt"
    nvinfer.write_text(
        """[property]
model-engine-file=../../models/peoplenet.engine
onnx-file=../../models/person.onnx
labelfile-path=../../models/labels.txt
infer-dims=3;544;960
network-type=0
num-detected-classes=3
cluster-mode=3
maintain-aspect-ratio=1
output-blob-names=output_bbox/BiasAdd:0;output_cov/Sigmoid:0
""",
        encoding="utf-8",
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "person: {enabled: true, config_file: configs/nvinfer/person.txt}",
            """person:
  enabled: true
  type: peoplenet
  config_file: configs/nvinfer/person.txt
  people_classes: {person: 0, bag: 1, face: 2}""",
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError, match="输出顺序"):
        validate_assets(load_config(config_path))
