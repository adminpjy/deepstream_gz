from __future__ import annotations

from pathlib import Path

import pytest

from deepstream_ai.config import InferComponentConfig, OutputConfig, load_config
from deepstream_ai.errors import AssetValidationError, ConfigurationError


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    config = tmp_path / "configs" / "config.yaml"
    config.parent.mkdir()
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "test.mp4").touch()
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "person.txt").touch()
    (tmp_path / "tracker.yml").touch()
    config.write_text(
        """
source:
  type: file
  camera_id: entrance
  path: videos/test.mp4
  nominal_fps: 25
inference:
  person_fps: 5
  face_fps: 2
  behavior_fps: 1
person:
  enabled: true
  config_file: models/person.txt
  unique_id: 1
face:
  enabled: false
tracker:
  backend: nvdcf
  config_file: tracker.yml
behavior:
  smoking:
    enabled: false
    model: models/missing.engine
runtime:
  strict_assets: true
"""
        + extra,
        encoding="utf-8",
    )
    return config


def test_load_single_source_and_compute_inference_intervals(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))

    assert config.enabled_sources[0].camera_id == "entrance"
    assert config.interval_for(config.inference.person_fps) == 4
    assert config.interval_for(config.inference.face_fps) == 11
    assert config.interval_for(config.inference.behavior_fps) == 24
    assert config.validate_assets() == []


def test_disabled_behavior_model_is_not_a_required_asset(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))

    paths = {path.name for _, path in config.required_assets()}

    assert "missing.engine" not in paths


def test_enabled_component_missing_asset_fails_fast(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "enabled: false\n    model",
            "enabled: true\n    config_file: models/smoking.txt\n    model",
        ),
        encoding="utf-8",
    )
    config = load_config(path)

    with pytest.raises(AssetValidationError, match="smoking"):
        config.validate_assets()


def test_adaface_requires_face_detector_and_database(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
face_recognition:
  enabled: true
  model: models/adaface.engine
""",
    )

    with pytest.raises(ConfigurationError, match="face.enabled"):
        load_config(path)


def test_sources_must_have_unique_camera_ids(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        """
sources:
  - {camera_id: duplicate, type: file, path: videos/test.mp4}
  - {camera_id: duplicate, type: file, path: videos/test.mp4}
person: {enabled: true, config_file: models/person.txt}
tracker: {config_file: tracker.yml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="camera_id"):
        load_config(path)


def test_x264_output_backend_is_explicitly_supported_for_h264() -> None:
    output = OutputConfig.from_mapping({"codec": "h264", "encoder": "x264"}, {})

    assert output.encoder == "x264"


@pytest.mark.parametrize(
    "mapping",
    [
        {"codec": "h264", "encoder": "unknown"},
        {"codec": "h265", "encoder": "x264"},
    ],
)
def test_invalid_output_encoder_contract_is_rejected(mapping: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError, match="output.encoder"):
        OutputConfig.from_mapping(mapping, {})


def test_peoplenet_class_ids_are_derived_from_typed_mapping() -> None:
    config = InferComponentConfig.from_mapping(
        {
            "type": "peoplenet",
            "people_classes": {"person": 0, "bag": 1, "face": 2},
        },
        default_enabled=True,
        default_unique_id=1,
        default_label="person",
    )

    assert config.detector_type == "peoplenet"
    assert config.people_classes == (("person", 0), ("bag", 1), ("face", 2))
    assert config.person_class_ids == (0,)


def test_peoplenet_person_id_must_match_people_classes() -> None:
    with pytest.raises(ConfigurationError, match="people_classes.person"):
        InferComponentConfig.from_mapping(
            {
                "type": "peoplenet",
                "person_class_ids": [2],
                "people_classes": {"person": 0, "bag": 1, "face": 2},
            },
            default_enabled=True,
            default_unique_id=1,
            default_label="person",
        )


def test_top_level_person_crop_is_loaded_into_snapshot_config(tmp_path: Path) -> None:
    config = load_config(
        _write_config(
            tmp_path,
            """
person_crop:
  padding_x_ratio: 0.15
  padding_top_ratio: 0.25
  upper_body_height_ratio: 0.70
  min_crop_width: 24
  min_crop_height: 48
  min_visible_ratio: 0.60
""",
        )
    )

    snapshot = config.output.snapshot
    assert snapshot.padding_x_ratio == 0.15
    assert snapshot.padding_top_ratio == 0.25
    assert snapshot.upper_body_height_ratio == 0.70
    assert snapshot.min_crop_width == 24
    assert snapshot.min_crop_height == 48
    assert snapshot.min_visible_ratio == 0.60
