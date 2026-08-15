from __future__ import annotations

from pathlib import Path

from deepstream_ai.production.contracts import FeatureSet

ROOT = Path(__file__).resolve().parents[1]


def test_phone_feature_is_removed_from_production_contract() -> None:
    assert not hasattr(FeatureSet(), "phone")
    assert not (ROOT / "configs/nvinfer/phone.txt").exists()

    converter = (ROOT / "scripts/prepare-local-behavior-models.py").read_text(
        encoding="utf-8"
    )
    assert 'name="phone"' not in converter
    assert 'source_name="phone.pt"' not in converter


def test_eat_drink_proxy_matches_opsvision_rule() -> None:
    parser = (ROOT / "native/yolo_parser/nvdsparsebbox_YoloDynamic.cpp").read_text(
        encoding="utf-8"
    )
    assert "constexpr float kMouthRegionRatio = 0.40F;" in parser

    expected = {
        "{39, kDrinking}",  # bottle
        "{40, kDrinking}",  # wine glass
        "{41, kDrinking}",  # cup
        "{45, kDrinking}",  # bowl
        "{46, kEating}",  # banana
        "{47, kEating}",  # apple
        "{48, kEating}",  # sandwich
        "{49, kEating}",  # orange
        "{52, kEating}",  # hot dog
        "{53, kEating}",  # pizza
        "{54, kEating}",  # donut
        "{55, kEating}",  # cake
    }
    for mapping in expected:
        assert mapping in parser

    # These broader proxy classes existed in an intermediate implementation but
    # are not part of the validated opsvision EatingDrinking rule.
    for mapping in (
        "{42, kEating}",  # fork
        "{43, kEating}",  # knife
        "{44, kEating}",  # spoon
        "{50, kEating}",  # broccoli
        "{51, kEating}",  # carrot
    ):
        assert mapping not in parser

    # Mirror Ultralytics/opsvision semantics: choose the highest COCO class first,
    # then ask whether that class is one of the business evidence classes.
    assert "for (int coco_class = 1; coco_class < kCocoClasses; ++coco_class)" in parser
    assert "if (mapping.first == best_coco_class)" in parser


def test_eat_drink_runtime_config_uses_reviewed_threshold_and_labels() -> None:
    nvinfer = (ROOT / "configs/nvinfer/eat-drink.txt").read_text(encoding="utf-8")
    assert "labelfile-path=eat-drink.labels.txt" in nvinfer
    assert "num-detected-classes=2" in nvinfer
    assert "parse-bbox-func-name=NvDsInferParseCustomYoloEatDrinkCoco" in nvinfer
    assert "pre-cluster-threshold=0.45" in nvinfer

    labels = (ROOT / "configs/nvinfer/eat-drink.labels.txt").read_text(
        encoding="utf-8"
    )
    assert labels.splitlines() == ["eating", "drinking"]

    app_config = (ROOT / "configs/config.yaml").read_text(encoding="utf-8")
    assert "threshold: 0.45" in app_config
    assert app_config.count("behavior_fps: 5.0") >= 6
