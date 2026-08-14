from __future__ import annotations

import configparser
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _person_config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read(_ROOT / "configs/nvinfer/person-peoplenet.txt", encoding="utf-8")
    return parser


def test_peoplenet_uses_calibrated_hybrid_postprocessing() -> None:
    parser = _person_config()
    properties = parser["property"]
    attributes = parser["class-attrs-all"]
    person_attributes = parser["class-attrs-0"]

    assert properties.getint("cluster-mode") == 3
    assert properties.getboolean("maintain-aspect-ratio")
    assert properties["output-blob-names"].split(";") == [
        "output_cov/Sigmoid:0",
        "output_bbox/BiasAdd:0",
    ]
    assert attributes.getfloat("pre-cluster-threshold") == pytest.approx(0.1037)
    assert attributes.getfloat("nms-iou-threshold") == pytest.approx(0.4842)
    assert attributes.getint("minBoxes") == 4
    assert attributes.getfloat("dbscan-min-score") == pytest.approx(1.1845)
    assert attributes.getfloat("eps") == pytest.approx(0.3207)
    assert attributes.getint("detected-min-w") == 20
    assert attributes.getint("detected-min-h") == 20
    assert person_attributes.getfloat("post-cluster-threshold") == pytest.approx(0.1894)


def test_nvdcf_uses_accuracy_association_and_nvidia_reid() -> None:
    tracker = yaml.safe_load((_ROOT / "configs/tracker/nvdcf.yml").read_text(encoding="utf-8"))

    assert tracker["VisualTracker"]["visualTrackerType"] == 1
    assert tracker["TargetManagement"]["maxShadowTrackingAge"] == 75
    assert tracker["TargetManagement"]["probationAge"] == 2
    assert tracker["TargetManagement"]["minIouDiff4NewTarget"] == pytest.approx(0.2)
    assert tracker["DataAssociator"]["associationMatcherType"] == 1
    assert "checkClassMatch" not in tracker["BaseConfig"]
    assert tracker["DataAssociator"]["checkClassMatch"] == 1
    assert tracker["DataAssociator"]["matchingScoreWeight4VisualSimilarity"] == pytest.approx(
        0.3951
    )
    assert tracker["DataAssociator"]["matchingScoreWeight4SizeSimilarity"] == pytest.approx(0.6003)
    assert tracker["DataAssociator"]["matchingScoreWeight4Iou"] == pytest.approx(0.4033)
    assert tracker["TrajectoryManagement"]["enableReAssoc"] == 1
    assert tracker["TrajectoryManagement"]["matchingScoreWeight4ReidSimilarity"] > 0

    reid = tracker["ReID"]
    assert reid["reidType"] == 2
    assert reid["outputReidTensor"] == 1
    assert reid["batchSize"] == 16
    assert reid["reidFeatureSize"] == 256
    assert reid["inferDims"] == [3, 256, 128]
    assert reid["networkMode"] == 1
    assert reid["inputOrder"] == 0
    assert reid["colorFormat"] == 0
    assert reid["addFeatureNormalization"] == 1
    assert reid["modelEngineFile"].endswith("_b16_gpu0_fp16.engine")
