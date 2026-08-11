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


def test_nvdcf_retains_short_detector_shadow_without_changing_track_ids() -> None:
    tracker = yaml.safe_load((_ROOT / "configs/tracker/nvdcf.yml").read_text(encoding="utf-8"))

    assert tracker["VisualTracker"]["visualTrackerType"] == 1
    assert tracker["TargetManagement"]["maxShadowTrackingAge"] == 75
