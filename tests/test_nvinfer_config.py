from __future__ import annotations

import configparser
from pathlib import Path

from deepstream_ai.pipeline.nvinfer_config import materialize_nvinfer_config


def test_materialize_sgie_config_applies_cadence_and_absolutizes_assets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "configs" / "face.txt"
    source.parent.mkdir()
    source.write_text(
        """[property]
model-engine-file=../models/face.engine
gie-unique-id=99
secondary-reinfer-interval=99

[class-attrs-all]
pre-cluster-threshold=0.5
""",
        encoding="utf-8",
    )
    destination = tmp_path / "output/.runtime/face.txt"

    materialize_nvinfer_config(
        source,
        destination,
        {"gie-unique-id": 2, "secondary-reinfer-interval": 12},
    )

    parser = configparser.ConfigParser()
    parser.read(destination, encoding="utf-8")
    assert parser["property"].getint("gie-unique-id") == 2
    assert parser["property"].getint("secondary-reinfer-interval") == 12
    assert Path(parser["property"]["model-engine-file"]).is_absolute()
    assert parser["class-attrs-all"].getfloat("pre-cluster-threshold") == 0.5
