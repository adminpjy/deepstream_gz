from __future__ import annotations

from datetime import UTC, datetime

import pytest

from deepstream_ai.behavior import (
    BehaviorConfigurationError,
    BehaviorMetadata,
    BehaviorMetadataRouter,
    BehaviorModelConfig,
)
from deepstream_ai.domain import BehaviorType, BoundingBox

NOW = datetime(2026, 8, 10, tzinfo=UTC)


def test_config_mapping_excludes_disabled_models_from_load_plan() -> None:
    router = BehaviorMetadataRouter.from_mapping(
        {
            "behavior": {
                "smoking": {
                    "enabled": True,
                    "model": "models/smoking.engine",
                    "unique_id": 31,
                    "threshold": 0.65,
                },
                "eating": {"enabled": False, "model": "models/eating.engine"},
            }
        }
    )

    assert [model.behavior for model in router.enabled_models] == [BehaviorType.SMOKING]
    assert router.enabled_models[0].gie_unique_id == 31
    assert len(router.configured_models) == 4


def test_router_uses_gie_metadata_and_threshold() -> None:
    router = BehaviorMetadataRouter.from_mapping(
        {
            "smoking": {
                "enabled": True,
                "model": "smoking.engine",
                "gie_unique_id": 22,
                "confidence_threshold": 0.6,
                "labels": {0: "smoking"},
            }
        }
    )
    metadata = BehaviorMetadata(
        camera_id="cam-a",
        track_id=5,
        timestamp=NOW,
        gie_unique_id=22,
        class_id=0,
        confidence=0.8,
        bbox=BoundingBox(1, 2, 20, 30),
        raw={"source": "nvinfer"},
    )

    result = router.route(metadata)
    assert result.behavior is BehaviorType.SMOKING
    assert result.model_name == "smoking"
    assert result.metadata["source"] == "nvinfer"
    assert (
        router.route(
            BehaviorMetadata(
                camera_id="cam-a",
                track_id=5,
                timestamp=NOW,
                gie_unique_id=22,
                class_id=0,
                confidence=0.59,
                bbox=BoundingBox(1, 2, 20, 30),
            )
        )
        is None
    )


def test_unknown_or_unmapped_metadata_is_ignored() -> None:
    router = BehaviorMetadataRouter.from_mapping(
        {
            "drinking": {
                "enabled": True,
                "model": "drinking.engine",
                "unique_id": 25,
                "labels": {2: "drinking"},
            }
        }
    )
    base = {
        "camera_id": "cam-a",
        "track_id": "track-1",
        "timestamp": NOW,
        "class_id": 0,
        "confidence": 0.9,
        "bbox": [0, 0, 10, 10],
    }
    assert router.route({**base, "component_id": 999}) is None
    assert router.route({**base, "component_id": 25}) is None


def test_duplicate_enabled_unique_ids_fail_fast() -> None:
    models = [
        BehaviorModelConfig(BehaviorType.SMOKING, True, "a.engine", 20),
        BehaviorModelConfig(BehaviorType.EATING, True, "b.engine", 20),
    ]
    with pytest.raises(BehaviorConfigurationError, match="duplicate gie_unique_id"):
        BehaviorMetadataRouter(models)
