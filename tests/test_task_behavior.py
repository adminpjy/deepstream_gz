from __future__ import annotations

import pytest

from deepstream_ai.domain import BehaviorType
from deepstream_ai.task_behavior import (
    allowed_behavior_types,
    behavior_model_enabled,
    normalize_task_behavior_features,
)


def test_drinking_enables_shared_eat_drink_model_only() -> None:
    features = normalize_task_behavior_features(
        {"smoking": False, "eating": False, "drinking": True}
    )

    assert behavior_model_enabled("eating", features) is True
    assert behavior_model_enabled("smoking", features) is False
    assert allowed_behavior_types(features) == frozenset({BehaviorType.DRINKING})


def test_smoking_and_eating_enable_both_behavior_models() -> None:
    features = normalize_task_behavior_features(
        {"smoking": True, "eating": True, "drinking": False}
    )

    assert behavior_model_enabled("eating", features) is True
    assert behavior_model_enabled("smoking", features) is True
    assert allowed_behavior_types(features) == frozenset(
        {BehaviorType.SMOKING, BehaviorType.EATING}
    )


def test_removed_phone_feature_cannot_be_reintroduced_into_stable_tasks() -> None:
    with pytest.raises(ValueError, match="phone"):
        normalize_task_behavior_features({"phone": True})
