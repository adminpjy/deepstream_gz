from __future__ import annotations

from deepstream_ai.production.contracts import FeatureSet
from deepstream_ai.production.feature_gate import FeatureRegistry


def test_feature_registry_is_per_source_and_independent() -> None:
    registry = FeatureRegistry()
    registry.register(
        0,
        "camera-a",
        FeatureSet(smoking=True, drinking=False, eating=False),
    )
    registry.register(
        1,
        "camera-b",
        FeatureSet(smoking=False, drinking=True, eating=True),
    )

    assert registry.enabled(0, "smoking") is True
    assert registry.enabled(0, "drinking") is False
    assert registry.enabled(1, "smoking") is False
    assert registry.enabled(1, "drinking") is True
    assert registry.enabled(1, "eating") is True

    registry.unregister(0)
    assert registry.binding(0) is None
    assert registry.enabled(0, "smoking") is False
    assert registry.enabled(1, "drinking") is True
