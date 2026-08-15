from __future__ import annotations

from deepstream_ai.production.contracts import FeatureSet
from deepstream_ai.production.feature_gate import FeatureRegistry


def test_feature_registry_is_per_source_and_independent() -> None:
    registry = FeatureRegistry()
    registry.register(
        0,
        "camera-a",
        FeatureSet(smoking=True, drinking=False, eating=False, phone=False),
    )
    registry.register(
        1,
        "camera-b",
        FeatureSet(smoking=False, drinking=True, eating=False, phone=True),
    )

    assert registry.enabled(0, "smoking") is True
    assert registry.enabled(0, "drinking") is False
    assert registry.enabled(0, "phone") is False
    assert registry.enabled(1, "smoking") is False
    assert registry.enabled(1, "drinking") is True
    assert registry.enabled(1, "phone") is True

    # The production config uses the `eating` model entry for one shared
    # eating/drinking TensorRT SGIE. Either independent business switch must
    # therefore enable that one inference element.
    assert registry.enabled(1, "eating") is True

    registry.unregister(0)
    assert registry.binding(0) is None
    assert registry.enabled(0, "smoking") is False
    assert registry.enabled(1, "drinking") is True


def test_shared_eat_drink_gate_does_not_enable_other_models() -> None:
    registry = FeatureRegistry()
    registry.register(
        0,
        "camera-drink-only",
        FeatureSet(drinking=True),
    )

    assert registry.enabled(0, "eating") is True
    assert registry.enabled(0, "drinking") is True
    assert registry.enabled(0, "smoking") is False
    assert registry.enabled(0, "phone") is False
