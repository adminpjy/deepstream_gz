from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from deepstream_ai.pipeline.adaptive import (
    AdaptiveInferenceConfig,
    _fraction_to_float,
    _queue_ratio,
    _skip_interval,
)


def test_adaptive_config_parses_profiles_and_thresholds(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
adaptive_inference:
  enabled: true
  initial_profile: balanced
  monitor:
    sample_interval_sec: 3
  load:
    gpu_high: 80
    gpu_critical: 90
  profiles:
    boost:
      person_fps: 9
      face_fps: 8
      behavior_fps: 2
    balanced:
      person_fps: 5
      face_fps: 4
      behavior_fps: 0.5
""",
        encoding="utf-8",
    )

    config = AdaptiveInferenceConfig.from_file(path)

    assert config.enabled is True
    assert config.initial_profile == "balanced"
    assert config.sample_interval_sec == pytest.approx(3.0)
    assert config.gpu_high == pytest.approx(80.0)
    assert config.gpu_critical == pytest.approx(90.0)
    profiles = {profile.name: profile for profile in config.profiles}
    assert profiles["boost"].person_fps == pytest.approx(9.0)
    assert profiles["balanced"].face_fps == pytest.approx(4.0)
    # Unspecified profiles retain safe defaults.
    assert profiles["emergency"].person_fps == pytest.approx(3.0)
    assert profiles["emergency"].face_fps == pytest.approx(3.0)


def test_interval_conversion_uses_detected_source_fps() -> None:
    # 30 fps source -> infer every sixth frame -> 5 fps.
    assert _skip_interval(30.0, 5.0) == 5
    assert 30.0 / (_skip_interval(30.0, 5.0) + 1) == pytest.approx(5.0)

    # 8 fps cannot be represented exactly as an integer interval at 30 fps;
    # the closest bounded cadence selected here is 7.5 fps.
    assert _skip_interval(30.0, 8.0) == 3
    assert 30.0 / (_skip_interval(30.0, 8.0) + 1) == pytest.approx(7.5)

    assert _skip_interval(30.0, 3.0) == 9
    assert _skip_interval(10.0, 20.0) == 0


def test_fraction_parser_handles_caps_style_values() -> None:
    assert _fraction_to_float("30000/1001") == pytest.approx(29.97002997)
    assert _fraction_to_float("30/1") == pytest.approx(30.0)
    # A zero fraction is parsed but later rejected as a negotiated FPS, causing
    # the controller to use the configured startup fallback instead.
    assert _fraction_to_float("0/1") == pytest.approx(0.0)


def test_queue_ratio_finds_analytics_queue_through_delegate_chain() -> None:
    analytics_queue: queue.Queue[object] = queue.Queue(maxsize=8)
    for _ in range(4):
        analytics_queue.put_nowait(object())
    analytics = SimpleNamespace(_queue=analytics_queue)
    wrapper = SimpleNamespace(delegate=SimpleNamespace(delegate=analytics))

    assert _queue_ratio(wrapper) == pytest.approx(0.5)


def test_invalid_threshold_order_is_rejected(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
adaptive_inference:
  enabled: true
  load:
    gpu_low: 80
    gpu_high: 70
    gpu_critical: 90
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gpu thresholds"):
        AdaptiveInferenceConfig.from_file(path)
