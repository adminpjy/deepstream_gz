from __future__ import annotations

from types import SimpleNamespace

from deepstream_ai.pipeline.adaptive import GpuLoad
from deepstream_ai.pipeline.adaptive_realtime import RealtimeAdaptiveInferenceController


class _Element:
    def __init__(self, name: str) -> None:
        self.name = name
        self.properties: list[tuple[str, int]] = []

    def set_property(self, name: str, value: int) -> None:
        if name == "secondary-reinfer-interval":
            raise AssertionError("runtime SGIE reinfer property must never be written")
        self.properties.append((name, int(value)))


def _controller(tmp_path) -> RealtimeAdaptiveInferenceController:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
adaptive_inference:
  enabled: true
  initial_profile: normal
  load:
    gpu_low: 55
    gpu_high: 85
    gpu_critical: 92
    nvdec_low: 60
    nvdec_high: 85
    nvdec_critical: 95
    queue_low_ratio: 0.25
    queue_high_ratio: 0.60
    queue_critical_ratio: 0.85
  profiles:
    boost: {person_fps: 8, face_fps: 2, behavior_fps: 1}
    normal: {person_fps: 5, face_fps: 2, behavior_fps: 1}
    balanced: {person_fps: 5, face_fps: 2, behavior_fps: 1}
    protect: {person_fps: 5, face_fps: 2, behavior_fps: 1}
    emergency: {person_fps: 4, face_fps: 2, behavior_fps: 1}
""",
        encoding="utf-8",
    )
    app_config = SimpleNamespace(
        config_path=config_path,
        pipeline=SimpleNamespace(streammux=SimpleNamespace(gpu_id=0)),
        inference=SimpleNamespace(person_fps=5.0, face_fps=2.0, behavior_fps=1.0),
    )
    graph = SimpleNamespace(inference_elements={})
    return RealtimeAdaptiveInferenceController(app_config, graph)


def test_runtime_adaptation_changes_only_primary_gie(tmp_path) -> None:
    controller = _controller(tmp_path)
    person = _Element("person-detector")
    face = _Element("face-detector")
    behavior = _Element("behavior-smoking")
    controller.graph.inference_elements = {
        "person": person,
        "face": face,
        "behavior:smoking": behavior,
    }

    controller._apply_profile(controller.current_profile, 30.0, reason="test")

    assert person.properties == [("interval", 5)]
    assert face.properties == []
    assert behavior.properties == []


def test_emergency_profile_cannot_reduce_person_below_configured_floor(tmp_path) -> None:
    controller = _controller(tmp_path)
    person = _Element("person-detector")
    controller.graph.inference_elements = {"person": person}
    emergency = next(profile for profile in controller.config.profiles if profile.name == "emergency")
    assert emergency.person_fps == 4.0

    controller._apply_profile(emergency, 30.0, reason="critical_load")

    # 30 FPS / (interval + 1) = 5 FPS, even though the profile asks for 4.
    assert person.properties == [("interval", 5)]


def test_gpu_spike_without_backlog_is_not_critical(tmp_path) -> None:
    controller = _controller(tmp_path)
    load = GpuLoad(
        gpu_util=99.0,
        memory_util=10.0,
        memory_used_mb=1000.0,
        memory_total_mb=24000.0,
        decoder_util=20.0,
    )

    assert not controller._is_critical(load, queue_ratio=0.0, drops_delta=0)


def test_queue_drop_is_critical_even_when_gpu_is_not_full(tmp_path) -> None:
    controller = _controller(tmp_path)
    load = GpuLoad(
        gpu_util=70.0,
        memory_util=10.0,
        memory_used_mb=1000.0,
        memory_total_mb=24000.0,
        decoder_util=20.0,
    )

    assert controller._is_critical(load, queue_ratio=0.1, drops_delta=1)


def test_high_gpu_with_real_queue_pressure_is_critical(tmp_path) -> None:
    controller = _controller(tmp_path)
    load = GpuLoad(
        gpu_util=95.0,
        memory_util=10.0,
        memory_used_mb=1000.0,
        memory_total_mb=24000.0,
        decoder_util=20.0,
    )

    assert controller._is_critical(load, queue_ratio=0.70, drops_delta=0)
