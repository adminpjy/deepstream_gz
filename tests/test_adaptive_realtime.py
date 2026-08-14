from __future__ import annotations

from types import SimpleNamespace

from deepstream_ai.pipeline.adaptive_realtime import RealtimeAdaptiveInferenceController


class _Element:
    def __init__(self, name: str) -> None:
        self.name = name
        self.properties: list[tuple[str, int]] = []

    def set_property(self, name: str, value: int) -> None:
        if name == "secondary-reinfer-interval":
            raise AssertionError("runtime SGIE reinfer property must never be written")
        self.properties.append((name, int(value)))


def test_runtime_adaptation_changes_only_primary_gie(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
adaptive_inference:
  enabled: true
  initial_profile: normal
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
    person = _Element("person-detector")
    face = _Element("face-detector")
    behavior = _Element("behavior-smoking")
    graph = SimpleNamespace(
        inference_elements={
            "person": person,
            "face": face,
            "behavior:smoking": behavior,
        }
    )
    controller = RealtimeAdaptiveInferenceController(app_config, graph)

    controller._apply_profile(controller.current_profile, 30.0, reason="test")

    assert person.properties == [("interval", 5)]
    assert face.properties == []
    assert behavior.properties == []
