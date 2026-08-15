from __future__ import annotations

from types import SimpleNamespace

import deepstream_ai.production.pipeline as production_pipeline
from deepstream_ai.config import SourceConfig
from deepstream_ai.production.contracts import FeatureSet
from deepstream_ai.production.feature_gate import FeatureRegistry
from deepstream_ai.production.multiuri_pipeline import MultiUriSourceController
from deepstream_ai.production.pipeline import DynamicSourceController


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


class _FakePad:
    def link(self, _other):
        return "ok"

    def unlink(self, _other):
        return True


class _FakeBin:
    def __init__(self, name: str) -> None:
        self.name = name
        self.src_pad = _FakePad()
        self.state = None

    def get_static_pad(self, name: str):
        return self.src_pad if name == "src" else None

    def sync_state_with_parent(self) -> bool:
        return True

    def set_state(self, state):
        self.state = state
        return None


class _FakeSourceBin:
    def __init__(self, _runtime, _config, source, index: int) -> None:
        self.config = source
        self.bin = _FakeBin(f"source-bin-{index:02d}")


class _FakeStreamMux:
    def __init__(self) -> None:
        self.pads: dict[str, _FakePad] = {}

    def request_pad_simple(self, name: str):
        pad = _FakePad()
        self.pads[name] = pad
        return pad

    def release_request_pad(self, pad) -> None:
        for name, current in list(self.pads.items()):
            if current is pad:
                self.pads.pop(name, None)


class _FakePipeline:
    """Mirror PyGObject Gst.Bin add/remove: success returns None, failure raises."""

    def __init__(self, streammux: _FakeStreamMux) -> None:
        self.streammux = streammux
        self.children: dict[str, _FakeBin] = {}

    def get_by_name(self, name: str):
        if name == "stream-muxer":
            return self.streammux
        return self.children.get(name)

    def add(self, child: _FakeBin) -> None:
        if child.name in self.children:
            raise RuntimeError(f"duplicate child: {child.name}")
        self.children[child.name] = child
        return None

    def remove(self, child: _FakeBin) -> None:
        if self.children.get(child.name) is not child:
            raise RuntimeError(f"child not found: {child.name}")
        self.children.pop(child.name, None)
        return None


def _source(camera_id: str) -> SourceConfig:
    return SourceConfig(
        camera_id=camera_id,
        type="rtsp",
        url="rtsp://127.0.0.1/live",
        enabled=True,
        nominal_fps=30.0,
        latency_ms=200,
        reconnect_interval_sec=10,
    )


def test_dynamic_source_accepts_none_return_and_recovers_orphan(monkeypatch) -> None:
    monkeypatch.setattr(production_pipeline, "SourceBin", _FakeSourceBin)
    gst = SimpleNamespace(
        PadLinkReturn=SimpleNamespace(OK="ok"),
        State=SimpleNamespace(NULL="null"),
    )
    runtime = SimpleNamespace(Gst=gst)
    streammux = _FakeStreamMux()
    pipeline = _FakePipeline(streammux)
    graph = SimpleNamespace(
        pipeline=pipeline,
        metadata_probe=SimpleNamespace(camera_by_pad={}),
        source_bins=(),
    )
    registry = FeatureRegistry()
    controller = DynamicSourceController(
        runtime,
        SimpleNamespace(),
        graph,
        registry,
        capacity=1,
    )

    # Gst.Bin.add succeeds with None. The controller must not interpret that as failure.
    assert controller.add(_source("camera-a"), FeatureSet()) == 0
    assert controller.active_count() == 1
    assert pipeline.get_by_name("source-bin-00") is not None

    # Gst.Bin.remove also succeeds with None.
    assert controller.remove("camera-a") is True
    assert controller.active_count() == 0
    assert pipeline.get_by_name("source-bin-00") is None

    # A stale unregistered bin from an interrupted old attach is reclaimed before reuse.
    pipeline.children["source-bin-00"] = _FakeBin("source-bin-00")
    assert controller.add(_source("camera-b"), FeatureSet(phone=True)) == 0
    assert controller.slot_for_camera("camera-b") == 0
    assert registry.enabled(0, "phone") is True


def test_multiuri_source_uses_official_rest_lifecycle_and_metrics_mapping(monkeypatch) -> None:
    graph = SimpleNamespace(
        metadata_probe=SimpleNamespace(camera_by_pad={}),
    )
    registry = FeatureRegistry()
    controller = MultiUriSourceController(
        SimpleNamespace(),
        SimpleNamespace(),
        graph,
        registry,
        capacity=4,
    )
    calls: list[tuple[str, str, dict | None]] = []

    def request_json(method, path, payload=None, **_kwargs):
        calls.append((method, path, payload))
        if method == "GET" and path == "/metrics":
            return {
                "status": "HTTP/1.1 200 OK",
                "reason": "GET_METRICS_INFO_SUCCESS",
                "metrics-info": {
                    "stream-count": 1,
                    "stream-stats": [
                        {
                            "sensor_id": "camera-prod",
                            "source_id": 3,
                            "fps": 30.0,
                            "frame_number": 2,
                        }
                    ],
                },
            }
        return {"status": "HTTP/1.1 200 OK", "reason": "STREAM_ADD_SUCCESS"}

    monkeypatch.setattr(controller, "_request_json", request_json)

    source = _source("camera-prod")
    assert controller.add(source, FeatureSet(smoking=True, phone=True)) == 3
    assert controller.active_count() == 1
    assert controller.slot_for_camera("camera-prod") == 3
    assert graph.metadata_probe.camera_by_pad == {3: "camera-prod"}
    assert registry.enabled(3, "smoking") is True
    assert registry.enabled(3, "phone") is True

    add_call = calls[0]
    assert add_call[0:2] == ("POST", "/stream/add")
    assert add_call[2]["value"]["camera_id"] == "camera-prod"
    assert add_call[2]["value"]["camera_url"] == source.url
    assert add_call[2]["value"]["change"] == "camera_add"

    assert controller.remove("camera-prod") is True
    assert controller.active_count() == 0
    assert graph.metadata_probe.camera_by_pad == {}
    assert registry.binding(3) is None

    remove_call = calls[-1]
    assert remove_call[0:2] == ("POST", "/stream/remove")
    assert remove_call[2]["value"]["camera_id"] == "camera-prod"
    assert remove_call[2]["value"]["change"] == "camera_remove"
