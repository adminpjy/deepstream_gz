from __future__ import annotations

import configparser
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from deepstream_ai.config import SourceConfig, load_config
from deepstream_ai.errors import ConfigurationError
from deepstream_ai.pipeline.builder import DeepStreamPipelineBuilder
from deepstream_ai.pipeline.metadata import MetadataProbe
from deepstream_ai.pipeline.runner import PipelineRunner
from deepstream_ai.pipeline.source import SourceBin


class _Factory:
    def __init__(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name


class _Child:
    def __init__(self, factory: str):
        self._factory = _Factory(factory)

    def get_factory(self) -> _Factory:
        return self._factory


def _source_bin(*, attach_system_timestamp: bool):
    calls: list[int] = []
    source = object.__new__(SourceBin)
    source.runtime = SimpleNamespace(
        pyds=SimpleNamespace(configure_source_for_ntp_sync=lambda pointer: calls.append(pointer))
    )
    source.config = SourceConfig(camera_id="rtsp-a", type="rtsp", url="rtsp://example.invalid/live")
    source._attach_system_timestamp = attach_system_timestamp
    source._ntp_synced_sources = set()
    return source, calls


def test_rtsp_rtcp_ntp_is_configured_once_when_mux_does_not_attach_clock() -> None:
    source, calls = _source_bin(attach_system_timestamp=False)
    child = _Child("rtspsrc")

    source._on_child_added(None, child, "source")
    source._on_child_added(None, child, "source")

    assert calls == [hash(child)]


def test_rtsp_rtcp_ntp_is_not_configured_when_mux_attaches_system_clock() -> None:
    source, calls = _source_bin(attach_system_timestamp=True)

    source._on_child_added(None, _Child("rtspsrc"), "source")

    assert calls == []


class _Loop:
    def __init__(self):
        self.quit_calls = 0

    def quit(self) -> None:
        self.quit_calls += 1

    def is_running(self) -> bool:
        return True


class _GLib:
    def __init__(self):
        self.loop = _Loop()
        self.timeouts: list[tuple[int, object]] = []

    def MainLoop(self) -> _Loop:
        return self.loop

    def timeout_add_seconds(self, delay: int, callback: object) -> int:
        self.timeouts.append((delay, callback))
        return 1


class _Event:
    @staticmethod
    def new_eos() -> object:
        return object()


class _Pipeline:
    def __init__(self, accepts_eos: bool):
        self.accepts_eos = accepts_eos
        self.events: list[object] = []

    def send_event(self, event: object) -> bool:
        self.events.append(event)
        return self.accepts_eos


def _runner(*, accepts_eos: bool) -> tuple[PipelineRunner, _GLib]:
    glib = _GLib()
    runtime = SimpleNamespace(GLib=glib, Gst=SimpleNamespace(Event=_Event))
    config = SimpleNamespace(runtime=SimpleNamespace(shutdown_timeout_sec=20))
    graph = SimpleNamespace(pipeline=_Pipeline(accepts_eos))
    return PipelineRunner(runtime, config, graph), glib


def test_runner_quits_immediately_when_pipeline_rejects_eos() -> None:
    runner, glib = _runner(accepts_eos=False)

    runner.stop()

    assert glib.loop.quit_calls == 1
    assert glib.timeouts == []


def test_second_stop_request_forces_exit_while_waiting_for_eos() -> None:
    runner, glib = _runner(accepts_eos=True)

    runner.stop()
    assert glib.loop.quit_calls == 0
    assert glib.timeouts[0][0] == 20

    runner.stop(force=True)
    assert glib.loop.quit_calls == 1


def test_identity_annotation_holds_batch_meta_lock_only_for_mutation() -> None:
    lock_events: list[tuple[str, object]] = []
    pyds = SimpleNamespace(
        nvds_acquire_meta_lock=lambda meta: lock_events.append(("acquire", meta)),
        nvds_release_meta_lock=lambda meta: lock_events.append(("release", meta)),
    )
    consumer = SimpleNamespace(identity_label=lambda _camera, _track: "alice")
    config = SimpleNamespace(enabled_sources=(), behavior=())
    probe = MetadataProbe(SimpleNamespace(pyds=pyds), config, consumer)
    obj = SimpleNamespace(text_params=SimpleNamespace(display_text="person"))
    batch_meta = object()

    probe._annotate_identities(batch_meta, "camera-a", [(obj, 7)])

    assert obj.text_params.display_text == "person alice"
    assert lock_events == [("acquire", batch_meta), ("release", batch_meta)]


def test_identity_annotation_decodes_pyds_display_text_pointer() -> None:
    pyds = SimpleNamespace(
        nvds_acquire_meta_lock=lambda _meta: None,
        nvds_release_meta_lock=lambda _meta: None,
        get_string=lambda pointer: "person 7" if pointer == 123456 else "unexpected",
    )
    consumer = SimpleNamespace(identity_label=lambda _camera, _track: "unknown sim=-1.000")
    config = SimpleNamespace(enabled_sources=(), behavior=())
    probe = MetadataProbe(SimpleNamespace(pyds=pyds), config, consumer)
    obj = SimpleNamespace(text_params=SimpleNamespace(display_text=123456))

    probe._annotate_identities(object(), "camera-a", [(obj, 7)])

    assert obj.text_params.display_text == "person 7 unknown sim=-1.000"


def test_probe_performance_reports_bounded_p95_and_logs_once(caplog) -> None:
    probe = MetadataProbe(
        SimpleNamespace(pyds=SimpleNamespace()),
        SimpleNamespace(enabled_sources=(), behavior=()),
        SimpleNamespace(),
    )
    for milliseconds in range(1, 101):
        probe._record_timing(
            milliseconds * 1_000_000,
            frames=1,
            queue_drops=1 if milliseconds == 1 else 0,
            errors=1 if milliseconds == 100 else 0,
        )

    result = probe.performance()
    assert result.callbacks == 100
    assert result.frames == 100
    assert result.queue_drops == 1
    assert result.errors == 1
    assert result.average_ms == pytest.approx(50.5)
    assert result.p95_ms == pytest.approx(95.0)
    assert result.max_ms == pytest.approx(100.0)

    with caplog.at_level("INFO"):
        probe.log_performance()
        probe.log_performance()
    assert caplog.text.count("========== Probe Performance ==========") == 1
    assert "Average Probe Time (ms):      50.500" in caplog.text
    assert "P95 Probe Time (ms):          95.000" in caplog.text


def test_probe_p95_covers_the_full_lifetime_not_only_a_recent_window() -> None:
    probe = MetadataProbe(
        SimpleNamespace(pyds=SimpleNamespace()),
        SimpleNamespace(enabled_sources=(), behavior=()),
        SimpleNamespace(),
    )
    for _ in range(5_000):
        probe._record_timing(100_000_000)
    for _ in range(5_000):
        probe._record_timing(1_000_000)

    result = probe.performance()
    assert result.callbacks == 10_000
    assert result.average_ms == pytest.approx(50.5)
    assert result.p95_ms == pytest.approx(100.0)


def test_face_sgie_template_declares_one_class_and_no_primary_interval() -> None:
    parser = configparser.ConfigParser()
    parser.read("configs/nvinfer/face.example.txt", encoding="utf-8")

    assert parser["property"].getint("num-detected-classes") == 1
    assert "interval" not in parser["property"]


def test_nonzero_runtime_gpu_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
source: {type: file, path: video.mp4}
person: {config_file: person.txt}
tracker: {config_file: tracker.yml}
pipeline:
  streammux: {gpu_id: 1}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="streammux.gpu_id=0"):
        load_config(config_path)


def test_tracker_rejects_non_reference_low_level_library(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
source: {type: file, path: video.mp4}
person: {config_file: person.txt}
tracker:
  backend: nvdcf
  config_file: tracker.yml
  library_file: /tmp/libcustomtracker.so
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="libnvds_nvmultiobjecttracker.so"):
        load_config(config_path)


class _Element:
    def __init__(self, name: str = "element"):
        self.name = name
        self.values: dict[str, object] = {}
        self.src_pad = _Pad()

    def find_property(self, _name: str) -> object:
        return object()

    def set_property(self, name: str, value: object) -> None:
        self.values[name] = value

    def get_static_pad(self, name: str) -> _Pad | None:
        return self.src_pad if name == "src" else None


class _Pad:
    def __init__(self) -> None:
        self.probes: list[tuple[object, object, object]] = []

    def add_probe(self, probe_type: object, callback: object, data: object) -> None:
        self.probes.append((probe_type, callback, data))


class _GraphPipeline:
    def __init__(self) -> None:
        self.elements: list[_Element] = []

    def add(self, element: _Element) -> None:
        self.elements.append(element)


@pytest.mark.parametrize(
    ("face_enabled", "expected_probe_name"),
    [(True, "face-detector"), (False, "snapshot-rgba-caps")],
)
def test_builder_converts_before_face_sgie_and_probes_last_metadata_source(
    monkeypatch: pytest.MonkeyPatch,
    face_enabled: bool,
    expected_probe_name: str,
) -> None:
    pipeline = _GraphPipeline()
    probe_type = object()
    Gst = SimpleNamespace(
        Pipeline=SimpleNamespace(new=lambda _name: pipeline),
        Caps=SimpleNamespace(from_string=lambda value: value),
        PadProbeType=SimpleNamespace(BUFFER=probe_type),
    )
    elements: dict[str, _Element] = {}
    linked_names: list[str] = []
    probe_callback = object()

    def element(name: str) -> _Element:
        result = _Element(name)
        elements[name] = result
        return result

    monkeypatch.setattr(
        "deepstream_ai.pipeline.builder.SourceBin",
        lambda *_args, **_kwargs: SimpleNamespace(bin=element("source-bin")),
    )
    monkeypatch.setattr("deepstream_ai.pipeline.builder.link_source_to_mux", lambda *_args: None)
    monkeypatch.setattr(
        "deepstream_ai.pipeline.builder.make_element",
        lambda _gst, _factory, name: element(name),
    )
    monkeypatch.setattr(
        "deepstream_ai.pipeline.builder.link_many",
        lambda chain: linked_names.extend(item.name for item in chain),
    )
    monkeypatch.setattr(
        "deepstream_ai.pipeline.builder.MetadataProbe",
        lambda *_args: SimpleNamespace(callback=probe_callback),
    )
    monkeypatch.setattr(
        DeepStreamPipelineBuilder,
        "_infer_element",
        lambda _self, name, *_args, **_kwargs: element(name),
    )
    monkeypatch.setattr(
        DeepStreamPipelineBuilder,
        "_tracker_element",
        lambda _self: element("person-tracker"),
    )
    monkeypatch.setattr(
        DeepStreamPipelineBuilder,
        "_configure_streammux",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        DeepStreamPipelineBuilder,
        "_configure_tiler",
        lambda *_args: None,
    )

    config = SimpleNamespace(
        enabled_sources=(SimpleNamespace(type="file"),),
        pipeline=SimpleNamespace(
            person=object(),
            face=SimpleNamespace(enabled=face_enabled),
            streammux=SimpleNamespace(gpu_id=0),
        ),
        inference=SimpleNamespace(person_fps=10.0, face_fps=2.0, behavior_fps=2.0),
        behavior=(),
        output=SimpleNamespace(enabled=False),
    )

    DeepStreamPipelineBuilder(SimpleNamespace(Gst=Gst), config, object()).build()

    expected_prefix = [
        "stream-muxer",
        "person-detector",
        "person-tracker",
        "snapshot-rgba-convert",
        "snapshot-rgba-caps",
    ]
    if face_enabled:
        expected_prefix.append("face-detector")
    assert linked_names[: len(expected_prefix)] == expected_prefix
    assert elements[expected_probe_name].src_pad.probes == [(probe_type, probe_callback, None)]


@pytest.mark.parametrize(
    (
        "backend",
        "codec",
        "expected_factory",
        "expected_caps",
        "expected_memory_type",
        "expected_bitrate",
    ),
    [
        ("x264", "h264", "x264enc", "video/x-raw,format=I420", 1, 8_000),
        (
            "nvidia",
            "h264",
            "nvv4l2h264enc",
            "video/x-raw(memory:NVMM),format=NV12",
            None,
            8_000_000,
        ),
    ],
)
def test_output_encoder_backend_materializes_the_required_memory_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    codec: str,
    expected_factory: str,
    expected_caps: str,
    expected_memory_type: int | None,
    expected_bitrate: int,
) -> None:
    created: list[tuple[str, _Element]] = []

    def create(_gst: object, factory: str, name: str) -> _Element:
        result = _Element(name)
        created.append((factory, result))
        return result

    monkeypatch.setattr("deepstream_ai.pipeline.builder.make_element", create)
    output = SimpleNamespace(
        enabled=True,
        path="result.mp4",
        codec=codec,
        encoder=backend,
        bitrate=8_000_000,
        sync=False,
    )
    config = SimpleNamespace(
        output=output,
        pipeline=SimpleNamespace(streammux=SimpleNamespace(gpu_id=0)),
        resolve_path=lambda _value: tmp_path / "result.mp4",
    )
    Gst = SimpleNamespace(Caps=SimpleNamespace(from_string=lambda value: value))

    elements = DeepStreamPipelineBuilder(
        SimpleNamespace(Gst=Gst), config, object()
    )._output_elements()

    factories = [factory for factory, _element in created]
    assert expected_factory in factories
    convert = elements[0]
    caps_filter = elements[1]
    encoder = elements[2]
    assert convert.values.get("nvbuf-memory-type") == expected_memory_type
    assert caps_filter.values["caps"] == expected_caps
    assert encoder.values["bitrate"] == expected_bitrate
    if backend == "x264":
        assert encoder.values["speed-preset"] == 3
        assert encoder.values["tune"] == 4
        assert encoder.values["key-int-max"] == 30
    else:
        assert encoder.values["insert-sps-pps"] is True
        assert encoder.values["iframeinterval"] == 30
        assert encoder.values["idrinterval"] == 30


def test_synchronized_file_output_adds_a_raw_frame_clock_pacer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: list[tuple[str, _Element]] = []

    def create(_gst: object, factory: str, name: str) -> _Element:
        result = _Element(name)
        created.append((factory, result))
        return result

    monkeypatch.setattr("deepstream_ai.pipeline.builder.make_element", create)
    output = SimpleNamespace(
        enabled=True,
        path="result.mp4",
        codec="h264",
        encoder="nvidia",
        bitrate=8_000_000,
        sync=True,
    )
    config = SimpleNamespace(
        output=output,
        pipeline=SimpleNamespace(streammux=SimpleNamespace(gpu_id=0)),
        resolve_path=lambda _value: tmp_path / "result.mp4",
    )
    Gst = SimpleNamespace(Caps=SimpleNamespace(from_string=lambda value: value))

    elements = DeepStreamPipelineBuilder(
        SimpleNamespace(Gst=Gst), config, object()
    )._output_elements()

    assert created[0][0] == "identity"
    assert elements[0].values["sync"] is True


def test_tracker_builder_uses_ds9_plugin_properties_only(monkeypatch: pytest.MonkeyPatch) -> None:
    element = _Element()
    monkeypatch.setattr(
        "deepstream_ai.pipeline.builder.make_element",
        lambda _gst, _factory, _name: element,
    )
    tracker = SimpleNamespace(
        width=960,
        height=544,
        gpu_id=0,
        config_file="tracker.yml",
        library_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        display_tracking_id=True,
    )
    config = SimpleNamespace(
        pipeline=SimpleNamespace(tracker=tracker),
        resolve_path=lambda value: Path(value),
    )
    builder = DeepStreamPipelineBuilder(SimpleNamespace(Gst=object()), config, object())

    assert builder._tracker_element() is element
    assert "enable-batch-process" not in element.values
    assert "enable-past-frame" not in element.values
    assert element.values["ll-config-file"] == "tracker.yml"


def test_compose_grace_period_exceeds_default_runtime_shutdown() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))

    # 20s Pipeline EOS + up to 60s analytics drain + shutdown margin.
    assert compose["services"]["app"]["stop_grace_period"] == "95s"
