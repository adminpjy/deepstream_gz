from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from deepstream_ai.pipeline.peoplenet_pretracker_guard import (
    PeopleNetPretrackerGuard,
    PeopleNetPretrackerGuardConfig,
)


def _object(
    *,
    component_id: int = 1,
    class_id: int = 0,
    confidence: float = 0.22,
    left: float = 20.0,
    top: float = 0.0,
    width: float = 880.0,
    height: float = 1035.0,
):
    return SimpleNamespace(
        unique_component_id=component_id,
        class_id=class_id,
        confidence=confidence,
        rect_params=SimpleNamespace(left=left, top=top, width=width, height=height),
    )


def _config(**overrides: object) -> PeopleNetPretrackerGuardConfig:
    values: dict[str, object] = {
        "enabled": True,
        "max_confidence": 0.55,
        "max_left_ratio": 0.03,
        "max_top_ratio": 0.02,
        "min_width_ratio": 0.40,
        "max_width_ratio": 0.50,
        "min_height_ratio": 0.93,
        "max_right_ratio": 0.52,
    }
    values.update(overrides)
    return PeopleNetPretrackerGuardConfig(**values)


def _guard(config: PeopleNetPretrackerGuardConfig | None = None) -> PeopleNetPretrackerGuard:
    runtime = SimpleNamespace(
        Gst=SimpleNamespace(PadProbeReturn=SimpleNamespace(OK="ok")),
        pyds=SimpleNamespace(),
    )
    return PeopleNetPretrackerGuard(
        runtime,
        config or _config(),
        pgie_unique_id=1,
        person_class_ids=(0,),
        frame_width=1920,
        frame_height=1080,
    )


def test_verified_low_confidence_left_top_full_height_giant_is_suppressed() -> None:
    assert _guard().should_suppress(_object())


@pytest.mark.parametrize(
    ("confidence", "left", "top", "width", "height"),
    [
        (0.2093505859375, 23.828115463256836, 0.0, 815.9375, 1028.9375),
        (0.2208251953125, 21.640615463256836, 0.0, 918.75, 1043.4219970703125),
        (0.2213134765625, 27.656240463256836, 0.0, 886.484375, 1045.0625),
    ],
)
def test_repeated_test2_rack_pgie_proposals_match_the_guard(
    confidence: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> None:
    assert _guard().should_suppress(
        _object(
            confidence=confidence,
            left=left,
            top=top,
            width=width,
            height=height,
        )
    )


@pytest.mark.parametrize(
    ("confidence", "left", "top", "width", "height"),
    [
        # Remaining full-EOS rack track 1: two detector refreshes.
        (0.37890625, 27.109365463256836, 0.0, 888.671875, 1048.34375),
        (0.422119140625, 32.57811737060547, 0.0, 905.8984375, 1055.453125),
        # Remaining full-EOS rack track 5.
        (0.5068359375, 36.95311737060547, 0.0, 875.2734375, 1042.875),
    ],
)
def test_full_eos_remaining_rack_proposals_match_the_guard(
    confidence: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> None:
    assert _guard().should_suppress(
        _object(
            confidence=confidence,
            left=left,
            top=top,
            width=width,
            height=height,
        )
    )


@pytest.mark.parametrize(
    ("confidence", "left", "top", "width", "height"),
    [
        # Confirmed close-up real track from e1a7776b: very large/full-height,
        # but centered and high-confidence.
        (0.98828125, 815.34375, 49.25780487060547, 878.0078125, 1017.4609375),
        # Confirmed real track from 859308c: top/full-height but on right edge.
        (0.430908203125, 1621.41015625, 5.179691314697266, 296.58984375, 1065.5860595703125),
        # Low-confidence real test2 proposal: neither left-edge nor giant.
        (0.1895751953125, 834.0, 136.0, 220.0, 320.0),
    ],
)
def test_confirmed_real_regression_proposals_do_not_match(
    confidence: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> None:
    assert not _guard().should_suppress(
        _object(
            confidence=confidence,
            left=left,
            top=top,
            width=width,
            height=height,
        )
    )


@pytest.mark.parametrize(
    "obj",
    [
        _object(component_id=2),
        _object(class_id=1),
        _object(confidence=0.551),
        _object(left=60.0),
        _object(top=25.0),
        _object(width=760.0),
        _object(width=1000.0),
        _object(height=990.0),
        _object(left=30.0, width=1000.0),
    ],
)
def test_no_single_signal_can_suppress_an_object(obj: object) -> None:
    assert not _guard().should_suppress(obj)


def test_disabled_guard_never_suppresses() -> None:
    assert not _guard(_config(enabled=False)).should_suppress(_object())


def test_config_is_loaded_from_top_level_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
person_pretracker_guard:
  enabled: true
  max_confidence: 0.225
  max_left_ratio: 0.025
  max_top_ratio: 0.015
  min_width_ratio: 0.41
  max_width_ratio: 0.49
  min_height_ratio: 0.94
  max_right_ratio: 0.51
""",
        encoding="utf-8",
    )

    config = PeopleNetPretrackerGuardConfig.from_file(path)

    assert config.enabled
    assert config.max_confidence == pytest.approx(0.225)
    assert config.min_width_ratio == pytest.approx(0.41)
    assert config.max_right_ratio == pytest.approx(0.51)


@pytest.mark.parametrize(
    "section",
    [
        "enabled: 'yes'",
        "max_confidence: 1.1",
        "min_width_ratio: 0.6\n  max_width_ratio: 0.5",
        "unknown_threshold: 0.1",
    ],
)
def test_invalid_guard_config_fails_fast(tmp_path: Path, section: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"person_pretracker_guard:\n  {section}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="person_pretracker_guard"):
        PeopleNetPretrackerGuardConfig.from_file(path)


class _Node:
    def __init__(self, data: object, next_node: _Node | None = None):
        self.data = data
        self.next = next_node


def _glist(values: list[object]) -> _Node | None:
    node = None
    for value in reversed(values):
        node = _Node(value, node)
    return node


def test_callback_removes_only_matching_pgie_person_metadata() -> None:
    giant = _object()
    real_person = _object(confidence=0.8, left=600.0, top=100.0, width=400.0, height=800.0)
    auxiliary = _object(class_id=1)
    frame_meta = SimpleNamespace(
        source_id=0,
        frame_num=123,
        obj_meta_list=_glist([giant, real_person, auxiliary]),
    )
    batch_meta = SimpleNamespace(frame_meta_list=_glist([frame_meta]))
    removed: list[tuple[object, object]] = []
    pyds = SimpleNamespace(
        gst_buffer_get_nvds_batch_meta=lambda _buffer_hash: batch_meta,
        NvDsFrameMeta=SimpleNamespace(cast=lambda value: value),
        NvDsObjectMeta=SimpleNamespace(cast=lambda value: value),
        nvds_remove_obj_meta_from_frame=lambda frame, obj: removed.append((frame, obj)),
    )
    runtime = SimpleNamespace(
        Gst=SimpleNamespace(PadProbeReturn=SimpleNamespace(OK="ok")),
        pyds=pyds,
    )
    guard = PeopleNetPretrackerGuard(
        runtime,
        _config(),
        pgie_unique_id=1,
        person_class_ids=(0,),
        frame_width=1920,
        frame_height=1080,
    )
    buffer = object()
    info = SimpleNamespace(get_buffer=lambda: buffer)

    assert guard.callback(None, info) == "ok"
    assert removed == [(frame_meta, giant)]
    assert guard.stats().frames == 1
    assert guard.stats().verified_people == 2
    assert guard.stats().suppressed == 1
    assert guard.stats().errors == 0
