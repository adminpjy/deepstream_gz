from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from deepstream_ai.activity import ActivityAwareConsumer, PersonActivityTracker
from deepstream_ai.domain import BoundingBox, Track
from deepstream_ai.pipeline.metadata import FramePacket

NOW = datetime(2026, 8, 11, tzinfo=UTC)
SECOND = 1_000_000_000


def packet(
    stream_time_ns: int | None,
    *,
    frame_number: int = 0,
    people: int = 0,
) -> FramePacket:
    tracks = tuple(
        Track(
            camera_id="camera-a",
            track_id=index + 1,
            timestamp=NOW,
            bbox=BoundingBox(1, 1, 8, 10),
            confidence=0.9,
        )
        for index in range(people)
    )
    return FramePacket(
        camera_id="camera-a",
        frame_number=frame_number,
        timestamp=NOW,
        image=np.zeros((12, 12, 4), dtype=np.uint8),
        tracks=tracks,
        faces=(),
        behaviors=(),
        stream_time_ns=stream_time_ns,
    )


def test_person_free_pts_triggers_at_exact_timeout() -> None:
    activity = PersonActivityTracker(10)

    assert not activity.observe(packet(5 * SECOND, frame_number=1))
    assert not activity.observe(packet(15 * SECOND - 1, frame_number=2))
    assert activity.observe(packet(15 * SECOND, frame_number=3))

    snapshot = activity.snapshot()
    assert snapshot.frames == 3
    assert snapshot.person_frames == 0
    assert snapshot.idle_seconds == 10.0
    assert snapshot.idle_triggered
    assert activity.idle_event.is_set()


def test_person_frame_resets_the_person_free_window() -> None:
    activity = PersonActivityTracker(10)

    assert not activity.observe(packet(0, frame_number=1))
    assert not activity.observe(packet(9 * SECOND, frame_number=2, people=2))
    assert not activity.observe(packet(19 * SECOND - 1, frame_number=3))
    assert activity.observe(packet(19 * SECOND, frame_number=4))

    snapshot = activity.snapshot()
    assert snapshot.frames == 4
    assert snapshot.person_frames == 1
    assert snapshot.person_detections == 2
    assert snapshot.last_person_stream_time_ns == 9 * SECOND
    assert snapshot.idle_seconds == 10.0


def test_pts_rollback_starts_a_fresh_idle_window() -> None:
    activity = PersonActivityTracker(10)

    assert not activity.observe(packet(100 * SECOND, frame_number=1))
    assert not activity.observe(packet(109 * SECOND, frame_number=2))

    # A seek or RTSP reconnect must not inherit nine seconds from the old PTS epoch.
    assert not activity.observe(packet(2 * SECOND, frame_number=3))
    assert not activity.observe(packet(12 * SECOND - 1, frame_number=4))
    assert activity.observe(packet(12 * SECOND, frame_number=5))

    snapshot = activity.snapshot()
    assert snapshot.current_stream_time_ns == 12 * SECOND
    assert snapshot.idle_seconds == 10.0


def test_no_frames_and_invalid_pts_never_advance_idle_time() -> None:
    activity = PersonActivityTracker(10)

    initial = activity.snapshot()
    assert initial.frames == 0
    assert initial.idle_seconds == 0
    assert not initial.idle_triggered

    assert not activity.observe(packet(None))
    assert not activity.observe(packet(-1))
    unchanged = activity.snapshot()
    assert unchanged.frames == 0
    assert unchanged.current_stream_time_ns is None
    assert unchanged.idle_seconds == 0
    assert not activity.idle_event.is_set()


class _Delegate:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted
        self.packets: list[FramePacket] = []

    def submit(self, value: FramePacket) -> bool:
        self.packets.append(value)
        return self.accepted

    def identity_label(self, camera_id: str, track_id: int | str) -> str:
        return f"{camera_id}:{track_id}"


class _Preview:
    def __init__(self) -> None:
        self.packets: list[FramePacket] = []

    def submit(self, value: FramePacket) -> None:
        self.packets.append(value)


def test_activity_is_observed_even_when_the_analytics_queue_rejects_a_frame() -> None:
    activity = PersonActivityTracker(10)
    delegate = _Delegate(accepted=False)
    preview = _Preview()
    consumer = ActivityAwareConsumer(delegate, activity, preview)
    value = packet(0, people=1)

    assert not consumer.submit(value)
    assert delegate.packets == [value]
    assert preview.packets == [value]
    assert activity.snapshot().person_detections == 1
    assert consumer.identity_label("camera-a", 7) == "camera-a:7"
