from __future__ import annotations

import json
import threading

from deepstream_ai.production.contracts import RecognitionEvent
from deepstream_ai.production.publishers import (
    JsonlResultPublisher,
    QueuedResultPublisher,
)


def _event(index: int = 1) -> RecognitionEvent:
    return RecognitionEvent.create(
        session_id="session-1",
        camera_id="camera-01",
        event_type="SMOKING",
        track_id=index,
        person_id="employee-8",
        confidence=0.91,
        extra={"source": "test"},
    )


def test_jsonl_result_publisher_uses_stable_internal_event_contract(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    publisher = JsonlResultPublisher(path)
    event = _event(42)

    publisher.publish(event)
    publisher.close()

    stored = json.loads(path.read_text(encoding="utf-8").strip())
    assert stored["eventId"] == event.event_id
    assert stored["sessionId"] == "session-1"
    assert stored["cameraId"] == "camera-01"
    assert stored["eventType"] == "SMOKING"
    assert stored["trackId"] == "42"
    assert stored["personId"] == "employee-8"
    assert stored["confidence"] == 0.91


def test_result_queue_backpressure_never_raises_into_recognition(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingPublisher:
        def publish(self, event: RecognitionEvent) -> None:
            del event
            started.set()
            release.wait(5)

        def close(self) -> None:
            return

    dead_letter = tmp_path / "dead-letter.jsonl"
    publisher = QueuedResultPublisher(
        BlockingPublisher(),
        queue_size=16,
        dead_letter_path=dead_letter,
    )
    publisher.publish(_event(0))
    assert started.wait(1)

    # One item is blocked in delivery, so filling more than queue capacity must
    # degrade to dead-letter persistence rather than raise to analytics code.
    for index in range(1, 20):
        publisher.publish(_event(index))

    assert publisher.stats()["queue_full"] >= 1
    assert publisher.stats()["dead_lettered"] >= 1
    assert dead_letter.is_file()

    release.set()
    publisher.close()
