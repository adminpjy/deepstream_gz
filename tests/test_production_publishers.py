from __future__ import annotations

import json

from deepstream_ai.production.contracts import RecognitionEvent
from deepstream_ai.production.publishers import JsonlResultPublisher


def test_jsonl_result_publisher_uses_stable_internal_event_contract(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    publisher = JsonlResultPublisher(path)
    event = RecognitionEvent.create(
        session_id="session-1",
        camera_id="camera-01",
        event_type="SMOKING",
        track_id=42,
        person_id="employee-8",
        confidence=0.91,
        extra={"source": "test"},
    )

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
