from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from deepstream_ai.analytics import AnalyticsDispatcher, _jsonable
from deepstream_ai.config import load_config
from deepstream_ai.domain import BoundingBox, FaceDetection, Track
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.snapshot import FilesystemSnapshotStore


def _config(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs/config.yaml"
    path.write_text(
        """
source: {type: file, path: videos/test.mp4}
person: {enabled: true, config_file: models/person.txt}
tracker: {config_file: configs/tracker.yml}
face: {enabled: false}
snapshot:
  enabled: true
  root: output/snapshot
  jpeg_quality: 90
  person_decision_delay_sec: 1
output:
  enabled: false
  events_enabled: true
  events_path: output/events.jsonl
runtime: {strict_assets: false}
""",
        encoding="utf-8",
    )
    return load_config(path)


def _packet(
    *,
    with_face: bool,
    timestamp: datetime | None = None,
    face_score: float = 0.95,
    face_bbox: BoundingBox | None = None,
    image_value: int = 0,
) -> FramePacket:
    timestamp = timestamp or datetime.now(UTC)
    track = Track("camera-a", 7, timestamp, BoundingBox(10, 10, 90, 110), 0.9)
    faces = (
        (
            FaceDetection(
                "camera-a",
                7,
                timestamp,
                face_bbox or BoundingBox(35, 20, 65, 50),
                face_score,
            ),
        )
        if with_face
        else ()
    )
    image = np.full((120, 120, 4), image_value, dtype=np.uint8)
    image[..., 3] = 255
    return FramePacket("camera-a", 1, timestamp, image, (track,), faces, ())


def test_person_without_face_saves_one_best_crop(tmp_path: Path) -> None:
    dispatcher = AnalyticsDispatcher(_config(tmp_path))
    dispatcher.start()
    assert dispatcher.submit(_packet(with_face=False))
    dispatcher.close()

    assert len(list((tmp_path / "output/snapshot/person").glob("*.jpg"))) == 1
    assert not (tmp_path / "output/snapshot/face").exists()


def test_jsonable_does_not_publish_private_track_metadata() -> None:
    timestamp = datetime.now(UTC)
    track = Track(
        "camera-a",
        7,
        timestamp,
        BoundingBox(10, 10, 90, 110),
        0.9,
        metadata={
            "class_id": 0,
            "_tracker_reid_embedding": np.ones(256, dtype=np.float32),
        },
    )

    payload = _jsonable(track)

    assert payload["metadata"] == {"class_id": 0}


def test_face_without_recognition_is_saved_as_unknown_upper_body(tmp_path: Path) -> None:
    dispatcher = AnalyticsDispatcher(_config(tmp_path))
    dispatcher.start()
    assert dispatcher.submit(_packet(with_face=True))
    dispatcher.close()

    assert len(list((tmp_path / "output/snapshot/face/unknow").glob("*.jpg"))) == 1
    assert not list((tmp_path / "output/snapshot/person").glob("*.jpg"))
    journal = (tmp_path / "output/events.jsonl").read_text(encoding="utf-8")
    assert '"event_type":"face"' in journal
    assert '"event_type":"snapshot"' in journal


def test_identity_event_is_emitted_once_per_track(tmp_path: Path) -> None:
    dispatcher = AnalyticsDispatcher(_config(tmp_path))
    packet = _packet(with_face=True)
    dispatcher.start()
    assert dispatcher.submit(packet)
    assert dispatcher.submit(packet)
    dispatcher.close()

    events = [
        json.loads(line)
        for line in (tmp_path / "output/events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    identities = [event for event in events if event["event_type"] == "identity"]
    assert len(identities) == 1
    assert identities[0]["camera_id"] == "camera-a"
    assert identities[0]["track_id"] == "7"


def test_identity_result_does_not_freeze_the_evidence_frame(tmp_path: Path) -> None:
    dispatcher = AnalyticsDispatcher(_config(tmp_path))
    first_timestamp = datetime(2026, 8, 11, tzinfo=UTC)
    better_timestamp = first_timestamp + timedelta(milliseconds=40)
    dispatcher.start()
    assert dispatcher.submit(
        _packet(
            with_face=True,
            timestamp=first_timestamp,
            face_score=0.2,
            face_bbox=BoundingBox(40, 25, 52, 37),
            image_value=20,
        )
    )
    assert dispatcher.submit(
        _packet(
            with_face=True,
            timestamp=better_timestamp,
            face_score=0.95,
            face_bbox=BoundingBox(30, 18, 70, 58),
            image_value=220,
        )
    )
    dispatcher.close()

    events = [
        json.loads(line)
        for line in (tmp_path / "output/events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    identities = [event for event in events if event["event_type"] == "identity"]
    snapshots = [event for event in events if event["event_type"] == "snapshot"]
    assert len(identities) == 1
    assert len(snapshots) == 1
    assert snapshots[0]["payload"]["source"] == "best_face"
    assert snapshots[0]["payload"]["timestamp"] == better_timestamp.isoformat()
    assert snapshots[0]["payload"]["person_bbox"] == [10.0, 10.0, 90.0, 110.0]
    assert snapshots[0]["payload"]["face_bbox"] == [30.0, 18.0, 70.0, 58.0]
    crop_bbox = snapshots[0]["payload"]["evidence_bbox"]
    face_bbox = snapshots[0]["payload"]["face_bbox"]
    assert crop_bbox[0] <= face_bbox[0]
    assert crop_bbox[1] <= face_bbox[1]
    assert crop_bbox[2] >= face_bbox[2]
    assert crop_bbox[3] >= face_bbox[3]
    assert snapshots[0]["payload"]["full_frame_fallback"] is False
    assert len(list((tmp_path / "output/snapshot/face/unknow").glob("*.jpg"))) == 1


def test_snapshot_disk_write_runs_only_on_analytics_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_threads: list[str] = []
    original_write = FilesystemSnapshotStore.write

    def recording_write(self, relative_path, payload):
        write_threads.append(threading.current_thread().name)
        return original_write(self, relative_path, payload)

    monkeypatch.setattr(FilesystemSnapshotStore, "write", recording_write)
    dispatcher = AnalyticsDispatcher(_config(tmp_path))
    dispatcher.start()
    assert dispatcher.submit(_packet(with_face=True))
    dispatcher.close()

    assert write_threads == ["analytics-worker"]
