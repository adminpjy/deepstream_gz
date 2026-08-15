from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

import deepstream_ai.face_registration as registration
from deepstream_ai.database import FaceVectorMatch, StoredFaceVector
from deepstream_ai.domain import BoundingBox, FaceDetection
from deepstream_ai.face_registration import FaceRegistrationPolicy, FaceRegistrationService

NOW = datetime(2026, 8, 15, tzinfo=UTC)


class FakeDetector:
    def __init__(self, detections):
        self.detections = tuple(detections)

    def detect(self, image_bgr):
        del image_bgr
        return self.detections


class FakeAligner:
    def align(self, crop, detection):
        del detection
        return np.ascontiguousarray(crop)


class FakeEmbedder:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def embed(self, _crop):
        vector = self.vector / np.linalg.norm(self.vector)
        return vector.astype(np.float32)


class FakeRepository:
    def __init__(self, existing=(), competitor=None):
        self.existing = list(existing)
        self.competitor = competitor
        self.added = []

    def list_worker(self, worker_id):
        return tuple(row for row in self.existing if row.worker_id == worker_id)

    def find_nearest_other(self, embedding, worker_id):
        del embedding, worker_id
        return self.competitor

    def add(self, worker_id, embedding, **metadata):
        row = StoredFaceVector(
            worker_id=worker_id,
            embedding=embedding,
            record_id=len(self.existing) + 1,
            created_at=NOW,
            sample_type=metadata.get("sample_type"),
            pose=metadata.get("pose"),
            quality=metadata.get("quality"),
            image_sha256=metadata.get("image_sha256"),
        )
        self.existing.append(row)
        self.added.append(row)
        return row


def face(*, track_id=1, score=0.94, frontal=0.94, blur=0.90, x1=170, y1=120, size=220):
    bbox = BoundingBox(x1, y1, x1 + size, y1 + size)
    cx = x1 + size / 2
    eye_y = y1 + size * 0.38
    eye_dx = size * 0.18
    landmarks = (
        (cx - eye_dx, eye_y),
        (cx + eye_dx, eye_y),
        (cx, y1 + size * 0.56),
        (cx - size * 0.12, y1 + size * 0.73),
        (cx + size * 0.12, y1 + size * 0.73),
    )
    return FaceDetection(
        camera_id="register",
        track_id=track_id,
        timestamp=NOW,
        bbox=bbox,
        score=score,
        landmarks=landmarks,
        metadata={"frontal_score": frontal, "blur_score": blur},
    )


def manager(detector, embedder, repository):
    return FaceRegistrationService(
        detector,
        embedder,
        repository,
        aligner=FakeAligner(),
        policy=FaceRegistrationPolicy(),
    )


def patch_image(monkeypatch):
    image = np.full((640, 720, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(registration, "_decode_image", lambda _payload: image)


def test_clear_single_frontal_face_is_stored_as_primary(monkeypatch):
    patch_image(monkeypatch)
    repo = FakeRepository()
    service = manager(FakeDetector([face()]), FakeEmbedder(np.ones(512)), repo)

    result = service.register("worker-1", b"jpeg", mode="primary", filename="front.jpg")

    assert result["accepted"] is True
    assert result["stored"] is True
    assert result["template_count"] == 1
    assert result["metrics"]["face_width"] == 220
    assert repo.added[0].sample_type == "primary"
    assert repo.added[0].pose == "front"


def test_multiple_faces_are_rejected_before_embedding(monkeypatch):
    patch_image(monkeypatch)
    repo = FakeRepository()
    service = manager(
        FakeDetector([face(track_id=1), face(track_id=2, x1=410, y1=160, size=170)]),
        FakeEmbedder(np.ones(512)),
        repo,
    )

    result = service.register("worker-1", b"jpeg", mode="primary")

    assert result["accepted"] is False
    assert result["stored"] is False
    assert any("2 张人脸" in item for item in result["issues"])
    assert repo.added == []


def test_small_or_blurry_primary_is_rejected_with_actionable_reasons(monkeypatch):
    patch_image(monkeypatch)
    repo = FakeRepository()
    service = manager(
        FakeDetector([face(score=0.68, frontal=0.60, blur=0.20, size=80)]),
        FakeEmbedder(np.ones(512)),
        repo,
    )

    result = service.register("worker-1", b"jpeg", mode="primary")

    joined = " ".join(result["issues"])
    assert result["accepted"] is False
    assert "太小" in joined
    assert "清晰度不足" in joined
    assert "角度过偏" in joined
    assert repo.added == []


def test_supplement_appends_pose_template_instead_of_replacing_primary(monkeypatch):
    patch_image(monkeypatch)
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    supplement = np.zeros(512, dtype=np.float32)
    supplement[0] = 0.80
    supplement[1] = 0.60
    existing = StoredFaceVector(
        worker_id="worker-1",
        embedding=base,
        record_id=1,
        created_at=NOW,
        sample_type="primary",
        pose="front",
        quality=0.90,
    )
    repo = FakeRepository(existing=[existing])
    service = manager(
        FakeDetector([face(frontal=0.66, blur=0.90)]),
        FakeEmbedder(supplement),
        repo,
    )

    result = service.register("worker-1", b"jpeg", mode="supplement", filename="left.jpg")

    assert result["accepted"] is True
    assert result["stored"] is True
    assert result["template_count"] == 2
    assert len(repo.existing) == 2
    assert repo.existing[0].sample_type == "primary"
    assert repo.existing[1].sample_type == "supplement"


def test_supplement_is_rejected_when_it_does_not_match_worker(monkeypatch):
    patch_image(monkeypatch)
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    wrong = np.zeros(512, dtype=np.float32)
    wrong[1] = 1.0
    existing = StoredFaceVector(worker_id="worker-1", embedding=base, record_id=1)
    repo = FakeRepository(existing=[existing])
    service = manager(
        FakeDetector([face(frontal=0.70, blur=0.90)]),
        FakeEmbedder(wrong),
        repo,
    )

    result = service.register("worker-1", b"jpeg", mode="supplement")

    assert result["accepted"] is False
    assert result["stored"] is False
    assert any("差异过大" in item for item in result["issues"])
    assert len(repo.existing) == 1


def test_supplement_is_rejected_if_another_worker_is_materially_closer(monkeypatch):
    patch_image(monkeypatch)
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    supplement = np.zeros(512, dtype=np.float32)
    supplement[0] = 0.70
    supplement[1] = np.sqrt(1.0 - 0.70**2)
    existing = StoredFaceVector(worker_id="worker-1", embedding=base, record_id=1)
    competitor = FaceVectorMatch(worker_id="worker-2", similarity=0.82, record_id=2)
    repo = FakeRepository(existing=[existing], competitor=competitor)
    service = manager(
        FakeDetector([face(frontal=0.70, blur=0.90)]),
        FakeEmbedder(supplement),
        repo,
    )

    result = service.register("worker-1", b"jpeg", mode="supplement")

    assert result["accepted"] is False
    assert result["stored"] is False
    assert any("更接近其他人员 worker-2" in item for item in result["issues"])
    assert len(repo.existing) == 1
