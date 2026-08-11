from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from deepstream_ai.database import FaceVectorMatch, StoredFaceVector
from deepstream_ai.domain import BoundingBox, FaceDetection
from deepstream_ai.face import (
    AdaFaceONNXAdapter,
    AdaFaceTensorRTAdapter,
    FaceFusionConfig,
    FaceQualityScorer,
    FaceQualityWeights,
    FaceRecognitionConfig,
    FaceRecognitionService,
    InvalidFaceInput,
    MultiFrameFaceFusion,
    normalize_embedding,
)

NOW = datetime(2026, 8, 10, tzinfo=UTC)


def face(
    score: float,
    *,
    frame: int = 0,
    track_id: int = 7,
    metadata: dict[str, float] | None = None,
) -> FaceDetection:
    return FaceDetection(
        camera_id="cam-1",
        track_id=track_id,
        timestamp=NOW + timedelta(milliseconds=frame * 40),
        bbox=BoundingBox(10, 10, 90, 90),
        score=score,
        landmarks=((25, 35), (75, 35), (50, 55), (32, 72), (68, 72)),
        metadata=metadata or {"blur_score": 1.0, "frontal_score": 1.0},
    )


def test_multi_frame_selector_scores_and_fuses_normalized_embeddings() -> None:
    config = FaceFusionConfig(
        min_candidates=2,
        max_candidates=3,
        weights=FaceQualityWeights(face_score=1.0, size=0.0, blur=0.0, frontal=0.0),
    )
    fusion = MultiFrameFaceFusion(config)
    first = np.zeros(512, dtype=np.float32)
    first[0] = 1
    second = np.zeros(512, dtype=np.float32)
    second[1] = 1

    fusion.add(face(0.25), embedding=first)
    fusion.add(face(0.75, frame=1), embedding=second)

    assert fusion.is_ready("cam-1", 7)
    assert fusion.best("cam-1", 7).detection.score == 0.75
    result = fusion.fused_embedding("cam-1", 7)
    assert result.shape == (512,)
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert result[1] > result[0] > 0


def test_candidate_buffer_is_bounded_and_expirable() -> None:
    fusion = MultiFrameFaceFusion(FaceFusionConfig(min_candidates=1, max_candidates=2))
    embedding = np.ones(512, dtype=np.float32)
    for index in range(3):
        fusion.add(face(0.8, frame=index), embedding=embedding)
    assert fusion.count("cam-1", 7) == 2
    assert fusion.expire(NOW + timedelta(seconds=10)) == (("cam-1", 7),)
    assert fusion.count("cam-1", 7) == 0


def test_face_sharpness_luminance_uses_deepstream_rgba_order() -> None:
    config = FaceFusionConfig(
        min_candidates=1,
        max_candidates=1,
        blur_reference=100_000.0,
        weights=FaceQualityWeights(face_score=0, size=0, blur=1, frontal=0),
        frame_color_space="rgba",
    )
    scorer = FaceQualityScorer(config)
    pattern = (np.indices((20, 20)).sum(axis=0) % 2).astype(np.uint8) * 255
    red = np.zeros((20, 20, 4), dtype=np.uint8)
    blue = np.zeros_like(red)
    red[..., 0], red[..., 3] = pattern, 255
    blue[..., 2], blue[..., 3] = pattern, 255
    detection = FaceDetection(
        "cam-1",
        7,
        NOW,
        BoundingBox(0, 0, 20, 20),
        0.9,
    )

    assert (
        scorer.score(detection, crop=red).blur_score > scorer.score(detection, crop=blue).blur_score
    )


class _Input:
    name = "images"


class _Session:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.feed = None

    def get_inputs(self):
        return [_Input()]

    def run(self, outputs, feed):
        assert outputs is None
        self.feed = feed
        return [self.output]


def test_onnx_adapter_uses_injected_runtime_and_returns_real_normalized_vector() -> None:
    raw = np.arange(1, 513, dtype=np.float32)[None, :]
    session = _Session(raw)
    adapter = AdaFaceONNXAdapter(session=session)

    embedding = adapter.embed(np.full((80, 60, 3), 127, dtype=np.uint8))

    assert session.feed["images"].shape == (1, 3, 112, 112)
    assert embedding.shape == (512,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    assert np.allclose(embedding, raw[0] / np.linalg.norm(raw[0]))


def test_default_preprocessor_preserves_deepstream_rgba_channel_order() -> None:
    raw = np.ones((1, 512), dtype=np.float32)
    session = _Session(raw)
    adapter = AdaFaceONNXAdapter(session=session)
    pixel = np.array([[[255, 127, 0, 255]]], dtype=np.uint8)

    adapter.embed(pixel)
    tensor = session.feed["images"]

    assert tensor[0, 0, 0, 0] == pytest.approx(1.0)
    assert tensor[0, 1, 0, 0] == pytest.approx((127 - 127.5) / 127.5)
    assert tensor[0, 2, 0, 0] == pytest.approx(-1.0)


def test_adapters_reject_invalid_model_output_instead_of_faking_embedding() -> None:
    adapter = AdaFaceONNXAdapter(session=_Session(np.zeros((1, 512), dtype=np.float32)))
    with pytest.raises(InvalidFaceInput, match="zero norm"):
        adapter.embed(np.zeros((112, 112, 3), dtype=np.uint8))

    class Runner:
        def infer(self, input_tensor):
            return np.ones((1, 128), dtype=np.float32)

    with pytest.raises(InvalidFaceInput, match="512"):
        AdaFaceTensorRTAdapter(runner=Runner()).embed(np.zeros((112, 112, 3), dtype=np.uint8))


class _Embedder:
    def embed(self, crop):
        return normalize_embedding(np.arange(1, 513, dtype=np.float32))


class _Repository:
    def add(self, worker_id, embedding):
        return StoredFaceVector(worker_id, embedding)

    def find_nearest(self, embedding, *, min_similarity=None):
        return FaceVectorMatch("worker-42", 0.82, 1, NOW)


def test_recognition_service_waits_for_multiple_frames_then_matches() -> None:
    config = FaceRecognitionConfig(
        similarity_threshold=0.6,
        fusion=FaceFusionConfig(min_candidates=2, max_candidates=3),
    )
    service = FaceRecognitionService(_Embedder(), _Repository(), config)
    crop = np.full((112, 112, 3), 128, dtype=np.uint8)

    assert service.observe(face(0.8), crop) is None
    result = service.observe(face(0.9, frame=1), crop)

    assert result.worker_id == "worker-42"
    assert result.known
    assert result.similarity == pytest.approx(0.82)
    assert result.sample_count == 2
    assert service.result_for("cam-1", 7) is result


def test_normalize_embedding_enforces_512_finite_nonzero_values() -> None:
    with pytest.raises(InvalidFaceInput, match="512"):
        normalize_embedding(np.ones(511))
    with pytest.raises(InvalidFaceInput, match="NaN"):
        values = np.ones(512)
        values[4] = np.nan
        normalize_embedding(values)
