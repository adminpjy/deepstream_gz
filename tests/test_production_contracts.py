from __future__ import annotations

import pytest

from deepstream_ai.production.contracts import FeatureSet, SessionRequest


def test_session_request_parses_production_controls() -> None:
    request = SessionRequest.from_mapping(
        {
            "cameraId": "room-a-01",
            "streamUrl": "rtsp://user:secret@10.0.0.8/live",
            "nominalFps": 25,
            "features": {
                "smoking": True,
                "eating": False,
                "drinking": True,
                "phone": True,
                "leftObject": True,
                "largeObjectMoving": False,
            },
            "exitPolicy": {"personAbsentSeconds": 30},
        }
    )

    assert request.camera_id == "room-a-01"
    assert request.features == FeatureSet(
        smoking=True,
        eating=False,
        drinking=True,
        phone=True,
        left_object=True,
        large_object_moving=False,
    )
    assert request.exit_policy.person_absent_seconds == 30
    assert "secret" not in request.as_dict(redact_url=True)["streamUrl"]


def test_phone_call_alias_is_supported() -> None:
    request = SessionRequest.from_mapping(
        {
            "cameraId": "room-a-01",
            "streamUrl": "rtsp://10.0.0.8/live",
            "features": {"phoneCall": True},
        }
    )
    assert request.features.phone is True
    assert request.as_dict()["features"]["phone"] is True


def test_core_recognition_cannot_be_disabled_through_feature_payload() -> None:
    request = SessionRequest.from_mapping(
        {
            "cameraId": "room-a-01",
            "streamUrl": "rtsp://10.0.0.8/live",
            "features": {
                "personDetection": False,
                "faceRecognition": False,
                "smoking": True,
            },
        }
    )

    assert request.features.smoking is True
    assert not hasattr(request.features, "person_detection")
    assert not hasattr(request.features, "face_recognition")


def test_session_request_rejects_invalid_rtsp_and_timeout() -> None:
    with pytest.raises(ValueError, match="rtsp"):
        SessionRequest.from_mapping(
            {"cameraId": "room-a-01", "streamUrl": "http://10.0.0.8/live"}
        )

    with pytest.raises(ValueError, match="personAbsentSeconds"):
        SessionRequest.from_mapping(
            {
                "cameraId": "room-a-01",
                "streamUrl": "rtsp://10.0.0.8/live",
                "exitPolicy": {"personAbsentSeconds": 0},
            }
        )
