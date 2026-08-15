from __future__ import annotations

import cv2
import numpy as np

from deepstream_ai.production.baseline import BaselineStore
from deepstream_ai.production.contracts import LeftObjectPolicy
from deepstream_ai.production.scenarios import SceneDiffer


def test_baseline_store_keeps_current_camera_image(tmp_path) -> None:
    image = np.full((120, 160, 3), 80, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    store = BaselineStore(tmp_path / "baselines")

    saved = store.save("camera-01", encoded.tobytes(), content_type="image/jpeg")
    current = store.current("camera-01")

    assert current is not None
    assert current.baseline_id == saved.baseline_id
    assert current.path.is_file()


def test_scene_differ_ignores_global_brightness_but_finds_new_object() -> None:
    policy = LeftObjectPolicy(
        pixel_threshold=28,
        min_area_ratio=0.005,
        min_component_area_ratio=0.001,
        confirm_frames=3,
        max_recent_frames=8,
    )
    differ = SceneDiffer(policy, analysis_width=320)
    before = np.full((240, 320, 3), 90, dtype=np.uint8)
    brighter = np.full((240, 320, 3), 105, dtype=np.uint8)
    after = brighter.copy()
    cv2.rectangle(after, (120, 100), (210, 190), (10, 10, 10), thickness=-1)

    before_prepared = differ.prepare(before)
    brightness_only = differ.compare_prepared(before_prepared, differ.prepare(brighter))
    new_object = differ.compare_prepared(before_prepared, differ.prepare(after))

    assert brightness_only.changed is False
    assert new_object.changed is True
    assert new_object.area_ratio >= policy.min_area_ratio
    assert new_object.boxes
