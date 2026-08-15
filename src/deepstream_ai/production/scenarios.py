"""Independent production scenario processors.

Each processor consumes the same normalized FramePacket contract and emits only
standard RecognitionEvent objects. Processors never call each other.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from deepstream_ai.domain import BehaviorType
from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.production.contracts import FeatureSet, LeftObjectPolicy, RecognitionEvent
from deepstream_ai.production.publishers import ResultPublisher

LOGGER = logging.getLogger(__name__)


class ScenarioProcessor(Protocol):
    def process(self, packet: FramePacket) -> None: ...
    def on_person_absent(self) -> None: ...
    def close(self) -> None: ...


class BehaviorScenarioProcessor:
    def __init__(
        self,
        *,
        session_id: str,
        camera_id: str,
        behavior: BehaviorType,
        publisher: ResultPublisher,
        cooldown_sec: float = 5.0,
    ) -> None:
        self.session_id = session_id
        self.camera_id = camera_id
        self.behavior = behavior
        self.publisher = publisher
        self.cooldown_sec = float(cooldown_sec)
        self._last_emit: dict[str, datetime] = {}

    def process(self, packet: FramePacket) -> None:
        for detection in packet.behaviors:
            if detection.behavior is not self.behavior:
                continue
            track_id = str(detection.track_id)
            previous = self._last_emit.get(track_id)
            if previous is not None:
                elapsed = (detection.timestamp - previous).total_seconds()
                if 0 <= elapsed < self.cooldown_sec:
                    continue
            self._last_emit[track_id] = detection.timestamp
            self.publisher.publish(
                RecognitionEvent.create(
                    session_id=self.session_id,
                    camera_id=self.camera_id,
                    event_type=self.behavior.value.upper(),
                    track_id=detection.track_id,
                    timestamp=detection.timestamp,
                    confidence=detection.confidence,
                    extra={"modelName": detection.model_name},
                )
            )

    def on_person_absent(self) -> None:
        return

    def close(self) -> None:
        self._last_emit.clear()


@dataclass(frozen=True, slots=True)
class SceneDiffResult:
    changed: bool
    area_ratio: float
    boxes: tuple[tuple[int, int, int, int], ...]
    mask: np.ndarray


class SceneDiffer:
    """Lighting-tolerant deterministic OpenCV scene difference."""

    def __init__(self, policy: LeftObjectPolicy, *, analysis_width: int = 960) -> None:
        self.policy = policy
        self.analysis_width = int(analysis_width)

    @staticmethod
    def _cv2():
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("OpenCV 不可用，无法执行物品遗留检测") from exc
        return cv2

    def prepare(self, image: np.ndarray) -> np.ndarray:
        cv2 = self._cv2()
        if image is None or image.size == 0:
            raise ValueError("scene image is empty")
        if image.ndim == 2:
            gray = image
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if width > self.analysis_width:
            scale = self.analysis_width / float(width)
            gray = cv2.resize(
                gray,
                (self.analysis_width, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def compare_prepared(self, before: np.ndarray, after: np.ndarray) -> SceneDiffResult:
        cv2 = self._cv2()
        if before.shape != after.shape:
            after = cv2.resize(
                after,
                (before.shape[1], before.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        delta = cv2.absdiff(before, after)
        _, mask = cv2.threshold(
            delta,
            self.policy.pixel_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        height, width = mask.shape
        frame_area = max(1, height * width)
        min_component = max(
            16,
            round(frame_area * self.policy.min_component_area_ratio),
        )
        boxes: list[tuple[int, int, int, int]] = []
        kept = np.zeros_like(mask)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        for index in range(1, count):
            x, y, w, h, area = [int(value) for value in stats[index]]
            if area < min_component:
                continue
            kept[y : y + h, x : x + w] = np.maximum(
                kept[y : y + h, x : x + w],
                mask[y : y + h, x : x + w],
            )
            boxes.append((x, y, w, h))
        area_ratio = float(np.count_nonzero(kept)) / float(frame_area)
        return SceneDiffResult(
            changed=area_ratio >= self.policy.min_area_ratio,
            area_ratio=area_ratio,
            boxes=tuple(boxes),
            mask=kept,
        )


class LeftObjectProcessor:
    """Compare the CPU-provided pre-entry baseline after confirmed person absence."""

    def __init__(
        self,
        *,
        session_id: str,
        camera_id: str,
        baseline_path: str | Path,
        output_dir: str | Path,
        policy: LeftObjectPolicy,
        publisher: ResultPublisher,
    ) -> None:
        self.session_id = session_id
        self.camera_id = camera_id
        self.baseline_path = Path(baseline_path).resolve()
        self.output_dir = Path(output_dir).resolve() / "left-object"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.publisher = publisher
        self.differ = SceneDiffer(policy)
        self._before_prepared = self._load_baseline()
        self._empty_frames: deque[tuple[datetime, np.ndarray, np.ndarray]] = deque(
            maxlen=policy.max_recent_frames
        )
        self._last_sample: datetime | None = None
        self._alarm_emitted = False

    def _load_baseline(self) -> np.ndarray:
        cv2 = self.differ._cv2()
        image = cv2.imread(str(self.baseline_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"无法读取物品遗留基准图片: {self.baseline_path}")
        return self.differ.prepare(image)

    def process(self, packet: FramePacket) -> None:
        if packet.tracks:
            self._empty_frames.clear()
            self._last_sample = None
            return
        if self._last_sample is not None:
            elapsed = (packet.timestamp - self._last_sample).total_seconds()
            if 0 <= elapsed < 0.5:
                return
        self._last_sample = packet.timestamp
        frame = np.array(packet.image, copy=True)
        prepared = self.differ.prepare(frame)
        self._empty_frames.append((packet.timestamp, prepared, frame))

    def on_person_absent(self) -> None:
        if self._alarm_emitted:
            return
        if len(self._empty_frames) < self.policy.confirm_frames:
            LOGGER.info(
                "[LEFT_OBJECT_SKIP] session=%s camera=%s reason=insufficient_stable_frames frames=%d",
                self.session_id,
                self.camera_id,
                len(self._empty_frames),
            )
            return
        candidates = list(self._empty_frames)[-self.policy.confirm_frames :]
        results = [
            self.differ.compare_prepared(self._before_prepared, prepared)
            for _timestamp, prepared, _frame in candidates
        ]
        if not all(result.changed for result in results):
            LOGGER.info(
                "[LEFT_OBJECT_CLEAR] session=%s camera=%s ratios=%s",
                self.session_id,
                self.camera_id,
                [round(result.area_ratio, 6) for result in results],
            )
            return

        latest_timestamp, _prepared, latest_frame = candidates[-1]
        latest = results[-1]
        median_ratio = float(statistics.median(r.area_ratio for r in results))
        before_out = self.output_dir / "before.jpg"
        after_out = self.output_dir / "after.jpg"
        diff_out = self.output_dir / "diff.jpg"
        cv2 = self.differ._cv2()
        original = cv2.imread(str(self.baseline_path), cv2.IMREAD_COLOR)
        if original is not None:
            cv2.imwrite(str(before_out), original)
        after_bgr = (
            cv2.cvtColor(latest_frame, cv2.COLOR_RGBA2BGR)
            if latest_frame.ndim == 3 and latest_frame.shape[2] == 4
            else latest_frame
        )
        cv2.imwrite(str(after_out), after_bgr)
        cv2.imwrite(str(diff_out), latest.mask)
        confidence = min(
            1.0,
            median_ratio / max(self.policy.min_area_ratio * 4.0, 1e-9),
        )
        self.publisher.publish(
            RecognitionEvent.create(
                session_id=self.session_id,
                camera_id=self.camera_id,
                event_type="LEFT_OBJECT",
                timestamp=latest_timestamp,
                confidence=confidence,
                snapshot=str(after_out),
                extra={
                    "beforeImage": str(before_out),
                    "afterImage": str(after_out),
                    "diffImage": str(diff_out),
                    "changeAreaRatio": median_ratio,
                    "boxes": [list(box) for box in latest.boxes],
                    "detectionMode": "pre_entry_vs_post_exit_scene_diff",
                },
            )
        )
        self._alarm_emitted = True
        LOGGER.warning(
            "[LEFT_OBJECT_ALARM] session=%s camera=%s area_ratio=%.6f boxes=%d",
            self.session_id,
            self.camera_id,
            median_ratio,
            len(latest.boxes),
        )

    def close(self) -> None:
        self._empty_frames.clear()


class ScenarioManager:
    def __init__(
        self,
        *,
        session_id: str,
        camera_id: str,
        features: FeatureSet,
        publisher: ResultPublisher,
        output_dir: str | Path,
        left_object_policy: LeftObjectPolicy,
        baseline_path: str | Path | None,
    ) -> None:
        if features.large_object_moving:
            raise NotImplementedError("largeObjectMoving is reserved but not implemented")
        processors: list[ScenarioProcessor] = []
        if features.smoking:
            processors.append(
                BehaviorScenarioProcessor(
                    session_id=session_id,
                    camera_id=camera_id,
                    behavior=BehaviorType.SMOKING,
                    publisher=publisher,
                )
            )
        if features.eating:
            processors.append(
                BehaviorScenarioProcessor(
                    session_id=session_id,
                    camera_id=camera_id,
                    behavior=BehaviorType.EATING,
                    publisher=publisher,
                )
            )
        if features.drinking:
            processors.append(
                BehaviorScenarioProcessor(
                    session_id=session_id,
                    camera_id=camera_id,
                    behavior=BehaviorType.DRINKING,
                    publisher=publisher,
                )
            )
        if features.phone:
            processors.append(
                BehaviorScenarioProcessor(
                    session_id=session_id,
                    camera_id=camera_id,
                    behavior=BehaviorType.PHONE,
                    publisher=publisher,
                )
            )
        if features.left_object:
            if baseline_path is None:
                raise ValueError("leftObject=true 时必须提供进入前基准图片")
            processors.append(
                LeftObjectProcessor(
                    session_id=session_id,
                    camera_id=camera_id,
                    baseline_path=baseline_path,
                    output_dir=output_dir,
                    policy=left_object_policy,
                    publisher=publisher,
                )
            )
        self.processors = tuple(processors)

    def process(self, packet: FramePacket) -> None:
        for processor in self.processors:
            try:
                processor.process(packet)
            except Exception:
                LOGGER.exception(
                    "场景处理失败 processor=%s camera=%s frame=%s",
                    type(processor).__name__,
                    packet.camera_id,
                    packet.frame_number,
                )

    def on_person_absent(self) -> None:
        for processor in self.processors:
            try:
                processor.on_person_absent()
            except Exception:
                LOGGER.exception(
                    "人员离场场景收尾失败 processor=%s",
                    type(processor).__name__,
                )

    def close(self) -> None:
        for processor in reversed(self.processors):
            try:
                processor.close()
            except Exception:
                LOGGER.exception("关闭场景处理器失败: %s", type(processor).__name__)


__all__ = [
    "BehaviorScenarioProcessor",
    "LeftObjectProcessor",
    "ScenarioManager",
    "ScenarioProcessor",
    "SceneDiffResult",
    "SceneDiffer",
]
