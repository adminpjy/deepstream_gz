"""Bounded asynchronous live-preview rendering for service tasks."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepstream_ai.domain import BoundingBox
from deepstream_ai.pipeline.metadata import FramePacket

LOGGER = logging.getLogger(__name__)
_STOP = object()


@dataclass(slots=True)
class _SecondaryBoxHold:
    bbox: BoundingBox
    parent_bbox: BoundingBox
    updated_at: float


class PreviewWriter:
    """Keep one pending frame and atomically publish the newest annotated JPEG."""

    def __init__(
        self,
        destination: str | Path,
        *,
        identity_label: Callable[[str, int | str], str | None],
        max_fps: float = 5.0,
        max_width: int = 960,
        jpeg_quality: int = 80,
        secondary_hold_sec: float = 1.5,
    ) -> None:
        if max_fps <= 0:
            raise ValueError("max_fps must be positive")
        if max_width < 160:
            raise ValueError("max_width must be at least 160")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not 0.0 <= secondary_hold_sec <= 10.0:
            raise ValueError("secondary_hold_sec must be between 0 and 10")
        self.destination = Path(destination)
        self.identity_label = identity_label
        self.max_fps = float(max_fps)
        self.max_width = int(max_width)
        self.jpeg_quality = int(jpeg_quality)
        self.secondary_hold_sec = float(secondary_hold_sec)
        self._queue: queue.Queue[FramePacket | object] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._throttle_lock = threading.Lock()
        self._next_frame_at = 0.0
        self._stats_lock = threading.Lock()
        self._frames_written = 0
        self._frames_dropped = 0
        self._errors = 0
        self._secondary_holds: dict[tuple[str, int | str], _SecondaryBoxHold] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name=f"preview-{self.destination.parent.name}",
            daemon=True,
        )
        self._thread.start()

    def submit(self, packet: FramePacket) -> None:
        now = time.monotonic()
        with self._throttle_lock:
            if now < self._next_frame_at or self._stopping.is_set():
                return
            self._next_frame_at = now + 1.0 / self.max_fps
        try:
            self._queue.put_nowait(packet)
            return
        except queue.Full:
            with self._stats_lock:
                self._frames_dropped += 1
        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            with self._stats_lock:
                self._frames_dropped += 1

    def close(self, timeout: float = 5.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stopping.set()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(_STOP)
        thread.join(timeout)
        if thread.is_alive():
            LOGGER.warning("实时预览线程未在 %.1f 秒内结束", timeout)
        self._thread = None
        self._secondary_holds.clear()

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "preview_frames": self._frames_written,
                "preview_drops": self._frames_dropped,
                "preview_errors": self._errors,
            }

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, FramePacket)
                try:
                    payload = self._render(item)
                    self._atomic_write(payload)
                    with self._stats_lock:
                        self._frames_written += 1
                except Exception:
                    with self._stats_lock:
                        self._errors += 1
                    LOGGER.exception(
                        "生成实时预览失败 camera_id=%s frame=%s",
                        item.camera_id,
                        item.frame_number,
                    )
            finally:
                self._queue.task_done()

    def _render(self, packet: FramePacket) -> bytes:
        import cv2  # type: ignore[import-not-found]

        rgba = np.asarray(packet.image)
        image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        source_height, source_width = image.shape[:2]
        scale = min(1.0, self.max_width / max(1, source_width))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (round(source_width * scale), round(source_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        thickness = max(1, round(2 * scale))
        font_scale = max(0.45, 0.65 * scale)
        now = time.monotonic()

        secondary_by_track = {}
        for item in packet.faces:
            key = (item.camera_id, item.track_id)
            current = secondary_by_track.get(key)
            if current is None or item.score > current.score:
                secondary_by_track[key] = item

        active_keys: set[tuple[str, int | str]] = set()
        secondary_boxes: list[BoundingBox] = []
        for track in packet.tracks:
            key = (track.camera_id, track.track_id)
            active_keys.add(key)
            x1, y1, x2, y2 = (round(value * scale) for value in track.bbox.as_tuple())
            color = (34, 211, 238)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            label = f"person #{track.track_id}"
            identity = self.identity_label(track.camera_id, track.track_id)
            if identity:
                label = f"{label} {identity}"
            cv2.putText(
                image,
                label,
                (max(0, x1), max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            detected = secondary_by_track.get(key)
            if detected is not None:
                self._secondary_holds[key] = _SecondaryBoxHold(
                    bbox=detected.bbox,
                    parent_bbox=track.bbox,
                    updated_at=now,
                )
                secondary_boxes.append(detected.bbox)
                continue

            hold = self._secondary_holds.get(key)
            if hold is None or now - hold.updated_at > self.secondary_hold_sec:
                continue
            projected = _project_box(
                hold.bbox,
                hold.parent_bbox,
                track.bbox,
                source_width,
                source_height,
            )
            if projected is not None:
                secondary_boxes.append(projected)

        stale_keys = [
            key
            for key, hold in self._secondary_holds.items()
            if key not in active_keys and now - hold.updated_at > self.secondary_hold_sec
        ]
        for key in stale_keys:
            self._secondary_holds.pop(key, None)

        for box in secondary_boxes:
            x1, y1, x2, y2 = (round(value * scale) for value in box.as_tuple())
            cv2.rectangle(image, (x1, y1), (x2, y2), (92, 224, 122), thickness)

        cv2.putText(
            image,
            f"{packet.camera_id}  frame {packet.frame_number}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not success or encoded is None or encoded.size == 0:
            raise RuntimeError("OpenCV returned an empty preview JPEG")
        return encoded.tobytes()

    def _atomic_write(self, payload: bytes) -> None:
        temporary = self.destination.with_name(f".{self.destination.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, self.destination)
        finally:
            temporary.unlink(missing_ok=True)


def _project_box(
    old_box: BoundingBox,
    old_parent: BoundingBox,
    new_parent: BoundingBox,
    frame_width: int,
    frame_height: int,
) -> BoundingBox | None:
    if old_parent.width <= 1 or old_parent.height <= 1:
        return None
    left = (old_box.x1 - old_parent.x1) / old_parent.width
    top = (old_box.y1 - old_parent.y1) / old_parent.height
    right = (old_box.x2 - old_parent.x1) / old_parent.width
    bottom = (old_box.y2 - old_parent.y1) / old_parent.height
    try:
        projected = BoundingBox(
            new_parent.x1 + left * new_parent.width,
            new_parent.y1 + top * new_parent.height,
            new_parent.x1 + right * new_parent.width,
            new_parent.y1 + bottom * new_parent.height,
        )
    except ValueError:
        return None
    return projected.clipped(frame_width, frame_height)


__all__ = ["PreviewWriter"]
