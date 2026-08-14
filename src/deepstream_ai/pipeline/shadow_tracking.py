"""Display-only NvDCF shadow-track metadata bridge.

Shadow targets are intentionally kept out of FramePacket business tracks.  The
tracker probe stores only lightweight bbox metadata for the live preview so a
short detector miss does not make the operator-visible box disappear.  Shadow
boxes never enter SCRFD, AdaFace, snapshots, alarms, or database work.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable

import yaml

from deepstream_ai.domain import BoundingBox, TrackId

LOGGER = logging.getLogger(__name__)
_SECTION = "shadow_tracking"
_MAX_FRAME_CACHE = 128


@dataclass(frozen=True, slots=True)
class ShadowTrackingConfig:
    enabled: bool = True
    display_max_age_sec: float = 1.5

    @classmethod
    def from_file(cls, config_path: str | Path) -> "ShadowTrackingConfig":
        try:
            raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        section = raw.get(_SECTION, {}) if isinstance(raw, dict) else {}
        if not isinstance(section, dict):
            section = {}
        result = cls(
            enabled=bool(section.get("enabled", True)),
            display_max_age_sec=float(section.get("display_max_age_sec", 1.5)),
        )
        if not 0.0 < result.display_max_age_sec <= 5.0:
            raise ValueError(f"{_SECTION}.display_max_age_sec must be in (0, 5]")
        return result


@dataclass(frozen=True, slots=True)
class ShadowDisplayTrack:
    camera_id: str
    track_id: TrackId
    raw_track_id: TrackId
    frame_number: int
    bbox: BoundingBox
    confidence: float
    tracker_age: int


@dataclass(frozen=True, slots=True)
class ShadowTrackingStats:
    frames: int
    objects: int
    hidden_by_age: int
    errors: int


_CURRENT: "ShadowTrackRegistry | None" = None
_CURRENT_LOCK = RLock()


def _iter_glist(head: Any, cast: Any):
    node = head
    while node is not None:
        try:
            yield cast(node.data)
        except StopIteration:
            return
        try:
            node = node.next
        except StopIteration:
            return


def _shadow_meta_type(pyds: Any) -> Any | None:
    direct = getattr(pyds, "NVDS_TRACKER_SHADOW_LIST_META", None)
    if direct is not None:
        return direct
    enum = getattr(pyds, "NvDsMetaType", None)
    return getattr(enum, "NVDS_TRACKER_SHADOW_LIST_META", None) if enum is not None else None


class ShadowTrackRegistry:
    """Read tracker shadow-list user meta and cache preview-safe current boxes."""

    def __init__(self, runtime: Any, config: Any, consumer: Any) -> None:
        self.runtime = runtime
        self.config = ShadowTrackingConfig.from_file(config.config_path)
        self.camera_by_pad = {
            index: source.camera_id for index, source in enumerate(config.enabled_sources)
        }
        person = config.pipeline.person
        self.person_class_ids = frozenset(int(value) for value in person.person_class_ids)
        presenter = getattr(consumer, "presentation_track_id", None)
        self.presenter: Callable[[str, TrackId], TrackId | None] | None = (
            presenter if callable(presenter) else None
        )
        self._frames: OrderedDict[tuple[str, int], tuple[ShadowDisplayTrack, ...]] = OrderedDict()
        self._shadow_since: dict[tuple[str, TrackId], float] = {}
        self._lock = RLock()
        self._frame_count = 0
        self._object_count = 0
        self._hidden_by_age = 0
        self._errors = 0
        self._warned_unavailable = False
        global _CURRENT
        with _CURRENT_LOCK:
            _CURRENT = self

    def callback(self, _pad: Any, info: Any, _user_data: Any = None) -> Any:
        Gst, pyds = self.runtime.Gst, self.runtime.pyds
        if not self.config.enabled:
            return Gst.PadProbeReturn.OK
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch_meta is None:
                return Gst.PadProbeReturn.OK
            meta_type = _shadow_meta_type(pyds)
            user_meta_type = getattr(pyds, "NvDsUserMeta", None)
            batch_type = getattr(pyds, "NvDsTargetMiscDataBatch", None)
            stream_type = getattr(pyds, "NvDsTargetMiscDataStream", None)
            object_type = getattr(pyds, "NvDsTargetMiscDataObject", None)
            if (
                meta_type is None
                or user_meta_type is None
                or batch_type is None
                or stream_type is None
                or object_type is None
            ):
                if not self._warned_unavailable:
                    LOGGER.warning(
                        "[TRACK_SHADOW_UNAVAILABLE] pinned PyDS does not expose shadow-list metadata"
                    )
                    self._warned_unavailable = True
                return Gst.PadProbeReturn.OK

            current_frames: dict[int, int] = {}
            for frame_meta in _iter_glist(batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast):
                current_frames[int(frame_meta.pad_index)] = int(frame_meta.frame_num)

            now = time.monotonic()
            by_camera: dict[str, list[ShadowDisplayTrack]] = {
                self.camera_by_pad[pad]: [] for pad in current_frames if pad in self.camera_by_pad
            }
            current_shadow_ids: dict[str, set[TrackId]] = {camera: set() for camera in by_camera}

            for user_meta in _iter_glist(batch_meta.batch_user_meta_list, user_meta_type.cast):
                if getattr(user_meta.base_meta, "meta_type", None) != meta_type:
                    continue
                shadow_batch = batch_type.cast(user_meta.user_meta_data)
                for stream in batch_type.list(shadow_batch):
                    pad_index = int(getattr(stream, "streamID", -1))
                    camera_id = self.camera_by_pad.get(pad_index)
                    frame_number = current_frames.get(pad_index)
                    if camera_id is None or frame_number is None:
                        continue
                    for target in stream_type.list(stream):
                        class_id = int(
                            getattr(target, "classId", getattr(target, "classID", -1))
                        )
                        if class_id not in self.person_class_ids:
                            continue
                        raw_id = int(getattr(target, "uniqueId"))
                        selected = None
                        for sample in object_type.list(target):
                            sample_frame = int(getattr(sample, "frameNum", -1))
                            if sample_frame == frame_number:
                                selected = sample
                                break
                        if selected is None:
                            continue
                        rect = selected.tBbox
                        try:
                            bbox = BoundingBox(
                                float(rect.left),
                                float(rect.top),
                                float(rect.left + rect.width),
                                float(rect.top + rect.height),
                            )
                        except (AttributeError, TypeError, ValueError):
                            continue
                        key = (camera_id, raw_id)
                        since = self._shadow_since.setdefault(key, now)
                        current_shadow_ids[camera_id].add(raw_id)
                        if now - since > self.config.display_max_age_sec:
                            self._hidden_by_age += 1
                            continue
                        presentation_id: TrackId | None = raw_id
                        if self.presenter is not None:
                            presentation_id = self.presenter(camera_id, raw_id)
                        if presentation_id is None:
                            continue
                        by_camera[camera_id].append(
                            ShadowDisplayTrack(
                                camera_id=camera_id,
                                track_id=presentation_id,
                                raw_track_id=raw_id,
                                frame_number=frame_number,
                                bbox=bbox,
                                confidence=max(
                                    0.0,
                                    min(1.0, float(getattr(selected, "confidence", 0.0))),
                                ),
                                tracker_age=max(0, int(getattr(selected, "age", 0))),
                            )
                        )

            with self._lock:
                for camera_id, frame_number in (
                    (self.camera_by_pad[pad], frame)
                    for pad, frame in current_frames.items()
                    if pad in self.camera_by_pad
                ):
                    tracks = tuple(by_camera.get(camera_id, ()))
                    self._frames[(camera_id, frame_number)] = tracks
                    self._frames.move_to_end((camera_id, frame_number))
                    self._frame_count += 1
                    self._object_count += len(tracks)
                while len(self._frames) > _MAX_FRAME_CACHE:
                    self._frames.popitem(last=False)

                for key in list(self._shadow_since):
                    camera_id, raw_id = key
                    if camera_id in current_shadow_ids and raw_id not in current_shadow_ids[camera_id]:
                        self._shadow_since.pop(key, None)
            return Gst.PadProbeReturn.OK
        except Exception:
            with self._lock:
                self._errors += 1
            LOGGER.exception("[TRACK_SHADOW_ERROR] failed to read tracker shadow metadata")
            return Gst.PadProbeReturn.OK

    def get(self, camera_id: str, frame_number: int) -> tuple[ShadowDisplayTrack, ...]:
        with self._lock:
            return self._frames.get((camera_id, int(frame_number)), ())

    def stats(self) -> ShadowTrackingStats:
        with self._lock:
            return ShadowTrackingStats(
                frames=self._frame_count,
                objects=self._object_count,
                hidden_by_age=self._hidden_by_age,
                errors=self._errors,
            )

    def close(self) -> None:
        global _CURRENT
        with _CURRENT_LOCK:
            if _CURRENT is self:
                _CURRENT = None
        with self._lock:
            self._frames.clear()
            self._shadow_since.clear()


def current_shadow_tracks(camera_id: str, frame_number: int) -> tuple[ShadowDisplayTrack, ...]:
    with _CURRENT_LOCK:
        registry = _CURRENT
    return () if registry is None else registry.get(camera_id, frame_number)


__all__ = [
    "ShadowDisplayTrack",
    "ShadowTrackRegistry",
    "ShadowTrackingConfig",
    "ShadowTrackingStats",
    "current_shadow_tracks",
]
