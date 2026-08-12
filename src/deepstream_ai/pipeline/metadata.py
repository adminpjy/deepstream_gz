"""Translate NvDs metadata into tracker-independent business contracts."""

from __future__ import annotations

import ctypes
import logging
import time
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Protocol

import numpy as np

from deepstream_ai.config import AppConfig
from deepstream_ai.domain import (
    BehaviorDetection,
    BehaviorType,
    BoundingBox,
    FaceDetection,
    Track,
)
from deepstream_ai.pipeline.scrfd import (
    assign_scrfd_landmarks,
    extract_scrfd_tensor_result,
    landmarks_from_scrfd_tensor,
)

LOGGER = logging.getLogger(__name__)
_NANOSECONDS = 1_000_000_000
_UNTRACKED_ID = (1 << 64) - 1
_MAX_GPU_SURFACE_BYTES = 512 * 1024 * 1024
_PROBE_HISTOGRAM_MAX_MS = 60_000


@dataclass(frozen=True, slots=True)
class FramePacket:
    camera_id: str
    frame_number: int
    timestamp: datetime
    image: np.ndarray
    tracks: tuple[Track, ...]
    faces: tuple[FaceDetection, ...]
    behaviors: tuple[BehaviorDetection, ...]
    stream_time_ns: int | None = None


@dataclass(frozen=True, slots=True)
class ProbePerformance:
    callbacks: int
    frames: int
    queue_drops: int
    errors: int
    average_ms: float
    p95_ms: float
    max_ms: float


class FramePacketConsumer(Protocol):
    def submit(self, packet: FramePacket) -> bool: ...

    def identity_label(self, camera_id: str, track_id: int | str) -> str | None: ...


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


def _timestamp(frame_meta: Any) -> datetime:
    raw = int(getattr(frame_meta, "ntp_timestamp", 0) or 0)
    # NTP timestamps emitted by nvstreammux are epoch nanoseconds. Ignore
    # obviously non-epoch values and use receipt time instead.
    if raw > 946684800 * _NANOSECONDS:  # 2000-01-01
        try:
            return datetime.fromtimestamp(raw / _NANOSECONDS, tz=UTC)
        except (OverflowError, OSError, ValueError):
            pass
    return datetime.now(UTC)


def _box(obj_meta: Any) -> BoundingBox | None:
    rect = obj_meta.rect_params
    try:
        return BoundingBox(
            float(rect.left),
            float(rect.top),
            float(rect.left + rect.width),
            float(rect.top + rect.height),
        )
    except ValueError:
        return None


def _confidence(obj_meta: Any) -> float:
    value = float(getattr(obj_meta, "confidence", 0.0))
    # Tracker-generated objects can expose -0.1 when detector inference was skipped.
    if value < 0:
        tracker_value = float(getattr(obj_meta, "tracker_confidence", -1.0))
        value = tracker_value if tracker_value >= 0 else 1.0
    return min(1.0, max(0.0, value))


def _hide_pgie_non_person_osd(
    obj_meta: Any,
    *,
    pgie_unique_id: int,
    person_class_ids: Sequence[int],
) -> bool:
    """Hide auxiliary PGIE classes from OSD without removing their metadata."""

    if int(getattr(obj_meta, "unique_component_id", -1)) != pgie_unique_id:
        return False
    if int(getattr(obj_meta, "class_id", -1)) in person_class_ids:
        return False
    obj_meta.rect_params.border_width = 0
    obj_meta.rect_params.has_bg_color = False
    obj_meta.text_params.display_text = ""
    obj_meta.text_params.set_bg_clr = False
    return True


def _track_id(obj_meta: Any, frame_number: int, index: int) -> int | str:
    value = int(getattr(obj_meta, "object_id", _UNTRACKED_ID))
    if value == _UNTRACKED_ID:
        return f"untracked-{frame_number}-{index}"
    return value


def _effective_nvdcf_track_id(obj_meta: Any) -> int | None:
    """Return only object IDs accepted as effective targets by NvDCF."""

    value = int(getattr(obj_meta, "object_id", _UNTRACKED_ID))
    return None if value == _UNTRACKED_ID else value


def _validate_gpu_surface_layout(
    dtype: Any,
    shape: Sequence[int],
    strides: Sequence[int],
    size: int,
) -> tuple[np.dtype[Any], tuple[int, int, int], tuple[int, int, int], int]:
    """Validate the strict packed-RGBA PyDS GPU surface contract."""

    resolved_dtype = np.dtype(dtype)
    resolved_shape = tuple(int(value) for value in shape)
    resolved_strides = tuple(int(value) for value in strides)
    resolved_size = int(size)
    if resolved_dtype != np.dtype(np.uint8):
        raise ValueError(f"NvBufSurface dtype must be uint8, got {resolved_dtype}")
    if len(resolved_shape) != 3 or resolved_shape[2] != 4:
        raise ValueError(f"NvBufSurface shape must be HxWx4, got {resolved_shape}")
    height, width, _channels = resolved_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"NvBufSurface dimensions must be positive, got {resolved_shape}")
    if len(resolved_strides) != 3:
        raise ValueError(f"NvBufSurface strides must have three values, got {resolved_strides}")
    pitch, pixel_stride, channel_stride = resolved_strides
    if pixel_stride != 4 or channel_stride != 1 or pitch < width * 4:
        raise ValueError(
            "NvBufSurface strides must describe pitched RGBA bytes, "
            f"got {resolved_strides} for {resolved_shape}"
        )
    minimum_size = (height - 1) * pitch + width * 4
    if resolved_size < minimum_size or resolved_size > _MAX_GPU_SURFACE_BYTES:
        raise ValueError(
            f"NvBufSurface size {resolved_size} is outside "
            f"[{minimum_size}, {_MAX_GPU_SURFACE_BYTES}]"
        )
    return resolved_dtype, resolved_shape, resolved_strides, resolved_size


def _face_landmarks(
    obj_meta: Any,
    bbox: BoundingBox,
    *,
    source: str,
    coordinates: str,
    scale: float,
    pyds: Any | None = None,
    unique_id: int = 0,
    threshold: float = 0.65,
) -> tuple[tuple[float, float], ...]:
    """Decode an explicit face-landmark transport without fabricating points.

    ``tensor`` is the production SCRFD route: nvinfer attaches raw secondary
    tensor output to the parent person, which is decoded and matched to this
    face. ``mask`` remains available only for existing custom bridges.
    """

    if source == "none":
        return ()
    if source == "tensor":
        if pyds is None:
            return ()
        return landmarks_from_scrfd_tensor(
            pyds,
            obj_meta,
            bbox,
            unique_id=unique_id,
            threshold=threshold,
        )
    if source != "mask":
        return ()
    mask = getattr(obj_meta, "mask_params", None)
    if mask is None or int(getattr(mask, "size", 0) or 0) <= 0:
        return ()
    getter = getattr(mask, "get_mask_array", None)
    if not callable(getter):
        return ()
    try:
        raw = np.asarray(getter(), dtype=np.float32).reshape(-1)
    except Exception:
        LOGGER.exception("无法读取 face landmark mask metadata")
        return ()
    if raw.size < 10 or not np.all(np.isfinite(raw[:10])):
        return ()
    values = raw[:10] / scale
    points: list[tuple[float, float]] = []
    for index in range(0, 10, 2):
        x, y = float(values[index]), float(values[index + 1])
        if coordinates == "bbox":
            x, y = bbox.x1 + x, bbox.y1 + y
        elif coordinates == "normalized":
            x, y = bbox.x1 + x * bbox.width, bbox.y1 + y * bbox.height
        points.append((x, y))
    return tuple(points)


class MetadataProbe:
    """Pad-probe callback that copies RGBA frames before their GPU buffer expires."""

    def __init__(self, runtime: Any, config: AppConfig, consumer: FramePacketConsumer):
        self.runtime = runtime
        self.config = config
        self.consumer = consumer
        self.camera_by_pad = {
            index: source.camera_id for index, source in enumerate(config.enabled_sources)
        }
        self.behavior_by_uid = {
            model.unique_id: model for model in config.behavior if model.enabled
        }
        self._cuda: Any | None = None
        self._cuda_context: Any | None = None
        self._timing_lock = Lock()
        # One-millisecond lifetime buckets keep P95 bounded in memory while
        # covering both short test clips and long-running streams.  The final
        # bucket contains callbacks taking 60 seconds or longer.
        self._timing_histogram = array("Q", [0]) * (_PROBE_HISTOGRAM_MAX_MS + 1)
        self._timing_callbacks = 0
        self._timing_frames = 0
        self._timing_queue_drops = 0
        self._timing_errors = 0
        self._timing_total_ns = 0
        self._timing_max_ns = 0
        self._timing_logged = False

    def callback(self, _pad: Any, info: Any, _user_data: Any = None) -> Any:
        started_ns = time.perf_counter_ns()
        frame_count = 0
        queue_drops = 0
        errors = 0
        Gst, pyds = self.runtime.Gst, self.runtime.pyds
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch_meta is None:
                return Gst.PadProbeReturn.OK
            for frame_meta in _iter_glist(batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast):
                frame_count += 1
                try:
                    packet, person_meta = self._packet(buffer, frame_meta)
                    accepted = self.consumer.submit(packet)
                    if not accepted:
                        queue_drops += 1
                        LOGGER.warning(
                            "分析队列已满，丢弃帧 camera_id=%s frame=%s",
                            packet.camera_id,
                            packet.frame_number,
                        )
                    self._annotate_identities(batch_meta, packet.camera_id, person_meta)
                except Exception:
                    errors += 1
                    # A malformed metadata item must not tear down the streaming thread.
                    LOGGER.exception("处理 DeepStream metadata 失败，当前帧已跳过")
            return Gst.PadProbeReturn.OK
        finally:
            self._record_timing(
                time.perf_counter_ns() - started_ns,
                frames=frame_count,
                queue_drops=queue_drops,
                errors=errors,
            )

    def _record_timing(
        self,
        elapsed_ns: int,
        *,
        frames: int = 0,
        queue_drops: int = 0,
        errors: int = 0,
    ) -> None:
        elapsed_ns = max(0, int(elapsed_ns))
        with self._timing_lock:
            self._timing_callbacks += 1
            self._timing_frames += max(0, int(frames))
            self._timing_queue_drops += max(0, int(queue_drops))
            self._timing_errors += max(0, int(errors))
            self._timing_total_ns += elapsed_ns
            self._timing_max_ns = max(self._timing_max_ns, elapsed_ns)
            bucket_ms = min(
                _PROBE_HISTOGRAM_MAX_MS,
                (elapsed_ns + 999_999) // 1_000_000,
            )
            self._timing_histogram[bucket_ms] += 1

    def performance(self) -> ProbePerformance:
        with self._timing_lock:
            callbacks = self._timing_callbacks
            average_ns = self._timing_total_ns / callbacks if callbacks else 0.0
            target = (callbacks * 95 + 99) // 100
            seen = 0
            p95_ms = 0
            for bucket_ms, count in enumerate(self._timing_histogram):
                seen += count
                if seen >= target:
                    p95_ms = bucket_ms
                    break
            return ProbePerformance(
                callbacks=callbacks,
                frames=self._timing_frames,
                queue_drops=self._timing_queue_drops,
                errors=self._timing_errors,
                average_ms=average_ns / 1_000_000,
                p95_ms=float(p95_ms),
                max_ms=self._timing_max_ns / 1_000_000,
            )

    def log_performance(self) -> ProbePerformance:
        with self._timing_lock:
            should_log = not self._timing_logged
            if should_log:
                self._timing_logged = True
        result = self.performance()
        if not should_log:
            return result
        LOGGER.info(
            "\n========== Probe Performance ==========\n"
            "Probe Callbacks:              %d\n"
            "Probe Frames:                 %d\n"
            "Probe Queue Drops:            %d\n"
            "Probe Errors:                 %d\n"
            "Average Probe Time (ms):      %.3f\n"
            "P95 Probe Time (ms):          %.3f\n"
            "Max Probe Time (ms):          %.3f\n"
            "=======================================",
            result.callbacks,
            result.frames,
            result.queue_drops,
            result.errors,
            result.average_ms,
            result.p95_ms,
            result.max_ms,
        )
        return result

    def _packet(
        self, gst_buffer: Any, frame_meta: Any
    ) -> tuple[FramePacket, list[tuple[Any, int | str]]]:
        pyds = self.runtime.pyds
        pad_index = int(frame_meta.pad_index)
        camera_id = self.camera_by_pad.get(pad_index, f"camera-{pad_index + 1}")
        frame_number = int(frame_meta.frame_num)
        timestamp = _timestamp(frame_meta)
        raw_stream_time = getattr(frame_meta, "buf_pts", None)
        try:
            stream_time_ns = int(raw_stream_time) if raw_stream_time is not None else -1
        except (TypeError, ValueError, OverflowError):
            stream_time_ns = -1
        if not 0 <= stream_time_ns < (1 << 63):
            source = self.config.enabled_sources[pad_index]
            stream_time_ns = round(frame_number / source.nominal_fps * _NANOSECONDS)
        image = self._copy_gpu_surface(gst_buffer, int(frame_meta.batch_id))
        # The probe caps and all downstream business adapters use RGBA. Keeping
        # this representation avoids an extra full-frame channel-swizzle copy.
        image.setflags(write=False)

        tracks: list[Track] = []
        faces: list[FaceDetection] = []
        behaviors: list[BehaviorDetection] = []
        persons_by_object_id: dict[int, Track] = {}
        person_meta: list[tuple[Any, int | str]] = []
        objects = list(_iter_glist(frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast))
        for index, obj_meta in enumerate(objects):
            uid = int(getattr(obj_meta, "unique_component_id", -1))
            class_id = int(getattr(obj_meta, "class_id", -1))
            _hide_pgie_non_person_osd(
                obj_meta,
                pgie_unique_id=self.config.pipeline.person.unique_id,
                person_class_ids=self.config.pipeline.person.person_class_ids,
            )
            if uid != self.config.pipeline.person.unique_id:
                continue
            if class_id not in self.config.pipeline.person.person_class_ids:
                continue
            native_track_id = _effective_nvdcf_track_id(obj_meta)
            # NvDCF leaves a PGIE proposal as UNTRACKED_OBJECT_ID while the
            # target is tentative or rejected. It is not an effective Person
            # Track and must not create a one-frame business lifecycle,
            # snapshot, or synthetic ID. The established mapping for every
            # accepted NvDCF object_id remains unchanged.
            if native_track_id is None:
                continue
            bbox = _box(obj_meta)
            if bbox is None:
                continue
            track_id = _track_id(obj_meta, frame_number, index)
            track = Track(
                camera_id=camera_id,
                track_id=track_id,
                timestamp=timestamp,
                bbox=bbox,
                confidence=_confidence(obj_meta),
                metadata={"class_id": class_id, "component_id": uid},
            )
            tracks.append(track)
            persons_by_object_id[native_track_id] = track
            person_meta.append((obj_meta, track_id))
            behaviors.extend(self._classifier_behaviors(obj_meta, track))

        tensor_landmarks = (
            self._tensor_face_landmarks(objects)
            if self.config.pipeline.face.enabled
            and self.config.pipeline.face.landmark_source == "tensor"
            else {}
        )
        for index, obj_meta in enumerate(objects):
            uid = int(getattr(obj_meta, "unique_component_id", -1))
            if uid == self.config.pipeline.face.unique_id and self.config.pipeline.face.enabled:
                bbox = _box(obj_meta)
                if bbox is None:
                    continue
                parent_track = self._parent_track(obj_meta, persons_by_object_id)
                if parent_track is None:
                    parent_track = self._nearest_track(bbox, tracks)
                if parent_track is None:
                    continue
                faces.append(
                    FaceDetection(
                        camera_id=camera_id,
                        track_id=parent_track.track_id,
                        timestamp=timestamp,
                        bbox=bbox,
                        score=_confidence(obj_meta),
                        landmarks=tensor_landmarks.get(index, ())
                        if self.config.pipeline.face.landmark_source == "tensor"
                        else _face_landmarks(
                            obj_meta,
                            bbox,
                            source=self.config.pipeline.face.landmark_source,
                            coordinates=self.config.pipeline.face.landmark_coordinates,
                            scale=self.config.pipeline.face.landmark_scale,
                            pyds=pyds,
                            unique_id=self.config.pipeline.face.unique_id,
                            threshold=self.config.pipeline.face.landmark_threshold,
                        ),
                        metadata={"component_id": uid},
                    )
                )
            elif uid in self.behavior_by_uid:
                bbox = _box(obj_meta)
                if bbox is None:
                    continue
                parent_track = self._parent_track(obj_meta, persons_by_object_id)
                if parent_track is None:
                    parent_track = self._nearest_track(bbox, tracks)
                if parent_track is None:
                    continue
                model = self.behavior_by_uid[uid]
                label = (
                    model.labels[int(obj_meta.class_id)]
                    if 0 <= int(obj_meta.class_id) < len(model.labels)
                    else model.name
                )
                try:
                    behavior_type = BehaviorType.parse(label)
                except ValueError:
                    LOGGER.warning("忽略未知行为标签: model=%s label=%s", model.name, label)
                    continue
                confidence = _confidence(obj_meta)
                if confidence >= model.threshold:
                    behaviors.append(
                        BehaviorDetection(
                            camera_id=camera_id,
                            track_id=parent_track.track_id,
                            timestamp=timestamp,
                            behavior=behavior_type,
                            confidence=confidence,
                            bbox=parent_track.bbox,
                            model_name=model.name,
                        )
                    )
        return (
            FramePacket(
                camera_id=camera_id,
                frame_number=frame_number,
                timestamp=timestamp,
                image=image,
                tracks=tuple(tracks),
                faces=tuple(faces),
                behaviors=tuple(behaviors),
                stream_time_ns=stream_time_ns,
            ),
            person_meta,
        )

    def _copy_gpu_surface(self, gst_buffer: Any, batch_id: int) -> np.ndarray:
        """Copy an RGBA NvBufSurface into owned host memory.

        The PyDS CPU helper exposes a NumPy view over ``dataPtr`` and can
        segfault when the SDK supplies device memory. The GPU helper gives us
        the same pointer and layout without dereferencing it on the CPU, so an
        explicit CUDA copy is safe for both device and unified surfaces.
        """
        dtype, shape, strides, capsule, size = self.runtime.pyds.get_nvds_buf_surface_gpu(
            hash(gst_buffer), batch_id
        )
        dtype, shape, strides, size = _validate_gpu_surface_layout(dtype, shape, strides, size)
        if self._cuda is None:
            import pycuda.driver as cuda  # type: ignore[import-not-found]

            cuda.init()
            self._cuda = cuda
            self._cuda_context = cuda.Device(
                self.config.pipeline.streammux.gpu_id
            ).retain_primary_context()

        pointer_getter = ctypes.pythonapi.PyCapsule_GetPointer
        pointer_getter.restype = ctypes.c_void_p
        pointer_getter.argtypes = (ctypes.py_object, ctypes.c_char_p)
        pointer = pointer_getter(capsule, None)
        if not pointer:
            raise RuntimeError("NvBufSurface CUDA pointer is null")

        host = np.empty(size, dtype=np.uint8)
        assert self._cuda_context is not None
        self._cuda_context.push()
        try:
            self._cuda.memcpy_dtoh(host, int(pointer))
        finally:
            self._cuda.Context.pop()

        view = np.ndarray(
            shape,
            dtype=dtype,
            buffer=host,
            strides=strides,
        )
        # ``view`` owns the host allocation through its buffer reference.  It
        # may be pitched, but downstream evidence code copies only selected
        # ROIs.  Returning the view avoids a second full-frame host copy in the
        # streaming thread.
        return view

    def _tensor_face_landmarks(
        self, objects: Sequence[Any]
    ) -> dict[int, tuple[tuple[float, float], ...]]:
        """Decode each parent tensor once and assign proposals one-to-one."""

        face_uid = self.config.pipeline.face.unique_id
        grouped: dict[int, tuple[Any, list[tuple[int, BoundingBox]]]] = {}
        for index, obj_meta in enumerate(objects):
            if int(getattr(obj_meta, "unique_component_id", -1)) != face_uid:
                continue
            bbox = _box(obj_meta)
            parent = getattr(obj_meta, "parent", None)
            parent_bbox = _box(parent) if parent is not None else None
            if bbox is None or parent is None or parent_bbox is None:
                continue
            native_id = int(getattr(parent, "object_id", _UNTRACKED_ID))
            key = native_id if native_id != _UNTRACKED_ID else hash(parent)
            entry = grouped.setdefault(key, (parent, []))
            entry[1].append((index, bbox))

        result: dict[int, tuple[tuple[float, float], ...]] = {}
        for parent, faces in grouped.values():
            parent_bbox = _box(parent)
            if parent_bbox is None:
                continue
            tensor = extract_scrfd_tensor_result(
                self.runtime.pyds,
                parent,
                unique_id=face_uid,
                threshold=self.config.pipeline.face.landmark_threshold,
            )
            if tensor is None:
                LOGGER.warning("SCRFD parent object has no decodable tensor meta")
                continue
            assigned = assign_scrfd_landmarks(
                [bbox for _, bbox in faces],
                parent_bbox,
                tensor,
            )
            for (index, _bbox), landmarks in zip(faces, assigned, strict=True):
                if landmarks:
                    result[index] = landmarks
        return result

    def _classifier_behaviors(self, obj_meta: Any, track: Track) -> Sequence[BehaviorDetection]:
        pyds = self.runtime.pyds
        result: list[BehaviorDetection] = []
        for classifier in _iter_glist(obj_meta.classifier_meta_list, pyds.NvDsClassifierMeta.cast):
            uid = int(classifier.unique_component_id)
            model = self.behavior_by_uid.get(uid)
            if model is None:
                continue
            for info in _iter_glist(classifier.label_info_list, pyds.NvDsLabelInfo.cast):
                probability = float(getattr(info, "result_prob", 0.0))
                if probability < model.threshold:
                    continue
                raw_label = str(getattr(info, "result_label", "") or model.name)
                try:
                    behavior = BehaviorType.parse(raw_label)
                except ValueError:
                    LOGGER.warning("忽略未知行为标签: model=%s label=%s", model.name, raw_label)
                    continue
                result.append(
                    BehaviorDetection(
                        camera_id=track.camera_id,
                        track_id=track.track_id,
                        timestamp=track.timestamp,
                        behavior=behavior,
                        confidence=min(1.0, max(0.0, probability)),
                        bbox=track.bbox,
                        model_name=model.name,
                    )
                )
        return result

    @staticmethod
    def _parent_track(obj_meta: Any, persons_by_object_id: dict[int, Track]) -> Track | None:
        parent = getattr(obj_meta, "parent", None)
        if parent is None:
            return None
        parent_id = int(getattr(parent, "object_id", _UNTRACKED_ID))
        return persons_by_object_id.get(parent_id)

    @staticmethod
    def _nearest_track(box: BoundingBox, tracks: Sequence[Track]) -> Track | None:
        x, y = box.center
        containing = [
            track
            for track in tracks
            if track.bbox.x1 <= x <= track.bbox.x2 and track.bbox.y1 <= y <= track.bbox.y2
        ]
        if not containing:
            return None
        return min(containing, key=lambda track: track.bbox.area)

    def _annotate_identities(
        self,
        batch_meta: Any,
        camera_id: str,
        values: Sequence[tuple[Any, int | str]],
    ) -> None:
        updates: list[tuple[Any, str]] = []
        for obj_meta, track_id in values:
            label = self.consumer.identity_label(camera_id, track_id)
            if label:
                updates.append((obj_meta, label))
        if not updates:
            return

        # Identity lookup may touch a worker-owned cache, so complete it before
        # taking DeepStream's batch metadata lock. Hold the lock only while
        # mutating NvOSD text pointers.
        pyds = self.runtime.pyds
        pyds.nvds_acquire_meta_lock(batch_meta)
        try:
            for obj_meta, label in updates:
                raw_text = obj_meta.text_params.display_text
                if isinstance(raw_text, str):
                    current = raw_text
                elif raw_text:
                    current = str(pyds.get_string(raw_text))
                else:
                    current = ""
                obj_meta.text_params.display_text = f"{current} {label}".strip()
        finally:
            pyds.nvds_release_meta_lock(batch_meta)
