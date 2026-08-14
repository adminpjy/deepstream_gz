"""Remove one verified PeopleNet false-positive shape before NvDCF sees it.

The guard operates only on NvDsObjectMeta.  It neither maps/copies video
surfaces nor changes tracker or business-layer IDs.  Every configured geometry
and confidence condition must match before metadata is removed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)
_SECTION = "person_pretracker_guard"
_ALLOWED_KEYS = {
    "enabled",
    "max_confidence",
    "max_left_ratio",
    "max_top_ratio",
    "min_width_ratio",
    "max_width_ratio",
    "min_height_ratio",
    "max_right_ratio",
}


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


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{_SECTION}.{name} must be true or false")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{_SECTION}.{name} must be numeric")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{_SECTION}.{name} must be numeric") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{_SECTION}.{name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class PeopleNetPretrackerGuardConfig:
    enabled: bool = False
    max_confidence: float = 0.55
    max_left_ratio: float = 0.03
    max_top_ratio: float = 0.02
    min_width_ratio: float = 0.40
    max_width_ratio: float = 0.50
    min_height_ratio: float = 0.93
    max_right_ratio: float = 0.52

    @classmethod
    def from_file(cls, config_path: str | Path) -> PeopleNetPretrackerGuardConfig:
        path = Path(config_path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"unable to read {_SECTION} from {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{_SECTION} requires a YAML mapping at the document root")
        section = raw.get(_SECTION, {})
        if not isinstance(section, dict):
            raise ValueError(f"{_SECTION} must be a mapping")
        unknown = sorted(set(section) - _ALLOWED_KEYS)
        if unknown:
            raise ValueError(f"{_SECTION} contains unknown keys: {', '.join(unknown)}")

        result = cls(
            enabled=_strict_bool(section.get("enabled", False), "enabled"),
            max_confidence=_finite_float(section.get("max_confidence", 0.55), "max_confidence"),
            max_left_ratio=_finite_float(section.get("max_left_ratio", 0.03), "max_left_ratio"),
            max_top_ratio=_finite_float(section.get("max_top_ratio", 0.02), "max_top_ratio"),
            min_width_ratio=_finite_float(section.get("min_width_ratio", 0.40), "min_width_ratio"),
            max_width_ratio=_finite_float(section.get("max_width_ratio", 0.50), "max_width_ratio"),
            min_height_ratio=_finite_float(
                section.get("min_height_ratio", 0.93), "min_height_ratio"
            ),
            max_right_ratio=_finite_float(section.get("max_right_ratio", 0.52), "max_right_ratio"),
        )
        for name in (
            "max_confidence",
            "max_left_ratio",
            "max_top_ratio",
            "min_width_ratio",
            "max_width_ratio",
            "min_height_ratio",
            "max_right_ratio",
        ):
            value = getattr(result, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{_SECTION}.{name} must be greater than 0 and at most 1")
        if result.min_width_ratio >= result.max_width_ratio:
            raise ValueError(f"{_SECTION}.min_width_ratio must be less than max_width_ratio")
        if result.max_left_ratio + result.min_width_ratio > result.max_right_ratio:
            raise ValueError(f"{_SECTION} left/width thresholds cannot fit below max_right_ratio")
        return result


@dataclass(frozen=True, slots=True)
class PeopleNetPretrackerGuardStats:
    frames: int
    verified_people: int
    suppressed: int
    errors: int


class PeopleNetPretrackerGuard:
    """Filter a narrow, measured PGIE false-positive signature before NvDCF."""

    def __init__(
        self,
        runtime: Any,
        config: PeopleNetPretrackerGuardConfig,
        *,
        pgie_unique_id: int,
        person_class_ids: tuple[int, ...],
        frame_width: int,
        frame_height: int,
    ) -> None:
        if pgie_unique_id <= 0:
            raise ValueError("pgie_unique_id must be positive")
        if not person_class_ids or any(class_id < 0 for class_id in person_class_ids):
            raise ValueError("person_class_ids must contain non-negative IDs")
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("streammux frame dimensions must be positive")
        self.runtime = runtime
        self.config = config
        self.pgie_unique_id = int(pgie_unique_id)
        self.person_class_ids = frozenset(int(value) for value in person_class_ids)
        self.frame_width = float(frame_width)
        self.frame_height = float(frame_height)
        self._lock = Lock()
        self._frames = 0
        self._verified_people = 0
        self._suppressed = 0
        self._errors = 0
        LOGGER.info(
            "[PRETRACKER_GUARD] enabled=%s pgie_uid=%d person_classes=%s "
            "max_conf=%.3f left<=%.3f top<=%.3f width=[%.3f,%.3f] "
            "height>=%.3f right<=%.3f frame=%dx%d",
            config.enabled,
            self.pgie_unique_id,
            sorted(self.person_class_ids),
            config.max_confidence,
            config.max_left_ratio,
            config.max_top_ratio,
            config.min_width_ratio,
            config.max_width_ratio,
            config.min_height_ratio,
            config.max_right_ratio,
            int(self.frame_width),
            int(self.frame_height),
        )

    def should_suppress(self, obj_meta: Any) -> bool:
        if not self.config.enabled:
            return False
        if int(getattr(obj_meta, "unique_component_id", -1)) != self.pgie_unique_id:
            return False
        if int(getattr(obj_meta, "class_id", -1)) not in self.person_class_ids:
            return False
        try:
            confidence = float(obj_meta.confidence)
            rect = obj_meta.rect_params
            left = float(rect.left)
            top = float(rect.top)
            width = float(rect.width)
            height = float(rect.height)
        except (AttributeError, TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in (confidence, left, top, width, height)):
            return False
        if confidence < 0.0 or width <= 0.0 or height <= 0.0 or left < 0.0 or top < 0.0:
            return False

        left_ratio = left / self.frame_width
        top_ratio = top / self.frame_height
        width_ratio = width / self.frame_width
        height_ratio = height / self.frame_height
        right_ratio = (left + width) / self.frame_width
        cfg = self.config
        return (
            confidence <= cfg.max_confidence
            and left_ratio <= cfg.max_left_ratio
            and top_ratio <= cfg.max_top_ratio
            and cfg.min_width_ratio <= width_ratio <= cfg.max_width_ratio
            and height_ratio >= cfg.min_height_ratio
            and right_ratio <= cfg.max_right_ratio
        )

    def callback(self, _pad: Any, info: Any, _user_data: Any = None) -> Any:
        Gst, pyds = self.runtime.Gst, self.runtime.pyds
        frames = 0
        verified_people = 0
        suppressed = 0
        errors = 0
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch_meta is None:
                return Gst.PadProbeReturn.OK
            for frame_meta in _iter_glist(batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast):
                frames += 1
                # Materialize first: removing an NvDsObjectMeta invalidates its
                # current GList node.
                objects = list(_iter_glist(frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast))
                for obj_meta in objects:
                    if (
                        int(getattr(obj_meta, "unique_component_id", -1)) == self.pgie_unique_id
                        and int(getattr(obj_meta, "class_id", -1)) in self.person_class_ids
                    ):
                        verified_people += 1
                    if not self.should_suppress(obj_meta):
                        continue
                    rect = obj_meta.rect_params
                    confidence = float(obj_meta.confidence)
                    bbox = (
                        float(rect.left),
                        float(rect.top),
                        float(rect.width),
                        float(rect.height),
                    )
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                    suppressed += 1
                    LOGGER.warning(
                        "[PRETRACKER_GUARD_DROP] source=%s frame=%s conf=%.4f "
                        "bbox=(%.1f,%.1f,%.1f,%.1f) total=%d",
                        getattr(frame_meta, "source_id", "unknown"),
                        getattr(frame_meta, "frame_num", "unknown"),
                        confidence,
                        *bbox,
                        self._suppressed + suppressed,
                    )
        except Exception:
            errors += 1
            # Metadata filtering must not tear down the streaming thread.
            LOGGER.exception("[PRETRACKER_GUARD_ERROR] metadata filtering failed")
        finally:
            with self._lock:
                self._frames += frames
                self._verified_people += verified_people
                self._suppressed += suppressed
                self._errors += errors
        return Gst.PadProbeReturn.OK

    def stats(self) -> PeopleNetPretrackerGuardStats:
        with self._lock:
            return PeopleNetPretrackerGuardStats(
                frames=self._frames,
                verified_people=self._verified_people,
                suppressed=self._suppressed,
                errors=self._errors,
            )


__all__ = [
    "PeopleNetPretrackerGuard",
    "PeopleNetPretrackerGuardConfig",
    "PeopleNetPretrackerGuardStats",
]
