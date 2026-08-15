"""Per-source gates for warm optional DeepStream SGIEs.

Production GPU workers keep optional models resident so starting a camera does
not deserialize TensorRT engines. A disabled feature must nevertheless consume
no SGIE inference for that camera. The gate temporarily masks the PGIE
component id while a buffer crosses one behavior SGIE and restores it
immediately afterwards. Face SGIE and all tuned core tracking/recognition run
before these gates and are not modified.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from deepstream_ai.production.contracts import FeatureSet

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceFeatureBinding:
    camera_id: str
    features: FeatureSet


class FeatureRegistry:
    def __init__(self) -> None:
        self._by_pad: dict[int, SourceFeatureBinding] = {}
        self._lock = threading.RLock()

    def register(self, pad_index: int, camera_id: str, features: FeatureSet) -> None:
        with self._lock:
            self._by_pad[int(pad_index)] = SourceFeatureBinding(camera_id, features)

    def unregister(self, pad_index: int) -> None:
        with self._lock:
            self._by_pad.pop(int(pad_index), None)

    def binding(self, pad_index: int) -> SourceFeatureBinding | None:
        with self._lock:
            return self._by_pad.get(int(pad_index))

    def enabled(self, pad_index: int, feature_name: str) -> bool:
        binding = self.binding(pad_index)
        if binding is None:
            return False
        return bool(getattr(binding.features, feature_name, False))

    def snapshot(self) -> dict[int, SourceFeatureBinding]:
        with self._lock:
            return dict(self._by_pad)


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


class BehaviorInferenceGate:
    """Install synchronous sink/src probes around one optional SGIE."""

    def __init__(
        self,
        runtime: Any,
        registry: FeatureRegistry,
        *,
        feature_name: str,
        person_unique_id: int,
        gate_unique_id: int,
    ) -> None:
        self.runtime = runtime
        self.registry = registry
        self.feature_name = feature_name
        self.person_unique_id = int(person_unique_id)
        self.mask_unique_id = 1_000_000 + int(gate_unique_id)
        self._masked_objects = 0
        self._lock = threading.Lock()

    def install(self, element: Any) -> None:
        Gst = self.runtime.Gst
        sink = element.get_static_pad("sink")
        src = element.get_static_pad("src")
        if sink is None or src is None:
            raise RuntimeError(
                f"behavior inference element {getattr(element, 'name', '?')} has no sink/src pad"
            )
        sink.add_probe(Gst.PadProbeType.BUFFER, self._before, None)
        src.add_probe(Gst.PadProbeType.BUFFER, self._after, None)
        LOGGER.info(
            "[OPTIONAL_GATE] feature=%s person_uid=%d mask_uid=%d",
            self.feature_name,
            self.person_unique_id,
            self.mask_unique_id,
        )

    def _before(self, _pad: Any, info: Any, _data: Any = None) -> Any:
        return self._rewrite(info, mask=True)

    def _after(self, _pad: Any, info: Any, _data: Any = None) -> Any:
        return self._rewrite(info, mask=False)

    def _rewrite(self, info: Any, *, mask: bool) -> Any:
        Gst, pyds = self.runtime.Gst, self.runtime.pyds
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK
        changed = 0
        pyds.nvds_acquire_meta_lock(batch_meta)
        try:
            for frame_meta in _iter_glist(
                batch_meta.frame_meta_list,
                pyds.NvDsFrameMeta.cast,
            ):
                pad_index = int(frame_meta.pad_index)
                if mask and self.registry.enabled(pad_index, self.feature_name):
                    continue
                for obj_meta in _iter_glist(
                    frame_meta.obj_meta_list,
                    pyds.NvDsObjectMeta.cast,
                ):
                    uid = int(getattr(obj_meta, "unique_component_id", -1))
                    if mask and uid == self.person_unique_id:
                        obj_meta.unique_component_id = self.mask_unique_id
                        changed += 1
                    elif not mask and uid == self.mask_unique_id:
                        obj_meta.unique_component_id = self.person_unique_id
                        changed += 1
        finally:
            pyds.nvds_release_meta_lock(batch_meta)
        if mask and changed:
            with self._lock:
                self._masked_objects += changed
        return Gst.PadProbeReturn.OK

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "feature": self.feature_name,
                "masked_objects": self._masked_objects,
            }


__all__ = [
    "BehaviorInferenceGate",
    "FeatureRegistry",
    "SourceFeatureBinding",
]
