"""Face-corroborated trust refresh for pose-sensitive NvDCF continuity.

The base continuity resolver remains authoritative for deciding whether a new
raw NvDCF ID may inherit an existing business ID.  This layer only refreshes the
trusted geometry/face anchor after the base resolver has already accepted a
borderline body-ReID merge using a real SCRFD face.

This matters when one physical person repeatedly changes pose: body ReID can
briefly fall below the strict business threshold even though the face and person
geometry are clearly continuous.  Without refreshing the trusted face anchor,
a later fragmentation is compared against an old face observation and can no
longer use the face-backed recovery path.

The body ReID gallery is never updated here.  The strict reid_update_min gate in
TrackContinuityResolver therefore remains unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from deepstream_ai.pipeline.metadata import FramePacket
from deepstream_ai.track_continuity import (
    _box_metrics,
    _cosine,
    _seconds_between,
    _track_reid_embedding,
)
from deepstream_ai.track_continuity_guard import GuardedTrackContinuityResolver

LOGGER = logging.getLogger(__name__)


class FaceAnchoredTrackContinuityResolver(GuardedTrackContinuityResolver):
    """Chain already-approved face-backed continuity across pose changes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._face_verified_raw: set[tuple[str, int, object]] = set()

    def begin_stream_generation(self, camera_id: str, generation: int) -> None:
        generation = max(0, int(generation))
        previous = self._stream_generations.get(camera_id)
        super().begin_stream_generation(camera_id, generation)
        if previous is not None and previous != generation:
            self._face_verified_raw = {
                key for key in self._face_verified_raw if key[0] != camera_id
            }

    def resolve(self, packet: FramePacket) -> FramePacket:
        resolved = super().resolve(packet)
        if not packet.tracks or not packet.faces or not resolved.tracks:
            return resolved

        source_tracks = {track.track_id: track for track in packet.tracks}
        faces_by_raw: dict[object, list] = {}
        for face in packet.faces:
            faces_by_raw.setdefault(face.track_id, []).append(face)

        generation = self._stream_generations.get(packet.camera_id, 0)
        config = self.config
        with self._lock:
            for track in resolved.tracks:
                raw_id = track.metadata.get("raw_track_id", track.track_id)
                source = source_tracks.get(raw_id)
                raw_faces = faces_by_raw.get(raw_id, ())
                if source is None or not raw_faces:
                    continue
                face = max(raw_faces, key=lambda item: item.score)
                if face.score < config.face_override_min_face_score:
                    continue

                state = self._states.get((packet.camera_id, track.track_id))
                if (
                    state is None
                    or state.trusted_bbox is None
                    or state.trusted_face_bbox is None
                    or state.trusted_face_seen is None
                ):
                    continue

                raw_key = (packet.camera_id, generation, raw_id)
                already_verified = raw_key in self._face_verified_raw
                incoming = _track_reid_embedding(source)
                similarity = (
                    _cosine(incoming, state.reid_embedding)
                    if incoming is not None and state.reid_embedding is not None
                    else None
                )

                # A new raw ID is admitted to this chaining path only after the
                # base resolver has mapped it to another logical ID and its real
                # body ReID is inside the configured face-override safety band.
                if not already_verified:
                    if track.track_id == raw_id:
                        continue
                    if similarity is None:
                        continue
                    if not (
                        config.face_override_min_reid
                        <= similarity
                        < config.reid_update_min
                    ):
                        continue

                person_iou, person_containment, person_center, person_area = _box_metrics(
                    state.trusted_bbox,
                    source.bbox,
                )
                face_iou, face_containment, face_center, face_area = _box_metrics(
                    state.trusted_face_bbox,
                    face.bbox,
                )
                face_gap = _seconds_between(face.timestamp, state.trusted_face_seen)
                if not (
                    0.0 <= face_gap <= config.face_override_max_gap_sec
                    and person_iou >= config.face_override_min_person_iou
                    and person_containment >= config.face_override_min_person_containment
                    and person_center <= config.face_override_max_center_distance_ratio
                    and config.face_override_min_area_ratio
                    <= person_area
                    <= config.face_override_max_area_ratio
                    and face_iou >= config.face_override_min_face_iou
                    and face_containment >= config.face_override_min_face_containment
                    and face_center <= config.face_override_max_face_center_distance_ratio
                    and config.face_override_min_face_area_ratio
                    <= face_area
                    <= config.face_override_max_face_area_ratio
                ):
                    continue

                first_refresh = raw_key not in self._face_verified_raw
                self._face_verified_raw.add(raw_key)
                # Refresh only spatial trust.  Never write state.reid_embedding:
                # the base resolver keeps its 0.85 gallery-update threshold.
                state.trusted_bbox = source.bbox
                state.trusted_seen = source.timestamp
                state.trusted_face_bbox = face.bbox
                state.trusted_face_seen = face.timestamp
                if first_refresh:
                    LOGGER.info(
                        "[TRACK_CONTINUITY_FACE_TRUST] camera=%s raw=%s logical=%s "
                        "reid_similarity=%.3f person_iou=%.3f face_iou=%.3f "
                        "status=trusted_geometry_only",
                        packet.camera_id,
                        raw_id,
                        track.track_id,
                        similarity if similarity is not None else -1.0,
                        person_iou,
                        face_iou,
                    )

        return resolved


__all__ = ["FaceAnchoredTrackContinuityResolver"]
