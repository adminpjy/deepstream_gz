"""Process-local RTSP stream generation tracking.

NvDCF raw object IDs can restart after an RTSP source reconnect.  Keep a tiny
per-process generation counter so the business continuity layer can invalidate
raw-ID aliases without discarding trusted ReID/geometry history.
"""

from __future__ import annotations

import logging
from threading import RLock

LOGGER = logging.getLogger(__name__)
_LOCK = RLock()
_GENERATIONS: dict[str, int] = {}


def current_stream_generation(camera_id: str) -> int:
    with _LOCK:
        return int(_GENERATIONS.get(camera_id, 0))


def bump_stream_generation(camera_id: str, *, reason: str) -> int:
    with _LOCK:
        generation = int(_GENERATIONS.get(camera_id, 0)) + 1
        _GENERATIONS[camera_id] = generation
    LOGGER.warning(
        "[SOURCE_GENERATION] camera=%s generation=%d reason=%s",
        camera_id,
        generation,
        reason,
    )
    return generation


def reset_stream_generations() -> None:
    """Test/service-process cleanup helper."""

    with _LOCK:
        _GENERATIONS.clear()


__all__ = [
    "bump_stream_generation",
    "current_stream_generation",
    "reset_stream_generations",
]
