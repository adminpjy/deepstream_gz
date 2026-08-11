"""In-process business event dispatch, independent of the pipeline runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any
from uuid import UUID, uuid4

from deepstream_ai.domain import TrackId

LOGGER = logging.getLogger(__name__)


class AnalyticsEventType(str, Enum):
    PERSON = "person"
    FACE = "face"
    IDENTITY = "identity"
    BEHAVIOR = "behavior"
    TRACK_ENDED = "track_ended"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True, slots=True)
class AnalyticsEvent:
    event_type: AnalyticsEventType
    camera_id: str
    track_id: TrackId
    timestamp: datetime
    payload: Any = field(compare=False)
    attributes: Mapping[str, Any] = field(default_factory=dict, compare=False)
    event_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", AnalyticsEventType(self.event_type))
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        if self.track_id == "" or self.track_id is None:
            raise ValueError("track_id cannot be empty")


EventHandler = Callable[[AnalyticsEvent], None]


@dataclass(frozen=True, slots=True)
class EventHandlerFailure:
    handler: EventHandler
    error: Exception


class EventDispatchError(RuntimeError):
    def __init__(self, failures: tuple[EventHandlerFailure, ...]) -> None:
        super().__init__(f"{len(failures)} event handler(s) failed")
        self.failures = failures


class EventManager:
    """Thread-safe synchronous pub/sub with optional failure isolation."""

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
        self._handlers: dict[AnalyticsEventType | None, list[EventHandler]] = {}
        self._lock = RLock()

    def subscribe(
        self,
        handler: EventHandler,
        event_type: AnalyticsEventType | None = None,
    ) -> Callable[[], None]:
        if not callable(handler):
            raise TypeError("event handler must be callable")
        event_type = None if event_type is None else AnalyticsEventType(event_type)
        with self._lock:
            handlers = self._handlers.setdefault(event_type, [])
            if handler not in handlers:
                handlers.append(handler)

        def unsubscribe() -> None:
            self.unsubscribe(handler, event_type)

        return unsubscribe

    def unsubscribe(
        self,
        handler: EventHandler,
        event_type: AnalyticsEventType | None = None,
    ) -> None:
        event_type = None if event_type is None else AnalyticsEventType(event_type)
        with self._lock:
            handlers = self._handlers.get(event_type)
            if handlers and handler in handlers:
                handlers.remove(handler)

    def publish(self, event: AnalyticsEvent) -> tuple[EventHandlerFailure, ...]:
        with self._lock:
            handlers = tuple(self._handlers.get(event.event_type, ())) + tuple(
                self._handlers.get(None, ())
            )
        failures: list[EventHandlerFailure] = []
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                LOGGER.exception(
                    "Event handler failed type=%s event_id=%s",
                    event.event_type.value,
                    event.event_id,
                )
                failures.append(EventHandlerFailure(handler, exc))
        result = tuple(failures)
        if result and self.strict:
            raise EventDispatchError(result)
        return result


EventBus = EventManager


__all__ = [
    "AnalyticsEvent",
    "AnalyticsEventType",
    "EventBus",
    "EventDispatchError",
    "EventHandler",
    "EventHandlerFailure",
    "EventManager",
]
