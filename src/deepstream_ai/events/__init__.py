"""Business event contracts and dispatch."""

from deepstream_ai.events.manager import (
    AnalyticsEvent,
    AnalyticsEventType,
    EventBus,
    EventDispatchError,
    EventHandler,
    EventHandlerFailure,
    EventManager,
)

__all__ = [
    "AnalyticsEvent",
    "AnalyticsEventType",
    "EventBus",
    "EventDispatchError",
    "EventHandler",
    "EventHandlerFailure",
    "EventManager",
]
