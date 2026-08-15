"""Replaceable production result-publishing boundary."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from deepstream_ai.alarm_dispatcher import AlarmNotification, AlarmPublisher
from deepstream_ai.production.contracts import RecognitionEvent

LOGGER = logging.getLogger(__name__)
_STOP = object()


class ResultPublisher(Protocol):
    def publish(self, event: RecognitionEvent) -> None: ...

    def close(self) -> None: ...


class NullResultPublisher:
    def publish(self, event: RecognitionEvent) -> None:
        del event

    def close(self) -> None:
        return


class JsonlResultPublisher:
    """Durable local event journal used regardless of external integration."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()

    def publish(self, event: RecognitionEvent) -> None:
        encoded = json.dumps(event.as_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._stream.write(encoded + "\n")

    def close(self) -> None:
        with self._lock:
            self._stream.flush()
            self._stream.close()


@dataclass(frozen=True, slots=True)
class HttpPublisherConfig:
    url: str
    token: str = ""
    timeout_sec: float = 3.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("result publish url must be http:// or https://")
        if not 0.2 <= self.timeout_sec <= 30:
            raise ValueError("result publisher timeout must be between 0.2 and 30 seconds")
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("result publisher max_attempts must be between 1 and 5")


class HttpResultPublisher:
    """REST adapter. Endpoint-specific DTO mapping lives only here."""

    def __init__(self, config: HttpPublisherConfig) -> None:
        self.config = config

    def publish(self, event: RecognitionEvent) -> None:
        payload = json.dumps(
            self.to_external_payload(event),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "deepstream-gz-production/1",
        }
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        last_error: BaseException | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            request = urllib.request.Request(
                self.config.url,
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                    status = int(getattr(response, "status", 200))
                    if not 200 <= status < 300:
                        raise RuntimeError(f"result endpoint returned HTTP {status}")
                    return
            except (OSError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.config.max_attempts:
                    time.sleep(min(1.0, 0.2 * (2 ** (attempt - 1))))
        assert last_error is not None
        raise RuntimeError("result publish failed after bounded retries") from last_error

    @staticmethod
    def to_external_payload(event: RecognitionEvent) -> dict[str, object]:
        """Formal-system field mapping is intentionally isolated in this adapter."""

        return event.as_dict()

    def close(self) -> None:
        return


class CompositeResultPublisher:
    """Publish to every destination while preserving each destination's failure."""

    def __init__(self, *publishers: ResultPublisher) -> None:
        self.publishers = tuple(publishers)

    def publish(self, event: RecognitionEvent) -> None:
        errors: list[BaseException] = []
        for publisher in self.publishers:
            try:
                publisher.publish(event)
            except BaseException as exc:
                errors.append(exc)
                LOGGER.exception(
                    "结果推送失败 publisher=%s event=%s",
                    type(publisher).__name__,
                    event.event_id,
                )
        # The queued boundary catches this and writes one dead-letter record.
        # Raising here is deliberate even when the local journal succeeded: it
        # lets operations distinguish "event persisted" from "formal endpoint
        # delivered" without ever propagating transport failure to recognition.
        if errors:
            raise RuntimeError("one or more result publishers failed") from errors[0]

    def close(self) -> None:
        for publisher in reversed(self.publishers):
            try:
                publisher.close()
            except Exception:
                LOGGER.exception("关闭结果 Publisher 失败: %s", type(publisher).__name__)


class QueuedResultPublisher:
    """Keep all result I/O and backpressure away from recognition threads.

    Recognition correctness has higher priority than downstream transport.  A
    saturated or unavailable formal endpoint therefore degrades to local
    dead-letter persistence; publish() never raises transport/backpressure
    errors into DeepStream, AdaFace, alarm lifecycle or scenario processors.
    """

    def __init__(
        self,
        delegate: ResultPublisher,
        *,
        queue_size: int = 1024,
        dead_letter_path: str | Path | None = None,
    ) -> None:
        if queue_size < 16:
            raise ValueError("publisher queue_size must be >= 16")
        self.delegate = delegate
        self._queue: queue.Queue[RecognitionEvent | object] = queue.Queue(queue_size)
        self._dead_letter = (
            JsonlResultPublisher(dead_letter_path) if dead_letter_path is not None else None
        )
        self._thread = threading.Thread(
            target=self._run,
            name="result-publisher",
            daemon=True,
        )
        self._closed = threading.Event()
        self._stats_lock = threading.RLock()
        self._queued = 0
        self._delivered = 0
        self._dead_lettered = 0
        self._queue_full = 0
        self._thread.start()

    def publish(self, event: RecognitionEvent) -> None:
        if self._closed.is_set():
            LOGGER.warning("结果 Publisher 已关闭，丢弃迟到事件 event=%s", event.event_id)
            return
        try:
            self._queue.put_nowait(event)
            with self._stats_lock:
                self._queued += 1
        except queue.Full:
            with self._stats_lock:
                self._queue_full += 1
            LOGGER.error(
                "结果队列已满，事件转入死信但不影响识别 event=%s",
                event.event_id,
            )
            self._write_dead_letter(event)

    def _write_dead_letter(self, event: RecognitionEvent) -> None:
        if self._dead_letter is None:
            LOGGER.error("结果事件无法投递且未配置死信文件 event=%s", event.event_id)
            return
        try:
            self._dead_letter.publish(event)
            with self._stats_lock:
                self._dead_lettered += 1
        except Exception:
            # Even disk failure must not escape into recognition. It is logged
            # loudly so the host's log/health monitoring can alert operations.
            LOGGER.exception("写入结果死信失败 event=%s", event.event_id)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, RecognitionEvent)
                try:
                    self.delegate.publish(item)
                    with self._stats_lock:
                        self._delivered += 1
                except Exception:
                    LOGGER.exception("结果推送最终失败 event=%s", item.event_id)
                    self._write_dead_letter(item)
            finally:
                self._queue.task_done()

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "queued": self._queued,
                "delivered": self._delivered,
                "dead_lettered": self._dead_lettered,
                "queue_full": self._queue_full,
                "pending": self._queue.qsize(),
            }

    def _spill_pending_to_dead_letter(self) -> None:
        """Bound shutdown time even when an external endpoint is unavailable."""

        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                if isinstance(item, RecognitionEvent):
                    self._write_dead_letter(item)
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        # Do not wait for a potentially long HTTP retry backlog during service
        # shutdown. Persist queued items locally, then stop the delivery thread.
        self._spill_pending_to_dead_letter()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            # A delivery may have raced with the spill; free one slot safely.
            self._spill_pending_to_dead_letter()
            self._queue.put_nowait(_STOP)
        self._thread.join(10.0)
        if self._thread.is_alive():
            LOGGER.error("结果 Publisher 未在 10 秒内退出；守护线程将在进程退出时结束")
        self.delegate.close()
        if self._dead_letter is not None:
            self._dead_letter.close()


class AlarmResultAdapter(AlarmPublisher):
    """Bridge the tuned person/face alarm lifecycle to ResultPublisher."""

    def __init__(
        self,
        publisher: ResultPublisher,
        session_id_for_camera,
        *,
        evidence_root: str | Path | None = None,
    ) -> None:
        self.publisher = publisher
        self.session_id_for_camera = session_id_for_camera
        self.evidence_root = Path(evidence_root).resolve() if evidence_root else None
        if self.evidence_root is not None:
            self.evidence_root.mkdir(parents=True, exist_ok=True)

    def _save_evidence(self, notification: AlarmNotification) -> str | None:
        if self.evidence_root is None or notification.image is None:
            return None
        try:
            import cv2  # type: ignore[import-not-found]

            image = notification.image
            if image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            camera_root = self.evidence_root / notification.camera_id
            camera_root.mkdir(parents=True, exist_ok=True)
            path = camera_root / f"{uuid4().hex}.jpg"
            if not cv2.imwrite(str(path), image):
                return None
            return str(path)
        except Exception:
            LOGGER.exception(
                "保存核心识别报警证据失败 camera=%s tracker=%s",
                notification.camera_id,
                notification.tracker_id,
            )
            return None

    def publish(self, notification: AlarmNotification) -> None:
        session_id = self.session_id_for_camera(notification.camera_id)
        if session_id is None:
            session_id = f"detached:{notification.camera_id}"
        snapshot = self._save_evidence(notification)
        event = RecognitionEvent.create(
            session_id=session_id,
            camera_id=notification.camera_id,
            event_type=notification.alarm_type.value,
            action=notification.action.value,
            track_id=notification.tracker_id,
            person_id=notification.worker_id,
            confidence=(
                None
                if notification.similarity is None
                else max(0.0, min(1.0, (notification.similarity + 1.0) / 2.0))
            ),
            timestamp=notification.timestamp,
            snapshot=snapshot,
            extra={
                "previousType": (
                    notification.previous_type.value
                    if notification.previous_type is not None
                    else None
                ),
                "rawTrackerId": (
                    None
                    if notification.raw_tracker_id is None
                    else str(notification.raw_tracker_id)
                ),
                "similarity": notification.similarity,
                "alarmActive": notification.alarm_active,
                "reason": notification.reason,
            },
        )
        self.publisher.publish(event)


def build_result_publisher(output_root: str | Path) -> QueuedResultPublisher:
    root = Path(output_root).resolve()
    local = JsonlResultPublisher(root / "recognition-events.jsonl")
    delegates: list[ResultPublisher] = [local]
    url = os.environ.get("RESULT_PUBLISH_URL", "").strip()
    if url:
        timeout = float(os.environ.get("RESULT_PUBLISH_TIMEOUT_SEC", "3"))
        attempts = int(os.environ.get("RESULT_PUBLISH_MAX_ATTEMPTS", "3"))
        delegates.append(
            HttpResultPublisher(
                HttpPublisherConfig(
                    url=url,
                    token=os.environ.get("RESULT_PUBLISH_TOKEN", ""),
                    timeout_sec=timeout,
                    max_attempts=attempts,
                )
            )
        )
    delegate: ResultPublisher = (
        delegates[0] if len(delegates) == 1 else CompositeResultPublisher(*delegates)
    )
    return QueuedResultPublisher(
        delegate,
        queue_size=int(os.environ.get("RESULT_PUBLISH_QUEUE_SIZE", "1024")),
        dead_letter_path=root / "result-dead-letter.jsonl",
    )


__all__ = [
    "AlarmResultAdapter",
    "CompositeResultPublisher",
    "HttpPublisherConfig",
    "HttpResultPublisher",
    "JsonlResultPublisher",
    "NullResultPublisher",
    "QueuedResultPublisher",
    "ResultPublisher",
    "build_result_publisher",
]
