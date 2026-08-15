"""Degraded production facade that keeps the validated legacy service available."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

from deepstream_ai.config import AppConfig
from deepstream_ai.production.baseline import BaselineStore
from deepstream_ai.production.capabilities import production_capabilities
from deepstream_ai.production.manager import ProductionServiceError


class UnavailableProductionRecognitionService:
    """Expose clear production failures without taking down legacy `/api/tasks`.

    Production workers are an additive capability. If their DeepStream dynamic
    source implementation cannot start on a host, the already-validated legacy
    file/RTSP task service must remain usable while production endpoints report
    ``GPU_WORKER_UNAVAILABLE``.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        output_root: str | Path,
        error: BaseException,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.baselines = BaselineStore(self.output_root / "baselines")
        self.capabilities = production_capabilities(config)
        self.error = f"{type(error).__name__}: {error}"

    def status(self) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "activeSessions": 0,
            "availableGpus": 0,
            "error": self.error,
            "degraded": True,
        }

    def gpu_status(self) -> list[dict[str, Any]]:
        return []

    def list_sessions(self) -> list[dict[str, Any]]:
        return []

    def _raise_unavailable(self) -> NoReturn:
        raise ProductionServiceError(
            "GPU_WORKER_UNAVAILABLE",
            "生产多 GPU 识别服务当前不可用；已保留原有单路 RTSP/文件识别服务",
            detail={"error": self.error},
        )

    def start_session(self, _request: Any) -> NoReturn:
        self._raise_unavailable()

    def stop_session(self, _session_id: str, *, reason: str = "requested") -> NoReturn:
        del reason
        self._raise_unavailable()

    def stop_camera(self, _camera_id: str) -> NoReturn:
        self._raise_unavailable()

    def session(self, _session_id: str) -> NoReturn:
        self._raise_unavailable()

    def preview_path(self, _session_id: str) -> NoReturn:
        self._raise_unavailable()

    def baseline(self, camera_id: str) -> dict[str, Any] | None:
        record = self.baselines.current(camera_id)
        return None if record is None else record.as_dict()

    def upload_baseline(
        self,
        camera_id: str,
        payload: bytes,
        content_type: str,
    ) -> dict[str, Any]:
        return self.baselines.save(
            camera_id,
            payload,
            content_type=content_type,
        ).as_dict()

    def close(self) -> None:
        return None


__all__ = ["UnavailableProductionRecognitionService"]
