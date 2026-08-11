"""Application composition root."""

from __future__ import annotations

import logging

from deepstream_ai.analytics import AnalyticsDispatcher
from deepstream_ai.config import AppConfig
from deepstream_ai.pipeline.builder import DeepStreamPipelineBuilder
from deepstream_ai.pipeline.runner import PipelineRunner
from deepstream_ai.pipeline.runtime import load_runtime, runtime_versions
from deepstream_ai.preflight import validate_assets

LOGGER = logging.getLogger(__name__)


class DeepStreamApplication:
    """Compose runtime adapters and business services from one typed config."""

    def __init__(self, config: AppConfig):
        self.config = config

    def run(self) -> None:
        reports = validate_assets(self.config)
        for report in reports:
            LOGGER.info(
                "模型预检通过 config=%s engine=%s source_models=%s",
                report.config_path,
                report.engine_path,
                [str(path) for path in report.source_models],
            )
        detector_type = self.config.pipeline.person.detector_type
        detector_name = "PeopleNet" if detector_type == "peoplenet" else detector_type
        tracker_backend = self.config.pipeline.tracker.backend
        tracker_name = "NvDCF" if tracker_backend == "nvdcf" else tracker_backend
        LOGGER.info("[DETECTOR] %s", detector_name)
        LOGGER.info("[TRACKER] %s", tracker_name)
        LOGGER.info("[TRACKER_CONFIG] %s", self.config.pipeline.tracker.config_file)
        LOGGER.info(
            "[PEOPLENET_CLASSES] %s",
            ", ".join(
                f"{name}={class_id}"
                for name, class_id in self.config.pipeline.person.people_classes
            ),
        )
        runtime = load_runtime()
        LOGGER.info("DeepStream Python runtime 已加载: %s", runtime_versions(runtime))
        dispatcher = AnalyticsDispatcher(
            self.config,
            queue_size=self.config.runtime.analytics_queue_size,
        )
        dispatcher.start()
        primary_error: BaseException | None = None
        try:
            graph = DeepStreamPipelineBuilder(runtime, self.config, dispatcher).build()
            PipelineRunner(runtime, self.config, graph).run()
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                dispatcher.close()
            except Exception:
                if primary_error is None:
                    raise
                LOGGER.exception("关闭分析工作线程时发生附加错误")
