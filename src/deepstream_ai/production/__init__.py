"""Production multi-GPU recognition sessions."""

# Compatibility bridge for the existing worker import surface. Keep the old
# implementation in production.pipeline for rollback/reference, but route the
# public builder/controller names used by the worker to NVIDIA's official
# nvmultiurisrcbin implementation. This limits the source-lifecycle refactor to
# one layer and leaves manager/session/result contracts untouched.
from deepstream_ai.production import pipeline as _production_pipeline
from deepstream_ai.production.contracts import (
    ExitPolicy,
    FeatureSet,
    LeftObjectPolicy,
    RecognitionEvent,
    SessionRequest,
    SessionState,
)
from deepstream_ai.production.manager import ProductionRecognitionService, ProductionServiceError
from deepstream_ai.production.multiuri_pipeline import (
    MultiUriPipelineBuilder,
    MultiUriSourceController,
)

_production_pipeline.WarmDynamicPipelineBuilder = MultiUriPipelineBuilder
_production_pipeline.DynamicSourceController = MultiUriSourceController

__all__ = [
    "ExitPolicy",
    "FeatureSet",
    "LeftObjectPolicy",
    "ProductionRecognitionService",
    "ProductionServiceError",
    "RecognitionEvent",
    "SessionRequest",
    "SessionState",
]
