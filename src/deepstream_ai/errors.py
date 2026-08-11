"""Application-specific exceptions with actionable user-facing messages."""


class DeepStreamAIError(RuntimeError):
    """Base class for expected platform failures."""


class ConfigurationError(DeepStreamAIError):
    """Raised when the YAML configuration is invalid."""


class AssetValidationError(DeepStreamAIError):
    """Raised when an enabled component is missing a required model or input."""


class RuntimeUnavailableError(DeepStreamAIError):
    """Raised when DeepStream/GStreamer runtime libraries are unavailable."""


class PipelineError(DeepStreamAIError):
    """Raised when the GStreamer pipeline cannot be built or run."""
