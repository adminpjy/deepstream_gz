"""Face-recognition specific exceptions."""


class FaceRecognitionError(RuntimeError):
    """Base error for recoverable face-recognition failures."""


class FaceRuntimeUnavailable(FaceRecognitionError):
    """The requested optional inference runtime is not installed."""


class InvalidFaceInput(FaceRecognitionError, ValueError):
    """A face image or model output does not satisfy the AdaFace contract."""


class FaceModelLoadError(FaceRecognitionError):
    """An AdaFace model or TensorRT engine could not be loaded."""
