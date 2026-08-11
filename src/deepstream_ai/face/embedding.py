"""AdaFace ONNX and TensorRT adapters.

Optional runtimes are imported only when an adapter needs to construct its own
session/engine.  Tests and alternative runtime owners can inject an object that
implements the small protocols below.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from deepstream_ai.face.errors import (
    FaceModelLoadError,
    FaceRuntimeUnavailable,
    InvalidFaceInput,
)
from deepstream_ai.face.quality import normalize_embedding

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class FaceEmbedder(Protocol):
    """Minimal embedding contract used by the recognition service."""

    def embed(self, face_crop: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class TensorRTRunner(Protocol):
    """Runtime-owner interface for an already deserialized TensorRT engine."""

    def infer(self, input_tensor: np.ndarray) -> np.ndarray | Sequence[np.ndarray]: ...


class AdaFacePreprocessor:
    """Convert an aligned face crop to AdaFace RGB NCHW float input.

    DeepStream surfaces are RGBA, which is the default. OpenCV callers can set
    ``input_color="bgr"`` (or use the backwards-compatible ``bgr_input=True``).
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (112, 112),
        *,
        input_color: str = "rgba",
        bgr_input: bool | None = None,
    ) -> None:
        if len(input_size) != 2 or min(input_size) <= 0:
            raise ValueError("input_size must contain two positive dimensions")
        self.input_size = int(input_size[0]), int(input_size[1])
        if bgr_input is not None:
            input_color = "bgr" if bgr_input else "rgb"
        input_color = input_color.strip().lower()
        if input_color not in {"rgb", "rgba", "bgr", "bgra"}:
            raise ValueError("input_color must be rgb, rgba, bgr, or bgra")
        self.input_color = input_color
        self.bgr_input = input_color.startswith("bgr")

    def __call__(self, face_crop: np.ndarray) -> np.ndarray:
        image = np.asarray(face_crop)
        if image.size == 0:
            raise InvalidFaceInput("face crop is empty")
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
            raise InvalidFaceInput("face crop must be HxW, HxWx1, HxWx3, or HxWx4")
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[..., :3]
        if not np.issubdtype(image.dtype, np.number):
            raise InvalidFaceInput("face crop must contain numeric pixels")
        if not np.all(np.isfinite(image)):
            raise InvalidFaceInput("face crop contains NaN or infinity")

        width, height = self.input_size
        image = _resize(image, width, height)
        if self.bgr_input:
            image = image[..., ::-1]
        image = image.astype(np.float32, copy=False)
        # Canonical AdaFace normalization maps uint8 [0, 255] to [-1, 1].
        image = (image - 127.5) / 127.5
        return np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    try:
        import cv2  # type: ignore[import-not-found]

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        # A dependency-free nearest-neighbour fallback keeps imports and CPU
        # unit tests functional. Production images normally use the cv2 path.
        y = np.linspace(0, image.shape[0] - 1, height).round().astype(np.intp)
        x = np.linspace(0, image.shape[1] - 1, width).round().astype(np.intp)
        return image[y[:, None], x[None, :]]


class BaseAdaFaceAdapter(ABC):
    embedding_size = 512

    def __init__(self, preprocessor: AdaFacePreprocessor | None = None) -> None:
        self.preprocessor = preprocessor or AdaFacePreprocessor()

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        tensor = self.preprocessor(face_crop)
        raw = self._infer(tensor)
        return _extract_embedding(raw)

    def embed_many(self, face_crops: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
        return tuple(self.embed(crop) for crop in face_crops)

    @abstractmethod
    def _infer(self, input_tensor: np.ndarray) -> Any:
        raise NotImplementedError


class AdaFaceONNXAdapter(BaseAdaFaceAdapter):
    """Run an AdaFace ONNX graph and return a normalized 512-vector."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        providers: Sequence[str] | None = None,
        session: Any | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
        preprocessor: AdaFacePreprocessor | None = None,
    ) -> None:
        super().__init__(preprocessor)
        if session is None:
            if model_path is None:
                raise ValueError("model_path is required when session is not injected")
            path = Path(model_path)
            if not path.is_file():
                raise FaceModelLoadError(f"AdaFace ONNX model not found: {path}")
            try:
                import onnxruntime as ort  # type: ignore[import-not-found]
            except ImportError as exc:
                raise FaceRuntimeUnavailable(
                    "onnxruntime is required for the AdaFace ONNX adapter"
                ) from exc
            selected_providers = (
                list(providers)
                if providers
                else [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
            )
            try:
                session = ort.InferenceSession(str(path), providers=selected_providers)
            except Exception as exc:  # Runtime error detail is preserved as the cause.
                raise FaceModelLoadError(f"failed to load AdaFace ONNX model: {path}") from exc
        self.session = session
        try:
            self.input_name = input_name or session.get_inputs()[0].name
        except (AttributeError, IndexError) as exc:
            raise FaceModelLoadError("ONNX session exposes no model input") from exc
        self.output_name = output_name

    def _infer(self, input_tensor: np.ndarray) -> Any:
        outputs = None if self.output_name is None else [self.output_name]
        try:
            return self.session.run(outputs, {self.input_name: input_tensor})
        except Exception as exc:
            raise InvalidFaceInput("AdaFace ONNX inference failed") from exc


class AdaFaceTensorRTAdapter(BaseAdaFaceAdapter):
    """AdaFace adapter for an injected runner or a serialized TensorRT engine."""

    def __init__(
        self,
        engine_path: str | Path | None = None,
        *,
        runner: TensorRTRunner | None = None,
        preprocessor: AdaFacePreprocessor | None = None,
    ) -> None:
        super().__init__(preprocessor)
        if runner is None:
            if engine_path is None:
                raise ValueError("engine_path is required when runner is not injected")
            runner = PyCudaTensorRTEngineRunner(engine_path)
        if not isinstance(runner, TensorRTRunner):
            raise TypeError("runner must provide infer(input_tensor)")
        self.runner = runner

    def _infer(self, input_tensor: np.ndarray) -> Any:
        try:
            return self.runner.infer(input_tensor)
        except (FaceRuntimeUnavailable, FaceModelLoadError, InvalidFaceInput):
            raise
        except Exception as exc:
            raise InvalidFaceInput("AdaFace TensorRT inference failed") from exc


class PyCudaTensorRTEngineRunner:
    """Small TensorRT/pycuda runner supporting legacy and TensorRT 10 APIs.

    DeepStream deployments may inject their own CUDA context-aware runner
    instead. This implementation owns allocations only for the duration of a
    call and never fabricates an output when runtime setup or execution fails.
    """

    def __init__(self, engine_path: str | Path) -> None:
        path = Path(engine_path)
        if not path.is_file():
            raise FaceModelLoadError(f"AdaFace TensorRT engine not found: {path}")
        try:
            # The CUDA primary context interoperates with DeepStream's context;
            # pycuda.autoinit would create a separate context in this process.
            import pycuda.autoprimaryctx  # type: ignore[import-not-found]  # noqa: F401
            import pycuda.driver as cuda  # type: ignore[import-not-found]
            import tensorrt as trt  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FaceRuntimeUnavailable(
                "tensorrt and pycuda are required to load an AdaFace TensorRT engine"
            ) from exc
        self._trt = trt
        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        try:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(path.read_bytes())
        except Exception as exc:
            raise FaceModelLoadError(f"failed to deserialize TensorRT engine: {path}") from exc
        if engine is None:
            raise FaceModelLoadError(f"TensorRT rejected engine: {path}")
        self._runtime = runtime
        self._logger = logger
        self._engine = engine
        self._context = engine.create_execution_context()
        if self._context is None:
            raise FaceModelLoadError("failed to create TensorRT execution context")

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
        if hasattr(self._engine, "num_io_tensors"):
            return self._infer_v3(tensor)
        return self._infer_v2(tensor)

    def _infer_v3(self, tensor: np.ndarray) -> np.ndarray:
        trt, cuda = self._trt, self._cuda
        engine, context = self._engine, self._context
        input_names: list[str] = []
        output_names: list[str] = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        if len(input_names) != 1 or not output_names:
            raise FaceModelLoadError("AdaFace engine must have one input and at least one output")
        input_name = input_names[0]
        if not context.set_input_shape(input_name, tensor.shape):
            raise InvalidFaceInput(f"TensorRT rejected input shape {tensor.shape}")
        stream = cuda.Stream()
        allocations: list[Any] = []
        try:
            host_input = cuda.pagelocked_empty(tensor.shape, tensor.dtype)
            np.copyto(host_input, tensor)
            device_input = cuda.mem_alloc(host_input.nbytes)
            allocations.append(device_input)
            context.set_tensor_address(input_name, int(device_input))
            cuda.memcpy_htod_async(device_input, host_input, stream)
            host_outputs: list[np.ndarray] = []
            output_devices: list[Any] = []
            for name in output_names:
                shape = tuple(context.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise FaceModelLoadError(
                        f"unresolved TensorRT output shape for {name}: {shape}"
                    )
                dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
                host = cuda.pagelocked_empty(shape, dtype)
                device = cuda.mem_alloc(host.nbytes)
                allocations.append(device)
                output_devices.append(device)
                context.set_tensor_address(name, int(device))
                host_outputs.append(host)
            if not context.execute_async_v3(stream.handle):
                raise InvalidFaceInput("TensorRT execution returned false")
            for device, host in zip(output_devices, host_outputs, strict=True):
                cuda.memcpy_dtoh_async(host, device, stream)
            stream.synchronize()
            return np.asarray(host_outputs[0]).copy()
        finally:
            for allocation in reversed(allocations):
                allocation.free()

    def _infer_v2(self, tensor: np.ndarray) -> np.ndarray:
        trt, cuda = self._trt, self._cuda
        engine, context = self._engine, self._context
        input_indices = [
            index for index in range(engine.num_bindings) if engine.binding_is_input(index)
        ]
        output_indices = [
            index for index in range(engine.num_bindings) if not engine.binding_is_input(index)
        ]
        if len(input_indices) != 1 or not output_indices:
            raise FaceModelLoadError("AdaFace engine must have one input and at least one output")
        input_index = input_indices[0]
        if any(dimension < 0 for dimension in engine.get_binding_shape(input_index)):
            context.set_binding_shape(input_index, tensor.shape)
        stream = cuda.Stream()
        bindings: list[int] = [0] * engine.num_bindings
        allocations: list[Any] = []
        host_outputs: list[np.ndarray] = []
        output_devices: list[Any] = []
        try:
            host_input = cuda.pagelocked_empty(tensor.shape, tensor.dtype)
            np.copyto(host_input, tensor)
            device_input = cuda.mem_alloc(host_input.nbytes)
            allocations.append(device_input)
            bindings[input_index] = int(device_input)
            cuda.memcpy_htod_async(device_input, host_input, stream)
            for index in output_indices:
                shape = tuple(context.get_binding_shape(index))
                if any(dimension < 0 for dimension in shape):
                    raise FaceModelLoadError(f"unresolved TensorRT output shape: {shape}")
                dtype = np.dtype(trt.nptype(engine.get_binding_dtype(index)))
                host = cuda.pagelocked_empty(shape, dtype)
                device = cuda.mem_alloc(host.nbytes)
                allocations.append(device)
                output_devices.append(device)
                bindings[index] = int(device)
                host_outputs.append(host)
            if not context.execute_async_v2(bindings=bindings, stream_handle=stream.handle):
                raise InvalidFaceInput("TensorRT execution returned false")
            for device, host in zip(output_devices, host_outputs, strict=True):
                cuda.memcpy_dtoh_async(host, device, stream)
            stream.synchronize()
            return np.asarray(host_outputs[0]).copy()
        finally:
            for allocation in reversed(allocations):
                allocation.free()


def _extract_embedding(raw_output: Any) -> np.ndarray:
    output = raw_output
    if isinstance(output, Mapping):
        if not output:
            raise InvalidFaceInput("AdaFace runtime returned no outputs")
        output = next(iter(output.values()))
    if isinstance(output, (list, tuple)):
        if not output:
            raise InvalidFaceInput("AdaFace runtime returned no outputs")
        output = output[0]
    array = np.asarray(output, dtype=np.float32)
    if array.ndim >= 2:
        if array.shape[0] != 1:
            raise InvalidFaceInput(f"single-face inference returned batch shape {array.shape}")
        array = array.reshape(array.shape[0], -1)[0]
    return normalize_embedding(array)


ONNXAdaFaceEmbedder = AdaFaceONNXAdapter
TensorRTAdaFaceEmbedder = AdaFaceTensorRTAdapter
AdaFaceONNXEmbedder = AdaFaceONNXAdapter
AdaFaceTensorRTEmbedder = AdaFaceTensorRTAdapter


__all__ = [
    "AdaFaceONNXAdapter",
    "AdaFaceONNXEmbedder",
    "AdaFacePreprocessor",
    "AdaFaceTensorRTAdapter",
    "AdaFaceTensorRTEmbedder",
    "BaseAdaFaceAdapter",
    "FaceEmbedder",
    "ONNXAdaFaceEmbedder",
    "PyCudaTensorRTEngineRunner",
    "TensorRTAdaFaceEmbedder",
    "TensorRTRunner",
]
