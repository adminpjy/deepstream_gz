from __future__ import annotations

import numpy as np

from deepstream_ai.face.embedding import PyCudaTensorRTEngineRunner


class _Allocation:
    _next = 100

    def __init__(self, size: int):
        self.size = size
        self.value = _Allocation._next
        _Allocation._next += 1
        self.freed = False

    def __int__(self) -> int:
        return self.value

    def free(self) -> None:
        self.freed = True


class _Stream:
    handle = 99

    def synchronize(self) -> None:
        return None


class _Cuda:
    def __init__(self) -> None:
        self.pinned: list[np.ndarray] = []

    Stream = _Stream

    def pagelocked_empty(self, shape, dtype):
        value = np.empty(shape, dtype=dtype)
        self.pinned.append(value)
        return value

    @staticmethod
    def mem_alloc(size: int):
        return _Allocation(size)

    def memcpy_htod_async(self, _device, host, _stream) -> None:
        assert any(host is item for item in self.pinned)

    def memcpy_dtoh_async(self, host, _device, _stream) -> None:
        assert any(host is item for item in self.pinned)
        host.fill(0.25)


class _TensorIOMode:
    INPUT = "input"


class _Trt:
    TensorIOMode = _TensorIOMode

    @staticmethod
    def nptype(_dtype):
        return np.float32


class _Engine:
    num_io_tensors = 2

    @staticmethod
    def get_tensor_name(index: int) -> str:
        return ("input", "embedding")[index]

    @staticmethod
    def get_tensor_mode(name: str) -> str:
        return _TensorIOMode.INPUT if name == "input" else "output"

    @staticmethod
    def get_tensor_dtype(_name: str):
        return "float32"


class _Context:
    def __init__(self) -> None:
        self.addresses = {}

    @staticmethod
    def set_input_shape(_name: str, _shape) -> bool:
        return True

    @staticmethod
    def get_tensor_shape(name: str):
        return (1, 512) if name == "embedding" else (1, 3, 112, 112)

    def set_tensor_address(self, name: str, address: int) -> None:
        self.addresses[name] = address

    @staticmethod
    def execute_async_v3(_handle: int) -> bool:
        return True


def test_tensor_rt_v3_uses_page_locked_host_memory_for_async_copies() -> None:
    runner = PyCudaTensorRTEngineRunner.__new__(PyCudaTensorRTEngineRunner)
    runner._trt = _Trt()
    runner._cuda = _Cuda()
    runner._engine = _Engine()
    runner._context = _Context()

    result = runner.infer(np.ones((1, 3, 112, 112), dtype=np.float32))

    assert result.shape == (1, 512)
    assert np.all(result == 0.25)
    assert len(runner._cuda.pinned) == 2
