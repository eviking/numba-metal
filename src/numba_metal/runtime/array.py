"""GPU-resident arrays backed by shared-storage-mode MTLBuffers.

Design note (honesty requirement): on Apple-silicon, MTLResourceStorageModeShared
buffers live in unified memory and are directly addressable from both CPU and
GPU -- there is no discrete VRAM to copy across. numba_metal.to_device() and
copy_to_host() therefore do not perform a PCIe-style transfer; they perform a
host-side memcpy into/out of a shared-mode MTLBuffer's own backing memory
(`buf.contents()`), which is unavoidable because NumPy arrays and MTLBuffers
are distinct allocations. This is documented honestly (not "zero-copy") in
docs/architecture.md and docs/limitations.md. A true zero-copy path (wrapping
an existing NumPy allocation directly, or handing the MTLBuffer's memory
to NumPy as its backing store) is listed in the roadmap; it is legal Metal
API surface via MTLDevice.newBufferWithBytesNoCopy_length_options_deallocator_,
but was left out of the MVP to keep buffer lifetime rules simple.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from numba_metal.errors import MetalRuntimeError
from numba_metal.runtime.context import get_context
from numba_metal.types import SUPPORTED_DTYPES

# MTLResourceStorageModeShared
_STORAGE_MODE_SHARED = 0


class DeviceNDArray:
    """A GPU-resident array backed by a shared-storage-mode MTLBuffer.

    Instances are created via `metal.to_device`, `metal.device_array`, or
    `metal.device_array_like`; not constructed directly by users.
    """

    __slots__ = ("_buffer", "shape", "dtype", "strides", "size", "ndim")

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype, buffer: Any):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape)) if self.shape else 1
        self.ndim = len(self.shape)
        # C-contiguous strides, in elements (not bytes) to match how kernel
        # index arithmetic is generated.
        strides = [1] * self.ndim
        for i in range(self.ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * self.shape[i + 1]
        self.strides = tuple(strides)
        self._buffer = buffer

    @property
    def nbytes(self) -> int:
        """Total size of this array's data in bytes."""
        return self.size * self.dtype.itemsize

    @property
    def buffer(self):
        """The underlying MTLBuffer. Internal use by the dispatcher."""
        return self._buffer

    def copy_to_host(self, out: np.ndarray | None = None) -> np.ndarray:
        """Copy this array's contents back into a new (or provided) NumPy array."""
        get_context().synchronize()
        ptr = self._buffer.contents()
        raw = ptr.as_buffer(self.nbytes)
        host = np.frombuffer(raw, dtype=self.dtype).reshape(self.shape).copy()
        if out is not None:
            out[...] = host
            return out
        return host

    def copy_to_device(self, host_array: np.ndarray) -> None:
        """Overwrite this device array's contents from a NumPy array of the
        same shape and dtype. Used internally and by `to_device`."""
        if host_array.shape != self.shape:
            raise MetalRuntimeError(
                f"Shape mismatch copying to device: array is {self.shape}, "
                f"host data is {host_array.shape}."
            )
        src = np.ascontiguousarray(host_array, dtype=self.dtype)
        ptr = self._buffer.contents()
        raw = ptr.as_buffer(self.nbytes)
        raw[: self.nbytes] = src.tobytes()

    def __repr__(self) -> str:
        return (
            f"DeviceNDArray(shape={self.shape}, dtype={self.dtype}, "
            f"nbytes={self.nbytes})"
        )


def _validate_dtype(dtype: np.dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype not in SUPPORTED_DTYPES:
        supported = ", ".join(sorted(d.name for d in SUPPORTED_DTYPES))
        raise MetalRuntimeError(
            f"Unsupported dtype {dtype!r} for device array. "
            f"Supported dtypes: {supported}."
        )
    return dtype


def _alloc_buffer(nbytes: int):
    device = get_context().device
    nbytes = max(nbytes, 1)
    buf = device.newBufferWithLength_options_(nbytes, _STORAGE_MODE_SHARED)
    if buf is None:
        raise MetalRuntimeError(f"Failed to allocate a {nbytes}-byte MTLBuffer.")
    return buf


def device_array(shape, dtype) -> DeviceNDArray:
    """Allocate an uninitialized GPU-resident array of the given shape/dtype."""
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)
    dtype = _validate_dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
    buf = _alloc_buffer(nbytes)
    return DeviceNDArray(shape, dtype, buf)


def device_array_like(array: np.ndarray) -> DeviceNDArray:
    """Allocate an uninitialized GPU-resident array with the same shape/dtype
    as the given NumPy array (or DeviceNDArray)."""
    return device_array(array.shape, array.dtype)


def to_device(host_array: np.ndarray) -> DeviceNDArray:
    """Allocate a GPU-resident array and copy `host_array`'s contents into it."""
    host_array = np.asarray(host_array)
    dtype = _validate_dtype(host_array.dtype)
    dev = device_array(host_array.shape, dtype)
    dev.copy_to_device(host_array)
    return dev
