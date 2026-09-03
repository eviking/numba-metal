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

# Host/device synchronization model (Workstream 3)

Unified memory means a shared-storage-mode `MTLBuffer`'s backing memory is
literally the same memory the GPU reads and writes -- there is no
coherence-protocol copy step to wait for, but there is no ordering
guarantee either: nothing stops the CPU from reading or overwriting that
memory *while* a previously-submitted, not-yet-complete kernel is still
executing against it, if the host access were allowed to happen without
first waiting. That is a genuine data race (the GPU could read a
half-written host update, or the host could read a value the GPU hasn't
finished computing yet, or both could write concurrently), not merely a
staleness/ordering inconvenience. This was verified concretely, not just
argued: a kernel reading a buffer for ~2000 iterations, immediately
followed by an unsynchronized `copy_to_device()` overwrite of that same
buffer from the host, was observed (before this fix) to have the kernel's
output reflect the *new*, overwritten values instead of the values that
were present when it was launched -- i.e. the host write actually raced
ahead of and corrupted the in-flight kernel's input. See
`tests/integration/test_host_device_sync.py`.

The MVP's synchronization model is deliberately conservative rather than
a fine-grained per-buffer dependency tracker (matching the assignment's
explicit preference for correctness over sophistication for this pass):
**every** host-side read of a device buffer (`copy_to_host`) and **every**
host-side write to one (`copy_to_device`, including the write inside
`to_device`) first calls `metal.synchronize()`, unconditionally. This
waits for *all* outstanding GPU work on *any* buffer, not just the one
being touched -- coarser than strictly necessary, but correct
unconditionally: since numba-metal has no way (yet) to know which
in-flight kernels reference which specific buffer without dedicated
per-buffer dependency tracking, waiting for all of them is the only sound
choice available without building that tracker. This is documented
explicitly, here and in docs/architecture.md/docs/limitations.md, as a
real, deliberate performance/simplicity tradeoff: a `copy_to_host()` or
`copy_to_device()` call is a synchronization boundary, full stop, even
when it touches a buffer no in-flight kernel is anywhere near.
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
        """Copy this array's contents back into a new (or provided) NumPy array.

        Synchronizes first (unconditionally, like `copy_to_device` -- see
        this module's docstring, "Host/device synchronization model"), so
        a preceding kernel's GPU-side failure (see
        `numba_metal.runtime.context.SubmissionRecord`/`synchronize`)
        cannot be silently bypassed by reading stale or in-flight results:
        `copy_to_host()` always surfaces it as a `MetalRuntimeError`
        instead of returning whatever partial/undefined data happened to
        be in the buffer.
        """
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
        same shape and dtype. Used internally and by `to_device`.

        Synchronizes first: without waiting for outstanding GPU work, this
        write could race a previously-submitted kernel that is still
        reading or writing the same shared-memory buffer (see this
        module's docstring, "Host/device synchronization model"). This is
        true for a brand-new buffer with no prior kernel activity too --
        the check is unconditional, not buffer-specific, since numba-metal
        does not (yet) track which in-flight kernels touch which buffers.
        """
        if host_array.shape != self.shape:
            raise MetalRuntimeError(
                f"Shape mismatch copying to device: array is {self.shape}, "
                f"host data is {host_array.shape}."
            )
        get_context().synchronize()
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
