"""Public `numba_metal.metal` API: the `metal` object kernels are written
against (`from numba_metal import metal`).

This module is intentionally a plain Python module (not a class) because
Numba's typing of `metal.grid(...)` inside a kernel relies on Numba
resolving `metal` as a real module global and `grid`/`gridsize` as its
module-level attributes (the same mechanism `numba.cuda`'s `cuda.grid()`
uses) -- see docs/architecture.md.
"""

from __future__ import annotations

import numpy as np

from numba_metal.compiler.intrinsics import (  # noqa: F401 (re-exported)
    atomic_add,
    atomic_compare_exchange,
    atomic_exchange,
    atomic_max,
    atomic_min,
    atomic_sub,
    barrier,
    grid,
    gridsize,
    local_array,
    shared_array,
    thread_in_threadgroup,
    threadgroup_position,
    threads_per_threadgroup,
)
from numba_metal.runtime.array import (
    DeviceNDArray,
    device_array,
    device_array_like,
    to_device,
)
from numba_metal.runtime.context import (
    begin_batch,
    discard_batch,
    end_batch,
    get_context,
)
from numba_metal.runtime.device import DeviceInfo, check_capable
from numba_metal.runtime.dispatcher import device_func, jit

__all__ = [
    "jit",
    "device_func",
    "batch",
    "grid",
    "gridsize",
    "threadgroup_position",
    "thread_in_threadgroup",
    "threads_per_threadgroup",
    "local_array",
    "shared_array",
    "barrier",
    "atomic_add",
    "atomic_sub",
    "atomic_min",
    "atomic_max",
    "atomic_exchange",
    "atomic_compare_exchange",
    "to_device",
    "device_array",
    "device_array_like",
    "DeviceNDArray",
    "synchronize",
    "is_available",
    "get_device_info",
    "config",
]


def synchronize() -> None:
    """Block until all previously launched kernels have completed."""
    get_context().synchronize()


class _BatchContextManager:
    """Implementation of `metal.batch()`; see that function's docstring."""

    def __enter__(self):
        begin_batch()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # An exception inside the `with` block means the caller's own
            # code failed partway through building this batch of
            # launches -- e.g. a KernelLaunchError from a bad argument on
            # the second of three intended launches. Committing the
            # partially-built command buffer anyway would silently run
            # only "however much got encoded before the exception," which
            # is not what the caller asked for and would be surprising
            # (and, for a caller who then retries, could double-submit
            # work). Discard it instead: the batch's command buffer is
            # simply never committed, so it never runs at all, and the
            # caller's exception propagates normally.
            discard_batch()
            return False
        end_batch()
        return False


def batch() -> _BatchContextManager:
    """`with metal.batch():` -- encode every kernel launch inside the
    block onto ONE shared Metal command buffer, committed once when the
    block exits, instead of the default one-command-buffer-per-launch
    behavior. Reduces command-buffer submission overhead for a group of
    launches meant to run as one unit of work.

    Thread-local and not nestable: only one `metal.batch()` block may be
    active per thread at a time (nesting raises `MetalRuntimeError`).
    Every launch inside the block still goes through the same argument
    validation, kernel compilation/caching, and scalar-buffer-pool reuse
    as an unbatched launch -- batching only changes when the command
    buffer is committed and how many `SubmissionRecord`s
    `metal.synchronize()` later sees for this group of launches (one,
    covering all of them, instead of one per launch). If any exception
    is raised inside the block (including from an invalid launch), the
    batch's command buffer is discarded uncommitted -- none of the
    launches encoded so far run -- and the exception propagates normally
    rather than silently running a partial batch.

    `metal.synchronize()` treats a batch's single `SubmissionRecord`
    exactly like any other: if the underlying command buffer fails, the
    error names the combined batch (listing every kernel that was
    encoded onto it), since Metal reports one status/error per command
    buffer, not per individual dispatch within it -- a failure inside a
    batch cannot be attributed to one specific launch out of several
    encoded onto the same command buffer, which is the real precision
    -vs-overhead tradeoff batching makes (see docs/architecture.md).
    """
    return _BatchContextManager()


def is_available() -> bool:
    """Return True if numba-metal can run on this machine (Apple-silicon
    Mac, macOS >= 14, Metal device present, Metal compiler toolchain
    present). Does not raise; use `get_device_info()` to see the error."""
    try:
        check_capable()
    except Exception:
        return False
    return True


def get_device_info() -> DeviceInfo:
    """Return DeviceInfo for the default Metal device, or raise
    UnsupportedPlatformError / MetalToolchainError with a precise reason."""
    return check_capable()


class _Config:
    """Runtime debug configuration.

    `dump_msl`: if True, print generated MSL source for every kernel
    compiled from this point on. Equivalent to setting
    `NUMBA_METAL_DUMP_MSL=1` before the process starts, except it can be
    toggled at runtime.
    """

    def __init__(self) -> None:
        import os

        self.dump_msl: bool = os.environ.get("NUMBA_METAL_DUMP_MSL") == "1"

    def __setattr__(self, name, value):
        if name == "dump_msl":
            import os

            os.environ["NUMBA_METAL_DUMP_MSL"] = "1" if value else "0"
        object.__setattr__(self, name, value)


config = _Config()

# Re-exported for convenience so kernels/benchmarks can do
# `from numba_metal.metal import float32` etc. if desired; not required by
# the mandated kernel-writing API (kernels use plain Python/NumPy dtypes).
float32 = np.float32
int32 = np.int32
uint32 = np.uint32
