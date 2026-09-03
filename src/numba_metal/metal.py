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
from numba_metal.runtime.context import get_context
from numba_metal.runtime.device import DeviceInfo, check_capable
from numba_metal.runtime.dispatcher import jit

__all__ = [
    "jit",
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
