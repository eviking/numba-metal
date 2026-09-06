"""GPU-side reductions: `reduce_sum`/`reduce_min`/`reduce_max` over a whole
device array, without a host round trip through `copy_to_host()` for the
per-element data.

This is a host-side Python helper built ENTIRELY out of existing, already-
tested primitives (`metal.shared_array`, `metal.barrier`, `metal.atomic_add`/
`atomic_min`/`atomic_max` -- see `tests/integration/test_local_and_shared_
memory.py` and `test_atomics.py`) -- no new compiler intrinsic or MSL codegen
was needed. `tests/integration/test_local_and_shared_memory.py`'s
`test_shared_array_isolated_per_threadgroup` already demonstrated stage 1 (a
per-threadgroup partial reduction via shared memory + a barrier); this module
adds stage 2 (combining every threadgroup's partial into one final scalar via
a single atomic op per threadgroup, avoiding a second kernel launch) and
packages both stages behind one call.

Design rationale -- one atomic per threadgroup, not one atomic per thread:
the naive approach (every thread does `metal.atomic_add(result, 0, a[i])`)
is correct but serializes the ENTIRE reduction through one contended memory
location at full thread count. Doing the O(log(threadgroup_size)) shared-
memory reduction first means only one atomic op is issued per threadgroup
instead of one per thread -- a 256x reduction in atomic contention at the
default 256-thread threadgroup size used elsewhere in this project's
benchmarks (see `benchmarks/common.py`'s launch-configuration convention).

Non-power-of-two array sizes are handled the same way every kernel in this
project's benchmark suite already handles a non-power-of-two grid: an
`if idx < n:` bounds check on the load into shared memory, with
out-of-bounds lanes contributing the reduction's identity element (0.0 for
sum, +inf for min, -inf for max) instead of skipping the shared-memory write
entirely -- skipping it would leave that slot's PREVIOUS launch's stale data
(Metal does not zero threadgroup memory between dispatches; see
`test_shared_array_repeated_launches_do_not_leak_stale_data`), which is a
real, easy-to-miss correctness bug avoided here by construction, not by
accident.

Only float32, int32, and uint32 are supported, matching the exact dtype
set MSL's native/CAS-loop atomics support (`_ATOMIC_DTYPES` in
`compiler/intrinsics.py`) -- there would be no way to combine per-
threadgroup partials for any other dtype.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from numba_metal.errors import NumbaMetalError
from numba_metal.runtime.array import DeviceNDArray

_REDUCE_THREADS = 256

_SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.uint32))

_IDENTITY = {
    "sum": {
        np.dtype(np.float32): 0.0,
        np.dtype(np.int32): 0,
        np.dtype(np.uint32): 0,
    },
    "min": {
        np.dtype(np.float32): float("inf"),
        np.dtype(np.int32): np.iinfo(np.int32).max,
        np.dtype(np.uint32): np.iinfo(np.uint32).max,
    },
    "max": {
        np.dtype(np.float32): float("-inf"),
        np.dtype(np.int32): np.iinfo(np.int32).min,
        np.dtype(np.uint32): 0,
    },
}

# One compiled kernel per (op, dtype) pair, built lazily on first use and
# cached here for the lifetime of the process -- matches the module-level
# lazy-kernel-construction pattern already used throughout benchmarks/*.py's
# `_make_metal_kernel()` helpers, just cached instead of rebuilt every call.
_KERNEL_CACHE: dict[tuple[str, np.dtype], object] = {}


#: `metal.shared_array(shape, dtype)`'s `dtype` argument must type as a
#: literal NumPy-scalar-type global (see `intrinsics._resolve_array_dtype`)
#: -- a closure over a `dtype.type` local variable does NOT satisfy this
#: (numba-metal has no closures over non-constant outer-scope variables at
#: all; see docs/limitations.md), so each concrete dtype needs its own
#: kernel function with the type spelled out literally in source, one
#: factory per (op, dtype) pair rather than one generic parameterized
#: kernel body.
def _make_sum_kernel_f32():
    from numba_metal import metal

    @metal.jit
    def kernel(a, result, n, identity_value):
        tid = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(_REDUCE_THREADS, np.float32)
        scratch[tid] = a[gid] if gid < n else identity_value
        metal.barrier()
        stride = tg_size // 2
        while stride > 0:
            if tid < stride:
                scratch[tid] = scratch[tid] + scratch[tid + stride]
            metal.barrier()
            stride = stride // 2
        if tid == 0:
            metal.atomic_add(result, 0, scratch[0])

    return kernel


def _make_sum_kernel_i32():
    from numba_metal import metal

    @metal.jit
    def kernel(a, result, n, identity_value):
        tid = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(_REDUCE_THREADS, np.int32)
        scratch[tid] = a[gid] if gid < n else identity_value
        metal.barrier()
        stride = tg_size // 2
        while stride > 0:
            if tid < stride:
                scratch[tid] = scratch[tid] + scratch[tid + stride]
            metal.barrier()
            stride = stride // 2
        if tid == 0:
            metal.atomic_add(result, 0, scratch[0])

    return kernel


def _make_sum_kernel_u32():
    from numba_metal import metal

    @metal.jit
    def kernel(a, result, n, identity_value):
        tid = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(_REDUCE_THREADS, np.uint32)
        scratch[tid] = a[gid] if gid < n else identity_value
        metal.barrier()
        stride = tg_size // 2
        while stride > 0:
            if tid < stride:
                scratch[tid] = scratch[tid] + scratch[tid + stride]
            metal.barrier()
            stride = stride // 2
        if tid == 0:
            metal.atomic_add(result, 0, scratch[0])

    return kernel


def _make_min_kernel_f32():
    from numba_metal import metal

    @metal.jit
    def kernel(a, result, n, identity_value):
        tid = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(_REDUCE_THREADS, np.float32)
        scratch[tid] = a[gid] if gid < n else identity_value
        metal.barrier()
        stride = tg_size // 2
        while stride > 0:
            if tid < stride:
                other = scratch[tid + stride]
                if other < scratch[tid]:
                    scratch[tid] = other
            metal.barrier()
            stride = stride // 2
        if tid == 0:
            metal.atomic_min(result, 0, scratch[0])

    return kernel


def _make_min_kernel_i32():
    from numba_metal import metal

    @metal.jit
    def kernel(a, result, n, identity_value):
        tid = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(_REDUCE_THREADS, np.int32)
        scratch[tid] = a[gid] if gid < n else identity_value
        metal.barrier()
        stride = tg_size // 2
        while stride > 0:
            if tid < stride:
                other = scratch[tid + stride]
                if other < scratch[tid]:
                    scratch[tid] = other
            metal.barrier()
            stride = stride // 2
        if tid == 0:
            metal.atomic_min(result, 0, scratch[0])

    return kernel


def _make_min_kernel_u32():
    from numba_metal import metal

    @metal.jit
    def kernel(a, result, n, identity_value):
        tid = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(_REDUCE_THREADS, np.uint32)
        scratch[tid] = a[gid] if gid < n else identity_value
        metal.barrier()
        stride = tg_size // 2
        while stride > 0:
            if tid < stride:
                other = scratch[tid + stride]
                if other < scratch[tid]:
                    scratch[tid] = other
            metal.barrier()
            stride = stride // 2
        if tid == 0:
            metal.atomic_min(result, 0, scratch[0])

    return kernel


def _make_max_kernel_f32():
    from numba_metal import metal

    @metal.jit
    def kernel(a, result, n, identity_value):
        tid = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(_REDUCE_THREADS, np.float32)
        scratch[tid] = a[gid] if gid < n else identity_value
        metal.barrier()
        stride = tg_size // 2
        while stride > 0:
            if tid < stride:
                other = scratch[tid + stride]
                if other > scratch[tid]:
                    scratch[tid] = other
            metal.barrier()
            stride = stride // 2
        if tid == 0:
            metal.atomic_max(result, 0, scratch[0])

    return kernel


def _make_max_kernel_i32():
    from numba_metal import metal

    @metal.jit
    def kernel(a, result, n, identity_value):
        tid = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(_REDUCE_THREADS, np.int32)
        scratch[tid] = a[gid] if gid < n else identity_value
        metal.barrier()
        stride = tg_size // 2
        while stride > 0:
            if tid < stride:
                other = scratch[tid + stride]
                if other > scratch[tid]:
                    scratch[tid] = other
            metal.barrier()
            stride = stride // 2
        if tid == 0:
            metal.atomic_max(result, 0, scratch[0])

    return kernel


def _make_max_kernel_u32():
    from numba_metal import metal

    @metal.jit
    def kernel(a, result, n, identity_value):
        tid = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(_REDUCE_THREADS, np.uint32)
        scratch[tid] = a[gid] if gid < n else identity_value
        metal.barrier()
        stride = tg_size // 2
        while stride > 0:
            if tid < stride:
                other = scratch[tid + stride]
                if other > scratch[tid]:
                    scratch[tid] = other
            metal.barrier()
            stride = stride // 2
        if tid == 0:
            metal.atomic_max(result, 0, scratch[0])

    return kernel


_KERNEL_FACTORIES = {
    ("sum", np.dtype(np.float32)): _make_sum_kernel_f32,
    ("sum", np.dtype(np.int32)): _make_sum_kernel_i32,
    ("sum", np.dtype(np.uint32)): _make_sum_kernel_u32,
    ("min", np.dtype(np.float32)): _make_min_kernel_f32,
    ("min", np.dtype(np.int32)): _make_min_kernel_i32,
    ("min", np.dtype(np.uint32)): _make_min_kernel_u32,
    ("max", np.dtype(np.float32)): _make_max_kernel_f32,
    ("max", np.dtype(np.int32)): _make_max_kernel_i32,
    ("max", np.dtype(np.uint32)): _make_max_kernel_u32,
}


def _reduce(op: Literal["sum", "min", "max"], a: DeviceNDArray):
    if not isinstance(a, DeviceNDArray):
        raise NumbaMetalError(
            "metal.reduce_* requires a DeviceNDArray (use metal.to_device() "
            "first); got a host array or other object directly."
        )
    if a.ndim != 1:
        raise NumbaMetalError(
            f"metal.reduce_* only supports 1D device arrays; got shape "
            f"{a.shape} ({a.ndim}D). Reshape/flatten before reducing."
        )
    dtype = np.dtype(a.dtype)
    if dtype not in _SUPPORTED_DTYPES:
        raise NumbaMetalError(
            f"metal.reduce_* only supports float32/int32/uint32 device "
            f"arrays (matches MSL's native atomic dtype set); got {dtype!r}."
        )
    if a.size == 0:
        raise NumbaMetalError("metal.reduce_* requires a non-empty array.")

    from numba_metal import metal

    key = (op, dtype)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _KERNEL_FACTORIES[key]()
        _KERNEL_CACHE[key] = kernel

    identity = dtype.type(_IDENTITY[op][dtype])
    result = metal.to_device(np.array([identity], dtype=dtype))
    n = a.size
    blocks = (n + _REDUCE_THREADS - 1) // _REDUCE_THREADS
    kernel[blocks, _REDUCE_THREADS](a, result, np.int32(n), identity)
    metal.synchronize()
    return result


def reduce_sum(a: DeviceNDArray) -> DeviceNDArray:
    """Sum every element of a 1D device array, entirely on the GPU.

    Returns a 1-element DeviceNDArray (call `.copy_to_host()[0]` for the
    scalar) rather than a Python float, so the result can itself stay
    device-resident and feed a later kernel launch without an intervening
    host round trip -- matching this project's existing device-array-in/
    device-array-out convention rather than introducing a special case
    where a reduction alone forces a host transfer.

    Supports float32, int32, and uint32 (the dtypes MSL provides native
    or CAS-loop atomics for). See module docstring for the two-stage
    (shared-memory tree + one atomic per threadgroup) algorithm.
    """
    return _reduce("sum", a)


def reduce_min(a: DeviceNDArray) -> DeviceNDArray:
    """Minimum of every element of a 1D device array, entirely on the GPU.
    See `reduce_sum` for the return-type rationale and supported dtypes."""
    return _reduce("min", a)


def reduce_max(a: DeviceNDArray) -> DeviceNDArray:
    """Maximum of every element of a 1D device array, entirely on the GPU.
    See `reduce_sum` for the return-type rationale and supported dtypes."""
    return _reduce("max", a)
