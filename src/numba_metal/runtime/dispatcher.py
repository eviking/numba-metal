"""Kernel launch: `kernel[blocks, threads](*args)`.

Implements CUDA-style launch-configuration syntax on top of the
compilation pipeline and Metal command encoding. Argument binding, launch
geometry validation, and command submission all happen here; MSL
generation and caching live in `numba_metal.compiler.pipeline`.
"""

from __future__ import annotations

import numpy as np
from numba.core import types as nb_types

from numba_metal.compiler.pipeline import CompiledKernel, KernelCache
from numba_metal.errors import KernelLaunchError
from numba_metal.runtime.array import DeviceNDArray
from numba_metal.runtime.context import get_context

_SCALAR_NUMPY_TO_NUMBA = {
    np.dtype(np.float32): nb_types.float32,
    np.dtype(np.float16): nb_types.float16,
    np.dtype(np.int32): nb_types.int32,
    np.dtype(np.uint32): nb_types.uint32,
    np.dtype(np.int64): nb_types.int64,
    np.dtype(np.bool_): nb_types.boolean,
}

_PY_SCALAR_TO_NUMBA = {
    bool: nb_types.boolean,
    int: nb_types.int64,
    float: nb_types.float32,
}


#: numba-metal supports 1D, 2D, and 3D device-array kernel arguments,
#: matching the ndim range already accepted by the position intrinsics
#: (metal.grid(ndim), metal.thread_in_threadgroup(ndim), etc.) -- see
#: msl_backend.py's `_classify_params`/`_emit_signature` for the
#: corresponding MSL-side flattened-indexing codegen, and
#: `runtime/array.py`'s `DeviceNDArray` (already shape-agnostic; this
#: was always the only real restriction) for why no runtime-array
#: change was needed to support this.
_MAX_ARRAY_NDIM = 3


def _infer_arg_type(value) -> nb_types.Type:
    if isinstance(value, DeviceNDArray):
        if value.ndim < 1 or value.ndim > _MAX_ARRAY_NDIM:
            raise KernelLaunchError(
                f"Kernel arguments must be 1D, 2D, or 3D device arrays; "
                f"got shape {value.shape} ({value.ndim}D)."
            )
        try:
            scalar = _SCALAR_NUMPY_TO_NUMBA[value.dtype]
        except KeyError as exc:
            raise KernelLaunchError(
                f"Unsupported device array dtype {value.dtype!r}."
            ) from exc
        if value.ndim == 1:
            return scalar[::1]
        if value.ndim == 2:
            return scalar[:, ::1]
        return scalar[:, :, ::1]
    if isinstance(value, np.generic):
        try:
            return _SCALAR_NUMPY_TO_NUMBA[np.dtype(type(value))]
        except KeyError as exc:
            raise KernelLaunchError(
                f"Unsupported scalar argument dtype {np.dtype(type(value))!r}."
            ) from exc
    if isinstance(value, bool):
        return nb_types.boolean
    if isinstance(value, int):
        return nb_types.int64
    if isinstance(value, float):
        return nb_types.float32
    if isinstance(value, np.ndarray):
        raise KernelLaunchError(
            "Kernel arguments must be device arrays (use metal.to_device() "
            "first); a host numpy.ndarray was passed directly. numba-metal "
            "never silently copies host arrays to the device at launch "
            "time -- transfers must be explicit."
        )
    raise KernelLaunchError(
        f"Unsupported kernel argument type {type(value)!r} (value={value!r})."
    )


_LaunchDim = int | tuple[int, int] | tuple[int, int, int]


class _LaunchConfigured:
    """Bound to a launch geometry via `kernel[blocks, threads]`; calling
    it with kernel arguments compiles (if needed) and dispatches.

    `blocks`/`threads` are each a plain int (1D launch; the kernel
    typically calls `metal.grid(1)`), a `(x, y)` int tuple (2D launch;
    `metal.grid(2)`), or a `(x, y, z)` int tuple (3D launch;
    `metal.grid(3)`) -- mirroring CUDA-style `kernel[blocks, threads]`
    syntax extended to 1D/2D/3D.
    """

    __slots__ = ("_dispatcher", "_blocks", "_threads")

    def __init__(
        self, dispatcher: KernelDispatcher, blocks: _LaunchDim, threads: _LaunchDim
    ):
        self._dispatcher = dispatcher
        self._blocks = blocks
        self._threads = threads

    def __call__(self, *args):
        self._dispatcher._launch(self._blocks, self._threads, args)


class KernelDispatcher:
    """The object returned by `@metal.jit`.

    Supports `kernel[blocks, threads](*args)` launch syntax and caches
    compiled kernels per argument-type signature (and, transitively, per
    device -- see `numba_metal.compiler.pipeline.KernelCache`).
    """

    def __init__(self, func):
        self.py_func = func
        self.__name__ = getattr(func, "__name__", "kernel")
        self.__doc__ = getattr(func, "__doc__", None)
        self._cache = KernelCache()

    def __getitem__(self, launch_config) -> _LaunchConfigured:
        blocks, threads = _parse_launch_config(launch_config)
        return _LaunchConfigured(self, blocks, threads)

    def _launch(self, blocks: _LaunchDim, threads: _LaunchDim, args: tuple) -> None:
        bx, by, bz = _as_3tuple(blocks)
        tx, ty, tz = _as_3tuple(threads)

        for label, val in (
            ("block count", bx),
            ("block count", by),
            ("block count", bz),
            ("threads-per-block", tx),
            ("threads-per-block", ty),
            ("threads-per-block", tz),
        ):
            if val <= 0:
                raise KernelLaunchError(
                    f"Invalid launch configuration: {label} must be "
                    f"positive, got {val}."
                )
        ctx = get_context()
        total_tg_threads = tx * ty * tz
        if total_tg_threads > ctx.info.max_threads_per_threadgroup:
            raise KernelLaunchError(
                f"Invalid launch configuration: threads-per-block "
                f"({tx}x{ty}x{tz}={total_tg_threads}) exceeds this "
                f"device's maximum ({ctx.info.max_threads_per_threadgroup})."
            )

        arg_types = tuple(_infer_arg_type(a) for a in args)
        compiled = self._cache.get_or_compile(self.py_func, arg_types)

        if len(args) != len(compiled.signature.param_order):
            raise KernelLaunchError(
                f"Kernel {self.__name__!r} expects "
                f"{len(compiled.signature.param_order)} arguments, got "
                f"{len(args)}."
            )

        self._dispatch(compiled, bx, by, bz, tx, ty, tz, args)

    def _dispatch(
        self,
        compiled: CompiledKernel,
        bx: int,
        by: int,
        bz: int,
        tx: int,
        ty: int,
        tz: int,
        args: tuple,
    ) -> None:
        import Metal

        from numba_metal.runtime.context import current_batch

        ctx = get_context()
        batch = current_batch()
        # Inside a `metal.batch()` block, every launch on this thread
        # encodes onto the SAME shared command buffer (committed once,
        # as a single SubmissionRecord, when the batch ends) instead of
        # each launch getting its own command buffer committed
        # immediately -- see context.py's `_BatchState`/`begin_batch`/
        # `end_batch`. Outside a batch, `cmdbuf` is a brand-new command
        # buffer as before, committed at the end of this method exactly
        # like every previous numba-metal release.
        cmdbuf = (
            batch.command_buffer if batch is not None else ctx.queue.commandBuffer()
        )
        encoder = cmdbuf.computeCommandEncoder()
        encoder.setComputePipelineState_(compiled.pipeline_state)

        buffer_index = 0
        # Every Metal resource this dispatch touches -- argument array
        # buffers, synthesized scalar/size constant buffers, and the
        # pipeline state -- is retained here and handed to
        # `register_submission` so it cannot be garbage-collected while
        # the GPU may still be reading/writing it, for as long as this
        # command buffer is outstanding (see runtime/context.py). This
        # replaces the previous MVP's `addCompletedHandler_` closure
        # trick, which only kept small scalar buffers alive and did
        # nothing to let a caller observe *when* or *whether* the work
        # actually completed successfully.
        resources: list[object] = [compiled.pipeline_state]
        # Scalar-argument and array-size constant buffers acquired from
        # (or, on a pool miss, freshly allocated for) `ctx`'s process
        # -wide scalar-buffer pool -- see runtime/context.py's
        # `acquire_scalar_buffer`/`_release_scalar_buffers`. Recorded
        # here as `(byte_size, buffer)` pairs and handed to
        # `register_submission` as `reusable_buffers`, so `synchronize()`
        # can return them to the pool once THIS submission specifically
        # is confirmed complete -- never sooner, since Metal has no
        # notion of "this buffer's contents are no longer needed" short
        # of the command buffer that reads it actually finishing.
        reusable_buffers: list[tuple[int, object]] = []
        device = ctx.device

        for (kind, name), value in zip(
            compiled.signature.param_order, args, strict=True
        ):
            if kind == "array":
                dev_array: DeviceNDArray = value
                resources.append(dev_array.buffer)
                encoder.setBuffer_offset_atIndex_(dev_array.buffer, 0, buffer_index)
                buffer_index += 1
                size_buf = _acquire_scalar_buffer(
                    ctx, device, np.uint32(dev_array.size)
                )
                resources.append(size_buf)
                reusable_buffers.append((4, size_buf))
                encoder.setBuffer_offset_atIndex_(size_buf, 0, buffer_index)
                buffer_index += 1
                # For ndim>1 arrays, bind one additional scalar buffer per
                # TRAILING dimension size (shape[1], shape[2], ... --
                # never shape[0], which a row-major flat-offset
                # computation never needs: flat = i0*shape[1]*shape[2]
                # + i1*shape[2] + i2 for 3D, or i0*shape[1] + i1 for 2D).
                # Order and count must exactly match
                # msl_backend.py's `_emit_signature`, which binds these
                # at the same buffer indices right after `_size`.
                for dim_size in dev_array.shape[1:]:
                    dim_buf = _acquire_scalar_buffer(ctx, device, np.uint32(dim_size))
                    resources.append(dim_buf)
                    reusable_buffers.append((4, dim_buf))
                    encoder.setBuffer_offset_atIndex_(dim_buf, 0, buffer_index)
                    buffer_index += 1
            else:
                info = next(s for s in compiled.signature.scalar_params if s[0] == name)
                np_dtype = _numba_scalar_to_numpy(info[1])
                np_scalar = np_dtype.type(value)
                scalar_buf = _acquire_scalar_buffer(ctx, device, np_scalar)
                resources.append(scalar_buf)
                reusable_buffers.append((np_scalar.itemsize, scalar_buf))
                encoder.setBuffer_offset_atIndex_(scalar_buf, 0, buffer_index)
                buffer_index += 1

        # Every `metal.shared_array()` declared in this kernel must have
        # its backing threadgroup memory explicitly allocated via
        # setThreadgroupMemoryLength:atIndex:, with `atIndex` matching
        # the `[[threadgroup(n)]]` attribute index the MSL backend
        # assigned it (see MSLKernelLowerer._emit_signature). Metal does
        # NOT allocate this automatically from the `threadgroup <type>
        # name[count]` parameter declaration alone -- omitting this call
        # silently leaves the memory unallocated, which was observed
        # directly while implementing this feature: every thread's read
        # of its own just-written value came back as 0, with no error
        # from Metal at compile, encode, or execution time.
        for tg_index, tg in enumerate(compiled.signature.threadgroup_arrays):
            encoder.setThreadgroupMemoryLength_atIndex_(tg.byte_size, tg_index)

        grid_x, grid_y, grid_z = bx * tx, by * ty, bz * tz
        grid_size = Metal.MTLSizeMake(grid_x, grid_y, grid_z)
        # The threadgroup size dispatched is EXACTLY (tx, ty, tz) as
        # requested by the caller -- `_launch` already validated
        # tx*ty*tz <= max_threads_per_threadgroup before reaching here, so
        # there is no need (and it would be actively wrong) to silently
        # rescale any dimension: a kernel that calls
        # metal.threads_per_threadgroup()/metal.thread_in_threadgroup()
        # must see the threadgroup size it was actually launched with, not
        # one numba-metal quietly substituted.
        tg_size = Metal.MTLSizeMake(tx, ty, tz)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        if batch is not None:
            # Do not commit or register yet -- accumulate this launch's
            # resources/reusable buffers/name into the shared batch
            # state; `end_batch()` (called when the `metal.batch()`
            # block exits) commits the ONE shared command buffer once
            # and registers a single SubmissionRecord covering every
            # launch encoded onto it in this batch.
            batch.resources.extend(resources)
            batch.reusable_buffers.extend(reusable_buffers)
            batch.kernel_names.append(compiled.name)
            batch.msl_sources.append(compiled.msl_source)
            return
        cmdbuf.commit()
        ctx.register_submission(
            command_buffer=cmdbuf,
            kernel_name=compiled.name,
            resources=resources,
            msl_source=compiled.msl_source,
            reusable_buffers=reusable_buffers,
        )


def _acquire_scalar_buffer(ctx, device, np_scalar):
    """Return an `MTLBuffer` containing exactly `np_scalar`'s bytes,
    reusing a same-sized buffer from `ctx`'s process-wide scalar-buffer
    pool if one is available (writing the new bytes into it via
    `contents()`, the same mechanism `DeviceNDArray.copy_to_device` uses
    for ordinary host writes), or allocating a fresh one on a pool miss.

    Safe by construction, not by convention: a buffer only becomes
    available from the pool after `context.py`'s `synchronize()` has
    confirmed the command buffer that last used it has completed (see
    that module's docstring) -- there is no path that returns a buffer
    to the pool, or hands one out from it, while GPU work could still be
    reading its old contents.
    """
    data = np.asarray(np_scalar).tobytes()
    pooled = ctx.acquire_scalar_buffer(len(data))
    if pooled is not None:
        ptr = pooled.contents()
        raw = ptr.as_buffer(len(data))
        raw[: len(data)] = data
        return pooled
    buf = device.newBufferWithBytes_length_options_(data, len(data), 0)
    if buf is None:
        raise KernelLaunchError("Failed to allocate a scalar argument buffer.")
    return buf


def _numba_scalar_to_numpy(ty: nb_types.Type) -> np.dtype:
    for np_dtype, numba_ty in _SCALAR_NUMPY_TO_NUMBA.items():
        if ty == numba_ty:
            return np_dtype
    raise KernelLaunchError(f"Unsupported scalar argument type {ty!r}.")


def _parse_launch_dim(value, label: str) -> _LaunchDim:
    if isinstance(value, bool):
        raise KernelLaunchError(f"Launch configuration {label} must be int, not bool.")
    if isinstance(value, int):
        return value
    if (
        isinstance(value, tuple)
        and len(value) in (2, 3)
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        return value
    raise KernelLaunchError(
        f"Launch configuration {label} must be an int (1D), a 2-tuple of "
        f"int (2D), or a 3-tuple of int (3D); got {value!r}."
    )


def _launch_dim_ndim(value: _LaunchDim) -> int:
    return 1 if isinstance(value, int) else len(value)


def _as_3tuple(value: _LaunchDim) -> tuple[int, int, int]:
    if isinstance(value, int):
        return value, 1, 1
    if len(value) == 2:
        return value[0], value[1], 1
    return value[0], value[1], value[2]


def _parse_launch_config(launch_config) -> tuple[_LaunchDim, _LaunchDim]:
    if not isinstance(launch_config, tuple) or len(launch_config) != 2:
        raise KernelLaunchError(
            "Launch configuration must be `kernel[blocks, threads]`; got "
            f"{launch_config!r}."
        )
    blocks, threads = launch_config
    blocks = _parse_launch_dim(blocks, "blocks")
    threads = _parse_launch_dim(threads, "threads")
    if _launch_dim_ndim(blocks) != _launch_dim_ndim(threads):
        raise KernelLaunchError(
            "Launch configuration blocks and threads must have the same "
            f"number of dimensions; got blocks={blocks!r} "
            f"({_launch_dim_ndim(blocks)}D), threads={threads!r} "
            f"({_launch_dim_ndim(threads)}D)."
        )
    return blocks, threads


def jit(func=None):
    """`@metal.jit` decorator: wraps a Python function as a Metal kernel.

    Compilation is deferred until first launch (the argument types
    determine the kernel signature, matching CUDA-style kernel jitting).
    """

    def wrap(f):
        return KernelDispatcher(f)

    if func is not None:
        return wrap(func)
    return wrap


#: CPUDispatcher (the object `@metal.device_func` returns and users call
#: directly from inside a kernel body, e.g. `helper(a[i])`) -> original
#: plain Python function. Populated by `device_func` below, consulted by
#: `msl_backend.py`'s `_call` to recognize a device-function call site
#: and recursively compile the ORIGINAL function through numba-metal's
#: own typed-IR-to-MSL pipeline (never the CPUDispatcher's own LLVM
#: lowering, which is only used to get Numba's frontend to type the call
#: site the same way it already types an ordinary `@njit` call).
#:
#: Module-level (not per-dispatcher) because a device function is a
#: plain global the user's kernel module defines once and may call from
#: multiple kernels/other device functions, exactly like `metal.grid`;
#: it is never mutated after `device_func()` returns, so this is safe to
#: share across every `KernelDispatcher`/compilation.
_DEVICE_FUNCTION_REGISTRY: dict[object, object] = {}


def device_func(func=None):
    """`@metal.device_func` decorator: wraps a scalar-argument,
    scalar-return helper function so it can be called from inside a
    `@metal.jit` kernel body (or from another `@metal.device_func`),
    e.g. `helper(a[i])`. Compiled (via numba-metal's own typed-IR-to-MSL
    pipeline, recursively) to a real, separate MSL function the first
    time a kernel that calls it is compiled -- not inlined.

    Returns a real `@njit`-compiled dispatcher (so Numba's own frontend
    types a call to it exactly the way it already types a call to an
    ordinary `@njit` function -- a `CPUDispatcher` global with a
    resolved call signature in `calltypes`); numba-metal never runs that
    dispatcher's own LLVM lowering, only uses it to get the call site
    typed, then recompiles the original plain function through its own
    pipeline to produce MSL (see `_DEVICE_FUNCTION_REGISTRY`).

    Scalar (int32/uint32/int64/float32/float16/bool) arguments and
    return type are supported, and so are 1D/2D/3D array arguments of a
    supported dtype (forwarded from the caller's own array argument --
    a device function cannot allocate or return an array itself). No
    `metal.local_array`/`shared_array`, no recursion (calling a device
    function from within itself, directly or transitively, is rejected
    at compile time). See docs/supported-features.md.
    """

    def wrap(f):
        from numba import njit

        compiled = njit(cache=False)(f)
        _DEVICE_FUNCTION_REGISTRY[compiled] = f
        return compiled

    if func is not None:
        return wrap(func)
    return wrap


def is_device_function(callee) -> bool:
    """True if `callee` is a `@metal.device_func`-decorated dispatcher
    (used by `msl_backend.py` to recognize a device-function call site)."""
    return callee in _DEVICE_FUNCTION_REGISTRY


def device_function_py_func(callee):
    """The original plain Python function behind a
    `@metal.device_func`-decorated dispatcher."""
    return _DEVICE_FUNCTION_REGISTRY[callee]
