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


def _infer_arg_type(value) -> nb_types.Type:
    if isinstance(value, DeviceNDArray):
        if value.ndim != 1:
            raise KernelLaunchError(
                f"Kernel arguments must be 1D device arrays in this MVP; "
                f"got shape {value.shape}."
            )
        try:
            scalar = _SCALAR_NUMPY_TO_NUMBA[value.dtype]
        except KeyError as exc:
            raise KernelLaunchError(
                f"Unsupported device array dtype {value.dtype!r}."
            ) from exc
        return scalar[::1]
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

        ctx = get_context()
        cmdbuf = ctx.queue.commandBuffer()
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
        device = ctx.device

        for (kind, name), value in zip(
            compiled.signature.param_order, args, strict=True
        ):
            if kind == "array":
                dev_array: DeviceNDArray = value
                resources.append(dev_array.buffer)
                encoder.setBuffer_offset_atIndex_(dev_array.buffer, 0, buffer_index)
                buffer_index += 1
                size_buf = _scalar_buffer(device, np.uint32(dev_array.size))
                resources.append(size_buf)
                encoder.setBuffer_offset_atIndex_(size_buf, 0, buffer_index)
                buffer_index += 1
            else:
                info = next(s for s in compiled.signature.scalar_params if s[0] == name)
                np_dtype = _numba_scalar_to_numpy(info[1])
                scalar_buf = _scalar_buffer(device, np_dtype.type(value))
                resources.append(scalar_buf)
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
        cmdbuf.commit()
        ctx.register_submission(
            command_buffer=cmdbuf,
            kernel_name=compiled.name,
            resources=resources,
            msl_source=compiled.msl_source,
        )


def _scalar_buffer(device, np_scalar):
    data = np.asarray(np_scalar).tobytes()
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
