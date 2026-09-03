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


_LaunchDim = int | tuple[int, int]


class _LaunchConfigured:
    """Bound to a launch geometry via `kernel[blocks, threads]`; calling
    it with kernel arguments compiles (if needed) and dispatches.

    `blocks`/`threads` are each either a plain int (1D launch; the kernel
    typically calls `metal.grid(1)`) or a `(x, y)` int tuple (2D launch;
    the kernel typically calls `metal.grid(2)`) -- mirroring CUDA-style
    `kernel[blocks, threads]` syntax extended to 2D per the task's
    "1D or 2D grids" requirement.
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
        bx, by = blocks if isinstance(blocks, tuple) else (blocks, 1)
        tx, ty = threads if isinstance(threads, tuple) else (threads, 1)

        for label, val in (
            ("block count", bx),
            ("block count", by),
            ("threads-per-block", tx),
            ("threads-per-block", ty),
        ):
            if val <= 0:
                raise KernelLaunchError(
                    f"Invalid launch configuration: {label} must be "
                    f"positive, got {val}."
                )
        ctx = get_context()
        total_tg_threads = tx * ty
        if total_tg_threads > ctx.info.max_threads_per_threadgroup:
            raise KernelLaunchError(
                f"Invalid launch configuration: threads-per-block "
                f"({tx}x{ty}={total_tg_threads}) exceeds this device's "
                f"maximum ({ctx.info.max_threads_per_threadgroup})."
            )

        arg_types = tuple(_infer_arg_type(a) for a in args)
        compiled = self._cache.get_or_compile(self.py_func, arg_types)

        if len(args) != len(compiled.signature.param_order):
            raise KernelLaunchError(
                f"Kernel {self.__name__!r} expects "
                f"{len(compiled.signature.param_order)} arguments, got "
                f"{len(args)}."
            )

        self._dispatch(compiled, bx, by, tx, ty, args)

    def _dispatch(
        self, compiled: CompiledKernel, bx: int, by: int, tx: int, ty: int, args: tuple
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

        grid_x, grid_y = bx * tx, by * ty
        grid_size = Metal.MTLSizeMake(grid_x, grid_y, 1)
        # Metal requires threadsPerThreadgroup's product not exceed the
        # pipeline's maxTotalThreadsPerThreadgroup; scale down y first if
        # needed since numba-metal's supported kernels are typically x-major.
        tg_x = min(tx, compiled.max_threads_per_threadgroup)
        tg_y = max(1, min(ty, compiled.max_threads_per_threadgroup // max(tg_x, 1)))
        tg_size = Metal.MTLSizeMake(tg_x, tg_y, 1)
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
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        return value
    raise KernelLaunchError(
        f"Launch configuration {label} must be an int (1D) or a 2-tuple "
        f"of int (2D); got {value!r}."
    )


def _parse_launch_config(launch_config) -> tuple[_LaunchDim, _LaunchDim]:
    if not isinstance(launch_config, tuple) or len(launch_config) != 2:
        raise KernelLaunchError(
            "Launch configuration must be `kernel[blocks, threads]`; got "
            f"{launch_config!r}."
        )
    blocks, threads = launch_config
    blocks = _parse_launch_dim(blocks, "blocks")
    threads = _parse_launch_dim(threads, "threads")
    if isinstance(blocks, tuple) != isinstance(threads, tuple):
        raise KernelLaunchError(
            "Launch configuration blocks and threads must both be 1D (int) "
            f"or both be 2D (tuple); got blocks={blocks!r}, threads={threads!r}."
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
