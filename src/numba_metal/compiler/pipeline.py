"""Ties the frontend (typed IR) and backend (MSL codegen) together, with a
compilation cache keyed by kernel source, signature, compiler options, and
device identity.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import threading
from dataclasses import dataclass

from numba.core import types as nb_types

from numba_metal.compiler.frontend import compile_to_typed_ir
from numba_metal.compiler.msl_backend import KernelSignatureInfo, MSLKernelLowerer
from numba_metal.errors import KernelCompilationError
from numba_metal.runtime.context import get_context

_MSL_PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n\n"

_name_counter = itertools.count()
_name_lock = threading.Lock()


def _next_kernel_name(func_name: str) -> str:
    with _name_lock:
        n = next(_name_counter)
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in func_name)
    return f"nbmtl_{safe}_{n}"


@dataclass
class CompiledKernel:
    """A fully compiled kernel: MSL source, its Metal compute pipeline
    state, and the metadata needed to bind launch arguments to it."""

    name: str
    msl_source: str
    signature: KernelSignatureInfo
    arg_types: tuple[nb_types.Type, ...]
    pipeline_state: object  # MTLComputePipelineState
    max_threads_per_threadgroup: int


def _cache_key(func, arg_types, device_registry_id: int) -> str:
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = repr(func)
    payload = "|".join(
        [
            source,
            repr(func.__qualname__),
            repr(arg_types),
            str(device_registry_id),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class KernelCache:
    """Compilation cache keyed by (kernel source, signature, device).

    A cache hit skips both the Numba typed-IR frontend and Metal shader
    compilation. Cache lifetime is process-local (in-memory); there is no
    on-disk persistent cache in the MVP (see docs/roadmap.md).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, CompiledKernel] = {}

    def get_or_compile(
        self, func, arg_types: tuple[nb_types.Type, ...]
    ) -> CompiledKernel:
        """Return the cached CompiledKernel for `func`+`arg_types` on the
        current device, compiling it (Numba typing + MSL codegen + Metal
        shader compilation) on a cache miss."""
        ctx = get_context()
        device_id = ctx.info.registry_id
        key = _cache_key(func, arg_types, device_id)
        with self._lock:
            hit = self._entries.get(key)
        if hit is not None:
            return hit

        compiled = _compile_kernel(func, arg_types)
        with self._lock:
            self._entries[key] = compiled
        return compiled

    def clear(self) -> None:
        """Discard all cached compiled kernels."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def _compile_kernel(func, arg_types: tuple[nb_types.Type, ...]) -> CompiledKernel:
    import os

    typed = compile_to_typed_ir(func, arg_types)
    kernel_name = _next_kernel_name(getattr(func, "__name__", "kernel"))
    lowerer = MSLKernelLowerer(kernel_name, typed)
    body_src = lowerer.lower()
    # Every @metal.device_func this kernel called (directly or
    # transitively) was compiled to its own standalone MSL function
    # during `lower()` (see MSLKernelLowerer._compile_device_function);
    # those must be declared/defined in the same compilation unit,
    # before the kernel body that calls them.
    device_functions_src = "\n".join(lowerer.device_function_sources)
    full_src = _MSL_PRELUDE + device_functions_src + body_src

    if os.environ.get("NUMBA_METAL_DUMP_MSL") == "1":
        print(f"// ---- numba-metal generated MSL: {kernel_name} ----")
        print(full_src)
        print("// ---- end ----")

    ctx = get_context()
    device = ctx.device

    import Metal

    opts = Metal.MTLCompileOptions.alloc().init()
    # Metal's default fast-math mode permits FMA fusion and reassociation
    # that can change results at the last representable bit relative to
    # ordinary (non-fused) float32 arithmetic. That is invisible for most
    # kernels but compounds significantly in chaotic iterative algorithms
    # (e.g. Mandelbrot escape-time), where it was observed to shift escape
    # iteration counts by tens of iterations for a small fraction of
    # pixels relative to the CPU reference -- see docs/limitations.md and
    # benchmarks/mandelbrot.py. Disabling it trades a small amount of
    # performance for results that match standard (non-fused) float32
    # semantics, which is the more defensible default for a correctness-
    # focused MVP.
    opts.setFastMathEnabled_(False)
    library, err = device.newLibraryWithSource_options_error_(full_src, opts, None)
    if library is None:
        raise KernelCompilationError(
            f"Metal shader compilation failed for kernel "
            f"{getattr(func, '__name__', func)!r}:\n{err}\n\n"
            f"Generated MSL:\n{full_src}"
        )
    function = library.newFunctionWithName_(kernel_name)
    if function is None:
        raise KernelCompilationError(
            f"Metal library compiled but function {kernel_name!r} was not "
            "found in it (internal numba-metal codegen error)."
        )
    pipeline_state, perr = device.newComputePipelineStateWithFunction_error_(
        function, None
    )
    if pipeline_state is None:
        raise KernelCompilationError(
            f"Failed to create Metal compute pipeline state for kernel "
            f"{getattr(func, '__name__', func)!r}: {perr}"
        )

    return CompiledKernel(
        name=kernel_name,
        msl_source=full_src,
        signature=lowerer.sig,
        arg_types=arg_types,
        pipeline_state=pipeline_state,
        max_threads_per_threadgroup=int(pipeline_state.maxTotalThreadsPerThreadgroup()),
    )
