"""Kernel-side intrinsics: thread/threadgroup indexing, local/threadgroup
memory allocation, barriers, and atomics.

These are typed using `numba.extending.intrinsic` purely so Numba's type
inference accepts calls to them inside a kernel and produces a normal typed
`ir.Expr.call` node with a known return type. Their `codegen` callbacks are
never invoked: numba-metal never runs Numba's LLVM lowering. Instead,
`numba_metal.compiler.msl_backend` pattern-matches calls to these exact
function objects while walking the typed IR and emits the corresponding MSL
expression/statement directly. This split (real typing, intercepted
lowering) was verified empirically against this Numba version before
committing to it -- see docs/architecture.md.
"""

from __future__ import annotations

from numba.core import types
from numba.core.errors import NumbaValueError, RequireLiteralValue, TypingError
from numba.core.typing import signature
from numba.extending import intrinsic

MAX_GRID_DIMS = 3

#: Scalar element dtypes accepted by metal.local_array()/shared_array().
#: Matches the set of array-element dtypes the MSL backend can already
#: emit a type for (numba_scalar_to_msl's domain) -- kept as an explicit
#: allow-list here (rather than deferring to the backend to reject a bad
#: dtype later) so an unsupported dtype fails at the intrinsic's own
#: typing step, at the call site, with a message naming this specific
#: function.
_ARRAY_ELEMENT_DTYPES = (
    types.float32,
    types.float16,
    types.int32,
    types.uint32,
    types.int64,
    types.boolean,
)


def _resolve_array_dtype(dtype_arg, fn_name: str) -> types.Type:
    # A dtype argument like `np.float32` types as a NumberClass wrapping
    # the concrete scalar type in `.instance_type` (verified empirically
    # against this Numba version -- see docs/architecture.md).
    instance_type = getattr(dtype_arg, "instance_type", None)
    if instance_type is None or instance_type not in _ARRAY_ELEMENT_DTYPES:
        raise TypingError(
            f"{fn_name}(shape, dtype): dtype must be one of "
            f"{[str(t) for t in _ARRAY_ELEMENT_DTYPES]} (e.g. np.float32), "
            f"got {dtype_arg!r}."
        )
    return instance_type


def _resolve_array_shape(shape_arg, fn_name: str) -> int:
    if not isinstance(shape_arg, types.IntegerLiteral):
        raise RequireLiteralValue(shape_arg)
    count = shape_arg.literal_value
    if count <= 0:
        raise NumbaValueError(
            f"{fn_name}(shape, dtype): shape must be a positive integer "
            f"literal; got {count}."
        )
    return count


def _unimplemented_codegen(_context, _builder, _sig, _args):  # pragma: no cover
    raise NotImplementedError(
        "numba-metal kernel intrinsics must be lowered by numba-metal's MSL "
        "backend, not Numba's LLVM lowering. If you see this error, the "
        "kernel was compiled through the wrong pipeline."
    )


def _dim_signature(ndim_literal, fn_name: str) -> types.Type:
    if not isinstance(ndim_literal, types.IntegerLiteral):
        raise RequireLiteralValue(ndim_literal)
    val = ndim_literal.literal_value
    if val == 1:
        return types.int64
    if val == 2:
        return types.UniTuple(types.int64, 2)
    if val == 3:
        return types.UniTuple(types.int64, 3)
    raise NumbaValueError(f"{fn_name}(ndim) supports ndim in (1, 2, 3); got {val}.")


@intrinsic(prefer_literal=True)
def grid(typingctx, ndim):
    """metal.grid(ndim): absolute thread position in the dispatch grid.

    ndim=1 returns a single int64. ndim=2/3 return a UniTuple(int64, ndim)
    of (x, y[, z]). Must be called with a literal (compile-time constant)
    ndim.
    """
    restype = _dim_signature(ndim, "metal.grid")
    sig = signature(restype, types.int32)
    return sig, _unimplemented_codegen


@intrinsic(prefer_literal=True)
def gridsize(typingctx, ndim):
    """metal.gridsize(ndim): total number of threads dispatched along each
    requested dimension (equivalent to the `blocks * threads` the kernel was
    launched with). Same ndim rules as grid()."""
    restype = _dim_signature(ndim, "metal.gridsize")
    sig = signature(restype, types.int32)
    return sig, _unimplemented_codegen


@intrinsic(prefer_literal=True)
def threadgroup_position(typingctx, ndim):
    """metal.threadgroup_position(ndim): the index of the current thread's
    threadgroup within the dispatch grid (MSL's
    `threadgroup_position_in_grid`). Same ndim rules as grid()."""
    restype = _dim_signature(ndim, "metal.threadgroup_position")
    sig = signature(restype, types.int32)
    return sig, _unimplemented_codegen


@intrinsic(prefer_literal=True)
def thread_in_threadgroup(typingctx, ndim):
    """metal.thread_in_threadgroup(ndim): the current thread's position
    within its own threadgroup (MSL's `thread_position_in_threadgroup`).
    Same ndim rules as grid()."""
    restype = _dim_signature(ndim, "metal.thread_in_threadgroup")
    sig = signature(restype, types.int32)
    return sig, _unimplemented_codegen


@intrinsic(prefer_literal=True)
def threads_per_threadgroup(typingctx, ndim):
    """metal.threads_per_threadgroup(ndim): the number of threads in one
    threadgroup along each requested dimension (MSL's
    `threads_per_threadgroup`). Same ndim rules as grid()."""
    restype = _dim_signature(ndim, "metal.threads_per_threadgroup")
    sig = signature(restype, types.int32)
    return sig, _unimplemented_codegen


@intrinsic(prefer_literal=True)
def local_array(typingctx, shape, dtype):
    """metal.local_array(shape, dtype): a fixed-size array private to the
    calling thread (MSL: an ordinary function-local array, one instance
    per GPU thread -- not shared with any other thread). `shape` must be
    a positive compile-time-constant integer; `dtype` must be a NumPy
    scalar type object (e.g. `np.float32`) naming one of numba-metal's
    supported array-element dtypes. Indexing the result with `[]` reads
    and writes exactly like an ordinary 1D array kernel argument.
    """
    _resolve_array_shape(
        shape, "metal.local_array"
    )  # validated here; re-read in msl_backend.py
    elem_ty = _resolve_array_dtype(dtype, "metal.local_array")
    restype = types.Array(elem_ty, 1, "C")
    sig = signature(restype, shape, dtype)
    return sig, _unimplemented_codegen


@intrinsic(prefer_literal=True)
def shared_array(typingctx, shape, dtype):
    """metal.shared_array(shape, dtype): a fixed-size array in Metal's
    `threadgroup` address space, shared by every thread in the same
    threadgroup (MSL: a `threadgroup`-qualified array declared once per
    kernel, at kernel-function-parameter scope -- MSL does not allow
    `threadgroup`-qualified locals inside a function body). Every thread
    in a threadgroup sees the SAME underlying storage: writes by one
    thread are visible to others in the same threadgroup only after a
    `metal.barrier()` call establishes the necessary happens-before
    ordering (see `metal.barrier`'s docstring) -- reading data another
    thread wrote without an intervening barrier is a data race, exactly
    as in MSL/CUDA/OpenCL. Same `shape`/`dtype` rules as
    `metal.local_array`. A kernel may declare more than one
    `shared_array`; each gets its own independent threadgroup allocation.
    """
    _resolve_array_shape(
        shape, "metal.shared_array"
    )  # validated here; re-read in msl_backend.py
    elem_ty = _resolve_array_dtype(dtype, "metal.shared_array")
    restype = types.Array(elem_ty, 1, "C")
    sig = signature(restype, shape, dtype)
    return sig, _unimplemented_codegen


@intrinsic
def barrier(typingctx):
    """metal.barrier(): synchronize every thread in the current
    threadgroup (MSL's `threadgroup_barrier(mem_flags::mem_threadgroup)`).

    Establishes both execution ordering (every thread in the threadgroup
    reaches the barrier before any thread proceeds past it) and memory
    ordering (writes to `metal.shared_array()` memory made before the
    barrier by any thread in the threadgroup are visible to every thread
    after the barrier). Required before reading threadgroup memory
    another thread wrote, and before reusing/overwriting threadgroup
    memory another thread may still be reading.

    Like MSL's own `threadgroup_barrier`, this must be reached by every
    thread in the threadgroup uniformly -- calling it from inside a
    branch that not every thread in the threadgroup takes is undefined
    behavior in MSL itself; numba-metal does not attempt to detect this
    at compile time (the same limitation applies to CUDA's
    `cuda.syncthreads()` and MSL's own compiler), so kernels must ensure
    every thread reaches every `metal.barrier()` call.
    """
    restype = types.void
    sig = signature(restype)
    return sig, _unimplemented_codegen
