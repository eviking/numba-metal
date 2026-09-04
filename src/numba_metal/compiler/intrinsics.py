"""Kernel-side intrinsics: thread/threadgroup indexing, local/threadgroup
memory allocation, barriers, and atomics.

These are typed using `numba.extending.intrinsic` purely so Numba's type
inference accepts calls to them inside a kernel and produces a normal typed
`ir.Expr.call` node with a known return type. For a call inside an ordinary
`@metal.jit` kernel body, their `codegen` callbacks are never invoked:
numba-metal never runs Numba's LLVM lowering for a kernel.
`numba_metal.compiler.msl_backend` pattern-matches calls to these exact
function objects while walking the typed IR and emits the corresponding MSL
expression/statement directly. This split (real typing, intercepted
lowering) was verified empirically against this Numba version before
committing to it -- see docs/architecture.md.

The atomic intrinsics below are the one exception with a REAL `codegen` (not
`_unimplemented_codegen`): `@metal.device_func` (runtime/dispatcher.py) wraps
its target with a genuine `njit` dispatcher so Numba's own frontend can type
a *caller's* call site the same way it types any ordinary function call --
but typing a call to a Numba Dispatcher requires that dispatcher to actually
produce a compiled overload (`Dispatcher.compile()`), which unavoidably runs
real CPU lowering of the device function's entire body, not just its typing.
A device function that calls an atomic intrinsic therefore forces that
intrinsic's `codegen` to run for real during this throwaway CPU compile
(never actually executed -- only the resulting overload's *type signature*
is consulted by the caller's frontend; the MSL backend always re-derives the
real GPU semantics from the original plain function's typed IR via
`msl_backend._compile_device_function`, never from this CPU-lowered
version). A correct single-threaded sequential CPU implementation is used
for exactly this reason: it is semantically valid standalone Python/Numba
(no real concurrency exists in this throwaway compile), safe to keep even if
future code called it directly, and avoids special-casing device-function
compilation around Numba Dispatcher internals just to suppress lowering.
`metal.grid`/`gridsize`/thread-position/`local_array`/`shared_array`/
`barrier` do not have this problem in practice because a device function
calling any of those has no meaningful CPU (or, for that matter, GPU
outside a real dispatch) semantics to fall back to and remains out of
scope -- calling one from inside a `@metal.device_func` continues to raise
via `_unimplemented_codegen` exactly as before.
"""

from __future__ import annotations

from numba.core import types
from numba.core.errors import NumbaValueError, RequireLiteralValue, TypingError
from numba.core.typing import signature
from numba.extending import intrinsic

MAX_GRID_DIMS = 3

#: Element dtypes accepted by every atomic intrinsic below. MSL's own
#: `atomic<T>` machinery natively supports only 32-bit int/uint/float
#: (no 16/64-bit atomics on any Apple GPU family); this is a real,
#: permanent MSL/hardware constraint, not an arbitrary numba-metal
#: restriction -- see docs/architecture.md's atomics section for the
#: exact capability probes this was verified against.
_ATOMIC_DTYPES = (types.int32, types.uint32, types.float32)

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


# -- atomics --------------------------------------------------------------


def _resolve_atomic_array(arr_ty, fn_name: str) -> types.Type:
    if not isinstance(arr_ty, types.Array):
        raise TypingError(
            f"{fn_name}(array, index, value, ...): first argument must be "
            f"a device array; got {arr_ty!r}."
        )
    if arr_ty.ndim != 1:
        raise TypingError(
            f"{fn_name}(array, index, value, ...): only 1D arrays are "
            f"supported; got {arr_ty.ndim}D."
        )
    if arr_ty.dtype not in _ATOMIC_DTYPES:
        raise TypingError(
            f"{fn_name}(array, index, value, ...): array dtype must be one "
            f"of {[str(t) for t in _ATOMIC_DTYPES]} (MSL's atomic<T> "
            f"machinery only supports 32-bit int/uint/float on any Apple "
            f"GPU family); got {arr_ty.dtype!r}."
        )
    return arr_ty.dtype


def _atomic_add_pyfunc(arr, idx, val):
    old = arr[idx]
    arr[idx] = old + val
    return old


def _atomic_sub_pyfunc(arr, idx, val):
    old = arr[idx]
    arr[idx] = old - val
    return old


def _atomic_min_pyfunc(arr, idx, val):
    old = arr[idx]
    arr[idx] = min(old, val)
    return old


def _atomic_max_pyfunc(arr, idx, val):
    old = arr[idx]
    arr[idx] = max(old, val)
    return old


def _atomic_exchange_pyfunc(arr, idx, val):
    old = arr[idx]
    arr[idx] = val
    return old


#: Sequential (non-atomic) Python equivalent for each fetch-and-modify
#: intrinsic, compiled via `context.compile_internal` -- see this module's
#: docstring for why a real, single-threaded CPU implementation (rather
#: than `_unimplemented_codegen`) is needed and safe here.
_FETCH_OP_PYFUNC = {
    "atomic_add": _atomic_add_pyfunc,
    "atomic_sub": _atomic_sub_pyfunc,
    "atomic_min": _atomic_min_pyfunc,
    "atomic_max": _atomic_max_pyfunc,
    "atomic_exchange": _atomic_exchange_pyfunc,
}


def _make_fetch_op_intrinsic(name: str, doc: str):
    @intrinsic
    def _fn(typingctx, arr, idx, val):
        elem_ty = _resolve_atomic_array(arr, f"metal.{name}")
        sig = signature(elem_ty, arr, idx, val)
        pyfunc = _FETCH_OP_PYFUNC[name]

        def codegen(context, builder, sig, args):
            return context.compile_internal(builder, pyfunc, sig, args)

        return sig, codegen

    _fn.__name__ = name
    _fn.__doc__ = doc
    return _fn


atomic_add = _make_fetch_op_intrinsic(
    "atomic_add",
    """metal.atomic_add(array, index, value): atomically add `value` to
    `array[index]` and return the value that was there immediately
    before the add (MSL's `atomic_fetch_add_explicit`, relaxed memory
    order). Supported dtypes: int32, uint32, float32 (float32 is
    natively supported on this device -- see docs/architecture.md for
    per-device capability probing; a device lacking native float atomics
    would need a documented CAS-loop fallback, not silent rejection or
    silent incorrectness -- not yet implemented for that case, see
    docs/roadmap.md).""",
)

atomic_sub = _make_fetch_op_intrinsic(
    "atomic_sub",
    """metal.atomic_sub(array, index, value): atomically subtract `value`
    from `array[index]` and return the value that was there immediately
    before the subtraction (MSL's `atomic_fetch_sub_explicit`). Same
    dtype support as `metal.atomic_add`.""",
)

atomic_min = _make_fetch_op_intrinsic(
    "atomic_min",
    """metal.atomic_min(array, index, value): atomically set
    `array[index]` to `min(array[index], value)` and return the value
    that was there immediately before. int32/uint32 use MSL's native
    `atomic_fetch_min_explicit`; float32 has NO native MSL atomic
    min/max of any kind on any Apple GPU family (verified directly: MSL
    rejects `atomic_fetch_min_explicit`/`atomic_fetch_max_explicit` for
    `atomic_float*` with "no matching function," a permanent language
    limitation, not a device-capability gap) -- float32 is instead
    lowered to a compare-and-swap retry loop, which is still genuinely
    race-free (every thread's CAS attempt either succeeds or retries
    against the latest value) but does more work under heavy contention
    than a native atomic instruction. See docs/architecture.md.""",
)

atomic_max = _make_fetch_op_intrinsic(
    "atomic_max",
    """metal.atomic_max(array, index, value): atomically set
    `array[index]` to `max(array[index], value)` and return the value
    that was there immediately before. Same dtype/implementation notes
    as `metal.atomic_min` (float32 via CAS-loop, int32/uint32 native).""",
)

atomic_exchange = _make_fetch_op_intrinsic(
    "atomic_exchange",
    """metal.atomic_exchange(array, index, value): atomically set
    `array[index]` to `value` and return the value that was there
    immediately before (MSL's `atomic_exchange_explicit`). Supported
    dtypes: int32, uint32, float32 (native on all three for this
    operation).""",
)


@intrinsic
def atomic_compare_exchange(typingctx, arr, idx, expected, desired):
    """metal.atomic_compare_exchange(array, index, expected, desired):
    atomically compare `array[index]` against `expected`; if equal, set
    it to `desired` (MSL's `atomic_compare_exchange_weak_explicit`,
    relaxed memory order on both success and failure). Returns a
    `(old_value, success)` tuple: `old_value` is `array[index]`'s value
    immediately before this call (whether or not the swap happened), and
    `success` is `True` iff the swap happened. Supported dtypes: int32,
    uint32, float32.

    Uses the *weak* compare-exchange (may spuriously fail even when
    `array[index] == expected`, per MSL/C++ semantics) -- callers
    building a retry loop (the overwhelmingly common use case, e.g.
    numba-metal's own float32 `atomic_min`/`atomic_max` CAS-loop lowering
    above) already retry on failure regardless of the failure's cause,
    so the weak form's better performance on Apple GPU hardware is pure
    upside with no correctness cost for that pattern. A caller needing
    exactly one comparison (no retry) should be aware of this."""
    elem_ty = _resolve_atomic_array(arr, "metal.atomic_compare_exchange")
    restype = types.Tuple([elem_ty, types.boolean])
    sig = signature(restype, arr, idx, expected, desired)

    def codegen(context, builder, sig, args):
        return context.compile_internal(
            builder, _atomic_compare_exchange_pyfunc, sig, args
        )

    return sig, codegen


def _atomic_compare_exchange_pyfunc(arr, idx, expected, desired):
    old = arr[idx]
    ok = old == expected
    if ok:
        arr[idx] = desired
    return old, ok
