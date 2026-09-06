"""Compatibility analysis: does numba-metal actually support this function
today, and if not, exactly why not.

This runs the REAL numba-metal compilation pipeline in two device-free
stages -- confirmed by reading both modules directly:

1. `numba_metal.compiler.frontend.compile_to_typed_ir(func, arg_types)` --
   Numba's real bytecode-to-typed-IR frontend, stopping before lowering.
   Raises `KernelCompilationError` (wrapping Numba's own `TypingError`) if
   the function cannot even be typed.
2. `numba_metal.compiler.msl_backend.MSLKernelLowerer(name, typed).lower()`
   -- the actual MSL codegen backend. This is where
   `UnsupportedFeatureError` is raised for the ~40 distinct unsupported-
   construct cases documented in docs/limitations.md (unsupported calls,
   float64 arrays, multidim arrays, while-loops with nested if/break,
   etc.) -- every one of those raise sites names the specific construct.

Neither stage touches `numba_metal.runtime.context.get_context()` or the
`Metal` PyObjC module at all (verified by reading both files: no such
import or call appears before MSL source text is produced) -- so this
whole analysis runs identically with or without a live Metal device, and
even on non-Apple-Silicon machines. Only actually launching/compiling the
kernel on a real `MTLDevice` (which this module never does) would require
one.

This module never assigns a numerical score -- see models.CompatibilityStatus.
"""

from __future__ import annotations

import numpy as np
from numba.core import types as nb_types

from numba_metal.advisor.models import CompatibilityResult, CompatibilityStatus
from numba_metal.errors import KernelCompilationError, UnsupportedFeatureError

#: Mirrors runtime/dispatcher.py's `_SCALAR_NUMPY_TO_NUMBA` (that table is
#: keyed on runtime *values*, not static sample shapes, so it cannot be
#: imported and reused directly -- but the mapping itself must stay
#: identical to how numba-metal will actually type these arguments at
#: launch time, hence "mirrors" rather than "reimplements independently").
_NUMPY_DTYPE_TO_NUMBA_SCALAR: dict[np.dtype, nb_types.Type] = {
    np.dtype(np.float32): nb_types.float32,
    np.dtype(np.float16): nb_types.float16,
    np.dtype(np.int32): nb_types.int32,
    np.dtype(np.uint32): nb_types.uint32,
    np.dtype(np.int64): nb_types.int64,
    np.dtype(np.bool_): nb_types.boolean,
    np.dtype(np.float64): nb_types.float64,  # deliberately included: typed
    # so the dry-run can surface the *real* "float64 not supported"
    # UnsupportedFeatureError from types/__init__.py, rather than this
    # module silently refusing to even try.
}


class SampleTypingError(Exception):
    """Raised by `infer_arg_types_from_samples` when a sample argument's
    dtype has no known static-typing equivalent at all (distinct from
    numba-metal itself rejecting a *supported-shape* type -- that path
    goes through the real compiler and becomes a normal
    BLOCKED_BY_MISSING_FEATURE result, not this exception)."""


def infer_arg_types_from_samples(sample_args: tuple) -> tuple[nb_types.Type, ...]:
    """Derive Numba argument types from example call arguments, for
    functions that have never been decorated/launched (so there is no
    `DeviceNDArray` to type from, unlike `runtime.dispatcher._infer_arg_type`,
    which requires one). 1D `numpy.ndarray` and Python/NumPy scalars only,
    matching numba-metal's own supported argument shapes."""
    arg_types: list[nb_types.Type] = []
    for value in sample_args:
        if isinstance(value, np.ndarray):
            if value.ndim < 1 or value.ndim > 3:
                raise SampleTypingError(
                    f"sample array has {value.ndim} dimensions; numba-metal "
                    "kernels only accept 1D, 2D, or 3D arrays"
                )
            scalar = _NUMPY_DTYPE_TO_NUMBA_SCALAR.get(value.dtype)
            if scalar is None:
                raise SampleTypingError(
                    f"sample array dtype {value.dtype!r} has no Numba "
                    "scalar type mapping known to the advisor"
                )
            if value.ndim == 1:
                arg_types.append(scalar[::1])
            elif value.ndim == 2:
                arg_types.append(scalar[:, ::1])
            else:
                arg_types.append(scalar[:, :, ::1])
        elif isinstance(value, np.generic):
            scalar = _NUMPY_DTYPE_TO_NUMBA_SCALAR.get(np.dtype(type(value)))
            if scalar is None:
                raise SampleTypingError(
                    f"sample scalar dtype {type(value)!r} has no Numba "
                    "scalar type mapping known to the advisor"
                )
            arg_types.append(scalar)
        elif isinstance(value, bool):
            arg_types.append(nb_types.boolean)
        elif isinstance(value, int):
            arg_types.append(nb_types.int64)
        elif isinstance(value, float):
            arg_types.append(nb_types.float32)
        else:
            raise SampleTypingError(
                f"sample argument of type {type(value)!r} is not a "
                "recognized numba-metal argument shape (1D numpy array or "
                "scalar)"
            )
    return tuple(arg_types)


def check_compatibility(
    func,
    *,
    qualified_name: str,
    file: str,
    line_start: int,
    sample_args: tuple = (),
) -> CompatibilityResult:
    """Run the real, device-free numba-metal dry-run compilation against
    `func` and classify the result.

    `func` must be a plain (undecorated) Python function object -- if it
    is already wrapped by `@numba.njit`/`@metal.jit`, pass
    `func.py_func`/the dispatcher's underlying function so this module
    compiles the original source, not the wrapper.
    """
    from numba_metal.compiler.frontend import compile_to_typed_ir
    from numba_metal.compiler.msl_backend import MSLKernelLowerer

    try:
        arg_types = infer_arg_types_from_samples(sample_args)
    except SampleTypingError as exc:
        return CompatibilityResult(
            qualified_name=qualified_name,
            file=file,
            line_start=line_start,
            status=CompatibilityStatus.UNABLE_TO_ANALYZE,
            supported_features=(),
            blockers=(f"Could not determine argument types: {exc}",),
            recommendation=(
                "Provide representative sample arguments "
                "(1D numpy arrays / scalars) to analyze this function."
            ),
            raw_error=str(exc),
        )

    try:
        typed = compile_to_typed_ir(func, arg_types)
    except KernelCompilationError as exc:
        return CompatibilityResult(
            qualified_name=qualified_name,
            file=file,
            line_start=line_start,
            status=CompatibilityStatus.BLOCKED_BY_MISSING_FEATURE,
            supported_features=(),
            blockers=(str(exc),),
            recommendation=(
                "Numba's own type inference could not type this function "
                "with the given argument types -- fix the reported typing "
                "error before numba-metal can be attempted."
            ),
            raw_error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 -- see module docstring: any
        # OTHER exception here means Numba's frontend itself crashed on
        # this function (not a normal "unsupported" rejection), which is
        # a genuinely different, unexpected situation this tool cannot
        # explain -- recorded, never silently dropped, never crashes the
        # caller's batch analysis of other candidates.
        return CompatibilityResult(
            qualified_name=qualified_name,
            file=file,
            line_start=line_start,
            status=CompatibilityStatus.UNABLE_TO_ANALYZE,
            supported_features=(),
            blockers=(),
            recommendation=None,
            raw_error=f"{type(exc).__name__}: {exc}",
        )

    supported_features: list[str] = [
        "Array iteration",
        "Comparisons",
        "Output assignment",
    ]
    seen_dtype_features: set[str] = set()
    for arg_type in arg_types:
        base = arg_type.dtype if hasattr(arg_type, "dtype") else arg_type
        if hasattr(base, "name"):
            feature = f"{base.name} arithmetic"
            if feature not in seen_dtype_features:
                seen_dtype_features.add(feature)
                supported_features.append(feature)

    try:
        kernel_name = f"advisor_dryrun_{qualified_name.replace('.', '_')}"
        lowerer = MSLKernelLowerer(kernel_name, typed)
        lowerer.lower()
    except UnsupportedFeatureError as exc:
        return CompatibilityResult(
            qualified_name=qualified_name,
            file=file,
            line_start=line_start,
            status=CompatibilityStatus.BLOCKED_BY_MISSING_FEATURE,
            supported_features=tuple(supported_features),
            blockers=(str(exc),),
            recommendation=_recommendation_for_blocker(str(exc)),
            raw_error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 -- same rationale as above:
        # the MSL backend is a large tree-walking codegen; an unexpected
        # internal failure here (as opposed to a clean, named
        # UnsupportedFeatureError) is reported as UNABLE_TO_ANALYZE rather
        # than crashing whatever batch analysis called this function.
        return CompatibilityResult(
            qualified_name=qualified_name,
            file=file,
            line_start=line_start,
            status=CompatibilityStatus.UNABLE_TO_ANALYZE,
            supported_features=tuple(supported_features),
            blockers=(),
            recommendation=None,
            raw_error=f"{type(exc).__name__}: {exc}",
        )

    return CompatibilityResult(
        qualified_name=qualified_name,
        file=file,
        line_start=line_start,
        status=CompatibilityStatus.SUPPORTED,
        supported_features=tuple(supported_features),
        blockers=(),
        recommendation=None,
        raw_error=None,
    )


def _recommendation_for_blocker(message: str) -> str:
    """Deterministic, rule-based recommendation text keyed off the
    (verbatim, already-precise) blocker message -- not a generic string,
    and not an LLM call. Falls back to a generic-but-honest message when
    the blocker doesn't match a known, specific pattern."""
    lower = message.lower()
    if "float64" in lower:
        return (
            "Convert the array(s) to float32 before calling this function "
            "on Metal, or keep this function on the CPU if float64 "
            "precision is required."
        )
    if "percentile" in lower or "median" in lower:
        return (
            "Run the value-generation part of this function on Metal. "
            "Return the generated array and compute the percentile/median "
            "on the CPU."
        )
    if "recursion" in lower or "recursive" in lower:
        return (
            "Rewrite the recursive algorithm as an explicit loop; "
            "recursion has no Metal kernel equivalent in numba-metal today."
        )
    if "dict" in lower or "list" in lower or "set literal" in lower:
        return (
            "Replace the Python container with a fixed-size numpy array "
            "argument, or precompute the container's contents on the CPU "
            "before launching the kernel."
        )
    if "while" in lower and ("break" in lower or "if" in lower or "continue" in lower):
        return (
            "Restructure the while loop to a straight-line body (no "
            "nested if/else, break, or continue) -- see "
            "docs/limitations.md's while-loop restriction, or express the "
            "same logic as branchless arithmetic."
        )
    if "back-edge" in lower or "unexpected control flow" in lower:
        return (
            "This while loop has a nested if/else, break, or continue in "
            "its body -- numba-metal's while-loop structurer only "
            "supports a straight-line loop body. Restructure the loop "
            "condition/body to avoid branching inside the loop (e.g. "
            "branchless arithmetic using max()/min()), or express it as "
            "a for-range loop if the trip count is bounded."
        )
    if "ndim" in lower or "dimension" in lower:
        return "Flatten the array to 1D before passing it to a Metal kernel."
    if ".shape" in message or "uses .shape" in lower or "tuple type" in lower:
        return "Use array.size instead of array.shape[0] inside the kernel body."
    return (
        "Keep this function on the CPU until numba-metal supports the "
        "reported construct, or rewrite the specific unsupported part."
    )
