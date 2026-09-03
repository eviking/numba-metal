"""Kernel-side intrinsics: metal.grid(), metal.gridsize().

These are typed using `numba.extending.intrinsic` purely so Numba's type
inference accepts calls to them inside a kernel and produces a normal typed
`ir.Expr.call` node with a known return type. Their `codegen` callbacks are
never invoked: numba-metal never runs Numba's LLVM lowering. Instead,
`numba_metal.compiler.msl_backend` pattern-matches calls to these exact
function objects while walking the typed IR and emits the corresponding MSL
thread-position expression directly. This split (real typing, intercepted
lowering) was verified empirically against this Numba version before
committing to it -- see docs/architecture.md.
"""

from __future__ import annotations

from numba.core import types
from numba.core.errors import NumbaValueError, RequireLiteralValue
from numba.core.typing import signature
from numba.extending import intrinsic

MAX_GRID_DIMS = 2


def _unimplemented_codegen(_context, _builder, _sig, _args):  # pragma: no cover
    raise NotImplementedError(
        "metal.grid()/metal.gridsize() must be lowered by numba-metal's MSL "
        "backend, not Numba's LLVM lowering. If you see this error, the "
        "kernel was compiled through the wrong pipeline."
    )


def _grid_signature(ndim_literal) -> types.Type:
    if not isinstance(ndim_literal, types.IntegerLiteral):
        raise RequireLiteralValue(ndim_literal)
    val = ndim_literal.literal_value
    if val == 1:
        return types.int64
    if val == 2:
        return types.UniTuple(types.int64, 2)
    raise NumbaValueError(
        f"metal.grid(ndim) supports ndim in (1, 2); got {val}. 3D grids are "
        "not implemented in this MVP -- see docs/limitations.md."
    )


@intrinsic(prefer_literal=True)
def grid(typingctx, ndim):
    """metal.grid(ndim): absolute thread position in the dispatch grid.

    ndim=1 returns a single int64. ndim=2 returns a UniTuple(int64, 2) of
    (x, y). Must be called with a literal (compile-time constant) ndim.
    """
    restype = _grid_signature(ndim)
    sig = signature(restype, types.int32)
    return sig, _unimplemented_codegen


@intrinsic(prefer_literal=True)
def gridsize(typingctx, ndim):
    """metal.gridsize(ndim): total number of threads dispatched along each
    requested dimension (equivalent to the `blocks * threads` the kernel was
    launched with). Same ndim rules as grid()."""
    restype = _grid_signature(ndim)
    sig = signature(restype, types.int32)
    return sig, _unimplemented_codegen
