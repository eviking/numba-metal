"""Tests asserting that unsupported Python constructs and types produce
specific, actionable compile-time errors -- never a silent fallback to
another execution path. These run without a GPU (they only exercise the
Numba frontend + MSL codegen, which raise before any Metal compilation is
attempted).
"""

from __future__ import annotations

import pytest
from numba.core import types

from numba_metal.compiler import intrinsics as metal
from numba_metal.compiler.frontend import compile_to_typed_ir
from numba_metal.compiler.msl_backend import MSLKernelLowerer
from numba_metal.errors import KernelCompilationError, UnsupportedFeatureError


def _lower(func, sig) -> str:
    typed = compile_to_typed_ir(func, sig)
    return MSLKernelLowerer("test_kernel", typed).lower()


def test_string_literal_rejected() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            s = "hello"
            out[i] = a[i]

    with pytest.raises(UnsupportedFeatureError):
        _lower(f, (types.float32[::1], types.float32[::1]))


def test_dict_literal_rejected() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            d = {}
            out[i] = a[i]

    with pytest.raises((UnsupportedFeatureError, KernelCompilationError)):
        _lower(f, (types.float32[::1], types.float32[::1]))


def test_list_literal_rejected() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            lst = [1, 2, 3]
            out[i] = a[i]

    with pytest.raises((UnsupportedFeatureError, KernelCompilationError)):
        _lower(f, (types.float32[::1], types.float32[::1]))


def test_float64_array_argument_rejected() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i]

    with pytest.raises(UnsupportedFeatureError, match="float64"):
        _lower(f, (types.float64[::1], types.float64[::1]))


def test_four_dimensional_array_argument_rejected() -> None:
    """2D and 3D array kernel arguments are supported (see
    tests/integration/test_multidim_arrays.py); 4D+ remains unsupported
    -- numba-metal's flattened-index codegen and metal.grid's own ndim
    range only go up to 3 (matching Metal's own MTLSize dispatch
    dimensionality limit)."""

    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i, i, i, i]

    with pytest.raises(UnsupportedFeatureError, match="1D, 2D, or 3D|dimension"):
        _lower(f, (types.float32[:, :, :, ::1], types.float32[::1]))


def test_negative_literal_index_rejected() -> None:
    """`a[-1]` (meaning "last element" in Python/NumPy) has no
    wraparound implementation in numba-metal's MSL codegen -- it used
    to compile silently and read/write whatever out-of-bounds offset
    the negative value produced in C-style pointer arithmetic. A
    literal negative index is statically detectable (unlike a
    runtime-variable one, e.g. `a[x - 1]`, which may or may not be
    negative depending on `x` and cannot be checked here), so it is
    now rejected at compile time instead of silently misbehaving --
    see docs/limitations.md."""

    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[-1]

    with pytest.raises(UnsupportedFeatureError, match="[Nn]egative"):
        _lower(f, (types.float32[::1], types.float32[::1]))


def test_negative_literal_index_rejected_setitem() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i == 0:
            out[-1] = a[0]

    with pytest.raises(UnsupportedFeatureError, match="[Nn]egative"):
        _lower(f, (types.float32[::1], types.float32[::1]))


def test_negative_literal_index_rejected_2d() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[-1, 0]

    with pytest.raises(UnsupportedFeatureError, match="[Nn]egative"):
        _lower(f, (types.float32[:, ::1], types.float32[::1]))


def test_unsupported_math_function_rejected() -> None:
    import math

    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = math.tan(a[i])

    with pytest.raises(UnsupportedFeatureError, match="tan"):
        _lower(f, (types.float32[::1], types.float32[::1]))


def test_print_statement_rejected() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            print(a[i])
            out[i] = a[i]

    with pytest.raises((UnsupportedFeatureError, KernelCompilationError)):
        _lower(f, (types.float32[::1], types.float32[::1]))


def test_shape_attribute_rejected_in_favor_of_size() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            s = a.shape
            out[i] = a[i]

    with pytest.raises((UnsupportedFeatureError, KernelCompilationError)):
        _lower(f, (types.float32[::1], types.float32[::1]))


def test_local_array_allocation_rejected() -> None:
    import numpy as np

    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            buf = np.zeros(4, dtype=np.float32)
            out[i] = a[i]

    with pytest.raises((UnsupportedFeatureError, KernelCompilationError)):
        _lower(f, (types.float32[::1], types.float32[::1]))


def test_grid_with_non_literal_ndim_rejected() -> None:
    def f(a, out, n):
        i = metal.grid(n)
        if i < out.size:
            out[i] = a[i]

    with pytest.raises((UnsupportedFeatureError, KernelCompilationError)):
        _lower(f, (types.float32[::1], types.float32[::1], types.int32))


def test_grid_with_unsupported_ndim_rejected() -> None:
    def f(a, out):
        w = metal.grid(4)
        out[0] = a[0]

    with pytest.raises((UnsupportedFeatureError, KernelCompilationError)):
        _lower(f, (types.float32[::1], types.float32[::1]))
