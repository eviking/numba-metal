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


def test_two_dimensional_array_argument_rejected() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i, i]

    with pytest.raises(UnsupportedFeatureError, match="1D|dimension"):
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
