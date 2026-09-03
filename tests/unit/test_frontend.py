"""Tests for the Numba typed-IR frontend adapter. These do not touch the
Metal device or compiler and can run on any machine with Numba installed.
"""

from __future__ import annotations

from numba.core import types

from numba_metal.compiler.frontend import TypedKernelIR, compile_to_typed_ir
from numba_metal.errors import KernelCompilationError


def test_compiles_simple_function_to_typed_ir() -> None:
    def f(a, out):
        i = 0
        out[i] = a[i] + 1.0

    sig = (types.float32[::1], types.float32[::1])
    result = compile_to_typed_ir(f, sig)
    assert isinstance(result, TypedKernelIR)
    assert result.arg_names == ("a", "out")
    assert result.typemap["a"] == types.float32[::1]


def test_typing_failure_raises_kernel_compilation_error() -> None:
    def f(a, out):
        out[0] = a[0] + "not a number"

    sig = (types.float32[::1], types.float32[::1])
    try:
        compile_to_typed_ir(f, sig)
    except KernelCompilationError:
        pass
    else:
        raise AssertionError("expected KernelCompilationError")


def test_typemap_reflects_arithmetic_result_type() -> None:
    def f(a, b, out):
        out[0] = a[0] * b[0]

    sig = (types.int32[::1], types.int32[::1], types.int32[::1])
    result = compile_to_typed_ir(f, sig)
    assert result.return_type == types.void
