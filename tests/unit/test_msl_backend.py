"""Tests for the typed-IR -> MSL structured code generator. These only
generate MSL text; they do not compile or run it on a GPU, so they run on
any machine with Numba installed (no Metal device required).
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from numba.core import types

from numba_metal.compiler import intrinsics as metal
from numba_metal.compiler.frontend import compile_to_typed_ir
from numba_metal.compiler.msl_backend import MSLKernelLowerer
from numba_metal.errors import UnsupportedFeatureError


def _lower(func, sig) -> str:
    typed = compile_to_typed_ir(func, sig)
    return MSLKernelLowerer("test_kernel", typed).lower()


def test_vector_add_generates_kernel_void_and_buffer_bindings() -> None:
    def vector_add(a, b, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] + b[i]

    sig = (types.float32[::1], types.float32[::1], types.float32[::1])
    src = _lower(vector_add, sig)
    assert "kernel void test_kernel(" in src
    assert "device float* arg_a [[buffer(0)]]" in src
    assert "thread_position_in_grid" in src


def test_if_else_emits_structured_braces() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            if a[i] > 0.0:
                out[i] = 1.0
            else:
                out[i] = -1.0

    sig = (types.float32[::1], types.float32[::1])
    src = _lower(f, sig)
    assert src.count("if (") >= 2
    assert "else {" in src
    assert "goto" not in src


def test_for_range_loop_emits_native_for() -> None:
    def f(a, out, n):
        i = metal.grid(1)
        if i < out.size:
            s = 0.0
            for j in range(n):
                s = s + a[j]
            out[i] = s

    sig = (types.float32[::1], types.float32[::1], types.int32)
    src = _lower(f, sig)
    assert "for (" in src
    assert "while (true)" not in src


def test_break_and_continue_supported_in_loop() -> None:
    def f(a, out, n):
        i = metal.grid(1)
        if i < out.size:
            count = 0
            for j in range(n):
                if a[j] < 0.0:
                    continue
                if a[j] > 100.0:
                    break
                count = count + 1
            out[i] = count

    sig = (types.float32[::1], types.int32[::1], types.int32)
    src = _lower(f, sig)
    assert "break;" in src
    assert "continue;" in src


def test_math_functions_map_to_msl_names() -> None:
    import math

    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = math.sqrt(a[i]) + math.exp(a[i]) + math.log(a[i])
            out[i] = out[i] + math.sin(a[i]) + math.cos(a[i])

    sig = (types.float32[::1], types.float32[::1])
    src = _lower(f, sig)
    for name in ("sqrt(", "exp(", "log(", "sin(", "cos("):
        assert name in src


def test_abs_min_max_supported() -> None:
    def f(a, b, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = min(max(abs(a[i]), 0.0), b[i])

    sig = (types.float32[::1], types.float32[::1], types.float32[::1])
    src = _lower(f, sig)
    assert "abs(" in src
    assert "min(" in src
    assert "max(" in src


def test_grid_2d_generates_long2_and_xy_unpack() -> None:
    def f(out, width, height):
        x, y = metal.grid(2)
        if x < width and y < height:
            out[y * width + x] = x + y

    sig = (types.int64[::1], types.int32, types.int32)
    src = _lower(f, sig)
    assert "long2(" in src
    assert "numba_metal_tid.y" in src


def test_unsupported_string_literal_raises_with_variable_name() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            s = "not supported"
            out[i] = a[i]

    sig = (types.float32[::1], types.float32[::1])
    with pytest.raises(UnsupportedFeatureError, match="unicode|string|no MSL"):
        _lower(f, sig)


def test_unsupported_float64_array_arg_raises() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i]

    sig = (types.float64[::1], types.float64[::1])
    with pytest.raises(UnsupportedFeatureError, match="float64"):
        _lower(f, sig)


def test_unsupported_2d_array_arg_raises() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i, 0]

    sig = (types.float32[:, ::1], types.float32[::1])
    with pytest.raises(UnsupportedFeatureError, match="1D|dimension"):
        _lower(f, sig)


def test_generated_kernel_name_is_deterministic_per_call() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i]

    sig = (types.float32[::1], types.float32[::1])
    src1 = _lower(f, sig)
    src2 = _lower(f, sig)
    assert "kernel void test_kernel(" in src1
    assert "kernel void test_kernel(" in src2


# A module-level global declared with an explicit numpy scalar type
# (e.g. `SIGMA = np.float32(5.67e-8)`) is the idiomatic way to pin a
# physical constant's dtype for a kernel -- and numpy scalar types are
# NOT subclasses of Python's own bool/int/float, so an isinstance check
# written against only the builtins silently misses them. Found via a
# real kernel that produced NaN: the constant's SSA name was never
# declared or assigned (by design -- non-callable globals are meant to
# be resolved to literal text at every use site instead), and the
# isinstance check gating that resolution didn't recognize a
# numpy.float32 value, so the reference fell through to an identifier
# for a variable that was never declared -- reading whatever garbage
# happened to be on the MSL kernel's stack at that point.
_NUMPY_FLOAT_GLOBAL = np.float32(5.67e-8)
_NUMPY_INT_GLOBAL = np.int32(42)
_NUMPY_BOOL_GLOBAL = np.bool_(True)


def test_numpy_float32_global_constant_is_emitted_as_a_literal() -> None:
    def f(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = _NUMPY_FLOAT_GLOBAL * a[i]

    sig = (types.float32[::1], types.float32[::1])
    src = _lower(f, sig)
    # The multiplication must use the resolved literal value (not
    # exactly "5.67e-08f" -- float32->Python-float->repr() introduces
    # harmless extra digits, e.g. "5.669999936230852e-08f" -- so check
    # numeric value instead of exact text), and the final assignment to
    # `arg_out` must not read the bare, never-assigned temporary the
    # global load declared (the actual shape of the bug: `arg_out[v_i]
    # = v__58load_global_0;`, referencing an SSA name with no
    # initializer anywhere in the function).
    match = re.search(r"([0-9.eE+-]+)f\s*\*", src)
    assert match is not None, f"no float literal multiplication found in:\n{src}"
    assert abs(float(match.group(1)) - 5.67e-8) < 1e-15
    assign_line = next(line for line in src.splitlines() if "arg_out[v_i] =" in line)
    assert "load_global" not in assign_line


def test_numpy_int32_global_constant_is_emitted_as_a_literal() -> None:
    def f(a, out, n):
        i = metal.grid(1)
        if i < n + _NUMPY_INT_GLOBAL:
            out[i] = a[i]

    sig = (types.float32[::1], types.float32[::1], types.int64)
    src = _lower(f, sig)
    assert "42" in src


def test_numpy_bool_global_constant_is_emitted_as_true_or_false() -> None:
    def f(a, out):
        i = metal.grid(1)
        if _NUMPY_BOOL_GLOBAL and i < out.size:
            out[i] = a[i]

    sig = (types.float32[::1], types.float32[::1])
    src = _lower(f, sig)
    assert "true" in src
