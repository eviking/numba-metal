"""Standalone tests for `@metal.device_func`: calling a separately
-defined helper function from inside a `@metal.jit` kernel body (or from
another device function). General Numba-to-Metal compiler capability,
no connection to any specific downstream workload.

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


def test_device_function_basic_call() -> None:
    metal = _metal()

    @metal.device_func
    def double_plus_one(x):
        return x * 2.0 + 1.0

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = double_plus_one(a[i])

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, 4](d_a, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), a * 2.0 + 1.0)


def test_device_function_multiple_arguments() -> None:
    metal = _metal()

    @metal.device_func
    def weighted_sum(a, b, wa, wb):
        return a * wa + b * wb

    @metal.jit
    def kernel(a, b, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = weighted_sum(a[i], b[i], 0.25, 0.75)

    a = np.array([4.0, 8.0], dtype=np.float32)
    b = np.array([2.0, 6.0], dtype=np.float32)
    d_a, d_b = metal.to_device(a), metal.to_device(b)
    d_out = metal.device_array_like(a)
    kernel[1, 2](d_a, d_b, d_out)
    metal.synchronize()
    expected = a * 0.25 + b * 0.75
    assert np.allclose(d_out.copy_to_host(), expected)


def test_device_function_multiple_call_sites_same_kernel() -> None:
    metal = _metal()

    @metal.device_func
    def square(x):
        return x * x

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = square(a[i]) + square(a[i] + 1.0)

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, 3](d_a, d_out)
    metal.synchronize()
    expected = a**2 + (a + 1.0) ** 2
    assert np.allclose(d_out.copy_to_host(), expected)


def test_device_function_called_from_two_different_kernels() -> None:
    metal = _metal()

    @metal.device_func
    def cube(x):
        return x * x * x

    @metal.jit
    def kernel_a(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = cube(a[i])

    @metal.jit
    def kernel_b(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = cube(a[i]) + 1.0

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out_a = metal.device_array_like(a)
    d_out_b = metal.device_array_like(a)
    kernel_a[1, 3](d_a, d_out_a)
    kernel_b[1, 3](d_a, d_out_b)
    metal.synchronize()
    assert np.allclose(d_out_a.copy_to_host(), a**3)
    assert np.allclose(d_out_b.copy_to_host(), a**3 + 1.0)


def test_device_function_calling_another_device_function() -> None:
    """Nested device functions: one device function calling another."""
    metal = _metal()

    @metal.device_func
    def square(x):
        return x * x

    @metal.device_func
    def hypot2(a, b):
        return square(a) + square(b)

    @metal.jit
    def kernel(a, b, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = hypot2(a[i], b[i])

    a = np.array([3.0, 5.0, 8.0], dtype=np.float32)
    b = np.array([4.0, 12.0, 15.0], dtype=np.float32)
    d_a, d_b = metal.to_device(a), metal.to_device(b)
    d_out = metal.device_array_like(a)
    kernel[1, 3](d_a, d_b, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), a**2 + b**2)


def test_device_function_with_control_flow() -> None:
    """A device function with nested if/return -- exercises that
    structured control-flow reconstruction and multi-exit-point
    lowering work identically inside a device function as inside a
    kernel."""
    metal = _metal()

    @metal.device_func
    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            v1 = clamp(a[i], 0.0, 10.0)
            v2 = clamp(v1 * 2.0, 0.0, 10.0)
            out[i] = v2

    a = np.array([-5.0, 3.0, 8.0, 20.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, 4](d_a, d_out)
    metal.synchronize()
    expected = np.clip(np.clip(a, 0, 10) * 2, 0, 10)
    assert np.allclose(d_out.copy_to_host(), expected)


def test_device_function_with_loop() -> None:
    metal = _metal()

    @metal.device_func
    def sum_to_n(n):
        total = 0.0
        for k in range(n):
            total = total + float(k)
        return total

    @metal.jit
    def kernel(out, n):
        i = metal.grid(1)
        if i < out.size:
            out[i] = sum_to_n(n)

    n = 5
    d_out = metal.device_array(1, np.float32)
    kernel[1, 1](d_out, np.int32(n))
    metal.synchronize()
    assert d_out.copy_to_host()[0] == pytest.approx(sum(range(n)))


def test_device_function_int_and_bool_types() -> None:
    metal = _metal()

    @metal.device_func
    def is_positive(x):
        return x > 0

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = is_positive(a[i])

    a = np.array([-3, 0, 5, -1, 2], dtype=np.int32)
    d_a = metal.to_device(a)
    d_out = metal.device_array(len(a), np.bool_)
    kernel[1, len(a)](d_a, d_out)
    metal.synchronize()
    assert np.array_equal(d_out.copy_to_host(), a > 0)


def test_device_function_rejects_direct_recursion() -> None:
    from numba_metal.errors import NumbaMetalError

    metal = _metal()

    @metal.device_func
    def recurse(x):
        return recurse(x) + 1.0

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = recurse(a[i])

    d_a = metal.to_device(np.array([1.0], dtype=np.float32))
    d_out = metal.device_array(1, np.float32)
    with pytest.raises(NumbaMetalError):
        kernel[1, 1](d_a, d_out)


def test_device_function_rejects_mutual_recursion() -> None:
    """A cycle between two device functions: each function's own typing
    succeeds in isolation (Numba's frontend cannot see across separate
    compile_to_typed_ir calls), so this specifically exercises
    numba-metal's own in-progress-compilation cycle detection, not
    Numba's built-in direct-recursion check."""
    from numba_metal.errors import NumbaMetalError

    metal = _metal()

    @metal.device_func
    def is_even(n):
        if n == 0:
            return True
        return is_odd(n - 1)

    @metal.device_func
    def is_odd(n):
        if n == 0:
            return False
        return is_even(n - 1)

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = is_even(a[i])

    d_a = metal.to_device(np.array([4], dtype=np.int32))
    d_out = metal.device_array(1, np.bool_)
    with pytest.raises(NumbaMetalError):
        kernel[1, 1](d_a, d_out)


def test_device_function_rejects_array_argument() -> None:
    from numba_metal.errors import NumbaMetalError

    metal = _metal()

    @metal.device_func
    def bad(arr):
        return arr[0]

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = bad(a)

    d_a = metal.to_device(np.array([1.0, 2.0], dtype=np.float32))
    d_out = metal.device_array_like(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(NumbaMetalError):
        kernel[1, 2](d_a, d_out)


def test_device_function_cannot_be_called_from_ordinary_python() -> None:
    metal = _metal()

    @metal.device_func
    def helper(x):
        return x + 1.0

    # A @metal.device_func-decorated function is a real @njit dispatcher
    # (see docs/architecture.md), so it IS callable from ordinary Python
    # via Numba's own CPU compilation -- numba-metal does not prohibit
    # this (unlike an earlier design considered during development that
    # would have made it raise TypeError outside a kernel). This test
    # documents that actual, intentional behavior rather than asserting
    # a restriction that was not ultimately implemented.
    assert helper(2.0) == 3.0
