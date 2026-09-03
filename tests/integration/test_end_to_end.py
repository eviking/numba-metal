"""End-to-end integration tests that compile MSL and execute real kernels
on the Metal GPU. Require a working Apple-silicon Metal device; see
tests/conftest.py for the skip behavior on unsupported machines.

Run with: pytest -m metal
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.fixture(autouse=True, scope="module")
def _metal_module():
    from numba_metal import metal

    return metal


def _metal():
    from numba_metal import metal

    return metal


def test_vector_add_matches_numpy() -> None:
    metal = _metal()

    @metal.jit
    def vector_add(a, b, output):
        i = metal.grid(1)
        if i < output.size:
            output[i] = a[i] + b[i]

    a = np.arange(1_000_000, dtype=np.float32)
    b = np.arange(1_000_000, dtype=np.float32)
    d_a = metal.to_device(a)
    d_b = metal.to_device(b)
    d_output = metal.device_array_like(a)
    threads = 256
    blocks = (a.size + threads - 1) // threads
    vector_add[blocks, threads](d_a, d_b, d_output)
    metal.synchronize()
    output = d_output.copy_to_host()
    assert np.allclose(output, a + b)


def test_scalar_arithmetic_and_comparison() -> None:
    metal = _metal()

    @metal.jit
    def k(a, out, threshold):
        i = metal.grid(1)
        if i < out.size:
            x = a[i] * 2.0 - 1.0
            if x > threshold:
                out[i] = 1.0
            else:
                out[i] = 0.0

    a = np.linspace(-1, 1, 1000).astype(np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array(1000, np.float32)
    k[4, 256](d_a, d_out, np.float32(0.0))
    metal.synchronize()
    expected = np.where(a * 2.0 - 1.0 > 0.0, 1.0, 0.0).astype(np.float32)
    assert np.array_equal(d_out.copy_to_host(), expected)


def test_boolean_operators() -> None:
    metal = _metal()

    @metal.jit
    def k(a, b, out):
        i = metal.grid(1)
        if i < out.size:
            cond = (a[i] > 0.0) and (b[i] > 0.0)
            out[i] = cond

    a = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    b = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    d_a, d_b = metal.to_device(a), metal.to_device(b)
    d_out = metal.device_array(4, np.bool_)
    k[1, 4](d_a, d_b, d_out)
    metal.synchronize()
    assert np.array_equal(d_out.copy_to_host(), np.array([True, False, False, False]))


def test_for_loop_with_runtime_bound() -> None:
    metal = _metal()

    @metal.jit
    def k(out, n):
        i = metal.grid(1)
        if i < out.size:
            s = 0
            for j in range(n):
                s = s + j
            out[i] = s

    d_out = metal.device_array(8, np.int32)
    k[1, 8](d_out, np.int32(10))
    metal.synchronize()
    assert np.all(d_out.copy_to_host() == sum(range(10)))


def test_math_functions_match_numpy() -> None:
    metal = _metal()

    @metal.jit
    def k(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = math.sqrt(a[i]) + math.sin(a[i]) + math.cos(a[i])

    a = np.linspace(0.1, 5.0, 2000).astype(np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array(2000, np.float32)
    k[8, 256](d_a, d_out)
    metal.synchronize()
    expected = np.sqrt(a) + np.sin(a) + np.cos(a)
    assert np.allclose(d_out.copy_to_host(), expected, rtol=1e-4, atol=1e-4)


def test_multiple_signatures_same_dispatcher() -> None:
    metal = _metal()

    @metal.jit
    def k(a, out, factor):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] * factor

    d_a_i32 = metal.to_device(np.arange(50, dtype=np.int32))
    d_out_i32 = metal.device_array(50, np.int32)
    k[1, 64](d_a_i32, d_out_i32, np.int32(3))
    metal.synchronize()
    assert np.array_equal(d_out_i32.copy_to_host(), np.arange(50, dtype=np.int32) * 3)

    d_a_f32 = metal.to_device(np.arange(50, dtype=np.float32))
    d_out_f32 = metal.device_array(50, np.float32)
    k[1, 64](d_a_f32, d_out_f32, np.float32(1.5))
    metal.synchronize()
    assert np.allclose(d_out_f32.copy_to_host(), np.arange(50, dtype=np.float32) * 1.5)


def test_device_array_lifetime_across_multiple_kernels() -> None:
    metal = _metal()

    @metal.jit
    def double_it(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] * 2.0

    a = np.ones(100, dtype=np.float32)
    d_a = metal.to_device(a)
    d_mid = metal.device_array_like(a)
    d_final = metal.device_array_like(a)
    double_it[1, 128](d_a, d_mid)
    double_it[1, 128](d_mid, d_final)
    metal.synchronize()
    assert np.allclose(d_final.copy_to_host(), a * 4.0)


def test_synchronize_is_idempotent() -> None:
    metal = _metal()
    metal.synchronize()
    metal.synchronize()


def test_repeated_launches_are_correct() -> None:
    metal = _metal()

    @metal.jit
    def increment(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] + 1.0

    a = np.zeros(1000, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    for _ in range(10):
        increment[4, 256](d_a, d_out)
        metal.synchronize()
        d_a.copy_to_device(d_out.copy_to_host())
    result = d_out.copy_to_host()
    assert np.allclose(result, 10.0)


def test_compilation_cache_reuses_pipeline() -> None:
    metal = _metal()
    from numba_metal.compiler.pipeline import KernelCache

    @metal.jit
    def k(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i]

    d_a = metal.to_device(np.zeros(10, dtype=np.float32))
    d_out = metal.device_array(10, np.float32)
    assert len(k._cache) == 0
    k[1, 16](d_a, d_out)
    metal.synchronize()
    assert len(k._cache) == 1
    k[1, 16](d_a, d_out)
    metal.synchronize()
    assert len(k._cache) == 1  # second launch with same signature is a cache hit
    assert isinstance(k._cache, KernelCache)


def test_invalid_launch_config_rejected() -> None:
    from numba_metal.errors import KernelLaunchError

    metal = _metal()

    @metal.jit
    def k(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i]

    d_a = metal.to_device(np.zeros(10, dtype=np.float32))
    d_out = metal.device_array(10, np.float32)
    with pytest.raises(KernelLaunchError):
        k[0, 16](d_a, d_out)
    with pytest.raises(KernelLaunchError):
        k[1, -1](d_a, d_out)


def test_host_array_rejected_without_explicit_transfer() -> None:
    from numba_metal.errors import NumbaMetalError

    metal = _metal()

    @metal.jit
    def k(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i]

    a_host = np.zeros(10, dtype=np.float32)
    d_out = metal.device_array(10, np.float32)
    with pytest.raises(NumbaMetalError):
        k[1, 16](a_host, d_out)


def test_grid_2d_matches_reference() -> None:
    metal = _metal()

    @metal.jit
    def k(out, width, height):
        x, y = metal.grid(2)
        if x < width and y < height:
            out[y * width + x] = x + y * width

    width, height = 16, 12
    d_out = metal.device_array(width * height, np.int64)
    k[(1, 1), (width, height)](d_out, np.int32(width), np.int32(height))
    metal.synchronize()
    result = d_out.copy_to_host().reshape(height, width)
    yy, xx = np.mgrid[0:height, 0:width]
    expected = xx + yy * width
    assert np.array_equal(result, expected)


def test_metal_compilation_error_reports_msl_and_diagnostic() -> None:
    """A kernel that fails Numba typing never reaches Metal compilation, so
    this test verifies the alternate honest failure path: an unsupported
    construct raises before any GPU work is attempted."""
    from numba_metal.errors import NumbaMetalError

    metal = _metal()

    with pytest.raises(NumbaMetalError):

        @metal.jit
        def bad(a, out):
            i = metal.grid(1)
            if i < out.size:
                d = {}
                out[i] = a[i]

        d_a = metal.to_device(np.zeros(10, dtype=np.float32))
        d_out = metal.device_array(10, np.float32)
        bad[1, 16](d_a, d_out)
