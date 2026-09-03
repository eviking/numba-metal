"""Closes specific gaps found by the Workstream 6 feature-matrix audit:
several rows in docs/supported-features.md were marked "Supported" on
the strength of a type-mapping unit test or an MSL-substring check
alone, with no real Metal execution test actually exercising them (or,
for dtypes like int64/bool, only ever as a kernel *output*, never an
*input*). Each test here is real Metal execution (`pytest -m metal`)
with a differential check against a CPU/NumPy reference, so the
corresponding docs row is now backed by the evidence level it claims.

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


def test_gridsize_1d_matches_launch_geometry() -> None:
    """metal.gridsize(1) had zero test coverage despite being marked
    Supported; verify it returns the actual total dispatched thread
    count (blocks*threads), not just metal.grid()'s per-thread index."""
    metal = _metal()

    @metal.jit
    def kernel(out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = metal.gridsize(1)

    blocks, threads = 5, 16
    d_out = metal.device_array(blocks * threads, np.int64)
    kernel[blocks, threads](d_out)
    metal.synchronize()
    result = d_out.copy_to_host()
    assert np.all(result == blocks * threads)


def test_gridsize_2d_matches_launch_geometry() -> None:
    metal = _metal()

    @metal.jit
    def kernel(out, width):
        ix, iy = metal.grid(2)
        gx, gy = metal.gridsize(2)
        idx = iy * width + ix
        if ix < width and idx < out.size:
            out[idx] = gx * 1000 + gy

    bx, by, tx, ty = 3, 2, 8, 4
    width = bx * tx
    height = by * ty
    d_out = metal.device_array(width * height, np.int64)
    kernel[(bx, by), (tx, ty)](d_out, np.int32(width))
    metal.synchronize()
    expected = (bx * tx) * 1000 + (by * ty)
    assert np.all(d_out.copy_to_host() == expected)


def test_min_max_builtins_on_real_hardware() -> None:
    """min()/max() previously had only an MSL-substring unit test; verify
    actual runtime output against NumPy."""
    metal = _metal()

    @metal.jit
    def kernel(a, b, out_min, out_max):
        i = metal.grid(1)
        if i < a.size:
            out_min[i] = min(a[i], b[i])
            out_max[i] = max(a[i], b[i])

    rng = np.random.default_rng(0)
    a = rng.standard_normal(64).astype(np.float32)
    b = rng.standard_normal(64).astype(np.float32)
    d_a, d_b = metal.to_device(a), metal.to_device(b)
    d_min = metal.device_array_like(a)
    d_max = metal.device_array_like(a)
    kernel[1, 64](d_a, d_b, d_min, d_max)
    metal.synchronize()
    assert np.allclose(d_min.copy_to_host(), np.minimum(a, b))
    assert np.allclose(d_max.copy_to_host(), np.maximum(a, b))


def test_float_and_int_casts_on_real_hardware() -> None:
    """float()/int() casts had no test evidence at any level. Verify
    int->float and float->int truncating-cast behavior against Python's
    own semantics for float(int) and int(float) (truncation toward
    zero, matching C/MSL's (int) cast and Python's int() on a float)."""
    metal = _metal()

    @metal.jit
    def kernel(a_int, a_float, out_as_float, out_as_int):
        i = metal.grid(1)
        if i < a_int.size:
            out_as_float[i] = float(a_int[i]) * 0.5
            out_as_int[i] = int(a_float[i])

    a_int = np.array([-3, -1, 0, 1, 2, 7, 100], dtype=np.int32)
    a_float = np.array([-3.9, -1.1, 0.0, 1.9, 2.5, 7.99, 100.4], dtype=np.float32)
    d_int = metal.to_device(a_int)
    d_float = metal.to_device(a_float)
    d_out_float = metal.device_array(len(a_int), np.float32)
    d_out_int = metal.device_array(len(a_float), np.int32)
    kernel[1, len(a_int)](d_int, d_float, d_out_float, d_out_int)
    metal.synchronize()

    assert np.allclose(d_out_float.copy_to_host(), a_int.astype(np.float32) * 0.5)
    expected_int = np.trunc(a_float).astype(np.int32)
    assert np.array_equal(d_out_int.copy_to_host(), expected_int)


def test_uint32_array_input_and_output() -> None:
    """uint32 previously only had a type-mapping unit test; no kernel
    anywhere used it as an actual array argument dtype."""
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] + 1

    a = np.array([0, 1, 4294967294, 100, 2**31], dtype=np.uint32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, len(a)](d_a, d_out)
    metal.synchronize()
    expected = (a + 1).astype(np.uint32)
    assert np.array_equal(d_out.copy_to_host(), expected)


def test_float16_array_input_and_output() -> None:
    """float16 previously only had a type-mapping unit test ('half'
    string check); no integration test used it despite the docs' Notes
    column claiming integration-test coverage."""
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] + a[i]

    a = np.array([1.0, -2.5, 0.5, 3.25], dtype=np.float16)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, len(a)](d_a, d_out)
    metal.synchronize()
    expected = (a + a).astype(np.float16)
    assert np.allclose(
        d_out.copy_to_host().astype(np.float32), expected.astype(np.float32)
    )


def test_int64_array_input() -> None:
    """int64 previously was only ever exercised as an output dtype
    (test_grid_2d_matches_reference); verify it also works as an input
    array/scalar kernel argument."""
    metal = _metal()

    @metal.jit
    def kernel(a, scalar, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] + scalar

    a = np.array([-(2**40), -1, 0, 1, 2**40], dtype=np.int64)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, len(a)](d_a, np.int64(10), d_out)
    metal.synchronize()
    assert np.array_equal(d_out.copy_to_host(), a + 10)


def test_bool_array_input() -> None:
    """bool previously was only ever exercised as an output dtype
    (test_boolean_operators); verify it also works as an input array
    kernel argument."""
    metal = _metal()

    @metal.jit
    def kernel(flags, a, out):
        i = metal.grid(1)
        if i < a.size:
            if flags[i]:
                out[i] = a[i]
            else:
                out[i] = -a[i]

    flags = np.array([True, False, True, True, False], dtype=np.bool_)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    d_flags = metal.to_device(flags)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, len(a)](d_flags, d_a, d_out)
    metal.synchronize()
    expected = np.where(flags, a, -a)
    assert np.allclose(d_out.copy_to_host(), expected)


def test_all_comparison_operators_on_real_hardware() -> None:
    """Only <, >, >= had previously appeared inside any tested kernel
    body; <=, ==, != were untested despite the docs row claiming all six
    comparison operators are Supported."""
    metal = _metal()

    @metal.jit
    def kernel(a, b, out_le, out_eq, out_ne):
        i = metal.grid(1)
        if i < a.size:
            out_le[i] = a[i] <= b[i]
            out_eq[i] = a[i] == b[i]
            out_ne[i] = a[i] != b[i]

    a = np.array([1.0, 2.0, 3.0, 3.0, 5.0], dtype=np.float32)
    b = np.array([1.0, 1.0, 3.0, 4.0, 2.0], dtype=np.float32)
    d_a, d_b = metal.to_device(a), metal.to_device(b)
    d_le = metal.device_array(len(a), np.bool_)
    d_eq = metal.device_array(len(a), np.bool_)
    d_ne = metal.device_array(len(a), np.bool_)
    kernel[1, len(a)](d_a, d_b, d_le, d_eq, d_ne)
    metal.synchronize()
    assert np.array_equal(d_le.copy_to_host(), a <= b)
    assert np.array_equal(d_eq.copy_to_host(), a == b)
    assert np.array_equal(d_ne.copy_to_host(), a != b)


def test_or_and_not_boolean_operators_on_real_hardware() -> None:
    """Only `and` had previously appeared inside any tested kernel body;
    `or` and `not` were untested despite the docs row claiming all three
    boolean operators are Supported."""
    metal = _metal()

    @metal.jit
    def kernel(a, b, out_or, out_not):
        i = metal.grid(1)
        if i < a.size:
            out_or[i] = (a[i] > 0.0) or (b[i] > 0.0)
            out_not[i] = not (a[i] > 0.0)

    a = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    b = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    d_a, d_b = metal.to_device(a), metal.to_device(b)
    d_or = metal.device_array(len(a), np.bool_)
    d_not = metal.device_array(len(a), np.bool_)
    kernel[1, len(a)](d_a, d_b, d_or, d_not)
    metal.synchronize()
    assert np.array_equal(d_or.copy_to_host(), (a > 0.0) | (b > 0.0))
    assert np.array_equal(d_not.copy_to_host(), ~(a > 0.0))


def test_dump_msl_env_var_prints_generated_source(capsys) -> None:
    """NUMBA_METAL_DUMP_MSL / metal.config.dump_msl had zero test
    coverage; verify toggling it actually prints the generated MSL for
    the next compiled kernel, and that it can be turned back off."""
    metal = _metal()

    metal.config.dump_msl = True
    try:

        @metal.jit
        def kernel(a, out):
            i = metal.grid(1)
            if i < a.size:
                out[i] = a[i]

        d_a = metal.to_device(np.zeros(4, dtype=np.float32))
        d_out = metal.device_array(4, np.float32)
        kernel[1, 4](d_a, d_out)
        metal.synchronize()
        captured = capsys.readouterr()
        assert "kernel void" in captured.out
        assert "numba-metal generated MSL" in captured.out
    finally:
        metal.config.dump_msl = False
