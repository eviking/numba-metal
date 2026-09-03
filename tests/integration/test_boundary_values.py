"""Boundary-value coverage required by the Workstream 6 feature-matrix
audit: signed/unsigned dtype limits, zero, negative values, bool arrays,
odd/non-power-of-two sizes, non-divisible grid sizes (threads don't
evenly divide the element count), zero-length arrays, and NaN/inf where
the dtype supports them. None of this was previously exercised anywhere
in the test suite.

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


def test_int32_signed_limits() -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i]

    info = np.iinfo(np.int32)
    a = np.array([info.min, info.min + 1, -1, 0, 1, info.max - 1, info.max], np.int32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, len(a)](d_a, d_out)
    metal.synchronize()
    assert np.array_equal(d_out.copy_to_host(), a)


def test_uint32_limits_including_wraparound() -> None:
    """Verify uint32 addition wraps the same way in MSL as it does for
    NumPy's uint32 (both are 32-bit unsigned modular arithmetic)."""
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] + 1

    info = np.iinfo(np.uint32)
    a = np.array([0, 1, info.max - 1, info.max], dtype=np.uint32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, len(a)](d_a, d_out)
    metal.synchronize()
    expected = (a + np.uint32(1)).astype(np.uint32)  # wraps info.max -> 0
    assert np.array_equal(d_out.copy_to_host(), expected)
    assert expected[-1] == 0  # sanity: the wraparound actually occurred


def test_float32_nan_and_inf_propagation() -> None:
    """NaN/inf inputs should propagate through arithmetic exactly as
    IEEE-754 float32 defines, matching NumPy's own float32 semantics."""
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] * 2.0

    a = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, len(a)](d_a, d_out)
    metal.synchronize()
    result = d_out.copy_to_host()
    expected = a * np.float32(2.0)
    assert np.isnan(result[0])
    assert np.array_equal(result[1:], expected[1:])


def test_float32_nan_comparisons_are_false() -> None:
    """IEEE-754: every ordered comparison against NaN is False, including
    NaN == NaN. Verify MSL's comparison operators preserve this rather
    than treating NaN as an ordinary sentinel value."""
    metal = _metal()

    @metal.jit
    def kernel(a, out_eq, out_lt):
        i = metal.grid(1)
        if i < a.size:
            out_eq[i] = a[i] == a[i]
            out_lt[i] = a[i] < 0.0

    a = np.array([np.nan, 1.0, -1.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_eq = metal.device_array(len(a), np.bool_)
    d_lt = metal.device_array(len(a), np.bool_)
    kernel[1, len(a)](d_a, d_eq, d_lt)
    metal.synchronize()
    eq_result = d_eq.copy_to_host()
    lt_result = d_lt.copy_to_host()
    assert eq_result[0] == np.bool_(False)  # NaN == NaN is False
    assert eq_result[1] == np.bool_(True)
    assert lt_result[0] == np.bool_(False)  # NaN < 0.0 is False
    assert lt_result[2] == np.bool_(True)


def test_bool_array_all_true_all_false_mixed() -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = not a[i]

    for a in (
        np.array([True, True, True], dtype=np.bool_),
        np.array([False, False, False], dtype=np.bool_),
        np.array([True, False, True, False], dtype=np.bool_),
    ):
        d_a = metal.to_device(a)
        d_out = metal.device_array_like(a)
        kernel[1, len(a)](d_a, d_out)
        metal.synchronize()
        assert np.array_equal(d_out.copy_to_host(), ~a)


def test_odd_and_prime_sized_arrays() -> None:
    """Sizes that don't divide evenly by common threadgroup sizes (e.g.
    256) and aren't powers of two -- verifies the out[i] < out.size guard
    pattern correctly masks off out-of-range threads with no off-by-one
    at the boundary."""
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] + 1.0

    for n in (1, 2, 3, 7, 13, 97, 257, 1009):  # includes primes
        a = np.arange(n, dtype=np.float32)
        d_a = metal.to_device(a)
        d_out = metal.device_array_like(a)
        threads = 64
        blocks = (n + threads - 1) // threads
        kernel[blocks, threads](d_a, d_out)
        metal.synchronize()
        assert np.array_equal(d_out.copy_to_host(), a + 1.0)


def test_non_divisible_grid_launch_does_not_overrun() -> None:
    """Launch geometry (blocks*threads) intentionally exceeds the element
    count by a non-trivial, non-power-of-two amount, so threads beyond
    out.size must be masked off by the kernel's own bounds check rather
    than relying on the launch geometry being exact."""
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] * 3.0

    n = 37
    a = np.arange(n, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    # 8 blocks x 16 threads = 128 total threads dispatched for 37 elements
    # -- 91 threads must see i >= out.size and do nothing.
    kernel[8, 16](d_a, d_out)
    metal.synchronize()
    assert np.array_equal(d_out.copy_to_host(), a * 3.0)


def test_single_element_array() -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] + 1.0

    a = np.array([41.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, 1](d_a, d_out)
    metal.synchronize()
    assert d_out.copy_to_host()[0] == 42.0


def test_zero_length_array_allocation_and_transfer() -> None:
    """A zero-length device array is a documented, deliberately supported
    edge case for allocation and transfer: to_device(), device_array(),
    device_array_like(), and copy_to_host() must all handle it without
    crashing, and copy_to_host() must return a correctly-shaped empty
    array."""
    metal = _metal()

    a = np.zeros(0, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    assert d_a.size == 0
    assert d_out.size == 0
    result = d_a.copy_to_host()
    assert result.shape == (0,)
    assert result.dtype == np.float32


def test_zero_blocks_launch_is_rejected_not_silently_skipped() -> None:
    """A zero-length array has no work to dispatch, but numba-metal does
    not special-case this: `kernel[0, threads](...)` is still rejected by
    the ordinary "block count must be positive" launch-geometry check
    (KernelLaunchError), the same as it would be for a non-empty array --
    there is no silent no-op launch path. A caller with a zero-length
    array must skip the launch entirely rather than pass 0 blocks."""
    from numba_metal.errors import KernelLaunchError

    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] + 1.0

    a = np.zeros(0, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    with pytest.raises(KernelLaunchError):
        kernel[0, 1](d_a, d_out)
