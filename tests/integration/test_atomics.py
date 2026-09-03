"""Standalone tests for `metal.atomic_add/sub/min/max/exchange` and
`metal.atomic_compare_exchange`: general Numba-to-Metal compiler/runtime
atomic primitives operating on device array elements, with no connection
to any specific downstream workload.

Every real-contention test here launches enough threads that a race
would, with overwhelming probability, produce a wrong result if the
underlying operation were not actually atomic -- these are not merely
"does it compile" checks.

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal

_HEAVY_CONTENTION_N = 500_000


def _metal():
    from numba_metal import metal

    return metal


def _launch_1d(n: int, threads: int = 256) -> tuple[int, int]:
    return (n + threads - 1) // threads, threads


# -- atomic_add / atomic_sub ----------------------------------------------


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float32])
def test_atomic_add_exact_under_heavy_contention(dtype) -> None:
    """Every one of N threads increments the SAME counter element by 1;
    exact-integer dtypes must match exactly, and float32 (natively
    atomic on this hardware) must also match exactly since each
    increment is representable and the total stays well within float32's
    exact-integer range."""
    metal = _metal()

    @metal.jit
    def kernel(counter, n):
        i = metal.grid(1)
        if i < n:
            metal.atomic_add(counter, 0, 1)

    n = _HEAVY_CONTENTION_N
    d_counter = metal.to_device(np.zeros(1, dtype=dtype))
    blocks, threads = _launch_1d(n)
    kernel[blocks, threads](d_counter, np.int32(n))
    metal.synchronize()
    assert d_counter.copy_to_host()[0] == dtype(n)


def test_atomic_add_returns_previous_value() -> None:
    """metal.atomic_add must return the value that was there immediately
    before the add, not the new value -- verified by reconstructing the
    full permutation of "who went before whom" from the returned old
    values and checking it is a valid total order (every old value
    distinct, covering exactly range(n))."""
    metal = _metal()

    @metal.jit
    def kernel(counter, out, n):
        i = metal.grid(1)
        if i < n:
            out[i] = metal.atomic_add(counter, 0, 1)

    n = 4096
    d_counter = metal.to_device(np.zeros(1, dtype=np.int32))
    d_out = metal.device_array(n, np.int32)
    blocks, threads = _launch_1d(n)
    kernel[blocks, threads](d_counter, d_out, np.int32(n))
    metal.synchronize()
    old_values = d_out.copy_to_host()
    assert d_counter.copy_to_host()[0] == n
    assert sorted(old_values.tolist()) == list(range(n))


def test_atomic_sub_exact_under_heavy_contention() -> None:
    metal = _metal()

    @metal.jit
    def kernel(counter, n):
        i = metal.grid(1)
        if i < n:
            metal.atomic_sub(counter, 0, 1)

    n = _HEAVY_CONTENTION_N
    d_counter = metal.to_device(np.array([n], dtype=np.int32))
    blocks, threads = _launch_1d(n)
    kernel[blocks, threads](d_counter, np.int32(n))
    metal.synchronize()
    assert d_counter.copy_to_host()[0] == 0


def test_atomic_add_uniform_distribution_across_few_pixels() -> None:
    """A Datashader-style scatter-reduce pattern: many threads, but only
    a handful of distinct target elements (uniform-but-clustered
    contention), not one single element -- verifies atomics are correct
    when contention is spread over multiple addresses simultaneously."""
    metal = _metal()

    @metal.jit
    def kernel(indices, counters, n):
        i = metal.grid(1)
        if i < n:
            metal.atomic_add(counters, indices[i], 1)

    n_records = 200_000
    n_bins = 16
    rng = np.random.default_rng(0)
    indices = rng.integers(0, n_bins, size=n_records).astype(np.int32)
    expected = np.bincount(indices, minlength=n_bins).astype(np.int32)

    d_indices = metal.to_device(indices)
    d_counters = metal.to_device(np.zeros(n_bins, dtype=np.int32))
    blocks, threads = _launch_1d(n_records)
    kernel[blocks, threads](d_indices, d_counters, np.int32(n_records))
    metal.synchronize()
    assert np.array_equal(d_counters.copy_to_host(), expected)


def test_atomic_add_all_records_hit_one_pixel() -> None:
    """The maximum-contention extreme: every single thread targets the
    exact same element (not just a shared bin among several)."""
    metal = _metal()

    @metal.jit
    def kernel(counter, n):
        i = metal.grid(1)
        if i < n:
            metal.atomic_add(counter, 0, 1)

    n = _HEAVY_CONTENTION_N
    d_counter = metal.to_device(np.zeros(1, dtype=np.int32))
    blocks, threads = _launch_1d(n)
    kernel[blocks, threads](d_counter, np.int32(n))
    metal.synchronize()
    assert d_counter.copy_to_host()[0] == n


@pytest.mark.parametrize("threads_per_block", [32, 64, 128, 256, 512])
def test_atomic_add_correct_across_threadgroup_sizes(threads_per_block) -> None:
    metal = _metal()

    @metal.jit
    def kernel(counter, n):
        i = metal.grid(1)
        if i < n:
            metal.atomic_add(counter, 0, 1)

    n = 100_000
    d_counter = metal.to_device(np.zeros(1, dtype=np.int32))
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](d_counter, np.int32(n))
    metal.synchronize()
    assert d_counter.copy_to_host()[0] == n


def test_atomic_add_repeated_executions_stay_consistent() -> None:
    """Repeated independent launches intended to expose any nondeterministic
    race: every run against a fresh zeroed counter must produce exactly n,
    every time, not just "usually"."""
    metal = _metal()

    @metal.jit
    def kernel(counter, n):
        i = metal.grid(1)
        if i < n:
            metal.atomic_add(counter, 0, 1)

    n = 200_000
    blocks, threads = _launch_1d(n)
    for _ in range(10):
        d_counter = metal.to_device(np.zeros(1, dtype=np.int32))
        kernel[blocks, threads](d_counter, np.int32(n))
        metal.synchronize()
        assert d_counter.copy_to_host()[0] == n


# -- atomic_min / atomic_max -----------------------------------------------


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_atomic_max_native_int_exact(dtype) -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out, n):
        i = metal.grid(1)
        if i < n:
            metal.atomic_max(out, 0, a[i])

    n = 100_000
    rng = np.random.default_rng(1)
    if dtype is np.uint32:
        a = rng.integers(0, 2**31, size=n).astype(dtype)
        init = np.array([0], dtype=dtype)
    else:
        a = rng.integers(-(2**30), 2**30, size=n).astype(dtype)
        init = np.array([np.iinfo(dtype).min], dtype=dtype)
    d_a = metal.to_device(a)
    d_out = metal.to_device(init)
    blocks, threads = _launch_1d(n)
    kernel[blocks, threads](d_a, d_out, np.int32(n))
    metal.synchronize()
    assert d_out.copy_to_host()[0] == a.max()


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_atomic_min_native_int_exact(dtype) -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out, n):
        i = metal.grid(1)
        if i < n:
            metal.atomic_min(out, 0, a[i])

    n = 100_000
    rng = np.random.default_rng(2)
    if dtype is np.uint32:
        a = rng.integers(0, 2**31, size=n).astype(dtype)
        init = np.array([np.iinfo(dtype).max], dtype=dtype)
    else:
        a = rng.integers(-(2**30), 2**30, size=n).astype(dtype)
        init = np.array([np.iinfo(dtype).max], dtype=dtype)
    d_a = metal.to_device(a)
    d_out = metal.to_device(init)
    blocks, threads = _launch_1d(n)
    kernel[blocks, threads](d_a, d_out, np.int32(n))
    metal.synchronize()
    assert d_out.copy_to_host()[0] == a.min()


def test_atomic_max_float32_cas_loop_exact_under_heavy_contention() -> None:
    """float32 has no native MSL atomic_fetch_max -- this exercises the
    CAS-retry-loop lowering under real, heavy contention and requires an
    EXACT match (not a tolerance), since max/min of a set of exactly
    representable float32 values has no rounding ambiguity."""
    metal = _metal()

    @metal.jit
    def kernel(a, out, n):
        i = metal.grid(1)
        if i < n:
            metal.atomic_max(out, 0, a[i])

    n = _HEAVY_CONTENTION_N
    rng = np.random.default_rng(3)
    a = rng.standard_normal(n).astype(np.float32)
    d_a = metal.to_device(a)
    d_out = metal.to_device(np.array([-np.inf], dtype=np.float32))
    blocks, threads = _launch_1d(n)
    kernel[blocks, threads](d_a, d_out, np.int32(n))
    metal.synchronize()
    assert d_out.copy_to_host()[0] == a.max()


def test_atomic_min_float32_cas_loop_exact_under_heavy_contention() -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out, n):
        i = metal.grid(1)
        if i < n:
            metal.atomic_min(out, 0, a[i])

    n = _HEAVY_CONTENTION_N
    rng = np.random.default_rng(4)
    a = rng.standard_normal(n).astype(np.float32)
    d_a = metal.to_device(a)
    d_out = metal.to_device(np.array([np.inf], dtype=np.float32))
    blocks, threads = _launch_1d(n)
    kernel[blocks, threads](d_a, d_out, np.int32(n))
    metal.synchronize()
    assert d_out.copy_to_host()[0] == a.min()


def test_atomic_max_returns_previous_value() -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out, prev, n):
        i = metal.grid(1)
        if i < n:
            prev[i] = metal.atomic_max(out, 0, a[i])

    a = np.array([3.0, 1.0, 5.0, 2.0, 4.0], dtype=np.float32)
    n = len(a)
    d_a = metal.to_device(a)
    d_out = metal.to_device(np.array([0.0], dtype=np.float32))
    d_prev = metal.device_array(n, np.float32)
    kernel[1, n](d_a, d_out, d_prev, np.int32(n))
    metal.synchronize()
    assert d_out.copy_to_host()[0] == 5.0
    # Whatever serialization order the GPU chose, every returned "prev"
    # value must have been a legitimate running max at some point --
    # i.e. every prev value must be <= the final max, and 0.0 (the
    # initial value) must appear at least once (the very first update).
    prev = d_prev.copy_to_host()
    assert np.all(prev <= 5.0)
    assert 0.0 in prev


# -- atomic_exchange ---------------------------------------------------


def test_atomic_exchange_produces_valid_permutation() -> None:
    """Every thread exchanges a distinct value into the same slot; the
    set of "previous value" results, together with the final slot value,
    must form exactly the full permutation of {initial} + {every thread's
    new value} minus whichever one is left in the slot at the end."""
    metal = _metal()

    @metal.jit
    def kernel(buf, out, n):
        i = metal.grid(1)
        if i < n:
            out[i] = metal.atomic_exchange(buf, 0, i)

    n = 256
    d_buf = metal.to_device(np.array([-1], dtype=np.int32))
    d_out = metal.device_array(n, np.int32)
    kernel[1, n](d_buf, d_out, np.int32(n))
    metal.synchronize()
    old_values = d_out.copy_to_host()
    final_value = d_buf.copy_to_host()[0]
    all_values = set(old_values.tolist()) | {final_value}
    assert all_values == set(range(-1, n))
    assert len(set(old_values.tolist())) == n  # every old value distinct


def test_atomic_exchange_float32() -> None:
    metal = _metal()

    @metal.jit
    def kernel(buf, out):
        i = metal.grid(1)
        out[i] = metal.atomic_exchange(buf, 0, float(i) + 0.5)

    n = 32
    d_buf = metal.to_device(np.array([-1.0], dtype=np.float32))
    d_out = metal.device_array(n, np.float32)
    kernel[1, n](d_buf, d_out)
    metal.synchronize()
    old_values = d_out.copy_to_host()
    final_value = d_buf.copy_to_host()[0]
    expected_new_values = {float(i) + 0.5 for i in range(n)}
    all_values = set(old_values.tolist()) | {float(final_value)}
    assert all_values == expected_new_values | {-1.0}


# -- atomic_compare_exchange -----------------------------------------------


def test_compare_exchange_exactly_one_success_under_contention() -> None:
    """Every thread races to CAS the same element from its initial value
    to a distinct new value; only the compare against the TRUE initial
    value can ever succeed, so across many threads racing, exactly one
    success must occur, repeatably."""
    metal = _metal()

    @metal.jit
    def kernel(buf, out_ok, n):
        i = metal.grid(1)
        if i < n:
            _old, ok = metal.atomic_compare_exchange(buf, 0, 0, i + 1)
            out_ok[i] = ok

    n = 256
    for _ in range(10):
        d_buf = metal.to_device(np.array([0], dtype=np.int32))
        d_out_ok = metal.device_array(n, np.bool_)
        kernel[1, n](d_buf, d_out_ok, np.int32(n))
        metal.synchronize()
        assert int(d_out_ok.copy_to_host().sum()) == 1


def test_compare_exchange_returns_old_value_on_failure() -> None:
    metal = _metal()

    @metal.jit
    def kernel(buf, out_old, out_ok):
        old, ok = metal.atomic_compare_exchange(buf, 0, 999, 42)
        out_old[0] = old
        out_ok[0] = ok

    d_buf = metal.to_device(np.array([7], dtype=np.int32))
    d_out_old = metal.device_array(1, np.int32)
    d_out_ok = metal.device_array(1, np.bool_)
    kernel[1, 1](d_buf, d_out_old, d_out_ok)
    metal.synchronize()
    assert d_out_old.copy_to_host()[0] == 7  # actual value, not `expected`
    assert d_out_ok.copy_to_host()[0] == np.bool_(False)
    assert d_buf.copy_to_host()[0] == 7  # unchanged on failure


def test_compare_exchange_succeeds_and_updates_on_match() -> None:
    metal = _metal()

    @metal.jit
    def kernel(buf, out_old, out_ok):
        old, ok = metal.atomic_compare_exchange(buf, 0, 7, 42)
        out_old[0] = old
        out_ok[0] = ok

    d_buf = metal.to_device(np.array([7], dtype=np.int32))
    d_out_old = metal.device_array(1, np.int32)
    d_out_ok = metal.device_array(1, np.bool_)
    kernel[1, 1](d_buf, d_out_old, d_out_ok)
    metal.synchronize()
    assert d_out_old.copy_to_host()[0] == 7
    assert d_out_ok.copy_to_host()[0] == np.bool_(True)
    assert d_buf.copy_to_host()[0] == 42


def test_compare_exchange_retry_loop_matches_atomic_add() -> None:
    """A hand-rolled CAS retry loop implementing add-by-1 must produce
    exactly the same exact-count result as the built-in metal.atomic_add,
    proving the CAS primitive itself is genuinely race-free (this is
    also exactly the retry-loop pattern metal.atomic_min/max use
    internally for float32). Exercises numba-metal's straight-line
    `while` loop support (see docs/limitations.md for exactly what
    `while` shapes are and are not supported -- a CAS retry loop is
    exactly the straight-line shape that is)."""
    metal = _metal()

    @metal.jit
    def kernel(counter, n):
        i = metal.grid(1)
        if i < n:
            done = False
            while not done:
                old = counter[0]
                _prev, ok = metal.atomic_compare_exchange(counter, 0, old, old + 1)
                done = ok

    for n in (4, 1000, _HEAVY_CONTENTION_N):
        d_counter = metal.to_device(np.zeros(1, dtype=np.int32))
        blocks, threads = _launch_1d(n)
        kernel[blocks, threads](d_counter, np.int32(n))
        metal.synchronize()
        assert d_counter.copy_to_host()[0] == n


def test_atomic_rejects_unsupported_dtype() -> None:
    from numba_metal.errors import NumbaMetalError

    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            metal.atomic_add(out, i, a[i])

    a = np.array([1.0], dtype=np.float16)
    d_a = metal.to_device(a)
    d_out = metal.device_array(1, np.float16)
    with pytest.raises(NumbaMetalError):
        kernel[1, 1](d_a, d_out)


def test_atomic_rejects_2d_array() -> None:
    """Atomics only support 1D arrays, matching numba-metal's general 1D-
    array-kernel-argument restriction; this test exercises the
    intrinsic's own typing-level rejection (before Numba even reaches
    Metal-specific lowering), independent of that broader restriction."""
    from numba.core import types

    from numba_metal.compiler.frontend import compile_to_typed_ir
    from numba_metal.compiler.intrinsics import atomic_add

    def f(a):
        return atomic_add(a, 0, 1)

    with pytest.raises(Exception):  # noqa: B017 - Numba wraps TypingError variably
        compile_to_typed_ir(f, (types.int32[:, ::1],))
