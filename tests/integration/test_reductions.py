"""Standalone tests for `metal.reduce_sum/reduce_min/reduce_max`: the
host-side two-stage (shared-memory tree + one atomic per threadgroup)
reduction helper built on top of already-tested primitives (see
`test_local_and_shared_memory.py` and `test_atomics.py`).

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


# -- reduce_sum --------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 7, 255, 256, 257, 1000, 100_000, 1_000_003])
def test_reduce_sum_float32_matches_numpy(n) -> None:
    metal = _metal()
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).astype(np.float32)
    d_a = metal.to_device(a)
    result = metal.reduce_sum(d_a).copy_to_host()[0]
    # float32 accumulation order differs from NumPy's own (pairwise)
    # summation -- a tolerance, not exact equality, is the correct check
    # here (unlike the exact-integer atomics tests in test_atomics.py).
    assert result == pytest.approx(float(a.sum()), rel=1e-3, abs=1e-2)


def test_reduce_sum_int32_exact() -> None:
    metal = _metal()
    rng = np.random.default_rng(1)
    a = rng.integers(-1000, 1000, size=100_000).astype(np.int32)
    d_a = metal.to_device(a)
    result = metal.reduce_sum(d_a).copy_to_host()[0]
    assert result == int(a.astype(np.int64).sum())


def test_reduce_sum_uint32_exact() -> None:
    metal = _metal()
    rng = np.random.default_rng(2)
    a = rng.integers(0, 1000, size=100_000).astype(np.uint32)
    d_a = metal.to_device(a)
    result = metal.reduce_sum(d_a).copy_to_host()[0]
    assert result == int(a.astype(np.uint64).sum())


def test_reduce_sum_single_element() -> None:
    metal = _metal()
    d_a = metal.to_device(np.array([42.0], dtype=np.float32))
    assert metal.reduce_sum(d_a).copy_to_host()[0] == pytest.approx(42.0)


def test_reduce_sum_exceeds_single_threadgroup() -> None:
    """Forces multiple threadgroups (each contributing one atomic_add),
    not just the single-threadgroup shape already covered by
    test_shared_array_cooperative_reduction_single_threadgroup."""
    metal = _metal()
    n = 10 * 256 + 37  # several full threadgroups plus a partial one
    a = np.ones(n, dtype=np.float32)
    d_a = metal.to_device(a)
    result = metal.reduce_sum(d_a).copy_to_host()[0]
    assert result == pytest.approx(float(n))


# -- reduce_min / reduce_max ---------------------------------------------


@pytest.mark.parametrize("n", [1, 255, 256, 257, 100_000])
def test_reduce_min_max_float32_matches_numpy(n) -> None:
    metal = _metal()
    rng = np.random.default_rng(3)
    a = rng.standard_normal(n).astype(np.float32)
    d_a = metal.to_device(a)
    assert metal.reduce_min(d_a).copy_to_host()[0] == pytest.approx(float(a.min()))
    assert metal.reduce_max(d_a).copy_to_host()[0] == pytest.approx(float(a.max()))


def test_reduce_min_max_int32_exact() -> None:
    metal = _metal()
    rng = np.random.default_rng(4)
    a = rng.integers(-100_000, 100_000, size=100_000).astype(np.int32)
    d_a = metal.to_device(a)
    assert metal.reduce_min(d_a).copy_to_host()[0] == a.min()
    assert metal.reduce_max(d_a).copy_to_host()[0] == a.max()


def test_reduce_min_max_uint32_exact() -> None:
    metal = _metal()
    rng = np.random.default_rng(5)
    a = rng.integers(0, 200_000, size=100_000).astype(np.uint32)
    d_a = metal.to_device(a)
    assert metal.reduce_min(d_a).copy_to_host()[0] == a.min()
    assert metal.reduce_max(d_a).copy_to_host()[0] == a.max()


def test_reduce_max_finds_single_outlier() -> None:
    """A single, deliberately extreme value amid otherwise-uniform data --
    catches an off-by-one in the tree reduction that would only miss one
    element out of many rather than being uniformly wrong."""
    metal = _metal()
    n = 10_000
    a = np.ones(n, dtype=np.float32)
    a[7777] = 999.0
    d_a = metal.to_device(a)
    assert metal.reduce_max(d_a).copy_to_host()[0] == pytest.approx(999.0)


# -- repeated launches (no stale-threadgroup-memory leakage) --------------


def test_reduce_sum_repeated_calls_do_not_leak_stale_data() -> None:
    """Mirrors test_shared_array_repeated_launches_do_not_leak_stale_data:
    each call must reflect only its own input, not a previous call's
    out-of-bounds shared-memory contents (Metal does not zero
    threadgroup memory between dispatches)."""
    metal = _metal()
    for trial in range(5):
        n = 100 + trial  # varying, non-power-of-two sizes across calls
        a = np.full(n, float(trial + 1), dtype=np.float32)
        d_a = metal.to_device(a)
        result = metal.reduce_sum(d_a).copy_to_host()[0]
        assert result == pytest.approx(float(n * (trial + 1)))


# -- error paths -----------------------------------------------------------


def test_reduce_rejects_2d_array() -> None:
    from numba_metal.errors import NumbaMetalError

    metal = _metal()
    d_a = metal.to_device(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(NumbaMetalError):
        metal.reduce_sum(d_a)


def test_reduce_rejects_unsupported_dtype() -> None:
    from numba_metal.errors import NumbaMetalError

    metal = _metal()
    d_a = metal.to_device(np.zeros(4, dtype=np.float16))
    with pytest.raises(NumbaMetalError):
        metal.reduce_sum(d_a)


def test_reduce_rejects_host_array() -> None:
    from numba_metal.errors import NumbaMetalError

    metal = _metal()
    with pytest.raises(NumbaMetalError):
        metal.reduce_sum(np.zeros(4, dtype=np.float32))


def test_reduce_rejects_empty_array() -> None:
    from numba_metal.errors import NumbaMetalError

    metal = _metal()
    d_a = metal.device_array(0, np.float32)
    with pytest.raises(NumbaMetalError):
        metal.reduce_sum(d_a)
