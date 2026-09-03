"""Standalone tests for `metal.local_array()`, `metal.shared_array()`,
and `metal.barrier()`: general Numba-to-Metal compiler/runtime memory
primitives, with no connection to any specific downstream workload.

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


# -- metal.local_array() -------------------------------------------------


def test_local_array_read_write_roundtrip() -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            buf = metal.local_array(4, np.float32)
            buf[0] = a[i]
            buf[1] = a[i] * 2.0
            buf[2] = a[i] * 3.0
            buf[3] = a[i] * 4.0
            total = 0.0
            for k in range(4):
                total = total + buf[k]
            out[i] = total

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, 4](d_a, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), a * 10.0)


def test_local_array_is_private_per_thread() -> None:
    """Each thread's metal.local_array() must be an independent
    allocation -- one thread's writes must never be visible to another
    thread's local_array of the same declared shape/dtype (unlike
    metal.shared_array(), which is intentionally shared)."""
    metal = _metal()

    @metal.jit
    def kernel(out):
        i = metal.grid(1)
        buf = metal.local_array(1, np.int32)
        buf[0] = int(i)
        # A delay-free race window: if local_array were accidentally
        # shared, a fast-finishing neighbor thread could overwrite buf[0]
        # before this thread reads it back. Real Metal hardware runs all
        # threads in this single-threadgroup dispatch concurrently, so
        # this is a genuine (not merely theoretical) check.
        if i < out.size:
            out[i] = buf[0]

    n = 64
    d_out = metal.device_array(n, np.int32)
    kernel[1, n](d_out)
    metal.synchronize()
    assert np.array_equal(d_out.copy_to_host(), np.arange(n, dtype=np.int32))


def test_local_array_int_dtype() -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            buf = metal.local_array(3, np.int32)
            buf[0] = a[i]
            buf[1] = a[i] + 1
            buf[2] = a[i] + 2
            out[i] = buf[0] + buf[1] + buf[2]

    a = np.array([10, 20, 30], dtype=np.int32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, 3](d_a, d_out)
    metal.synchronize()
    assert np.array_equal(d_out.copy_to_host(), a * 3 + 3)


def test_local_array_rejects_non_literal_shape() -> None:
    from numba_metal.errors import NumbaMetalError

    metal = _metal()

    @metal.jit
    def kernel(a, out, n):
        i = metal.grid(1)
        if i < a.size:
            buf = metal.local_array(n, np.float32)
            buf[0] = a[i]
            out[i] = buf[0]

    d_a = metal.to_device(np.zeros(4, dtype=np.float32))
    d_out = metal.device_array(4, np.float32)
    with pytest.raises(NumbaMetalError):
        kernel[1, 4](d_a, d_out, np.int32(4))


# -- metal.shared_array() + metal.barrier() -------------------------------


def test_shared_array_visible_across_threads_after_barrier() -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        tid = metal.thread_in_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(8, np.float32)
        scratch[tid] = a[gid]
        metal.barrier()
        # Every thread reads the NEXT thread's value (wrapping), which is
        # only correct if every write became visible to every thread
        # before any read -- i.e. the barrier actually synchronized.
        next_tid = (tid + 1) % 8
        out[gid] = scratch[next_tid]

    a = np.arange(1, 9, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array(8, np.float32)
    kernel[1, 8](d_a, d_out)
    metal.synchronize()
    expected = np.roll(a, -1)
    assert np.array_equal(d_out.copy_to_host(), expected)


def test_shared_array_cooperative_reduction_single_threadgroup() -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        tid = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(8, np.float32)
        if gid < a.size:
            scratch[tid] = a[gid]
        metal.barrier()
        if tid == 0:
            total = 0.0
            for k in range(tg_size):
                total = total + scratch[k]
            out[metal.threadgroup_position(1)] = total

    a = np.arange(1, 9, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array(1, np.float32)
    kernel[1, 8](d_a, d_out)
    metal.synchronize()
    assert d_out.copy_to_host()[0] == pytest.approx(float(a.sum()))


def test_shared_array_isolated_per_threadgroup() -> None:
    """Verifies each threadgroup gets its own independent shared_array
    allocation -- a per-threadgroup partial sum must not see any other
    threadgroup's data."""
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        tid = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        gid = metal.grid(1)
        tgid = metal.threadgroup_position(1)
        scratch = metal.shared_array(4, np.float32)
        if gid < a.size:
            scratch[tid] = a[gid]
        metal.barrier()
        if tid == 0:
            total = 0.0
            for k in range(tg_size):
                total = total + scratch[k]
            out[tgid] = total

    n_groups, group_size = 6, 4
    a = np.arange(1, n_groups * group_size + 1, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array(n_groups, np.float32)
    kernel[n_groups, group_size](d_a, d_out)
    metal.synchronize()
    expected = a.reshape(n_groups, group_size).sum(axis=1)
    assert np.allclose(d_out.copy_to_host(), expected)


def test_two_independent_shared_arrays_in_one_kernel() -> None:
    """Two metal.shared_array() declarations in the same kernel must get
    distinct [[threadgroup(n)]] allocations (different dtypes/sizes),
    neither clobbering the other."""
    metal = _metal()

    @metal.jit
    def kernel(a, b, out):
        tid = metal.thread_in_threadgroup(1)
        gid = metal.grid(1)
        scratch_a = metal.shared_array(4, np.float32)
        scratch_b = metal.shared_array(4, np.int32)
        scratch_a[tid] = a[gid]
        scratch_b[tid] = b[gid]
        metal.barrier()
        out[gid] = scratch_a[tid] + float(scratch_b[tid])

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10, 20, 30, 40], dtype=np.int32)
    d_a, d_b = metal.to_device(a), metal.to_device(b)
    d_out = metal.device_array(4, np.float32)
    kernel[1, 4](d_a, d_b, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), a + b.astype(np.float32))


def test_shared_array_repeated_launches_do_not_leak_stale_data() -> None:
    """Each dispatch must see fresh threadgroup memory semantics: a
    thread that doesn't write its slot before the barrier must not
    observe a stale value from a PREVIOUS, unrelated launch (Metal does
    not guarantee zero-initialized threadgroup memory, so this test
    writes every slot every launch and only checks values that were
    actually written are read back correctly across repeated launches
    with different data)."""
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        tid = metal.thread_in_threadgroup(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(8, np.float32)
        scratch[tid] = a[gid]
        metal.barrier()
        out[gid] = scratch[tid]

    d_out = metal.device_array(8, np.float32)
    for trial in range(5):
        a = (np.arange(8, dtype=np.float32) + trial * 100.0).astype(np.float32)
        d_a = metal.to_device(a)
        kernel[1, 8](d_a, d_out)
        metal.synchronize()
        assert np.array_equal(d_out.copy_to_host(), a)


def test_barrier_with_multiple_threadgroups_only_syncs_within_group() -> None:
    """A barrier only synchronizes threads within the SAME threadgroup;
    this test confirms cross-threadgroup results are still independent
    and correct even though metal.barrier() is called by every
    threadgroup in the dispatch."""
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        tid = metal.thread_in_threadgroup(1)
        tgid = metal.threadgroup_position(1)
        gid = metal.grid(1)
        scratch = metal.shared_array(4, np.float32)
        scratch[tid] = a[gid] * float(tgid + 1)
        metal.barrier()
        out[gid] = scratch[tid]

    n_groups, group_size = 4, 4
    a = np.ones(n_groups * group_size, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array(n_groups * group_size, np.float32)
    kernel[n_groups, group_size](d_a, d_out)
    metal.synchronize()
    result = d_out.copy_to_host().reshape(n_groups, group_size)
    for g in range(n_groups):
        assert np.all(result[g] == float(g + 1))
