"""Standalone tests for the scalar-argument buffer-reuse pool
(`runtime/context.py`'s `acquire_scalar_buffer`/`_release_scalar_buffers`)
and `metal.batch()` command-buffer batching. General Numba-to-Metal
runtime capabilities, no connection to any specific downstream workload.

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


# -- scalar-buffer reuse pool ----------------------------------------------


def test_repeated_launches_with_varying_scalars_are_correct() -> None:
    """The core correctness property buffer reuse must never violate:
    each launch's scalar argument value must be the one actually passed
    to THAT launch, never a stale value left over from a pooled buffer's
    previous use."""
    metal = _metal()

    @metal.jit
    def kernel(a, out, scale):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] * scale

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    for scale in (2.0, 3.0, 4.0, 5.0, 0.5):
        kernel[1, 3](d_a, d_out, np.float32(scale))
        metal.synchronize()
        assert np.allclose(d_out.copy_to_host(), a * scale)


def test_scalar_buffer_pool_stabilizes_not_grows_unboundedly() -> None:
    """After enough synchronized launches of the same kernel signature,
    the pool must reuse existing buffers rather than accumulating a new
    one per launch -- verified by pool size staying bounded across many
    repetitions, not merely "small on the first check"."""
    from numba_metal.runtime.context import get_context

    metal = _metal()

    @metal.jit
    def kernel(a, out, scale):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] * scale

    ctx = get_context()
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)

    # One untimed launch first to reach this test's own steady state
    # (the pool is process-wide/shared across the whole test session --
    # see context.py -- so its absolute size also reflects every OTHER
    # test that ran earlier; only growth *within this test*, after this
    # kernel's own buffer sizes are already pooled, is meaningful).
    kernel[1, 3](d_a, d_out, np.float32(0))
    metal.synchronize()
    steady_state_size = ctx.scalar_buffer_pool_size()

    for i in range(1, 50):
        kernel[1, 3](d_a, d_out, np.float32(i))
        metal.synchronize()

    assert ctx.scalar_buffer_pool_size() == steady_state_size


def test_unsynchronized_rapid_launches_remain_correct() -> None:
    """Buffers can only be reused after synchronize() confirms they are
    safe (see context.py's pooling docstring); back-to-back launches
    with NO synchronize() in between must still each get correct,
    independent scalar values -- verifying the pool never hands out a
    buffer that might still be in flight."""
    metal = _metal()

    @metal.jit
    def kernel(a, out, scale):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] * scale

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)

    for i in range(20):
        kernel[1, 3](d_a, d_out, np.float32(i))
    metal.synchronize()
    # Only the LAST launch's scale should be reflected (the serial
    # command queue guarantees in-order execution; each launch
    # overwrites d_out).
    assert np.allclose(d_out.copy_to_host(), a * 19.0)


def test_buffer_pool_correct_across_multiple_kernel_signatures() -> None:
    """Two kernels with different scalar-argument byte sizes must not
    have their pooled buffers cross-contaminate (e.g. an int32 scalar
    buffer being handed out where a float32 one -- same byte size,
    different meaning -- is expected is actually fine since the pool key
    is byte size only and the buffer is fully overwritten before use;
    this test verifies that overwrite is genuinely complete)."""
    metal = _metal()

    @metal.jit
    def kernel_float(a, out, scale):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] * scale

    @metal.jit
    def kernel_int(a, out, offset):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] + offset

    a_f = np.array([1.0, 2.0], dtype=np.float32)
    a_i = np.array([10, 20], dtype=np.int32)
    d_a_f = metal.to_device(a_f)
    d_out_f = metal.device_array_like(a_f)
    d_a_i = metal.to_device(a_i)
    d_out_i = metal.device_array_like(a_i)

    for _ in range(5):
        kernel_float[1, 2](d_a_f, d_out_f, np.float32(3.0))
        metal.synchronize()
        kernel_int[1, 2](d_a_i, d_out_i, np.int32(7))
        metal.synchronize()

    assert np.allclose(d_out_f.copy_to_host(), a_f * 3.0)
    assert np.array_equal(d_out_i.copy_to_host(), a_i + 7)


# -- metal.batch() ----------------------------------------------------------


def test_batch_two_kernels_produces_one_submission() -> None:
    from numba_metal.runtime.context import get_context

    metal = _metal()

    @metal.jit
    def add_one(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] + 1.0

    @metal.jit
    def double_it(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] * 2.0

    ctx = get_context()
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_mid = metal.device_array_like(a)
    d_out = metal.device_array_like(a)

    outstanding_before = ctx.outstanding_count()
    with metal.batch():
        add_one[1, 3](d_a, d_mid)
        double_it[1, 3](d_mid, d_out)
    outstanding_after = ctx.outstanding_count()

    assert outstanding_after == outstanding_before + 1
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), (a + 1.0) * 2.0)


def test_batch_preserves_in_order_execution_dependency() -> None:
    """A later launch in a batch reading an earlier launch's output must
    see that output correctly -- proving the batched encodes on one
    command buffer still execute in submission order, not concurrently
    or out of order."""
    metal = _metal()

    @metal.jit
    def stage1(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] + 10.0

    @metal.jit
    def stage2(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] * 3.0

    @metal.jit
    def stage3(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] - 1.0

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_b = metal.device_array_like(a)
    d_c = metal.device_array_like(a)
    d_out = metal.device_array_like(a)

    with metal.batch():
        stage1[1, 4](d_a, d_b)
        stage2[1, 4](d_b, d_c)
        stage3[1, 4](d_c, d_out)
    metal.synchronize()

    expected = ((a + 10.0) * 3.0) - 1.0
    assert np.allclose(d_out.copy_to_host(), expected)


def test_batch_many_launches() -> None:
    """A larger batch (more than 2-3 launches) to exercise accumulation
    of resources/reusable buffers across many dispatches onto one
    command buffer."""
    metal = _metal()

    @metal.jit
    def increment(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] + 1.0

    a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    d_current = metal.to_device(a)
    d_next = metal.device_array_like(a)

    n_launches = 20
    with metal.batch():
        for _ in range(n_launches):
            increment[1, 3](d_current, d_next)
            d_current, d_next = d_next, d_current
    metal.synchronize()

    assert np.allclose(d_current.copy_to_host(), a + n_launches)


def test_batch_nesting_is_rejected() -> None:
    from numba_metal.errors import MetalRuntimeError

    metal = _metal()

    with metal.batch():
        with pytest.raises(MetalRuntimeError):
            with metal.batch():
                pass


def test_batch_exception_discards_uncommitted_work() -> None:
    """If an exception propagates out of a `metal.batch()` block, none
    of the launches encoded so far must run, and no phantom submission
    may appear -- and the thread-local batch state must not be left
    stuck, so subsequent (non-batched) launches still work normally."""
    from numba_metal.runtime.context import current_batch, get_context

    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] + 1.0

    a = np.array([1.0, 2.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.to_device(np.array([-99.0, -99.0], dtype=np.float32))

    ctx = get_context()
    outstanding_before = ctx.outstanding_count()

    with pytest.raises(ValueError):
        with metal.batch():
            kernel[1, 2](d_a, d_out)
            raise ValueError("simulated user error mid-batch")

    assert ctx.outstanding_count() == outstanding_before
    assert current_batch() is None

    # The discarded batch's launch must not have run: d_out should still
    # hold its original sentinel value once we synchronize (there is
    # nothing outstanding to wait on, but this also confirms no stray
    # completed-but-unaccounted work silently mutated it).
    metal.synchronize()
    assert np.array_equal(
        d_out.copy_to_host(), np.array([-99.0, -99.0], dtype=np.float32)
    )

    # Normal (non-batched) launches must still work after a discarded batch.
    kernel[1, 2](d_a, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), a + 1.0)


def test_empty_batch_is_a_no_op() -> None:
    from numba_metal.runtime.context import get_context

    metal = _metal()
    ctx = get_context()
    outstanding_before = ctx.outstanding_count()

    with metal.batch():
        pass  # no launches at all

    assert ctx.outstanding_count() == outstanding_before
    metal.synchronize()  # must not raise / hang on an empty batch


def test_batch_combined_with_buffer_reuse_across_rounds() -> None:
    """Buffer reuse and batching must compose correctly: buffers used
    inside a batch only return to the pool once the WHOLE batch's single
    SubmissionRecord is confirmed complete, not per-launch-within-the
    -batch."""
    from numba_metal.runtime.context import get_context

    metal = _metal()

    @metal.jit
    def scale_by(a, out, scale):
        i = metal.grid(1)
        if i < a.size:
            out[i] = a[i] * scale

    ctx = get_context()
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)

    # Run one round first (untimed) so the pool already contains
    # whatever this test's own buffer sizes need before measuring
    # growth -- the pool is process-wide/shared across the whole test
    # session (see context.py), so its absolute size at any point also
    # reflects every OTHER test that happened to run earlier in this
    # session; only growth *within this test*, after its own steady
    # state is reached, is a meaningful assertion.
    with metal.batch():
        scale_by[1, 3](d_a, d_out, np.float32(1.0))
        scale_by[1, 3](d_out, d_out, np.float32(2.0))
    metal.synchronize()
    steady_state_size = ctx.scalar_buffer_pool_size()

    for round_num in range(1, 6):
        with metal.batch():
            scale_by[1, 3](d_a, d_out, np.float32(round_num + 1))
            scale_by[1, 3](d_out, d_out, np.float32(2.0))
        metal.synchronize()
        expected = a * (round_num + 1) * 2.0
        assert np.allclose(d_out.copy_to_host(), expected)

    # Pool must have stabilized after the first round, not grown further
    # per subsequent round.
    assert ctx.scalar_buffer_pool_size() == steady_state_size
