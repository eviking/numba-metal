"""Real-Metal integration tests for the host/device synchronization model
(Workstream 3): `copy_to_device`/`copy_to_host` must not race outstanding
GPU work, per the conservative "every host touch synchronizes first"
model documented in `numba_metal/runtime/array.py` and
`docs/architecture.md`.

Requires a working Metal device; run with `pytest -m metal`.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


# 1. Launch a kernel and immediately call copy_to_device() on an involved
# buffer without explicitly calling metal.synchronize().
def test_copy_to_device_without_explicit_sync_is_still_correct() -> None:
    metal = _metal()

    @metal.jit
    def slow_accumulate(a, out, n):
        i = metal.grid(1)
        if i < out.size:
            acc = 0.0
            for _j in range(n):
                acc = acc + a[i]
            out[i] = acc

    n_elems = 500_000
    a = np.ones(n_elems, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)

    # Launch a kernel that keeps reading d_a for a while (no synchronize).
    slow_accumulate[2000, 256](d_a, d_out, 2000)

    # Immediately overwrite d_a from the host -- copy_to_device() must
    # synchronize first so this write cannot race the still-running
    # kernel's reads of the old contents.
    new_a = np.full(n_elems, 5.0, dtype=np.float32)
    d_a.copy_to_device(new_a)

    # The first kernel must have seen the ORIGINAL a (all ones) throughout
    # -- not a torn mix of old and new values -- because copy_to_device()
    # waited for it to finish before overwriting.
    metal.synchronize()
    first_result = d_out.copy_to_host()
    assert np.allclose(first_result, 2000.0), (
        "first kernel's output shows it read a torn/mixed 'a' buffer "
        f"instead of the original all-ones values: {first_result[:5]}"
    )

    # A second launch against the now-overwritten d_a must see the new
    # values consistently.
    slow_accumulate[2000, 256](d_a, d_out, 2000)
    metal.synchronize()
    second_result = d_out.copy_to_host()
    assert np.allclose(second_result, 2000.0 * 5.0)


# 2. Launch a kernel and immediately call copy_to_host().
def test_copy_to_host_without_explicit_sync_is_still_correct() -> None:
    metal = _metal()

    @metal.jit
    def slow_fill(out, n):
        i = metal.grid(1)
        if i < out.size:
            acc = 0.0
            for _j in range(n):
                acc = acc + 1.0
            out[i] = acc

    d_out = metal.device_array(500_000, np.float32)
    slow_fill[2000, 256](d_out, 3000)
    # No explicit metal.synchronize() -- copy_to_host() must synchronize
    # internally and return the fully-computed result, never a partial
    # or garbage value from reading mid-computation.
    result = d_out.copy_to_host()
    assert np.allclose(result, 3000.0)


# 3. Reuse a device array across several asynchronously submitted kernels.
def test_device_array_reused_across_multiple_async_kernels() -> None:
    metal = _metal()

    @metal.jit
    def increment(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] + 1.0

    a = np.zeros(10_000, dtype=np.float32)
    d_a = metal.to_device(a)
    d_b = metal.device_array_like(a)
    d_c = metal.device_array_like(a)

    # Chain: d_a -> d_b -> d_c -> d_a, all launched without intermediate
    # synchronize() calls; each stage reuses a buffer written by the
    # previous stage while it may still be "in flight" from the queue's
    # perspective.
    increment[40, 256](d_a, d_b)
    increment[40, 256](d_b, d_c)
    increment[40, 256](d_c, d_a)
    increment[40, 256](d_a, d_b)
    result = d_b.copy_to_host()  # implicit synchronize
    assert np.allclose(result, 4.0)


# 4. Delete Python references to input arrays before synchronization and
# verify execution remains correct.
def test_deleting_host_reference_before_sync_does_not_corrupt_execution() -> None:
    metal = _metal()

    @metal.jit
    def scale(a, out, factor):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] * factor

    a = np.arange(200_000, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    scale[800, 256](d_a, d_out, np.float32(2.0))

    # Drop every host-side reference to the original NumPy array before
    # synchronizing; the DeviceNDArray/MTLBuffer's own lifetime (retained
    # by the SubmissionRecord -- see Workstream 2) must be independent of
    # this and keep the GPU-visible data intact.
    del a
    gc.collect()

    metal.synchronize()
    result = d_out.copy_to_host()
    expected = np.arange(200_000, dtype=np.float32) * 2.0
    assert np.allclose(result, expected)


# 5. Repeated host/device updates in a loop.
def test_repeated_host_device_updates_in_a_loop() -> None:
    metal = _metal()

    @metal.jit
    def add_one(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] + 1.0

    d_a = metal.device_array(1000, np.float32)
    host_val = 0.0
    for iteration in range(20):
        host_arr = np.full(1000, host_val, dtype=np.float32)
        d_a.copy_to_device(host_arr)
        d_out = metal.device_array_like(host_arr)
        add_one[4, 256](d_a, d_out)
        result = d_out.copy_to_host()
        assert np.allclose(result, host_val + 1.0), f"iteration {iteration}"
        host_val = float(result[0])


# 6. Copying into a supplied host output array.
def test_copy_to_host_into_supplied_output_array() -> None:
    metal = _metal()

    @metal.jit
    def double_it(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] * 2.0

    a = np.arange(1000, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    double_it[4, 256](d_a, d_out)

    preallocated = np.empty(1000, dtype=np.float32)
    returned = d_out.copy_to_host(out=preallocated)
    assert returned is preallocated
    assert np.allclose(preallocated, a * 2.0)


# 7. Shape and dtype mismatch behavior.
def test_copy_to_device_shape_mismatch_raises() -> None:
    metal = _metal()
    from numba_metal.errors import MetalRuntimeError

    d_a = metal.device_array(100, np.float32)
    wrong_shape = np.zeros((50,), dtype=np.float32)
    with pytest.raises(MetalRuntimeError, match="Shape mismatch"):
        d_a.copy_to_device(wrong_shape)


def test_copy_to_device_dtype_is_cast_not_silently_corrupted() -> None:
    """copy_to_device casts via np.ascontiguousarray(..., dtype=self.dtype)
    -- verify this produces a correct, intentional cast (not silent byte
    reinterpretation) for a same-shape, different-dtype host array."""
    metal = _metal()

    d_a = metal.device_array(10, np.float32)
    int_host = np.arange(10, dtype=np.int32)
    d_a.copy_to_device(int_host)
    result = d_a.copy_to_host()
    assert np.allclose(result, int_host.astype(np.float32))
