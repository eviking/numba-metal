"""Real-Metal integration tests for command-buffer tracking (Workstream
2): multiple real kernel launches tracked through to real completion,
successful cleanup of the outstanding-submission registry, and resource
lifetime across real asynchronous GPU execution.

Failure-status handling itself is unit-tested with a fake command buffer
in tests/unit/test_command_buffer_tracking.py (see that file's docstring
for why a deliberately failing real Metal submission cannot be
constructed safely/deterministically). This file only exercises the
success path against real hardware.

Requires a working Metal device; run with `pytest -m metal`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


def test_multiple_launches_then_one_synchronize_real_metal() -> None:
    metal = _metal()
    from numba_metal.runtime.context import get_context

    @metal.jit
    def increment(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] + 1.0

    ctx = get_context()
    a = np.zeros(1000, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)

    for _ in range(10):
        increment[4, 256](d_a, d_out)

    assert ctx.outstanding_count() >= 1
    metal.synchronize()
    assert ctx.outstanding_count() == 0
    # Correctness of the final launch's actual result (each launch reads
    # the same d_a, so all 10 produce the same output -- this also proves
    # the tracked buffers really executed, not just that no error fired).
    assert np.allclose(d_out.copy_to_host(), 1.0)


def test_repeated_synchronize_with_no_pending_work_real_metal() -> None:
    metal = _metal()
    metal.synchronize()
    metal.synchronize()
    metal.synchronize()


def test_outstanding_registry_drains_after_successful_completion() -> None:
    metal = _metal()
    from numba_metal.runtime.context import get_context

    @metal.jit
    def noop_write(out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = 1

    ctx = get_context()
    d_out = metal.device_array(64, np.int32)
    noop_write[1, 64](d_out)
    metal.synchronize()
    assert ctx.outstanding_count() == 0
    result = d_out.copy_to_host()
    assert np.all(result == 1)


def test_resource_lifetime_through_real_async_completion() -> None:
    """Buffers used by a launch must survive until the GPU actually
    finishes with them, even if this test doesn't hold any extra
    reference beyond the DeviceNDArray objects themselves and the
    dispatcher's internal tracking."""
    metal = _metal()

    @metal.jit
    def scale(a, out, factor):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] * factor

    a = np.arange(10_000, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    scale[64, 256](d_a, d_out, np.float32(3.0))
    # No explicit synchronize() here: copy_to_host() must itself
    # synchronize first (Workstream 3), and the tracked resources
    # (argument buffers, scalar constant buffer for `factor`) must still
    # be alive for the GPU to actually read when it gets scheduled.
    result = d_out.copy_to_host()
    assert np.allclose(result, a * 3.0)


def test_many_repeated_launches_do_not_leak_unbounded_outstanding_entries() -> None:
    """A long-running loop of launches interspersed with periodic
    synchronize() calls should not grow the outstanding registry without
    bound -- each synchronize() should fully drain what's pending at that
    point."""
    metal = _metal()
    from numba_metal.runtime.context import get_context

    @metal.jit
    def add_one(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] + 1.0

    ctx = get_context()
    a = np.zeros(256, dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)

    for batch in range(5):
        for _ in range(20):
            add_one[1, 256](d_a, d_out)
        metal.synchronize()
        assert ctx.outstanding_count() == 0, f"leak after batch {batch}"
