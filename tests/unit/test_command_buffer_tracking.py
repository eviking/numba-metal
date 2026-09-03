"""Unit tests for command-buffer failure-status handling, using a faithful
fake `MTLCommandBuffer`-like object -- no GPU required.

A deliberately, deterministically failing *real* Metal submission could
not be constructed safely: out-of-bounds GPU memory access is undefined
behavior on Apple GPUs and does not reliably surface as a command-buffer
error status (verified empirically -- see docs/architecture.md), and
deliberately relying on undefined behavior to test error handling would
itself be an unsafe, non-reproducible test. Per the assignment's own
fallback guidance, failure-status handling is unit-tested here against a
fake command buffer whose `status()`/`error()` are fully controlled, and
real-hardware success-path tracking/cleanup is tested separately in
tests/integration/test_command_buffer_tracking_metal.py.
"""

from __future__ import annotations

import threading

import pytest

from numba_metal.errors import MetalRuntimeError
from numba_metal.runtime.context import _MetalContext


class _FakeCommandBuffer:
    """A minimal stand-in for MTLCommandBuffer with a scriptable
    status()/error(), so failure handling can be tested without any GPU
    or Metal framework dependency."""

    def __init__(self, status: int, error=None):
        self._status = status
        self._error = error
        self.wait_calls = 0

    def waitUntilCompleted(self):
        self.wait_calls += 1

    def status(self):
        return self._status

    def error(self):
        return self._error


_STATUS_COMPLETED = 4
_STATUS_ERROR = 5


def _fresh_context_with_fake_device() -> _MetalContext:
    """A _MetalContext whose device/queue init is bypassed (set directly)
    so registration/synchronization logic can be tested without touching
    real Metal or requiring a GPU."""
    ctx = _MetalContext()
    ctx._device = object()  # sentinel, never dereferenced by these tests
    ctx._queue = object()
    ctx._info = object()
    return ctx


def test_single_successful_submission_clears_outstanding() -> None:
    ctx = _fresh_context_with_fake_device()
    cb = _FakeCommandBuffer(_STATUS_COMPLETED)
    ctx.register_submission(cb, "my_kernel", resources=[])
    assert ctx.outstanding_count() == 1
    ctx.synchronize()
    assert ctx.outstanding_count() == 0
    assert cb.wait_calls == 1


def test_multiple_launches_then_one_synchronize() -> None:
    ctx = _fresh_context_with_fake_device()
    buffers = [_FakeCommandBuffer(_STATUS_COMPLETED) for _ in range(5)]
    for i, cb in enumerate(buffers):
        ctx.register_submission(cb, f"kernel_{i}", resources=[])
    assert ctx.outstanding_count() == 5
    ctx.synchronize()
    assert ctx.outstanding_count() == 0
    assert all(cb.wait_calls == 1 for cb in buffers)


def test_repeated_synchronize_with_no_pending_work_is_a_noop() -> None:
    ctx = _fresh_context_with_fake_device()
    ctx.synchronize()
    ctx.synchronize()
    ctx.synchronize()
    assert ctx.outstanding_count() == 0


def test_synchronize_does_not_rewait_already_cleared_buffers() -> None:
    ctx = _fresh_context_with_fake_device()
    cb = _FakeCommandBuffer(_STATUS_COMPLETED)
    ctx.register_submission(cb, "k", resources=[])
    ctx.synchronize()
    assert cb.wait_calls == 1
    ctx.synchronize()  # nothing new pending; must not wait on cb again
    assert cb.wait_calls == 1


def test_resource_lifetime_extends_through_synchronize() -> None:
    ctx = _fresh_context_with_fake_device()
    released = []

    class _Tracked:
        def __init__(self, name):
            self.name = name

        def __del__(self):
            released.append(self.name)

    cb = _FakeCommandBuffer(_STATUS_COMPLETED)
    resource = _Tracked("buf_a")
    ctx.register_submission(cb, "k", resources=[resource])
    del resource  # drop the caller's own reference
    # The record inside ctx must be the only thing keeping it alive until
    # synchronize() -- if resources were dropped early, this would already
    # be finalized.
    assert released == []
    ctx.synchronize()
    # After synchronize() completes and the record is dropped, the
    # resource may now be collected (CPython: immediately, by refcount).
    del cb
    import gc

    gc.collect()
    assert released == ["buf_a"]


def test_failing_submission_raises_metal_runtime_error_naming_kernel() -> None:
    ctx = _fresh_context_with_fake_device()
    fake_error = "simulated MTLCommandBufferError: out of memory"
    cb = _FakeCommandBuffer(_STATUS_ERROR, error=fake_error)
    ctx.register_submission(cb, "failing_kernel", resources=[])
    with pytest.raises(MetalRuntimeError) as exc_info:
        ctx.synchronize()
    message = str(exc_info.value)
    assert "failing_kernel" in message
    assert "submission #" in message
    assert fake_error in message


def test_failing_submission_error_includes_sequence_number() -> None:
    ctx = _fresh_context_with_fake_device()
    ok1 = _FakeCommandBuffer(_STATUS_COMPLETED)
    bad = _FakeCommandBuffer(_STATUS_ERROR, error="boom")
    ctx.register_submission(ok1, "ok_kernel_1", resources=[])
    seq_bad = ctx.register_submission(bad, "bad_kernel", resources=[]).sequence
    with pytest.raises(MetalRuntimeError) as exc_info:
        ctx.synchronize()
    assert f"submission #{seq_bad}" in str(exc_info.value)


def test_outstanding_cleared_even_when_a_failure_is_raised() -> None:
    """A failed synchronize() must still drain the outstanding list --
    otherwise a subsequent synchronize() would re-wait on (and re-raise
    for) a buffer whose failure was already reported once, which would
    look like a hang or a duplicate error rather than a clean failure."""
    ctx = _fresh_context_with_fake_device()
    cb = _FakeCommandBuffer(_STATUS_ERROR, error="boom")
    ctx.register_submission(cb, "bad_kernel", resources=[])
    with pytest.raises(MetalRuntimeError):
        ctx.synchronize()
    assert ctx.outstanding_count() == 0
    # A second synchronize() must not re-raise or hang.
    ctx.synchronize()


def test_earlier_failure_not_hidden_by_later_success() -> None:
    """If an early buffer fails but a later one (e.g. a subsequent
    barrier-like no-op) succeeds, synchronize() must still report the
    earlier failure -- this is the exact bug the barrier-only design had:
    a successful barrier could make an earlier real failure invisible."""
    ctx = _fresh_context_with_fake_device()
    bad = _FakeCommandBuffer(_STATUS_ERROR, error="first one failed")
    good = _FakeCommandBuffer(_STATUS_COMPLETED)
    ctx.register_submission(bad, "first_kernel", resources=[])
    ctx.register_submission(good, "second_kernel", resources=[])
    with pytest.raises(MetalRuntimeError) as exc_info:
        ctx.synchronize()
    assert "first_kernel" in str(exc_info.value)
    # Both buffers must have been waited on regardless of order.
    assert bad.wait_calls == 1
    assert good.wait_calls == 1


def test_multiple_failures_all_reported() -> None:
    ctx = _fresh_context_with_fake_device()
    bad1 = _FakeCommandBuffer(_STATUS_ERROR, error="err1")
    bad2 = _FakeCommandBuffer(_STATUS_ERROR, error="err2")
    ctx.register_submission(bad1, "kernel_a", resources=[])
    ctx.register_submission(bad2, "kernel_b", resources=[])
    with pytest.raises(MetalRuntimeError) as exc_info:
        ctx.synchronize()
    message = str(exc_info.value)
    assert "kernel_a" in message
    assert "kernel_b" in message


def test_thread_safety_of_registration_and_synchronize() -> None:
    """Concurrent registration from multiple threads must not corrupt the
    outstanding list or lose a submission (register_submission/synchronize
    both take the dedicated _outstanding_lock)."""
    ctx = _fresh_context_with_fake_device()
    n_threads = 8
    per_thread = 20
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()
        for _i in range(per_thread):
            cb = _FakeCommandBuffer(_STATUS_COMPLETED)
            ctx.register_submission(cb, "concurrent_kernel", resources=[])

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert ctx.outstanding_count() == n_threads * per_thread
    ctx.synchronize()
    assert ctx.outstanding_count() == 0
