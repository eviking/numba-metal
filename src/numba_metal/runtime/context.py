"""Process-wide Metal device/command-queue context, with explicit
tracking of every submitted command buffer through to completion.

There is one MTLDevice and one MTLCommandQueue per process, created lazily
on first use and reused for the process lifetime. This is one of two
pieces of module-level mutable state in numba-metal (the other being the
outstanding-submission list below); it mirrors how every GPU runtime
(CUDA, Metal itself) treats the device/queue as a process-wide resource,
and avoids the alternative of threading a context object through every
public API call for no practical benefit in a single-GPU MVP.

# Command-buffer tracking and error propagation

The MVP originally implemented `metal.synchronize()` by committing a
separate, empty "barrier" command buffer and waiting on it, relying on
the single serial MTLCommandQueue's in-order execution guarantee to infer
that everything submitted earlier had also finished. That establishes
*ordering* correctly, but it does not *inspect* the status of any of the
actual kernel command buffers that ran before the barrier -- so a kernel
that failed (driver-level Metal error, not a GPU-side logic bug) could
have its error silently discarded as long as the barrier itself
succeeded, which it always would (an empty command buffer with no work
has nothing that can fail).

This module replaces that with explicit tracking: every submitted
command buffer is registered in `_MetalContext._outstanding` as a
`SubmissionRecord` (kernel name, the command buffer itself, the argument
resources that must stay alive until it completes, a monotonic sequence
number, and the MSL source for diagnostics), retained for as long as it
is outstanding. `synchronize()` waits on *every* outstanding command
buffer (not just a barrier), checks each one's `status`/`error`
individually, and raises `MetalRuntimeError` naming the specific kernel
and submission sequence number that failed -- and it does this by
inspecting `MTLCommandBuffer.status()`/`.error()` synchronously in the
calling thread after `waitUntilCompleted()` returns, not by relying on a
Python exception raised inside an `addCompletedHandler_` callback (Metal
invokes completion handlers on its own internal dispatch queue; an
exception raised there does not propagate to the Python thread that
called `synchronize()` -- pyobjc has no mechanism to re-raise it there,
so relying on that would silently swallow the very errors this subsystem
exists to surface).
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from numba_metal.errors import MetalRuntimeError
from numba_metal.runtime.device import check_capable

# MTLCommandBufferStatus.MTLCommandBufferStatusError
_MTL_COMMAND_BUFFER_STATUS_ERROR = 5

_sequence_counter = itertools.count(1)

# Optional profiling hooks, installed by numba_metal.advisor.profiler.
# Empty by default -- normal execution never calls into these lists, so
# there is no cost when profiling is not active beyond the two `if`
# checks below (see advisor/metal_events.py's module docstring and
# docs/advisor.md's "How to add new instrumentation events" for the
# full hook contract). Nothing here changes register_submission's or
# synchronize's existing behavior or return value.
_submission_hooks: list[Any] = []
_sync_hooks: list[Any] = []


@dataclass
class SubmissionRecord:
    """Bookkeeping for one submitted (but not yet confirmed-complete)
    command buffer.

    `resources` retains every Metal object (argument buffers, the
    pipeline state) that must not be garbage-collected before the GPU
    finishes reading/writing them -- Metal does not extend Python object
    lifetimes on its own; only pyobjc's normal reference counting does,
    so this list is what keeps them alive across the async GPU execution
    window between `commit()` and `waitUntilCompleted()`.

    `reusable_buffers` is the subset of `resources` (if any) that are
    eligible to be returned to `_MetalContext`'s scalar-buffer pool once
    this submission is confirmed complete (see
    `runtime/dispatcher.py`'s buffer-reuse pool) -- kept as a separate
    list rather than reusing all of `resources`, since `resources` also
    includes the pipeline state and device-array buffers, which are
    never pool-managed (a device array's buffer is owned by its
    `DeviceNDArray` for that array's entire lifetime, not per-dispatch).
    Each entry is `(byte_size, buffer)`.
    """

    sequence: int
    kernel_name: str
    command_buffer: Any
    resources: list[Any] = field(default_factory=list)
    msl_source: str | None = None
    reusable_buffers: list[tuple[int, Any]] = field(default_factory=list)


class _MetalContext:
    """Lazily-initialized holder for the MTLDevice and MTLCommandQueue,
    and the registry of outstanding command-buffer submissions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._device = None
        self._queue = None
        self._info = None
        # Guards _outstanding specifically (separate from the
        # initialization lock above, since submission/synchronization
        # happen far more often than one-time device setup and should not
        # contend with each other unnecessarily).
        self._outstanding_lock = threading.Lock()
        self._outstanding: list[SubmissionRecord] = []
        # Process-wide pool of small scalar-argument MTLBuffers
        # (byte_size -> free buffers of that exact size), reused across
        # dispatches instead of allocating a fresh MTLBuffer for every
        # scalar kernel argument on every single launch (see
        # runtime/dispatcher.py's `_acquire_scalar_buffer`). A buffer
        # only re-enters this pool once `synchronize()` has confirmed
        # every outstanding command buffer that might still be reading
        # it has completed -- see `synchronize()` below and
        # `SubmissionRecord.reusable_buffers`. This is the ONLY point at
        # which a pooled buffer becomes available again, so reuse never
        # races an in-flight kernel that could still be reading the old
        # contents: the exact same "wait before touching shared memory"
        # discipline already used for `copy_to_host`/`copy_to_device`
        # (see runtime/array.py), applied here to buffer *recycling*
        # rather than host access.
        self._scalar_buffer_pool_lock = threading.Lock()
        self._scalar_buffer_pool: dict[int, list[Any]] = {}

    def _ensure_initialized(self) -> None:
        if self._device is not None:
            return
        with self._lock:
            if self._device is not None:
                return
            info = check_capable()
            import Metal

            device = Metal.MTLCreateSystemDefaultDevice()
            if device is None:
                raise MetalRuntimeError("Failed to create Metal device.")
            queue = device.newCommandQueue()
            if queue is None:
                raise MetalRuntimeError("Failed to create Metal command queue.")
            self._device = device
            self._queue = queue
            self._info = info

    @property
    def device(self):
        """The process-wide MTLDevice, created on first access."""
        self._ensure_initialized()
        return self._device

    @property
    def queue(self):
        """The process-wide MTLCommandQueue, created on first access."""
        self._ensure_initialized()
        return self._queue

    @property
    def info(self):
        """DeviceInfo for the selected Metal device."""
        self._ensure_initialized()
        return self._info

    def register_submission(
        self,
        command_buffer: Any,
        kernel_name: str,
        resources: list[Any],
        msl_source: str | None = None,
        reusable_buffers: list[tuple[int, Any]] | None = None,
    ) -> SubmissionRecord:
        """Register a just-submitted (already committed, or about to be
        committed by the caller) command buffer for tracking. Must be
        called for every command buffer numba-metal submits, before or
        immediately after `commit()`, so `synchronize()` can find it.

        `reusable_buffers`: `(byte_size, buffer)` pairs eligible to be
        returned to the scalar-buffer pool once this submission is
        confirmed complete with no error (see `synchronize()` and
        `acquire_scalar_buffer`).
        """
        record = SubmissionRecord(
            sequence=next(_sequence_counter),
            kernel_name=kernel_name,
            command_buffer=command_buffer,
            resources=resources,
            msl_source=msl_source,
            reusable_buffers=reusable_buffers or [],
        )
        with self._outstanding_lock:
            self._outstanding.append(record)
        if _submission_hooks:
            for hook in _submission_hooks:
                hook(record)
        return record

    def acquire_scalar_buffer(self, byte_size: int):
        """Pop a pooled MTLBuffer of exactly `byte_size` bytes, or return
        `None` if the pool has none available (caller must allocate a
        fresh one). See `__init__`'s docstring for why a pooled buffer is
        always safe to reuse the moment it is popped -- it can only have
        been placed here by `synchronize()`, after confirming no
        outstanding command buffer might still be reading it."""
        with self._scalar_buffer_pool_lock:
            bucket = self._scalar_buffer_pool.get(byte_size)
            if bucket:
                return bucket.pop()
        return None

    def _release_scalar_buffers(self, buffers: list[tuple[int, Any]]) -> None:
        with self._scalar_buffer_pool_lock:
            for byte_size, buf in buffers:
                self._scalar_buffer_pool.setdefault(byte_size, []).append(buf)

    def scalar_buffer_pool_size(self) -> int:
        """Total number of buffers currently sitting in the reuse pool
        (across all sizes) -- exposed for tests verifying pooling
        behavior, not part of the normal execution path."""
        with self._scalar_buffer_pool_lock:
            return sum(len(v) for v in self._scalar_buffer_pool.values())

    def synchronize(self) -> None:
        """Block until all previously submitted command buffers have
        completed, then inspect each one's status and raise
        `MetalRuntimeError` naming the specific kernel and submission
        sequence number for the *first* one that failed -- checking every
        one of them, not just the most recently submitted, so an earlier
        failure cannot be hidden by a later, unrelated buffer completing
        successfully.

        Safe to call with no pending work (a no-op) and safe to call
        repeatedly (each call only waits on buffers submitted since the
        last successful synchronize -- completed records are removed
        below, so there is nothing to double-wait-on and no deadlock risk
        from waiting on an already-completed buffer, which returns
        immediately).
        """
        self._ensure_initialized()
        with self._outstanding_lock:
            pending = list(self._outstanding)
            self._outstanding.clear()

        # Timed unconditionally (one cheap monotonic-clock pair) so a
        # profiling hook installed mid-call still sees a start time; the
        # sync_start/sync_duration values are only ever consumed if
        # _sync_hooks is non-empty (see the hook-firing block below,
        # reached only on the success path -- matching this method's
        # existing raise-or-return contract exactly).
        sync_start_ns = time.monotonic_ns()
        failures: list[tuple[SubmissionRecord, Any]] = []
        reclaimed_buffers: list[tuple[int, Any]] = []
        for record in pending:
            record.command_buffer.waitUntilCompleted()
            status = record.command_buffer.status()
            if status == _MTL_COMMAND_BUFFER_STATUS_ERROR:
                failures.append((record, record.command_buffer.error()))
            elif record.reusable_buffers:
                # Only a cleanly-completed submission's buffers are
                # returned to the pool -- a buffer touched by a failed
                # command buffer is in an unknown/ambiguous state (the
                # GPU may have partially written it, or the failure may
                # itself be buffer-related), so it is simply dropped
                # (garbage-collected normally) rather than risked in a
                # future dispatch.
                reclaimed_buffers.extend(record.reusable_buffers)
        sync_duration_ns = time.monotonic_ns() - sync_start_ns
        if _sync_hooks and pending:
            for hook in _sync_hooks:
                hook(sync_start_ns, sync_duration_ns, len(pending))

        if reclaimed_buffers:
            self._release_scalar_buffers(reclaimed_buffers)

        # Resources are released here (by falling out of scope) only after
        # every buffer has been confirmed complete -- this is what
        # prevents a use-after-free/race on shared-storage-mode buffers
        # the GPU might still be touching; see runtime/array.py for the
        # host-side half of this synchronization boundary.

        if failures:
            first_record, first_error = failures[0]
            detail = f": {first_error}" if first_error is not None else ""
            others = ""
            if len(failures) > 1:
                other_names = ", ".join(
                    f"{r.kernel_name!r} (submission #{r.sequence})"
                    for r, _ in failures[1:]
                )
                others = (
                    f" ({len(failures) - 1} other failure(s) also pending: "
                    f"{other_names})"
                )
            raise MetalRuntimeError(
                f"Metal command buffer for kernel {first_record.kernel_name!r} "
                f"(submission #{first_record.sequence}) failed{detail}.{others}"
            )

    def outstanding_count(self) -> int:
        """Number of command buffers submitted but not yet confirmed
        complete. Exposed for tests verifying tracking/cleanup behavior."""
        with self._outstanding_lock:
            return len(self._outstanding)


_context = _MetalContext()


def get_context() -> _MetalContext:
    """Return the process-wide Metal context, initializing it if needed."""
    return _context


def add_submission_hook(hook) -> None:
    """Register a callable invoked with the `SubmissionRecord` every time
    `register_submission` is called (i.e. once per kernel launch, or once
    per `metal.batch()` block). Intended for `numba_metal.advisor`'s
    profiler; empty by default and never invoked unless something has
    called this. See module docstring's "_submission_hooks" note."""
    _submission_hooks.append(hook)


def add_sync_hook(hook) -> None:
    """Register a callable invoked as `hook(start_ns, duration_ns,
    pending_count)` every time `synchronize()` waits on at least one
    outstanding command buffer. Intended for `numba_metal.advisor`'s
    profiler; empty by default."""
    _sync_hooks.append(hook)


def clear_hooks() -> None:
    """Remove every registered submission/sync hook. Intended for
    `numba_metal.advisor.profiler.deactivate()`."""
    _submission_hooks.clear()
    _sync_hooks.clear()


class _BatchState:
    """Accumulates multiple kernel launches' encoded work onto ONE
    shared `MTLCommandBuffer`, committed and registered as a single
    `SubmissionRecord` when the batch ends (see `metal.batch()` in
    `metal.py`), instead of `dispatcher.py`'s normal one-command-buffer
    -per-launch behavior.

    Thread-local (via `_batch_state` below, a `threading.local`): two
    threads batching concurrently must never see or contend over each
    other's in-progress command buffer -- each thread gets its own
    independent batch state, matching this project's existing
    thread-safety conventions (the outstanding-submission registry and
    scalar-buffer pool are already lock-protected shared state; a batch
    -in-progress command buffer is instead kept OUT of shared state
    entirely, avoiding the need for a lock at all for the common case of
    one thread batching its own sequential launches).
    """

    def __init__(self, command_buffer: Any) -> None:
        self.command_buffer = command_buffer
        self.resources: list[Any] = []
        self.reusable_buffers: list[tuple[int, Any]] = []
        self.kernel_names: list[str] = []
        self.msl_sources: list[str] = []


_batch_state_local = threading.local()


def current_batch() -> _BatchState | None:
    """The calling thread's in-progress `metal.batch()` state, or `None`
    if the calling thread is not currently inside a `metal.batch()`
    block. Consulted by `dispatcher.py`'s `_dispatch` to decide whether
    to encode onto an existing shared command buffer instead of creating
    and committing its own."""
    return getattr(_batch_state_local, "state", None)


def begin_batch() -> _BatchState:
    """Start a new batch on the calling thread: allocates one shared
    `MTLCommandBuffer` that subsequent launches on this thread will
    encode onto, until `end_batch()` commits and registers it. Raises if
    a batch is already in progress on this thread (`metal.batch()`
    blocks are not reentrant/nestable -- see that function's docstring
    for why)."""
    if current_batch() is not None:
        raise MetalRuntimeError(
            "metal.batch() blocks cannot be nested on the same thread; "
            "a batch is already in progress."
        )
    ctx = get_context()
    cmdbuf = ctx.queue.commandBuffer()
    state = _BatchState(cmdbuf)
    _batch_state_local.state = state
    return state


def discard_batch() -> None:
    """Abandon the calling thread's in-progress batch without committing
    its command buffer -- used when an exception propagates out of a
    `metal.batch()` block (see `metal.py`'s `_BatchContextManager`).
    None of the launches encoded onto the batch's command buffer so far
    ever run: an uncommitted `MTLCommandBuffer` is simply released
    (garbage-collected normally) with no GPU-side effect at all."""
    _batch_state_local.state = None


def end_batch() -> SubmissionRecord | None:
    """End the calling thread's in-progress batch: commits the shared
    command buffer (if at least one kernel was launched inside the
    batch; an empty `metal.batch()` block commits nothing, matching
    `synchronize()`'s existing "safe no-op on nothing pending" contract)
    and registers exactly one `SubmissionRecord` covering every launch
    that was encoded onto it. Returns that record, or `None` if the
    batch was empty."""
    state = current_batch()
    if state is None:  # pragma: no cover - defensive; metal.batch() always pairs these
        raise MetalRuntimeError("end_batch() called with no batch in progress.")
    _batch_state_local.state = None
    if not state.kernel_names:
        return None
    ctx = get_context()
    state.command_buffer.commit()
    combined_name = (
        state.kernel_names[0]
        if len(state.kernel_names) == 1
        else f"batch[{', '.join(state.kernel_names)}]"
    )
    return ctx.register_submission(
        command_buffer=state.command_buffer,
        kernel_name=combined_name,
        resources=state.resources,
        msl_source="\n\n".join(state.msl_sources),
        reusable_buffers=state.reusable_buffers,
    )
