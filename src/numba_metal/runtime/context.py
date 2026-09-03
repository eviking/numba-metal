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
from dataclasses import dataclass, field
from typing import Any

from numba_metal.errors import MetalRuntimeError
from numba_metal.runtime.device import check_capable

# MTLCommandBufferStatus.MTLCommandBufferStatusError
_MTL_COMMAND_BUFFER_STATUS_ERROR = 5

_sequence_counter = itertools.count(1)


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
    """

    sequence: int
    kernel_name: str
    command_buffer: Any
    resources: list[Any] = field(default_factory=list)
    msl_source: str | None = None


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
    ) -> SubmissionRecord:
        """Register a just-submitted (already committed, or about to be
        committed by the caller) command buffer for tracking. Must be
        called for every command buffer numba-metal submits, before or
        immediately after `commit()`, so `synchronize()` can find it."""
        record = SubmissionRecord(
            sequence=next(_sequence_counter),
            kernel_name=kernel_name,
            command_buffer=command_buffer,
            resources=resources,
            msl_source=msl_source,
        )
        with self._outstanding_lock:
            self._outstanding.append(record)
        return record

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

        failures: list[tuple[SubmissionRecord, Any]] = []
        for record in pending:
            record.command_buffer.waitUntilCompleted()
            status = record.command_buffer.status()
            if status == _MTL_COMMAND_BUFFER_STATUS_ERROR:
                failures.append((record, record.command_buffer.error()))

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
