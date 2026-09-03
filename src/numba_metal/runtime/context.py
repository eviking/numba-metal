"""Process-wide Metal device/command-queue context.

There is one MTLDevice and one MTLCommandQueue per process, created lazily
on first use and reused for the process lifetime. This is the one piece of
module-level mutable state in numba-metal; it mirrors how every GPU runtime
(CUDA, Metal itself) treats the device/queue as a process-wide resource, and
avoids the alternative of threading a context object through every public
API call for no practical benefit in a single-GPU MVP.
"""

from __future__ import annotations

import threading

from numba_metal.errors import MetalRuntimeError
from numba_metal.runtime.device import check_capable


class _MetalContext:
    """Lazily-initialized holder for the MTLDevice and MTLCommandQueue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._device = None
        self._queue = None
        self._info = None

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

    def synchronize(self) -> None:
        """Block until all previously committed command buffers complete.

        Implemented by committing a barrier command buffer with no work and
        waiting on it; because a single serial MTLCommandQueue executes
        command buffers in submission order, waiting on this one guarantees
        everything submitted before it has finished.
        """
        self._ensure_initialized()
        cmdbuf = self._queue.commandBuffer()
        cmdbuf.commit()
        cmdbuf.waitUntilCompleted()
        status = cmdbuf.status()
        # MTLCommandBufferStatusError == 5
        if status == 5:
            err = cmdbuf.error()
            raise MetalRuntimeError(f"Metal command buffer failed: {err}")


_context = _MetalContext()


def get_context() -> _MetalContext:
    """Return the process-wide Metal context, initializing it if needed."""
    return _context
