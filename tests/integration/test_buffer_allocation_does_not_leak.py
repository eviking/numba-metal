"""Regression test for a real native-memory leak found and fixed in
``runtime/array.py``'s ``_alloc_buffer``: ``MTLDevice.newBufferWithLength_options_``
is a Cocoa `new`-family factory method, so its caller owns an extra +1
reference beyond the one PyObjC's own proxy already holds -- verified
directly via ``buf.retainCount()`` reading 2 immediately after creation,
with no `.release()`, this extra +1 is never balanced by anything and is
permanently leaked at the Metal-driver level on every single call whose
buffer is actually written to. See ``_alloc_buffer``'s docstring for the
full root-cause investigation, including why a superficially similar fix
attempted on ``MTLCommandQueue.commandBuffer()``/
``MTLCommandBuffer.computeCommandEncoder()`` turned out to be UNSAFE
(a real segfault under concurrent use) despite an identical
``retainCount() == 2`` reading, and was deliberately NOT applied there.

Uses ``MTLDevice.currentAllocatedSize()`` -- Apple's own live-allocation
counter, not a Python-side proxy for it -- as the authoritative signal,
since Python-level reference counting and even process RSS can both be
misleading here (RSS specifically was observed to only grow proportional
to how much of a buffer was actually WRITTEN to, not merely allocated,
since untouched pages may never be committed at the OS level at all).

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def test_repeated_to_device_does_not_leak_metal_allocation():
    from numba_metal import metal
    from numba_metal.runtime.context import get_context

    device = get_context().device
    nbytes = 1_000_000
    data = np.random.default_rng(0).random(nbytes // 4).astype(np.float32)

    # One warm-up call so any one-time, genuinely-retained process-wide
    # state (e.g. the scalar-buffer pool reaching its steady-state size)
    # doesn't get misread as part of the per-call leak this test targets.
    d = metal.to_device(data)
    del d

    baseline = device.currentAllocatedSize()
    for _ in range(20):
        d = metal.to_device(data)
        _ = d.copy_to_host()
        del d

    after = device.currentAllocatedSize()
    growth = after - baseline
    assert growth < nbytes, (
        f"MTLDevice.currentAllocatedSize() grew by {growth} bytes over 20 "
        f"repeated to_device()/copy_to_host()/drop cycles of a "
        f"{nbytes}-byte array (baseline={baseline}, after={after}) -- "
        f"expected it to stay flat (well under one buffer's worth of "
        f"growth) once every buffer from this loop has been dereferenced "
        f"and garbage collected. This is the exact native-memory leak "
        f"_alloc_buffer's docstring documents; if this test starts "
        f"failing, the buf.release() fix there may have been reverted or "
        f"a new unreleased `new`-family Metal allocation was introduced."
    )


def test_repeated_device_array_does_not_leak_metal_allocation():
    """Same check via metal.device_array (no host data involved at all),
    confirming the leak (and the fix) is in allocation/write, not
    anything specific to to_device's host->device copy path."""
    from numba_metal import metal
    from numba_metal.runtime.context import get_context

    device = get_context().device
    n = 250_000  # 1,000,000 bytes at float32

    d = metal.device_array(n, np.float32)
    del d

    baseline = device.currentAllocatedSize()
    for _ in range(20):
        d = metal.device_array(n, np.float32)
        d.copy_to_device(np.ones(n, dtype=np.float32))
        del d

    after = device.currentAllocatedSize()
    growth = after - baseline
    assert growth < n * 4, (
        f"MTLDevice.currentAllocatedSize() grew by {growth} bytes over 20 "
        f"repeated device_array()/copy_to_device()/drop cycles "
        f"(baseline={baseline}, after={after})."
    )
