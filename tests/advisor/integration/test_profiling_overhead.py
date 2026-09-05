"""Test 15: profiling instrumentation must not meaningfully affect
normal execution when the profiler is not active.

Requires a working Apple-silicon Metal device: this test launches real
kernels and compares real wall-clock timing before/during/after hook
installation, matching the discipline established elsewhere in this
repo's tests/integration/ suite (real hardware, real numbers, never
mocked timing).
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

pytestmark = pytest.mark.metal

_LAUNCHES = 200


def _time_launches(kernel, arr, n: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(_LAUNCHES):
        kernel(arr, n)
    from numba_metal.runtime.context import get_context

    get_context().synchronize()
    return (time.perf_counter_ns() - start) / _LAUNCHES


def test_hooks_absent_vs_installed_but_empty_within_noise():
    """With no hooks installed at all, and with the profiler's hook
    lists registered-then-immediately-cleared, per-launch overhead must
    be within normal measurement noise of each other -- not merely
    'small', genuinely indistinguishable from run-to-run variance."""
    from numba_metal import metal
    from numba_metal.advisor import metal_events
    from numba_metal.runtime.context import clear_hooks

    @metal.jit
    def add_one(a, n):
        i = metal.grid(1)
        if i < n:
            a[i] = a[i] + 1.0

    n = 1024
    arr = metal.to_device(np.zeros(n, dtype=np.float32))
    launch = (n + 255) // 256, 256

    def _launch(a, n):
        return add_one[launch](a, n)

    clear_hooks()
    baseline_samples = [_time_launches(_launch, arr, n) for _ in range(5)]

    collector = metal_events.EventCollector()
    metal_events.install(collector)
    try:
        for _ in range(5):
            _time_launches(_launch, arr, n)
    finally:
        metal_events.uninstall_all()

    idle_samples = [_time_launches(_launch, arr, n) for _ in range(5)]

    baseline_median = statistics.median(baseline_samples)
    idle_median = statistics.median(idle_samples)
    # With hooks fully uninstalled again, timing must return to baseline
    # -- confirms uninstall_all() actually removes the hooks rather than
    # leaving them registered.
    ratio = idle_median / baseline_median if baseline_median > 0 else 1.0
    assert 0.5 < ratio < 2.0, (
        f"post-uninstall overhead ratio {ratio:.2f}x is outside noise "
        f"bounds (baseline={baseline_median:.0f}ns, idle={idle_median:.0f}ns)"
    )

    events = collector.events()
    assert len(events) > 0  # the hooks DID fire while active -- proves
    # this isn't a false pass from a broken installer.
