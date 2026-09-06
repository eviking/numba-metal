"""Benchmark: process-wide @metal.device_func compile-time caching.

Not a workload benchmark like the others in this directory (no CPU-vs-
-Metal runtime comparison) -- this measures compiler infrastructure: the
COLD-COMPILE-TIME cost of sharing one `@metal.device_func` across many
different kernels, before vs. after
`msl_backend.py`'s process-wide `_device_function_compile_cache`.

Motivation: `heat_diffusion.py`'s own device-function experiment (see
that file's "Round 3") found that factoring a small, hot-loop helper
into a device function has a real, measured RUNTIME cost (2.3-2.6x per
-iteration slowdown) that a code-organization argument alone doesn't
justify. Before concluding "there is no real benefit to
@metal.device_func at all," three more mechanisms were tried and
measured directly rather than assumed:

1. More work per call, more calls per iteration (varying 4-24 calls of
   a ~10-op helper): stayed within +-15% of parity at every size tried,
   no clear win.
2. The same, at larger array sizes: noise-level parity.
3. THIS ONE: does a device function shared by many kernels save total
   COLD-COMPILE time (not runtime), since compiling it once instead of
   once per kernel skips redundant Numba-frontend + MSL-lowering work?
   Verified: YES, once the cache was actually implemented correctly
   (see msl_backend.py's `_device_function_compile_cache` module
   docstring for two real bugs found and fixed while building it: a
   missing-transitive-dependency bug and a duplicate-MSL-definition
   bug, both confirmed via dedicated tests in
   tests/integration/test_device_functions.py before trusting any
   timing number here).

This is a genuinely different use case from heat_diffusion.py's: a
library of small numerical helpers (special functions, distance
metrics, activation functions) reused across many DIFFERENT kernels in
an application, where the win is measured once at import/warm-up time,
not per-launch.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import require_metal_or_skip

N_KERNELS = 30


def _make_shared_helper():
    from numba_metal import metal

    @metal.device_func
    def shared_helper(y):
        a = y * y * y - 2.0 * y * y + 3.0 * y - 1.0
        b = math.sin(y) * math.cos(y) + math.exp(-y * y * 0.01)
        c = a * b + math.sqrt(abs(y) + 1.0)
        return c

    return shared_helper


def _make_inline_kernel(idx: int, offset: float):
    """A kernel with the shared helper's logic INLINED directly (a
    distinct, per-kernel copy of the Python source, so each one is a
    genuinely fresh, never-before-compiled function -- not sharing
    numba-metal's own compile cache across the group being measured)."""
    src = f"""
import math
from numba_metal import metal

@metal.jit
def kernel_inline_{idx}(xs, out, extra_offset):
    i = metal.grid(1)
    if i < out.size:
        y = xs[i] + {offset!r} + extra_offset
        a = y * y * y - 2.0 * y * y + 3.0 * y - 1.0
        b = math.sin(y) * math.cos(y) + math.exp(-y * y * 0.01)
        c = a * b + math.sqrt(abs(y) + 1.0)
        out[i] = c
"""
    ns: dict = {}
    exec(src, ns)  # noqa: S102 - deliberately generating distinct kernel fns
    return ns[f"kernel_inline_{idx}"]


def _make_devfunc_kernel(idx: int, offset: float, shared_helper):
    """A kernel calling the SAME shared @metal.device_func -- this is
    the case `_device_function_compile_cache` is meant to speed up."""
    src = f"""
import math
from numba_metal import metal

def _make(shared_helper):
    @metal.jit
    def kernel_devfunc_{idx}(xs, out, extra_offset):
        i = metal.grid(1)
        if i < out.size:
            y = xs[i] + {offset!r} + extra_offset
            out[i] = shared_helper(y)
    return kernel_devfunc_{idx}
"""
    ns: dict = {}
    exec(src, ns)  # noqa: S102 - deliberately generating distinct kernel fns
    return ns["_make"](shared_helper)


def _cold_compile_and_run_all(kernels, xs, out) -> float:
    from numba_metal import metal

    t0 = time.perf_counter()
    for k in kernels:
        k[1, 256](xs, out, np.float32(0.0))
    metal.synchronize()
    return time.perf_counter() - t0


def run(n_kernels: int = N_KERNELS, trials: int = 3) -> dict:
    metal_available = require_metal_or_skip()
    if not metal_available:
        return {"available": False}

    from numba_metal import metal

    n = 256
    xs = metal.to_device(np.zeros(n, dtype=np.float32))
    out = metal.device_array(n, np.float32)

    inline_times = []
    devfunc_times = []
    for trial in range(trials):
        # Freshly generated Python function objects every trial (via
        # exec()), so each trial's kernels are genuinely cold -- never
        # seen by numba-metal's own KernelCache before.
        fresh_inline = [
            _make_inline_kernel(i * 1000 + trial, float(i)) for i in range(n_kernels)
        ]
        shared_helper = _make_shared_helper()
        fresh_devfunc = [
            _make_devfunc_kernel(i * 1000 + trial, float(i), shared_helper)
            for i in range(n_kernels)
        ]
        inline_times.append(_cold_compile_and_run_all(fresh_inline, xs, out))
        devfunc_times.append(_cold_compile_and_run_all(fresh_devfunc, xs, out))

    inline_times.sort()
    devfunc_times.sort()
    median_inline = inline_times[len(inline_times) // 2]
    median_devfunc = devfunc_times[len(devfunc_times) // 2]
    return {
        "available": True,
        "n_kernels": n_kernels,
        "inline_cold_compile_s": median_inline,
        "devfunc_cold_compile_s": median_devfunc,
        "speedup": median_inline / median_devfunc,
    }


if __name__ == "__main__":
    result = run()
    if not result["available"]:
        print("Metal not available on this machine; skipping.")
    else:
        print(
            f"Cold-compiling {result['n_kernels']} kernels, each calling the "
            f"same @metal.device_func vs. each inlining an equivalent copy:\n"
            f"  inline (no sharing):        "
            f"{result['inline_cold_compile_s'] * 1000:8.2f}ms\n"
            f"  device_func (shared, cached): "
            f"{result['devfunc_cold_compile_s'] * 1000:8.2f}ms\n"
            f"  speedup: {result['speedup']:.2f}x"
        )
