"""The `advisor_workload()` contract: what a target script must expose
for `numba-metal advisor compare` to run a real, measured CPU-vs-Metal
comparison against it.

This is a deliberately small, explicit contract (a single dataclass of
callables) rather than the advisor trying to auto-detect "the CPU
version" and "the Metal version" of a function via static analysis --
that kind of guessing is exactly the kind of unsupported speculation the
governing spec forbids ("Never claim that Metal is faster based only on
static analysis"). The script author states explicitly which callable is
which; the advisor only measures and compares what it's told to.

See docs/advisor.md's worked example (benchmarks/mandelbrot.py) for a
full `advisor_workload()` implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class AdvisorWorkload:
    """Returned by a target script's `advisor_workload()` function.

    `qualified_name`: display name for this workload (e.g.
        "mandelbrot.compute_metal").
    `cpu_fn`: zero-argument callable that runs the CPU reference
        implementation once. Required.
    `metal_warm_fn`: zero-argument callable that launches an
        ALREADY-COMPILED Metal kernel once and calls
        `metal.synchronize()` before returning. Required.
    `get_cpu_result` / `get_metal_result`: zero-argument callables
        returning the CPU/Metal result arrays for correctness comparison
        (called once each, AFTER `cpu_fn`/`metal_warm_fn` have run at
        least once). Optional -- if either is None, no correctness check
        is performed and `compare` will not recommend USE_METAL (see
        recommendations.py's correctness gate).
    `py_func_for_cold` / `arg_types_for_cold`: the plain (undecorated)
        kernel function and its Numba argument-type tuple, used to
        measure a genuinely cold compilation (see comparison.py's
        `time_metal_cold`). Optional -- if either is None, no COLD
        regime is measured.
    """

    qualified_name: str
    cpu_fn: Callable[[], None]
    metal_warm_fn: Callable[[], None]
    get_cpu_result: Callable[[], object] | None = None
    get_metal_result: Callable[[], object] | None = None
    py_func_for_cold: Callable | None = None
    arg_types_for_cold: tuple | None = None
